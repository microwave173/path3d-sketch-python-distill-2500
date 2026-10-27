from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter

from .geometry import SampledStroke, normalize_strokes, sample_stroke
from .schema import Path3DScene


@dataclass(frozen=True)
class Camera:
    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    focal: float = 2.6
    projection: str = "perspective"


DEFAULT_CAMERAS = (
    Camera("front", (0.0, -3.6, 0.15)),
    Camera("side", (3.6, 0.0, 0.15)),
    Camera("top", (0.01, -0.01, 3.8), up=(0.0, 1.0, 0.0)),
    Camera("perspective", (2.8, -3.2, 2.4)),
)


def _camera_coordinates(points: np.ndarray, camera: Camera) -> np.ndarray:
    if not np.isfinite(points).all():
        raise ValueError("render geometry contains non-finite coordinates")
    position = np.asarray(camera.position, dtype=float)
    target = np.asarray(camera.target, dtype=float)
    up_hint = np.asarray(camera.up, dtype=float)
    forward = target - position
    if not np.isfinite(position).all() or not np.isfinite(target).all() or not np.isfinite(up_hint).all():
        raise ValueError("camera contains non-finite coordinates")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - position
    # Explicit reductions avoid platform BLAS warnings for many tiny 3D polylines.
    result = np.stack((np.sum(relative * right, axis=1), np.sum(relative * up, axis=1), np.sum(relative * forward, axis=1)), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("camera transform produced non-finite coordinates")
    return result


def _project(points: np.ndarray, camera: Camera, width: int, height: int, margin: float) -> tuple[np.ndarray, np.ndarray]:
    camera_points = _camera_coordinates(points, camera)
    depth = np.maximum(camera_points[:, 2], 1e-4)
    projected = camera_points[:, :2] if camera.projection == "orthographic" else camera.focal * camera_points[:, :2] / depth[:, None]
    scale = min(width, height) * (0.5 - margin)
    pixels = np.empty_like(projected)
    pixels[:, 0] = width / 2.0 + projected[:, 0] * scale
    pixels[:, 1] = height / 2.0 - projected[:, 1] * scale
    return pixels, depth


def _rgba(color: str, opacity: float) -> tuple[int, int, int, int]:
    rgb = ImageColor.getrgb(color)
    return rgb[0], rgb[1], rgb[2], round(255 * opacity)


def _is_glow_color(color: str) -> bool:
    red, green, blue = ImageColor.getrgb(color)
    return max(red, green, blue) >= 140 and max(red, green, blue) - min(red, green, blue) >= 45


def view_metrics(scene: Path3DScene, camera: Camera, *, width: int = 512, height: int = 512, margin: float = 0.12, normalize: bool = True) -> dict:
    sampled = tuple(sample_stroke(stroke) for stroke in scene.strokes)
    if normalize:
        sampled = normalize_strokes(sampled)
    points = np.concatenate([line for item in sampled for line in item.polylines], axis=0)
    camera_points = _camera_coordinates(points, camera)
    pixels, _ = _project(points, camera, width, height, margin)
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    clipped = {"left": bool(low[0] < 0), "right": bool(high[0] >= width), "top": bool(low[1] < 0), "bottom": bool(high[1] >= height)}
    visible_low = np.maximum(low, [0, 0])
    visible_high = np.minimum(high, [width, height])
    visible_size = np.maximum(visible_high - visible_low, 0)
    return {
        "camera": camera.name,
        "projected_bounds": {"min": low.tolist(), "max": high.tolist(), "width": float(high[0] - low[0]), "height": float(high[1] - low[1])},
        "width_fraction": float(visible_size[0] / width),
        "height_fraction": float(visible_size[1] / height),
        "area_fraction": float((visible_size[0] * visible_size[1]) / (width * height)),
        "clipped": clipped,
        "behind_camera_fraction": float(np.mean(camera_points[:, 2] <= 0)),
    }


def render_view(
    scene: Path3DScene,
    camera: Camera,
    output_path: str | Path,
    *,
    width: int = 512,
    height: int = 512,
    supersample: int = 3,
    margin: float = 0.12,
    normalize: bool = True,
    subtle_glow: bool = False,
) -> Path:
    sampled = tuple(sample_stroke(stroke) for stroke in scene.strokes)
    if normalize:
        sampled = normalize_strokes(sampled)
    render_width, render_height = width * supersample, height * supersample
    layers: list[tuple[float, SampledStroke, np.ndarray]] = []
    for item in sampled:
        for line in item.polylines:
            pixels, depths = _project(line, camera, render_width, render_height, margin)
            layers.append((float(depths.mean()), item, pixels))
    layers.sort(key=lambda layer: layer[0], reverse=True)

    image = Image.new("RGBA", (render_width, render_height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    if subtle_glow:
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        for _, item, pixels in layers:
            if not _is_glow_color(item.stroke.stroke):
                continue
            coordinates = [(float(x), float(y)) for x, y in pixels]
            rgb = ImageColor.getrgb(item.stroke.stroke)
            glow_draw.line(coordinates, fill=(*rgb, round(80 * item.stroke.opacity)), width=max(2, round(item.stroke.stroke_width * supersample * 3)), joint="curve")
        glow = glow.filter(ImageFilter.GaussianBlur(radius=3.5 * supersample))
        image.alpha_composite(glow)
    for _, item, pixels in layers:
        coordinates = [(float(x), float(y)) for x, y in pixels]
        line_width = max(1, round(item.stroke.stroke_width * supersample))
        draw.line(
            coordinates,
            fill=_rgba(item.stroke.stroke, item.stroke.opacity),
            width=line_width,
            joint="curve",
        )
    image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def render_scene_views(
    scene: Path3DScene,
    output_dir: str | Path,
    *,
    width: int = 512,
    height: int = 512,
    cameras: tuple[Camera, ...] = DEFAULT_CAMERAS,
    normalize: bool = True,
    subtle_glow: bool = False,
) -> tuple[Path, ...]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        render_view(scene, camera, output / f"view_{camera.name}.png", width=width, height=height, normalize=normalize, subtle_glow=subtle_glow)
        for camera in cameras
    )
    _contact_sheet(paths, output / "contact_sheet.png", width=width, height=height)
    return paths


def _contact_sheet(paths: tuple[Path, ...], output: Path, *, width: int, height: int) -> None:
    sheet = Image.new("RGB", (width * 2, height * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths[:4]):
        image = Image.open(path).convert("RGB")
        x = (index % 2) * width
        y = (index // 2) * height
        sheet.paste(image, (x, y))
        draw.rectangle((x + 8, y + 8, x + 132, y + 34), fill="white")
        draw.text((x + 14, y + 12), path.stem.removeprefix("view_"), fill="#111111")
    sheet.save(output)
