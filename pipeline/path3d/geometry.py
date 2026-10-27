from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parser import parse_path3d
from .schema import Path3DStroke


@dataclass(frozen=True)
class SampledStroke:
    stroke: Path3DStroke
    polylines: tuple[np.ndarray, ...]


def _quadratic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps + 1)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _cubic(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    steps: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps + 1)[:, None]
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )


def sample_stroke(stroke: Path3DStroke, *, curve_steps: int = 24) -> SampledStroke:
    commands = parse_path3d(stroke.path)
    polylines: list[np.ndarray] = []
    current: np.ndarray | None = None
    start: np.ndarray | None = None
    points: list[np.ndarray] = []

    def flush() -> None:
        nonlocal points
        if len(points) >= 2:
            polylines.append(np.stack(points))
        points = []

    for item in commands:
        values = np.asarray(item.values, dtype=float)
        if item.command == "M":
            flush()
            current = values[:3]
            start = current.copy()
            points = [current.copy()]
        elif item.command == "L":
            current = values[:3]
            points.append(current.copy())
        elif item.command == "Q":
            if current is None:
                raise ValueError("Q requires a current point")
            segment = _quadratic(current, values[:3], values[3:6], curve_steps)
            points.extend(segment[1:])
            current = values[3:6]
        elif item.command == "C":
            if current is None:
                raise ValueError("C requires a current point")
            segment = _cubic(current, values[:3], values[3:6], values[6:9], curve_steps)
            points.extend(segment[1:])
            current = values[6:9]
        elif item.command == "Z":
            if start is not None and current is not None and not np.allclose(current, start):
                points.append(start.copy())
            current = start.copy() if start is not None else None
            flush()
            start = None
    flush()
    if not polylines:
        raise ValueError(f"stroke {stroke.id!r} contains no drawable segment")
    return SampledStroke(stroke, tuple(polylines))


def normalize_strokes(strokes: tuple[SampledStroke, ...], *, extent: float = 1.6) -> tuple[SampledStroke, ...]:
    all_points = np.concatenate([line for stroke in strokes for line in stroke.polylines], axis=0)
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    center = (low + high) / 2.0
    span = float(np.max(high - low)) or 1.0
    scale = extent / span
    return tuple(
        SampledStroke(item.stroke, tuple((line - center) * scale for line in item.polylines))
        for item in strokes
    )
