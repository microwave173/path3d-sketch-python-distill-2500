from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from sketch_agent.model_config import get_reasoning_effort, reasoning_options

from .parser import parse_path3d
from .schema import Path3DScene


SPATIAL_S_CURVE_C3_EXAMPLE_PATH = "M -0.8 -0.5 -0.3 C3 -0.3 0.9 0.6 0.3 -0.9 -0.5 0.8 0.5 0.75"
SPATIAL_CIRCLE_Q3_EXAMPLE_PATH = "M 0.678 0.254 0.339 Q3 0.562 0.564 0.339 0.281 0.709 0.240 Q3 0.000 0.854 0.141 -0.281 0.749 0.000 Q3 -0.562 0.644 -0.141 -0.678 0.350 -0.240 Q3 -0.794 0.056 -0.339 -0.678 -0.254 -0.339 Q3 -0.562 -0.564 -0.339 -0.281 -0.709 -0.240 Q3 0.000 -0.854 -0.141 0.281 -0.749 0.000 Q3 0.562 -0.644 0.141 0.678 -0.350 0.240 Q3 0.794 -0.056 0.339 0.678 0.254 0.339"


SYSTEM_PROMPT = f"""You are a 3D spatial sketch artist. Draw by moving a virtual pen through three-dimensional space.

Balance beauty, coordination, prompt fidelity, and distinctive identity cues. Diagnose both excessive complexity and excessive simplification. Prefer small reversible steps. Preserve the minimum distinctive details that make the requested subject recognizable while using clean abstract curves when literal detail would harm coordination.

Use a Path3D language based on SVG path semantics:
- M x y z: move the pen without drawing.
- L x y z: draw a straight line to a 3D point.
- Q3 cx cy cz x y z: draw a quadratic 3D Bezier. Q3 always has exactly 6 numbers, grouped as two xyz triplets in this order: [control], [endpoint].
- C3 c1x c1y c1z c2x c2y c2z x y z: draw a cubic 3D Bezier. C3 always has exactly 9 numbers, grouped as three xyz triplets in this order: [control 1], [control 2], [endpoint]. Control 1 sets the direction leaving the current point; control 2 sets the direction approaching the endpoint.
- Z: close the current subpath back to its latest M point.
- Commands are absolute and uppercase. A path may contain multiple subpaths by using M again.

World axes: +x points right, +y points away from the viewer/deeper into the scene, and +z points up. Use real coordinates roughly in [-1, 1]. The renderer recenters and uniformly scales the whole scene, so prioritize correct relative geometry and genuine depth. Do not place every stroke on one plane when the target is three-dimensional.

Each stroke is one editable semantic part. Use continuous paths when one pen movement naturally expresses a contour, ring, frame, or connected part. Use a restrained number of clean strokes. Different strokes may meet at shared 3D coordinates.

Do not use the 2D SVG commands Q or C. Use the explicitly three-dimensional commands Q3 and C3.
Write the command again for every new curve segment. Never put an incomplete second segment after a Q3 or C3. Even when a curve lies in a plane, include all three coordinates in every control point and endpoint.

Choose the simplest curve with enough control:
- Prefer Q3 for circular arcs, wheels, rings, rounded cross-sections, and simple one-direction bends. Several explicit Q3 segments can form one smooth closed contour.
- Use C3 for S-shaped curves, tails, necks, flight paths, or a curve whose departure and arrival directions need independent control.
- Use L for genuinely straight structural members; do not approximate an intended smooth organic curve with a faceted chain of L segments.

Format-only Q3 example: a smooth closed contour made from eight consecutive segments. Every segment repeats Q3 and contains [one xyz control], [one xyz endpoint]:
{SPATIAL_CIRCLE_Q3_EXAMPLE_PATH}

Format-only C3 example: one non-planar spatial S curve. The C3 contains exactly [control 1], [control 2], [endpoint]; its two controls lie on opposite sides so the bend reverses:
{SPATIAL_S_CURVE_C3_EXAMPLE_PATH}

These examples teach syntax and curve selection only. Do not copy their subject or coordinates unless they fit the target.

Return only one valid JSON object with exactly this top-level structure:
{{"prompt":"...","strokes":[{{"id":"unique_id","path":"M ...","description":"...","stroke":"#111111","stroke_width":3,"opacity":1,"group":"component"}}],"metadata":{{}}}}
Do not use SVG tags, markdown fences, fill, surfaces, meshes, primitives, transforms, or camera commands."""


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip().removeprefix("```json").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model returned no JSON object")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


class Path3DGenerator:
    def __init__(self, *, model: str | None = None, client: Any | None = None) -> None:
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
        self.model = model or os.getenv("MODEL", "gpt-5.6-terra")
        self.reasoning_effort = get_reasoning_effort()
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))
        self.client = client

    def generate_raw(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Draw this as a coherent 3D spatial line sketch: {prompt}"},
            ],
            max_output_tokens=5000,
            **reasoning_options(self.reasoning_effort),
        )
        return response.output_text or ""

    def parse_response(self, raw: str, *, prompt: str) -> Path3DScene:
        scene = Path3DScene.from_dict(_json_object(raw), prompt=prompt)
        for stroke in scene.strokes:
            parse_path3d(stroke.path)
        return scene

    def generate(self, prompt: str) -> tuple[Path3DScene, str]:
        raw = self.generate_raw(prompt)
        return self.parse_response(raw, prompt=prompt), raw
