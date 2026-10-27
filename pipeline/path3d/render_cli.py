from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import parse_path3d
from .renderer import render_scene_views
from .schema import Path3DScene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render an existing Path3D scene JSON")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args(argv)

    value = json.loads(args.scene.read_text(encoding="utf-8"))
    scene = Path3DScene.from_dict(value)
    for stroke in scene.strokes:
        parse_path3d(stroke.path)
    paths = render_scene_views(scene, args.output, width=args.width, height=args.height)
    print(json.dumps({
        "status": "complete",
        "views": [str(path.resolve()) for path in paths],
        "contact_sheet": str((args.output / "contact_sheet.png").resolve()),
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
