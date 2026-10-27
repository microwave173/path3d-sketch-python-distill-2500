from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

from .generator import Path3DGenerator
from .renderer import render_scene_views


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-shot Path3D generator and renderer")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    started = perf_counter()
    try:
        args = build_parser().parse_args(argv)
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        args.output.mkdir(parents=True)
        generator = Path3DGenerator(model=args.model)
        raw = generator.generate_raw(args.prompt)
        (args.output / "raw_response.txt").write_text(raw, encoding="utf-8")
        scene = generator.parse_response(raw, prompt=args.prompt)
        (args.output / "scene.json").write_text(scene.to_json(), encoding="utf-8")
        paths = render_scene_views(scene, args.output / "views", width=args.width, height=args.height)
        result = {
            "status": "complete",
            "scene": str((args.output / "scene.json").resolve()),
            "views": [str(path.resolve()) for path in paths],
            "contact_sheet": str((args.output / "views" / "contact_sheet.png").resolve()),
            "stroke_count": len(scene.strokes),
            "run_wall_seconds": perf_counter() - started,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }, ensure_ascii=False, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    sys.exit(main())
