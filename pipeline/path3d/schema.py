from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


CommandName = Literal["M", "L", "Q", "C", "Z"]


@dataclass(frozen=True)
class Path3DCommand:
    command: CommandName
    values: tuple[float, ...] = ()


@dataclass(frozen=True)
class Path3DStroke:
    id: str
    path: str
    description: str
    stroke: str = "#111111"
    stroke_width: float = 3.0
    opacity: float = 1.0
    group: str = "default"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Path3DStroke":
        required = ("id", "path", "description")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"3D stroke missing required fields: {missing}")
        fields = cls.__dataclass_fields__
        stroke = cls(**{key: value[key] for key in fields if key in value})
        if not 0 < float(stroke.stroke_width) <= 64:
            raise ValueError(f"stroke {stroke.id!r} has invalid stroke_width")
        if not 0 <= float(stroke.opacity) <= 1:
            raise ValueError(f"stroke {stroke.id!r} has invalid opacity")
        return stroke


@dataclass(frozen=True)
class Path3DScene:
    prompt: str
    strokes: tuple[Path3DStroke, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, prompt: str = "") -> "Path3DScene":
        strokes = tuple(Path3DStroke.from_dict(item) for item in value.get("strokes", []))
        if not strokes:
            raise ValueError("Path3D scene requires at least one stroke")
        ids = [stroke.id for stroke in strokes]
        if len(ids) != len(set(ids)):
            raise ValueError("Path3D stroke IDs must be unique")
        return cls(
            prompt=str(value.get("prompt") or prompt).strip(),
            strokes=strokes,
            metadata=value.get("metadata") if isinstance(value.get("metadata"), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "strokes": [asdict(stroke) for stroke in self.strokes],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
