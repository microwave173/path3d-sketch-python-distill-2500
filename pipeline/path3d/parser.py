from __future__ import annotations

import math
import re

from .schema import Path3DCommand


_TOKEN = re.compile(r"Q3|C3|q3|c3|[MLQCZmlqcz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_ARITY = {"M": 3, "L": 3, "Q": 6, "Q3": 6, "C": 9, "C3": 9, "Z": 0}


def parse_path3d(path: str) -> tuple[Path3DCommand, ...]:
    """Parse an absolute Path3D string using SVG-like M/L/Q/C/Z commands."""
    tokens = _TOKEN.findall(path.replace(",", " "))
    residue = _TOKEN.sub("", path.replace(",", " "))
    if residue.strip():
        raise ValueError(f"invalid Path3D syntax near {residue.strip()!r}")
    if not tokens:
        raise ValueError("empty Path3D path")

    commands: list[Path3DCommand] = []
    index = 0
    while index < len(tokens):
        raw = tokens[index]
        if raw.upper() not in _ARITY:
            raise ValueError(f"expected Path3D command, found {raw!r}")
        if raw != raw.upper():
            raise ValueError("Path3D v1 supports absolute uppercase commands only")
        command = raw
        index += 1
        arity = _ARITY[command]
        if command == "Z":
            commands.append(Path3DCommand(command))
            continue

        group_index = 0
        while index < len(tokens) and tokens[index].upper() not in _ARITY:
            if index + arity > len(tokens):
                raise ValueError(f"command {command} requires groups of {arity} numeric values")
            raw_values = tokens[index:index + arity]
            if any(value.upper() in _ARITY for value in raw_values):
                raise ValueError(f"command {command} requires groups of {arity} numeric values")
            values = tuple(float(value) for value in raw_values)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Path3D coordinates must be finite")
            normalized = {"Q3": "Q", "C3": "C"}.get(command, command)
            effective = "L" if command == "M" and group_index else normalized
            commands.append(Path3DCommand(effective, values))
            index += arity
            group_index += 1
        if group_index == 0:
            raise ValueError(f"command {command} requires {arity} numeric values")

    if commands[0].command != "M":
        raise ValueError("Path3D path must begin with M")
    current_subpath = False
    for command in commands:
        if command.command == "M":
            current_subpath = True
        elif not current_subpath:
            raise ValueError(f"command {command.command} requires an active subpath")
        elif command.command == "Z":
            current_subpath = False
    return tuple(commands)
