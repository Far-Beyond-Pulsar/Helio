#!/usr/bin/env python3
"""Generate Rust constants from Eric Lengyel's official Transvoxel.cpp.

The generated file is checked in so normal Helio builds never require Python,
network access, or the upstream C++ source. Re-run this tool only when auditing
or deliberately updating the pinned upstream revision.
"""

from __future__ import annotations

import argparse
import pathlib
import re


TABLES = (
    "regularCellClass",
    "regularCellData",
    "regularVertexData",
    "transitionCellClass",
    "transitionCellData",
    "transitionCornerData",
    "transitionVertexData",
)


def initializer(source: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*\[[^=;]+\]\s*=\s*\{{", source)
    if match is None:
        raise ValueError(f"could not find initializer for {name}")
    start = source.index("{", match.start())
    depth = 0
    for offset, character in enumerate(source[start:], start=start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise ValueError(f"unterminated initializer for {name}")


TOKEN = re.compile(r"\s*(\{|\}|,|0[xX][0-9a-fA-F]+|[0-9]+)")


def parse_initializer(text: str) -> list[object]:
    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = TOKEN.match(text, position)
        if match is None:
            raise ValueError(f"unexpected initializer text at {text[position:position + 32]!r}")
        tokens.append(match.group(1))
        position = match.end()

    cursor = 0

    def value() -> object:
        nonlocal cursor
        token = tokens[cursor]
        if token != "{":
            cursor += 1
            return int(token, 0)
        cursor += 1
        values: list[object] = []
        while tokens[cursor] != "}":
            values.append(value())
            if tokens[cursor] == ",":
                cursor += 1
                if tokens[cursor] == "}":
                    break
        cursor += 1
        return values

    parsed = value()
    if cursor != len(tokens) or not isinstance(parsed, list):
        raise ValueError("initializer did not parse as one complete array")
    return parsed


def numbers(values: list[object]) -> list[int]:
    if not all(isinstance(value, int) for value in values):
        raise ValueError("expected a flat numeric initializer")
    return [int(value) for value in values]


def padded(values: list[object], length: int) -> list[int]:
    row = numbers(values)
    if len(row) > length:
        raise ValueError(f"row has {len(row)} entries, expected at most {length}")
    return row + [0] * (length - len(row))


def rust_scalar(name: str, rust_type: str, values: list[int], width: int) -> str:
    rendered = []
    for offset in range(0, len(values), 16):
        chunk = values[offset : offset + 16]
        rendered.append("    " + ", ".join(f"0x{value:0{width}X}" for value in chunk) + ",")
    return (
        "#[rustfmt::skip]\n"
        f"pub const {name}: [{rust_type}; {len(values)}] = [\n"
        + "\n".join(rendered)
        + "\n];\n"
    )


def rust_matrix(
    name: str, rust_type: str, rows: list[list[int]], columns: int, width: int
) -> str:
    rendered = []
    for row in rows:
        values = ", ".join(f"0x{value:0{width}X}" for value in row)
        rendered.append(f"    [{values}],")
    return (
        "#[rustfmt::skip]\n"
        f"pub const {name}: [[{rust_type}; {columns}]; {len(rows)}] = [\n"
        + "\n".join(rendered)
        + "\n];\n"
    )


def cell_data(values: list[object], count: int, indices: int) -> tuple[list[int], list[list[int]]]:
    if len(values) != count:
        raise ValueError(f"cell data has {len(values)} rows, expected {count}")
    geometry: list[int] = []
    topology: list[list[int]] = []
    for value in values:
        if not isinstance(value, list) or len(value) != 2 or not isinstance(value[0], int):
            raise ValueError("malformed cell data row")
        if not isinstance(value[1], list):
            raise ValueError("malformed cell index row")
        geometry.append(value[0])
        topology.append(padded(value[1], indices))
    return geometry, topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    parsed = {name: parse_initializer(initializer(source, name)) for name in TABLES}

    regular_class = numbers(parsed["regularCellClass"])
    regular_geometry, regular_indices = cell_data(parsed["regularCellData"], 16, 15)
    regular_vertices = [padded(row, 12) for row in parsed["regularVertexData"]]
    transition_class = numbers(parsed["transitionCellClass"])
    transition_geometry, transition_indices = cell_data(parsed["transitionCellData"], 56, 36)
    transition_corners = numbers(parsed["transitionCornerData"])
    transition_vertices = [padded(row, 12) for row in parsed["transitionVertexData"]]

    expected = {
        "regularCellClass": (len(regular_class), 256),
        "regularVertexData": (len(regular_vertices), 256),
        "transitionCellClass": (len(transition_class), 512),
        "transitionCornerData": (len(transition_corners), 13),
        "transitionVertexData": (len(transition_vertices), 512),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f"{name} has {actual} rows, expected {wanted}")

    sections = [
        "// @generated by tools/generate_transvoxel_tables.py; do not edit by hand.\n",
        "//\n",
        "// Source: https://github.com/EricLengyel/Transvoxel\n",
        f"// Revision: {args.revision}\n",
        "// Copyright (c) 2009 Eric Lengyel; used under the MIT License.\n",
        "// See ../../../LICENSES/Transvoxel-MIT.txt.\n\n",
        f'pub const TRANSVOXEL_UPSTREAM_REVISION: &str = "{args.revision}";\n\n',
        rust_scalar("REGULAR_CELL_CLASS", "u8", regular_class, 2),
        "\n",
        rust_scalar("REGULAR_CELL_GEOMETRY_COUNTS", "u8", regular_geometry, 2),
        "\n",
        rust_matrix("REGULAR_CELL_VERTEX_INDEX", "u8", regular_indices, 15, 2),
        "\n",
        rust_matrix("REGULAR_VERTEX_DATA", "u16", regular_vertices, 12, 4),
        "\n",
        rust_scalar("TRANSITION_CELL_CLASS", "u8", transition_class, 2),
        "\n",
        rust_scalar("TRANSITION_CELL_GEOMETRY_COUNTS", "u8", transition_geometry, 2),
        "\n",
        rust_matrix("TRANSITION_CELL_VERTEX_INDEX", "u8", transition_indices, 36, 2),
        "\n",
        rust_scalar("TRANSITION_CORNER_DATA", "u8", transition_corners, 2),
        "\n",
        rust_matrix("TRANSITION_VERTEX_DATA", "u16", transition_vertices, 12, 4),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(sections), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
