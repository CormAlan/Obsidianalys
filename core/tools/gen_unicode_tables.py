#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genererar core/src/unicode_tables.hpp ur Pythons egen Unicode-databas.

C++-kärnan ska ge exakt samma siffror som Python-versionen, så tabellerna
beskriver precis det Python gör:
- \\w  (ordtecken i re, Unicode-läge)
- \\s  (blanktecken, samma som str.isspace)
- str.lower()
- str.casefold()

Kör: python3 core/tools/gen_unicode_tables.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

MAX_CP = 0x110000
OUT = Path(__file__).resolve().parent.parent / "src" / "unicode_tables.hpp"

WORD_RE = re.compile(r"\w")
SPACE_RE = re.compile(r"\s")


def is_surrogate(c: int) -> bool:
    return 0xD800 <= c <= 0xDFFF


def ranges(pred) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for c in range(MAX_CP):
        ok = not is_surrogate(c) and pred(chr(c))
        if ok and start is None:
            start = c
        elif not ok and start is not None:
            out.append((start, c - 1))
            start = None
    if start is not None:
        out.append((start, MAX_CP - 1))
    return out


def mappings(fn) -> list[tuple[int, list[int]]]:
    out = []
    for c in range(MAX_CP):
        if is_surrogate(c):
            continue
        ch = chr(c)
        m = fn(ch)
        if m != ch:
            assert 1 <= len(m) <= 3, (hex(c), m)
            out.append((c, [ord(x) for x in m]))
    return out


def fmt_ranges(name: str, rs: list[tuple[int, int]]) -> str:
    body = ",\n".join(f"    {{0x{lo:X}, 0x{hi:X}}}" for lo, hi in rs)
    return f"inline constexpr Range {name}[] = {{\n{body}\n}};\n"


def fmt_maps(name: str, ms: list[tuple[int, list[int]]]) -> str:
    rows = []
    for c, to in ms:
        padded = to + [0] * (3 - len(to))
        rows.append(f"    {{0x{c:X}, {{0x{padded[0]:X}, 0x{padded[1]:X}, 0x{padded[2]:X}}}, {len(to)}}}")
    body = ",\n".join(rows)
    return f"inline constexpr Mapping {name}[] = {{\n{body}\n}};\n"


def main() -> None:
    word = ranges(lambda ch: WORD_RE.match(ch) is not None)
    space = ranges(lambda ch: ch.isspace())
    # re:s \s och str.isspace ska vara samma mängd – annars stämmer inte strip() mot \s+
    assert space == ranges(lambda ch: SPACE_RE.match(ch) is not None)
    lower = mappings(str.lower)
    casefold = mappings(str.casefold)

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    parts = [
        f"// Genererad av core/tools/gen_unicode_tables.py (Python {py}, Unicode {unicodedata.unidata_version}).\n",
        "// Redigera inte för hand – kör skriptet igen.\n",
        "#pragma once\n\n",
        "namespace uni {\n\n",
        "struct Range { char32_t lo, hi; };\n",
        "struct Mapping { char32_t from; char32_t to[3]; unsigned char n; };\n\n",
        "// \\w i Pythons re (Unicode)\n",
        fmt_ranges("kWord", word),
        "\n// \\s i Pythons re == str.isspace()\n",
        fmt_ranges("kSpace", space),
        "\n// str.lower() (endast tecken som ändras)\n",
        fmt_maps("kLower", lower),
        "\n// str.casefold() (endast tecken som ändras)\n",
        fmt_maps("kCasefold", casefold),
        "\n}  // namespace uni\n",
    ]
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Skrev {OUT} – word={len(word)} space={len(space)} lower={len(lower)} casefold={len(casefold)}")


if __name__ == "__main__":
    main()
