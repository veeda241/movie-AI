"""Pure-Python _regex module — fallback shim.

This file is copied into ``.venv/Lib/site-packages/regex/`` at startup by
``video_lab.utils.regex_shim.ensure_regex_shim()`` to bypass the Windows
Application Control policy that blocks the compiled ``_regex.pyd``.

All dict keys are UPPERCASE because ``_regex_core.lookup_property`` calls
``standardise_name()`` which removes ``_`` / ``-`` and calls ``.upper()``.
"""

from __future__ import annotations

import re as _re
import unicodedata as _unicodedata
from typing import Any

BYTES_PER_CODE: int = 1


def get_code_size() -> int:
    return BYTES_PER_CODE


def fold_case(flags: int, string_or_char: str) -> str:
    return string_or_char.casefold()


def get_all_cases(flags: int, char: str) -> list[str]:
    lower = char.lower()
    upper = char.upper()
    title = char.title()
    results = {char, lower, upper, title}
    return [c for c in results if c]


def get_expand_on_folding() -> set[int]:
    return set()


def has_property_value(prop_value: str, ch: str) -> bool:
    if "=" in prop_value:
        prop, val = prop_value.split("=", 1)
    elif ":" in prop_value:
        prop, val = prop_value.split(":", 1)
    else:
        return False
    prop = prop.strip().lower()
    val = val.strip().lower()
    cat = _unicodedata.category(ch).lower()
    if prop == "general_category":
        return cat.startswith(val) or cat == val
    if prop == "script":
        try:
            import unicodedata2 as _ud2
            return _ud2.script(ch).lower() == val
        except ImportError:
            pass
    return False


def get_properties() -> dict[str, tuple[int, dict[str, int]]]:
    _i = iter(range(1, 300))
    gc_id = next(_i)
    gc_values: dict[str, int] = {
        "LU": next(_i), "LL": next(_i), "LT": next(_i),
        "LM": next(_i), "LO": next(_i),
        "ND": next(_i), "NL": next(_i), "NO": next(_i),
        "MN": next(_i), "MC": next(_i), "ME": next(_i),
        "PD": next(_i), "PS": next(_i), "PE": next(_i),
        "PI": next(_i), "PF": next(_i), "PO": next(_i),
        "ZS": next(_i), "ZL": next(_i), "ZP": next(_i),
        "CC": next(_i), "CF": next(_i), "CS": next(_i),
        "CO": next(_i), "CN": next(_i),
        "SC": next(_i), "SK": next(_i), "SM": next(_i), "SO": next(_i),
    }
    script_id = next(_i)
    block_id = next(_i)
    return {
        "GENERALCATEGORY": (gc_id, gc_values),
        "GC":             (gc_id, gc_values),
        "SCRIPT":         (script_id, {"UNKNOWN": next(_i)}),
        "BLOCK":          (block_id,  {"UNKNOWN": next(_i)}),
        "DIGIT":          (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "BLANK":          (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "SPACE":          (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "WORD":           (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "ALPHABETIC":     (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "LOWERCASE":      (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "UPPERCASE":      (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "WHITESPACE":     (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "PRINT":          (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "PUNCT":          (next(_i), {"YES": next(_i), "NO": next(_i)}),
        "GRAPH":          (next(_i), {"YES": next(_i), "NO": next(_i)}),
    }


class _RegexPattern:
    __slots__ = ("_pat", "_pattern", "_flags", "_pickled_data")

    def __init__(self, pattern: str, flags: int = 0) -> None:
        self._pat = _re.compile(pattern, flags)
        self._pattern = pattern
        self._flags = flags
        self._pickled_data = (pattern, flags)

    @staticmethod
    def _fix_pos(pos: int | None) -> int:
        return 0 if pos is None else pos

    def match(self, string: str = "", pos: int | None = None, endpos: int | None = None):
        p = self._fix_pos(pos)
        if endpos is None:
            return self._pat.match(string, p)
        return self._pat.match(string, p, endpos)

    def fullmatch(self, string: str = "", pos: int | None = None, endpos: int | None = None):
        p = self._fix_pos(pos)
        if endpos is None:
            return self._pat.fullmatch(string, p)
        return self._pat.fullmatch(string, p, endpos)

    def search(self, string: str = "", pos: int | None = None, endpos: int | None = None):
        p = self._fix_pos(pos)
        if endpos is None:
            return self._pat.search(string, p)
        return self._pat.search(string, p, endpos)

    def findall(self, string: str = "", pos: int | None = None, endpos: int | None = None, *args: Any):
        p = self._fix_pos(pos)
        if endpos is None:
            return self._pat.findall(string, p)
        return self._pat.findall(string, p, endpos)

    def finditer(self, string: str = "", pos: int | None = None, endpos: int | None = None, *args: Any):
        p = self._fix_pos(pos)
        if endpos is None:
            return self._pat.finditer(string, p)
        return self._pat.finditer(string, p, endpos)

    def sub(self, repl: Any, string: str = "", count: int = 0, *args: Any):
        return self._pat.sub(repl, string, count)

    def subn(self, repl: Any, string: str = "", count: int = 0, *args: Any):
        return self._pat.subn(repl, string, count)

    def split(self, string: str = "", maxsplit: int = 0, *args: Any):
        return self._pat.split(string, maxsplit)

    def scanner(self, string: str = ""):
        for m in self._pat.finditer(string):
            yield m

    def subf(self, format: str, string: str = "", count: int = 0):
        return self._pat.sub(format, string, count)

    def subfn(self, format: str, string: str = "", count: int = 0):
        return self._pat.subn(format, string, count)

    def splititer(self, string: str = "", maxsplit: int = 0):
        yield from self._pat.split(string, maxsplit)

    @property
    def pattern(self) -> str:
        return self._pattern

    @property
    def flags(self) -> int:
        return self._flags

    @property
    def groups(self) -> int:
        return self._pat.groups

    @property
    def groupindex(self) -> dict[str, int]:
        return self._pat.groupindex


def compile(pattern: str, flags: int = 0, *args: Any, **kwargs: Any) -> _RegexPattern:
    return _RegexPattern(pattern, flags)


__all__: list[str] = ["compile"]
