"""Provenance manifest: one JSON line per processed image."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


START = '{"boottool_version": "'         # every record starts so (keys are written in sorted order)


class _End(Exception):
    pass


def _json_prefix(text: str) -> bool:
    """Could `text` be the start of a valid JSON value (cut off anywhere)?"""
    i = 0

    def peek():
        if i >= len(text):
            raise _End
        return text[i]

    def ws():
        nonlocal i
        while i < len(text) and text[i] in " \t\r\n":
            i += 1

    def lit(word):
        nonlocal i
        for ch in word:
            if peek() != ch:
                raise ValueError
            i += 1

    def string():
        nonlocal i
        lit('"')
        while True:
            ch = peek()
            i += 1
            if ch == '"':
                return
            if ch == "\\":
                esc = peek()
                i += 1
                if esc == "u":
                    for _ in range(4):
                        if peek() not in "0123456789abcdefABCDEF":
                            raise ValueError
                        i += 1
                elif esc not in '"\\/bfnrt':
                    raise ValueError
            elif ord(ch) < 0x20:
                raise ValueError

    def number():
        nonlocal i
        start = i
        while i < len(text) and text[i] in "+-0123456789.eE":
            i += 1
        if i >= len(text):                       # cut off: it must still be the start of a number
            if re.fullmatch(r"-?(?:(?:0|[1-9]\d*)(?:\.|\.\d+(?:[eE][+-]?\d*)?|[eE][+-]?\d*)?)?", text[start:]):
                raise _End
            raise ValueError
        if not re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?", text[start:i]):
            raise ValueError

    def value():
        nonlocal i
        ws()
        ch = peek()
        if ch == "{":
            i += 1
            ws()
            if peek() == "}":
                i += 1
                return
            while True:
                ws()
                string()
                ws()
                lit(":")
                value()
                ws()
                ch = peek()
                i += 1
                if ch == "}":
                    return
                if ch != ",":
                    raise ValueError
        elif ch == "[":
            i += 1
            ws()
            if peek() == "]":
                i += 1
                return
            while True:
                value()
                ws()
                ch = peek()
                i += 1
                if ch == "]":
                    return
                if ch != ",":
                    raise ValueError
        elif ch == '"':
            string()
        elif ch == "-" and text[i + 1:i + 2] in ("I", ""):     # Python writes -Infinity for -inf
            lit("-")
            lit("Infinity")
        elif ch in "-0123456789":
            number()
        else:
            for word in ("true", "false", "null", "NaN", "Infinity"):   # json.dumps writes NaN and Infinity
                if word.startswith(text[i:i + len(word)]) and text[i:i + 1] == word[0]:
                    lit(word)
                    return
            raise ValueError

    try:
        value()
    except _End:
        return True
    except ValueError:
        return False
    ws()
    return i >= len(text)                # a whole value: it would have parsed, so it isn't cut off


def _cut_record(text: str) -> bool:
    """Does this look like the start of one of our records, cut off?"""
    if START.startswith(text):
        return True
    if not (text.startswith(START) and _json_prefix(text)):
        return False
    return not re.search(r'"(source|status|created_at|svg)"\s*:\s*[^"\s]', text)   # these are always text


class Manifest:
    def __init__(self, path):
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            lines = list(enumerate(f, 1))
        last = max((n for n, line in lines if line.strip()), default=0)
        out = []
        for n, line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if n == last and not line.endswith("\n") and _cut_record(line):
                    break                                # a run stopped while writing its last line: ignore it
                raise ValueError(f"{self.path}: line {n} is damaged") from None
            if not (isinstance(rec, dict) and isinstance(rec.get("source"), str) and "boottool_version" in rec
                    and rec.get("status") in ("ok", "low_score", "error", "saving")):
                raise ValueError(f"{self.path} doesn't look like a boottool manifest (line {n})")
            out.append(rec)
        return out

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._drop_unfinished_line()
        with open(self.path, "ab") as f:
            f.write((json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))

    def _drop_unfinished_line(self) -> None:
        """A run stopped while writing leaves a last line without its newline: remove it if it is
        broken, or just end it if it is a whole record."""
        if not self.path.exists():
            return
        with open(self.path, "r+b") as f:
            size = f.seek(0, 2)
            if not size:
                return
            f.seek(size - 1)
            if f.read(1) == b"\n":
                return
            keep, pos = 0, size
            while pos > 0:
                step = min(65536, pos)
                pos -= step
                f.seek(pos)
                i = f.read(step).rfind(b"\n")
                if i >= 0:
                    keep = pos + i + 1
                    break
            f.seek(keep)
            raw = f.read()
            tail = raw.decode("utf-8", "replace")
            if not tail.strip():                     # only spaces after the last record: nothing to keep
                f.truncate(keep)
                return
            try:
                json.loads(raw.decode("utf-8"))
                f.seek(0, 2)
                f.write(b"\n")
            except (ValueError, UnicodeDecodeError):
                if not _cut_record(tail):                                      # not one of ours: leave it
                    raise ValueError(f"{self.path} doesn't look like a boottool manifest") from None
                f.truncate(keep)

    def done_keys(self) -> set[tuple[str, str]]:
        """(sha256, params_key) pairs already processed without error."""
        return {
            (r["sha256"], r["params_key"])
            for r in self.load()
            if r.get("status") in ("ok", "low_score") and r.get("sha256")
        }


def latest_by_source(records: list[dict]) -> list[dict]:
    """Keep the newest record per source file (later lines win)."""
    latest: dict[str, dict] = {}
    for r in records:
        latest[r["source"]] = r
    return list(latest.values())


def load_provenance(csv_path) -> dict[str, dict]:
    """CSV with a `file` column (path relative to the input dir) plus any of:
    license, source, author, year."""
    out: dict[str, dict] = {}
    if not csv_path:
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("file") or "").strip().replace("\\", "/")
            if key:
                out[key] = {k: (v or "").strip() for k, v in row.items() if k != "file"}
    return out
