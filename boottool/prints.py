"""The Print library: finished prints saved from the Print tab, kept in one fixed folder.

The folder is ~/BootTool/Prints (set BOOTTOOL_PRINTS to use another place, mainly for tests).
Each print is a transparent PNG picture of the design (in its ink colours) plus a small JSON file
with its real size in millimetres. Simulate puts these on a shirt.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

from . import filelock

_SAFE = re.compile(r"[^A-Za-z0-9 _.()+\-]")
_turns = filelock.Turns()
LOCK = ".boottool-prints.lock"      # the folder's lock, shared with any other Boot Tool open on it (see filelock)


@contextlib.contextmanager
def _busy(change: bool = True):
    """One thing at a time touches the Print library, here (in the order asked) and in any other Boot
    Tool open on the same folder. Reading goes ahead even if another one keeps the folder for too long."""
    with _turns.held(folder() / LOCK, change):
        yield


MAX_PNG = 40 * 1024 * 1024


def folder() -> Path:
    p = Path(os.environ.get("BOOTTOOL_PRINTS") or (Path.home() / "BootTool" / "Prints"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_stem(name: str) -> str:
    stem = _SAFE.sub("_", Path(str(name)).stem).strip(" .")
    return (stem or "print")[:100]


def _unique(stem: str) -> str:
    d = folder()
    n = 1
    while True:
        s = stem if n == 1 else f"{stem}-{n}"
        if not (d / f"{s}.png").exists():
            return s
        n += 1


def save(name: str, png_b64: str, width_mm: float, height_mm: float, garment: str = "") -> str:
    data = base64.b64decode(png_b64.split(",")[-1], validate=False)
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("That is not a PNG picture.")
    if len(data) > MAX_PNG:
        raise ValueError("That picture is too large to store.")
    if not (1 <= float(width_mm) <= 5000 and 1 <= float(height_mm) <= 5000):
        raise ValueError("The design size is not valid.")
    with _busy():
        stem = _unique(clean_stem(name))
        d = folder()
        meta = {"width_mm": round(float(width_mm), 2), "height_mm": round(float(height_mm), 2),
                "garment": garment if re.fullmatch(r"#[0-9a-fA-F]{6}", garment or "") else "", "created": int(time.time()),
                "id": uuid.uuid4().hex}      # each save is its own print, even with the same picture and size
        # The size first, the picture last: a save that is cut short never leaves a picture without its size.
        (d / f"{stem}.json.tmp").write_text(json.dumps(meta), encoding="utf-8")
        (d / f"{stem}.json.tmp").replace(d / f"{stem}.json")
        (d / f"{stem}.png.tmp").write_bytes(data)
        (d / f"{stem}.png.tmp").replace(d / f"{stem}.png")
        return f"{stem}.png"


def _path(fname: str) -> Path:
    if not fname or fname != Path(fname).name or not fname.lower().endswith(".png") or fname.startswith("."):
        raise ValueError("bad file name")
    return folder() / fname


STALE = "that print has changed; the list has been refreshed"
_seen: dict[str, tuple] = {}        # file name -> (the picture file as last read, its digest)


def _snapshot(png: Path, want_data: bool = False):
    """(id, size note, picture bytes or None) of a print, all from one reading of its files (lock
    held). A print is its picture file, the picture and its note: a different print that takes an old
    name (even an identical copy put in its place) has another id, so an out-of-date list can't put
    it on a shirt at the old size, or delete it."""
    try:
        extra = png.with_suffix(".json").read_bytes()
    except OSError:
        extra = b""
    try:
        note = json.loads(extra.decode("utf-8")) if extra else {}
    except ValueError:
        note = {}
    with open(png, "rb") as f:
        st = os.fstat(f.fileno())
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        known = _seen.get(png.name)
        data = None
        if want_data or not (known and known[0] == key):
            data = f.read()
            _seen[png.name] = (key, hashlib.sha256(data).hexdigest())
        digest = _seen[png.name][1]
    item = hashlib.sha256(f"{digest}\0{st.st_dev}:{st.st_ino}\0".encode() + extra).hexdigest()[:16]
    return item, note if isinstance(note, dict) else {}, data


def _is_print(p: Path) -> bool:
    try:
        return p.is_file() and not p.name.startswith(".")
    except OSError:
        return False


def listing() -> list[dict]:
    out = []
    with _busy(change=False):                        # one turn for the whole list: a save or delete is never seen half done
        for f in folder().glob("*.png"):
            try:
                if not _is_print(f):
                    continue
                item, meta, _ = _snapshot(f)
                out.append({"name": f.name, "width_mm": meta.get("width_mm", 0), "height_mm": meta.get("height_mm", 0),
                            "garment": meta.get("garment", ""), "mtime": int(f.stat().st_mtime), "id": item})
            except OSError:
                continue
    out.sort(key=lambda x: -x["mtime"])
    return out


def read(fname: str, item: str | None = None) -> bytes:
    """`item`: the id the list showed for it, so an out-of-date list doesn't hand over another print."""
    p = _path(fname)
    with _busy(change=False):
        if not _is_print(p):
            raise KeyError(STALE if item is not None else "no such print")
        found, _, data = _snapshot(p, want_data=True)   # this very file and these very bytes are checked and returned
        if item is not None and found != item:
            raise KeyError(STALE)
        return data


def delete(fname: str, item: str | None = None) -> None:
    p = _path(fname)
    with _busy():
        if item is not None and (not _is_print(p) or _snapshot(p, want_data=True)[0] != item):
            raise KeyError(STALE)
        for f in (p, p.with_suffix(".json")):
            if f.is_file():
                f.unlink()
        _seen.pop(p.name, None)
