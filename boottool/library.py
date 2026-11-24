"""The library: every SVG the Convert tab makes is kept in one fixed folder on this computer.

The folder is ~/BootTool/Library (set BOOTTOOL_LIBRARY to use another place, mainly for tests).
Compose picks graphics from here. Files are plain SVGs, so Illustrator or Inkscape can open them.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

from . import filelock

_SAFE = re.compile(r"[^A-Za-z0-9 _.()+\-]")
_turns = filelock.Turns()           # one thing at a time touches the library's files and the tables below
_rev_lock = threading.Lock()        # the counter only, so a busy library never holds up a new result
_rev = 0

# What Boot Tool knows about the folder. An item is one graphic: it keeps its id when Boot Tool saves over
# it or renames it; a file with other contents (swapped or edited outside Boot Tool, or a new item that
# took an old name) is a different item with a new id. Lists show ids, and pictures open in Convert are
# tied to their item's id, so neither an out-of-date list nor an old picture can act on another item.
_items: dict[str, dict] = {}        # item id -> {"name": file name, "meta": size and date, "digest": contents}
_names: dict[str, str] = {}         # file name -> item id
_keys: dict[str, str] = {}          # save key (one picture in one session) -> item id
_used: dict[str, int] = {}          # save key -> when it last saved or renamed (old ones are forgotten)
_gone: dict[str, int] = {}          # save key -> when its item was deleted (see tick)
_loaded: Path | None = None         # the folder these tables are about
_state = None                       # the state file as last read or written (to notice another writer)
_writing = False                    # the folder's lock is held: the tables may be written down
_blind: str | None = None           # why the state file can't be read just now (nothing is written down meanwhile)
_stuck = ""                         # why the change below couldn't be finished the last time it was tried
_intent: dict | None = None         # the change in progress (a save, a rename, a delete), written down before it is
                                    # made, so one that was cut short (Boot Tool killed) is sorted out next time

# The tables are kept in this file in the library folder, so they survive a restart of Boot Tool: a
# picture still open in the browser keeps its item, and a deleted item stays deleted.
STATE = ".boottool-library.json"
KEEP = 30 * 24 * 3600 * 1_000_000   # how long (microseconds) an unused save key or a deletion is remembered


def folder() -> Path:
    p = Path(os.environ.get("BOOTTOOL_LIBRARY") or (Path.home() / "BootTool" / "Library"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def tick() -> int:
    """The next number in one order shared by new Convert results and library deletions, so a result
    made before its item was deleted (whichever tab deleted it) doesn't bring the item back. The
    numbers follow the clock (microseconds), so the order also holds across a restart."""
    global _rev
    with _rev_lock:
        _rev = max(_rev + 1, int(time.time() * 1_000_000))
        return _rev


LOCK = ".boottool-library.lock"     # the folder's lock (see filelock); the file itself stays, empty


@contextlib.contextmanager
def _busy(change: bool = True, wait: float | None = None):
    """One thing at a time touches the library: within this Boot Tool, in the order asked, and across
    every Boot Tool open on the same folder (the folder's lock, see filelock). A change needs the
    folder's lock. `change` False is for just looking: if the lock can't be had (another Boot Tool
    keeps it too long, or the folder can't be written) it goes ahead without it, and then only looks:
    nothing is written down and nothing unfinished is tidied up. `wait`: how long to wait for another
    Boot Tool (seconds)."""
    global _writing
    with _turns.held(folder() / LOCK, change, wait) as got:
        _writing = bool(got)
        try:
            _load()
            if _writing and _intent and not _blind:
                _recover()
            if change and _blind:
                raise OSError(f"the library's own notes ({STATE}) can't be read just now ({_blind}); try again in a moment")
            if change and _intent:           # nothing else is changed until that one is sorted out
                raise OSError(f"an earlier change to {_intent['name']} in the library isn't finished yet ({_stuck}); "
                              "try again in a moment")
            yield
        finally:
            _writing = False


def _stamp(d: Path):
    try:
        st = (d / STATE).stat()
        return (st.st_ino, st.st_size, st.st_mtime_ns)
    except OSError:
        return None


def _load() -> None:
    """Read the tables for the current library folder (lock held): the first time, and again whenever
    the file was written by someone else (a second Boot Tool open on the same library)."""
    global _loaded, _state, _intent, _blind
    d = folder()
    stamp = _stamp(d)
    if _loaded == d and _state == stamp and not _blind:
        return
    try:
        text = (d / STATE).read_text(encoding="utf-8")
    except FileNotFoundError:
        text = None                      # nothing written down yet
    except UnicodeDecodeError:
        text = ""                        # not ours
    except OSError as e:
        # It is there but can't be read just now (in use, not allowed). That is not "nothing written
        # down": what is known stays as it is, nothing is written, and it is tried again next time.
        if _loaded != d:
            _intent = None
            for table in (_items, _names, _keys, _used, _gone):
                table.clear()
        _loaded, _state, _blind = d, stamp, str(e) or type(e).__name__
        return
    _intent, _blind = None, None
    for table in (_items, _names, _keys, _used, _gone):
        table.clear()
    _loaded, _state = d, stamp
    if text is None:
        return
    try:
        data = json.loads(text)
        old = int(time.time() * 1_000_000) - KEEP
        items, names, keys, used, gone = {}, {}, {}, {}, {}
        for item, it in data["items"].items():
            name, digest, file = it["name"], it["digest"], it.get("file")
            if (isinstance(name, str) and name == Path(name).name and name.lower().endswith(".svg")
                    and isinstance(digest, str) and name not in names):
                file = tuple(int(x) for x in file) if isinstance(file, list) and len(file) == 2 else None
                items[str(item)] = {"name": name, "meta": None, "digest": digest, "file": file}
                names[name] = str(item)
        for key, (item, when) in data["keys"].items():
            if item in items and old < int(when) < old + 2 * KEEP:
                keys[str(key)], used[str(key)] = item, int(when)
        for key, rev in data["gone"].items():
            if old < int(rev) < old + 2 * KEEP:
                gone[str(key)] = int(rev)
        intent = data.get("intent")
        if intent is not None:
            intent = {"do": str(intent["do"]), "item": str(intent["item"]), "name": _plain(intent["name"]),
                      "to": _plain(intent.get("to") or intent["name"]), "digest": str(intent["digest"]),
                      "file": [int(x) for x in intent["file"]], "key": intent.get("key") and str(intent["key"])}
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return                           # not ours: start afresh (the folder is untouched)
    for table, loaded in ((_items, items), (_names, names), (_keys, keys), (_used, used), (_gone, gone)):
        table.update(loaded)
    _intent = intent


def _plain(name) -> str:
    if not (isinstance(name, str) and name == Path(name).name and name.lower().endswith(".svg") and not name.startswith(".")):
        raise ValueError("not a library file name")
    return name


def _is_file(name: str, file, digest: str) -> bool:
    """Is the file at this name that very file, with those contents? (No, if that can't be told.)"""
    path = folder() / name
    try:
        if not _is_item(path) or path.is_symlink() or list(_meta(path)[:2]) != list(file):
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == digest
    except OSError:
        return False


def _discard(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _settle() -> bool:
    """The change that was written down is over (made, or given up): strike the note out (folder's lock
    held). If that can't be written, the note stays, here and in the file, and is sorted out the next
    time (see _recover); until then nothing else in the library is changed (see _busy)."""
    global _intent, _stuck
    job, _intent = _intent, None
    try:
        _store(must=True)
        return True
    except OSError as e:
        _intent, _stuck = job, str(e) or type(e).__name__
        return False


def _recover() -> None:
    """Sort out a change that was written down but cut short (folder's lock held). What can't be told
    about a file counts as "not done". Only a delete that can't be finished just now, or a result
    that can't be written down, stays for the next time."""
    global _stuck
    job, d = _intent, folder()
    try:
        if job["do"] == "delete":        # finish it, if that very file is still there
            if _is_file(job["name"], job["file"], job["digest"]):
                (d / job["name"]).unlink()
            _forget(job["item"])
        elif job["do"] == "save":        # did the new file get into place?
            if _is_file(job["name"], job["file"], job["digest"]):
                _put(job["name"], _meta(d / job["name"]), job["digest"], job["item"])
                if job["key"]:
                    _keys[job["key"]], _used[job["key"]] = job["item"], int(time.time() * 1_000_000)
            elif _is_file(job["name"] + ".tmp", job["file"], job["digest"]):
                _discard(d / (job["name"] + ".tmp"))     # it was still waiting beside it
        elif job["do"] == "move":        # did the file get its new name?
            # Judged by the folder's own list of names and by the file itself: the old name may be taken
            # again by something else, and on Mac and Windows the old spelling still opens the file.
            # The new name is looked at first; the old one only if the file is there.
            names = os.listdir(d)
            if (job["to"] in names and _is_file(job["to"], job["file"], job["digest"])
                    and not (job["name"] != job["to"] and job["name"] in names
                             and _is_file(job["name"], job["file"], job["digest"]))):
                if _names.get(job["name"]) == job["item"]:
                    del _names[job["name"]]
                _put(job["to"], _meta(d / job["to"]), job["digest"], job["item"])
    except OSError as e:
        _stuck = str(e) or type(e).__name__
        return
    _settle()


def _store(must: bool = False) -> None:
    """Write the tables down (lock held). Items whose file is gone and long-unused records are
    dropped. `must`: fail (OSError) if it can't be written, for changes that must not be forgotten."""
    global _state
    if not _writing or _blind:
        if must:
            raise OSError("the library folder is in use by another Boot Tool; try again in a moment")
        return
    d = folder()
    old = int(time.time() * 1_000_000) - KEEP
    keep = {_intent["item"]} if _intent else set()   # (an item being changed may be between two names)
    for item, it in list(_items.items()):
        if item not in keep and not os.path.lexists(d / it["name"]):
            _forget(item)
    for key in [k for k, v in _keys.items() if v not in _items or _used.get(k, 0) <= old]:
        del _keys[key]
        _used.pop(key, None)
    for key in [k for k, rev in _gone.items() if rev <= old]:
        del _gone[key]
    data = {"items": {i: {"name": it["name"], "digest": it["digest"], "file": it.get("file")} for i, it in _items.items()},
            "keys": {k: [v, _used.get(k, 0)] for k, v in _keys.items()}, "gone": _gone}
    if _intent:
        data["intent"] = _intent
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".boottool-library-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        for again in (0.05, 0.1, 0.2, None):
            try:
                os.replace(tmp, d / STATE)
                break
            except PermissionError:      # Windows: something has the file open for a moment (a virus scanner, a reader)
                if again is None:
                    raise
                time.sleep(again)
        _state = _stamp(d)
    except OSError:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if must:
            raise                        # (otherwise the library still works; it just won't remember after a restart)


def clean_stem(name: str) -> str:
    stem = _SAFE.sub("_", Path(str(name)).stem).strip(" .")
    return (stem or "graphic")[:100]


def _unique(stem: str) -> str:
    d = folder()
    n = 1
    while True:
        fname = f"{stem}.svg" if n == 1 else f"{stem}-{n}.svg"
        if not os.path.lexists(d / fname):
            return fname
        n += 1


def _meta(path: Path) -> tuple:
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _forget(item: str) -> None:
    it = _items.pop(item, None)
    if it and _names.get(it["name"]) == item:
        del _names[it["name"]]


def _put(fname: str, meta: tuple, digest: str, item: str | None = None) -> str:
    """Record that the file at this name is this item (a new one unless `item` is given)."""
    old = _names.get(fname)
    if old and old != item:
        _forget(old)                     # whatever was known under this name is not there any more
    item = item or uuid.uuid4().hex[:12]
    _items[item] = {"name": fname, "meta": meta, "digest": digest, "file": tuple(meta[:2])}
    _names[fname] = item
    return item


def _id(fname: str, sure: bool = False, seen: tuple | None = None) -> str:
    """The id of the item now at this name (lock held; the file must be a regular file). A file that
    was only touched keeps its id; other contents, or another file put in its place (even an identical
    copy), are another item. `sure`: read the contents even
    if size and date are unchanged (before acting on the item). `seen`: (meta, digest) of contents the
    caller has read itself; those are what is judged, whatever is on the disk by now."""
    path = folder() / fname
    item = _names.get(fname)
    known = _items.get(item) if item else None
    if seen:
        meta, digest = seen
    else:
        meta = _meta(path)
        if known and known["meta"] == meta and not sure:
            return item
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if known and known["digest"] == digest and known.get("file") in (None, tuple(meta[:2])):
        known["meta"], known["file"] = meta, tuple(meta[:2])
        return item
    return _put(fname, meta, digest)


def _own(key: str | None) -> str | None:
    """The file name of the item this save key's picture owns, if it is still there and still that
    item (lock held)."""
    it = _items.get(_keys.get(key, "")) if key else None
    if not it:
        return None
    path = folder() / it["name"]
    if not (path.is_file() and not path.is_symlink()):
        return None
    if _id(it["name"], sure=True) != _keys[key]:
        return None
    _used[key] = int(time.time() * 1_000_000)
    return it["name"]


def _move(fname: str, stem: str) -> str:
    """Rename an item (lock held): to <stem>.svg, or <stem>-2.svg ... if another item has that name.
    Changing only the case is the same file on Mac and Windows, and needs no -2."""
    target = f"{stem}.svg"
    if target == fname:
        return fname
    src, dest = folder() / fname, folder() / target
    # The same entry under another spelling (Mac, Windows): the folder lists only one of the two names.
    # A hard link to it is listed too, and is another entry.
    same_entry = (target.casefold() == fname.casefold() and target not in os.listdir(folder())
                  and dest.exists() and os.path.samefile(src, dest))
    if os.path.lexists(dest) and not same_entry:
        if re.fullmatch(re.escape(stem) + r"-\d+\.svg", fname):
            return fname                 # the name is taken and this item already is its numbered twin
        target = _unique(stem)
    global _intent
    item = _id(fname)
    it = _items[item]
    _intent = {"do": "move", "item": item, "name": fname, "to": target, "digest": it["digest"],
               "file": list(_meta(src)[:2]), "key": None}
    try:
        _store(must=True)                # written down first, so a rename cut short is recognised next time;
    except OSError:                      # if it can't be written down, nothing is renamed
        _intent = None
        raise
    try:
        src.rename(folder() / target)
    except OSError:
        _settle()
        raise
    if _names.get(fname) == item:
        del _names[fname]
    _put(target, _meta(folder() / target), it["digest"], item)
    _settle()
    return target


def save(name: str, svg: str, key: str | None = None, rev: int | None = None, rename: bool = False) -> str:
    """Write an SVG into the library and return its file name (see save_named)."""
    return save_named(name, svg, key, rev, rename)[0]


def save_named(name: str, svg: str, key: str | None = None, rev: int | None = None, rename: bool = False) -> tuple[str, bool]:
    """Write an SVG into the library. Returns its file name, and whether the item has the name asked
    for. The same key keeps overwriting the same item (so moving a slider updates one item instead of
    adding a new one each time). `rev`: the result's number from tick(); a result made before its
    item was deleted doesn't bring the item back. `rename`: the picture was given this name, so its
    item takes it. The graphic is saved first; if the item then can't be renamed, it is saved all the
    same, under the name it still has, and the second value is False."""
    if "<svg" not in svg[:2000]:
        raise ValueError("That is not an SVG.")
    data = svg.encode("utf-8")
    with _busy():
        if key and rev is not None and rev <= _gone.get(key, 0):
            raise KeyError("deleted from the library")
        d = folder()
        fname = _own(key)
        item = _keys.get(key) if fname else None
        had = bool(fname)
        if not fname:
            fname = _unique(clean_stem(name))
        global _intent
        digest = hashlib.sha256(data).hexdigest()
        tmp = d / (fname + ".tmp")
        item = item or uuid.uuid4().hex[:12]
        try:
            tmp.write_bytes(data)
            _intent = {"do": "save", "item": item, "name": fname, "to": fname, "digest": digest,
                       "file": list(_meta(tmp)[:2]), "key": key}
            _store(must=True)            # written down first, so a save cut short is recognised next time;
        except OSError:                  # if it can't be written down, nothing is saved
            _intent = None
            _discard(tmp)
            raise
        try:
            tmp.replace(d / fname)
        except OSError:
            _discard(tmp)
            _settle()
            raise
        _put(fname, _meta(d / fname), digest, item)
        if key:
            _keys[key], _used[key] = item, int(time.time() * 1_000_000)
        named = _settle() or not (rename and had)
        if named and rename and had:
            try:
                fname = _move(fname, clean_stem(name))
            except OSError:
                named = False            # saved, under the name it had
        return fname, named


def _path(fname: str) -> Path:
    if not fname or fname != Path(fname).name or not fname.lower().endswith(".svg") or fname.startswith("."):
        raise ValueError("bad file name")
    return folder() / fname


def _is_item(path: Path) -> bool:
    try:
        return path.is_file() and not path.name.startswith(".")
    except OSError:
        return False


def listing() -> list[dict]:
    # Read new or changed files before taking the lock, so a big library doesn't hold up saves.
    with _busy(change=False, wait=0):     # (no waiting here: the wait, if any, is below)
        pass
    fresh = {}
    for f in folder().glob("*.svg"):
        try:
            if not _is_item(f):
                continue                 # a folder called something.svg, a broken link...
            meta = _meta(f)
            known = _items.get(_names.get(f.name, ""))
            if not known or known["meta"] != meta:
                fresh[f.name] = (meta, hashlib.sha256(f.read_bytes()).hexdigest())
        except OSError:
            continue
    out = []
    with _busy(change=False):
        before = set(_items)
        for f in folder().glob("*.svg"):
            try:
                if not _is_item(f):
                    continue
                st, seen = f.stat(), fresh.get(f.name)
                if seen and seen[0] != _meta(f):
                    seen = None          # changed again since it was read: read it now
                item = _id(f.name, seen=seen)
            except OSError:
                continue
            out.append({"name": f.name, "size": st.st_size, "mtime": int(st.st_mtime), "id": item})
        if set(_items) != before:
            _store()                     # so the ids just given out are still the items' ids after a restart
    out.sort(key=lambda x: -x["mtime"])
    return out


STALE = "that library item has changed; the list has been refreshed"


def read(fname: str, item: str | None = None) -> bytes:
    """`item`: the id the list showed for it, so an out-of-date list doesn't hand over another item."""
    p = _path(fname)
    with _busy(change=False):
        if not _is_item(p):
            raise KeyError(STALE if item is not None else "no such library item")
        with open(p, "rb") as f:                     # this very file and these very bytes are checked and returned
            st = os.fstat(f.fileno())
            data = f.read()
        if item is not None:
            known = _items.get(item)
            if (not known or _names.get(fname) != item or known["digest"] != hashlib.sha256(data).hexdigest()
                    or known.get("file") not in (None, (st.st_dev, st.st_ino))):
                if known and _names.get(fname) == item:
                    known["meta"] = None     # look at the file afresh for the next list
                raise KeyError(STALE)    # not the item the list showed (or it changed just now)
        return data


def _entry(fname: str) -> Path:
    """The item as the disk names it: on Mac and Windows line.svg and Line.svg are one file."""
    p = _path(fname)
    if _is_item(p):
        names = [f.name for f in folder().iterdir()]
        if fname not in names:
            for n in names:
                if n.casefold() == fname.casefold() and os.path.samefile(folder() / n, p):
                    return folder() / n
    return p


def _listed(p: Path, item: str | None) -> None:
    """Stop if the list this request came from is out of date (lock held)."""
    if item is not None and (not _is_item(p) or _id(p.name, sure=True) != item):
        raise KeyError(STALE)


def rename(fname: str, new_stem: str, key: str | None = None, item: str | None = None) -> str:
    """Rename a library item. With a save key (a picture open in Convert) the item is that picture's
    own, whatever it is called by now; `fname` is not used. `item`: the id the list showed for it,
    so an out-of-date list can't rename a different item that has since taken the name."""
    with _busy():
        if key is not None:
            fname = _own(key) or ""
            if not fname:
                raise KeyError("no such library item")
        else:
            src = _entry(fname)
            _listed(src, item)
            if not _is_item(src):
                raise KeyError("no such library item")
            fname = src.name
        return _move(fname, clean_stem(new_stem))


def delete(fname: str, item: str | None = None) -> None:
    """`item`: the id the list showed for it (see rename)."""
    global _rev, _intent
    with _busy():                        # not while a save is half done
        p = _entry(fname)
        _listed(p, item)
        gone = _id(p.name) if _is_item(p) else None
        with _rev_lock:
            _rev = max(_rev + 1, int(time.time() * 1_000_000))
            now = _rev
        before = (dict(_keys), dict(_used), dict(_gone))
        for k, v in list(_keys.items()):
            if v == gone or v not in _items:
                del _keys[k]
                _used.pop(k, None)
                if v == gone:
                    _gone[k] = now       # results made before now don't bring it back

        def put_back(was):
            for table, rows in zip((_keys, _used, _gone), was):
                table.clear()
                table.update(rows)
        if not gone:
            _store()
            return
        it = _items[gone]
        _intent = {"do": "delete", "item": gone, "name": p.name, "to": p.name, "digest": it["digest"],
                   "file": list(_meta(p)[:2]), "key": None}
        try:
            _store(must=True)            # written down first: if that fails, nothing is deleted
        except OSError:
            _intent = None
            put_back(before)
            raise
        try:
            p.unlink()
        except OSError as e:             # not deleted: everything as it was, and written down so
            after = (dict(_keys), dict(_used), dict(_gone))
            put_back(before)
            if not _settle():            # ...or, if that can't be written either, it is still to be done
                put_back(after)
                raise OSError(f"{p.name} couldn't be deleted just now ({e}); Boot Tool will delete it as soon as it can") from e
            raise                        # (stopped in any other way: what is written down is finished next time)
        _forget(gone)
        _settle()
