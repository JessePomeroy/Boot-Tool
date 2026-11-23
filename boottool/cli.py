from __future__ import annotations

import base64
import errno
import hashlib
import html
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import urllib.parse
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import click

from . import __version__
from .config import Params
from .manifest import Manifest, latest_by_source, load_provenance, sha256_file
from .pipeline import run_pipeline

IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
NO_CHECK = ("Note: the self-check was skipped because no SVG rasteriser was found. The SVGs are fine; "
            "to get scores, install librsvg (rsvg-convert) or `pip install cairosvg`.")

_PARAM_OPTIONS = [
    click.option("--mode", type=click.Choice(["auto", "lineart", "color"]), default="auto",
                 show_default=True, help="auto picks line art vs colour per image."),
    click.option("--backend", type=click.Choice(["builtin", "vtracer"]), default="builtin",
                 show_default=True),
    click.option("--colors", type=int, default=8, show_default=True, help="Palette size (colour)."),
    click.option("--threshold", type=click.Choice(["otsu", "adaptive"]), default="otsu",
                 show_default=True, help="Binarization for line art."),
    click.option("--ink", default="#000000", show_default=True,
                 help='Line-art ink colour, or "auto" to sample it from the source.'),
    click.option("--upscale", type=float, default=1.0, show_default=True,
                 help="Scale factor before thresholding, e.g. 3 for low-res scans."),
    click.option("--upscale-style", type=click.Choice(["smooth", "pixel"]), default="smooth",
                 show_default=True, help="pixel = stair-stepped, crunchier edges."),
    click.option("--smooth", "smooth_sigma", type=float, default=0.6, show_default=True,
                 help="Edge smoothing before tracing. 0 = raw, jagged."),
    click.option("--corner", "corner_deg", type=float, default=125.0, show_default=True,
                 help="Angles sharper than this stay corners."),
    click.option("--level", type=click.IntRange(0, 254), default=0, show_default=True,
                 help="Manual threshold like Photoshop (higher = more ink). 0 = automatic."),
    click.option("--min-side", type=int, default=768, show_default=True),
    click.option("--max-side", type=int, default=4096, show_default=True,
                 help="Cap on working size after upscaling."),
    click.option("--speckle", type=int, default=6, show_default=True,
                 help="Drop blobs / fill holes smaller than N px."),
    click.option("--epsilon", type=float, default=0.75, show_default=True,
                 help="Path simplification tolerance (px). Higher = fewer nodes."),
    click.option("--smooth-sigma", type=float, default=0.6, show_default=True,
                 help="Edge smoothing before contouring. 0 = hard pixel edges."),
    click.option("--corner-deg", type=float, default=125.0, show_default=True,
                 help="Vertices sharper than this angle stay corners."),
    click.option("--min-score", type=float, default=0.90, show_default=True,
                 help="ssim_target below this is flagged low_score."),
    click.option("--denoise/--no-denoise", default=True, show_default=True),
    click.option("--normalize-background/--no-normalize-background", default=True,
                 show_default=True, help="Flatten uneven paper tone (line art)."),
]


def params_options(f):
    for opt in reversed(_PARAM_OPTIONS):
        f = opt(f)
    return f


def make_params(kw: dict) -> Params:
    return Params(**{k: v for k, v in kw.items() if k in Params.__dataclass_fields__})


@click.group()
@click.version_option(__version__)
def main():
    """boottool: clean raster graphics into tidy SVG and curate a vector dataset."""


# --------------------------------------------------------------------------- ui

@main.command()
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("--no-browser", is_flag=True, help="Don't open a browser tab automatically.")
def ui(port, no_browser):
    """Open the interactive slider interface in your browser."""
    from .ui import serve
    serve(port=port, open_browser=not no_browser)


# --------------------------------------------------------------------------- trace

@main.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.argument("dst", type=click.Path(dir_okay=False))
@params_options
def trace(src, dst, **kw):
    """Trace one image to SVG and print its scores."""
    p = make_params(kw)
    res = run_pipeline(src, p)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(res.svg, encoding="utf-8")
    if res.scores.get("ssim_target") is None:
        click.echo(NO_CHECK, err=True)
    click.echo(json.dumps(
        {"mode": res.mode, "size": res.size, "stats": res.stats, "scores": res.scores}, indent=2
    ))


# --------------------------------------------------------------------------- run

_IGNORED = dict.fromkeys([0x200C, 0x200D, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x206A, 0x2070), 0xFEFF])


def _fold(name: str) -> str:
    """How Mac and Windows compare names: ignoring case, accents typed two ways, and the invisible
    characters older Mac disks skip. Folding a bit more than they do is safe: it only means a longer name."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", name.translate(_IGNORED)).casefold())


GOOD = ("ok", "low_score")
TEMP_DIR = ".boottool-unsaved"                       # in OUT: SVGs being written, before they're moved into svg/
MARK = "made-by-boottool"                            # inside each folder Boot Tool makes for unsaved SVGs


MARK_TEXT = "Unsaved SVGs from boottool run; safe to delete.\n"


def _work_folder(parent: Path, prefix: str) -> Path:
    """A new folder for unsaved SVGs, marked so a later run knows it may clear it."""
    d = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
    (d / MARK).write_text(MARK_TEXT, encoding="utf-8")
    return d


def _clear_work_folders(parent: Path, pattern: str, files: str) -> None:
    """Tidy folders a stopped run left behind: in Boot Tool's own marked folders, remove only the files
    it puts there (`files`), then the folder if nothing else is in it."""
    try:
        found = list(parent.glob(pattern))
    except OSError:
        return
    for d in found:
        try:
            if not d.is_dir() or d.is_symlink() or (d / MARK).read_text(encoding="utf-8") != MARK_TEXT:
                continue
            for f in d.glob(files):
                if f.is_file() and not f.is_symlink():
                    f.unlink()
            if [e.name for e in d.iterdir()] == [MARK]:
                (d / MARK).unlink()
                d.rmdir()
        except (OSError, UnicodeDecodeError):
            pass


def _same_disk(a, b) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def _put_aside(path: Path, free=None) -> Path:
    """Keep a file Boot Tool must not destroy: rename it, in its own folder, to <name>.replaced-N.svg.
    One step on the same disk, so it stays the very same file (links and folders included, and
    relative links still point where they did). `free(dest)`: False for names something else lays
    claim to (an SVG any record ever wrote, or one this run is about to write)."""
    stem = path.name[:-4] if path.name.lower().endswith(".svg") else path.name
    n = 1
    while True:
        tail, short = f".replaced-{n}.svg", stem
        while short and _length(short + tail) > 255:     # shorten the name, never the marker
            short = short[:-1]
        dest = path.parent / (short + tail)
        if not os.path.lexists(dest) and (free is None or free(dest)):
            break
        n += 1
    os.rename(path, dest)
    return dest


_UMASK = os.umask(0)
os.umask(_UMASK)


def _discard(path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass                                         # cleared out at the start of the next run


def _reproduces(rec: dict, sha: str, picture: str) -> bool:
    """Is `sha` what tracing the picture (where it is now) with the record's own settings gives, so
    the file there is that older trace, unedited?"""
    try:
        return hashlib.sha256(run_pipeline(picture, Params(**rec["params"])).svg.encode("utf-8")).hexdigest() == sha
    except Exception:  # noqa: BLE001 - can't tell: treat it as possibly edited
        return False


# Every SVG `run` writes ends with a stamp: the id of the save that wrote it (nothing else, so sharing an
# SVG shares no paths or details). The manifest's record for that save says which picture it is.
_STAMP = re.compile(rb"\n?<!-- boottool-run save ([0-9a-f]{32}) -->\s*\Z")


def _stamped(trace: bytes, save_id: str) -> bytes:
    return trace + b"\n<!-- boottool-run save " + save_id.encode("ascii") + b" -->\n"


def _read_svg(path: Path):
    """(fingerprint of the trace, the save id in its stamp or None, its first bytes) for a regular
    file, or None if there isn't one."""
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode):
            return None
        blob = path.read_bytes()
    except OSError:
        return None
    m = _STAMP.search(blob)
    save_id = None
    if m:
        save_id = m.group(1).decode("ascii")
        blob = blob[:m.start()]
    return hashlib.sha256(blob).hexdigest(), save_id, blob[:512].lstrip()


def _ident(out_root: Path, svg: str):
    """Which physical file a recorded name is (two names can be one file through linked folders)."""
    try:
        st = os.stat(out_root / svg)
        return ("file", st.st_dev, st.st_ino)
    except OSError:
        return ("name", _fold(svg))


def _settle(records: list[dict], out_root: Path):
    """The manifest as what was really saved, and the saves to write down. Saving is: a "saving" note
    (the whole record, with the save's id), the stamped SVG moved into place, then the finished record.
    A note without its finished record means the run stopped (or writing the record failed) in between:
    if the file there carries that save's stamp, the save happened and the note's record is used;
    otherwise the note stays as "saving" (not saved, but still a claim on that file, so an older record
    can't take the file over). Returns (history, adopted)."""
    finished = {r.get("save_id") for r in records if r.get("status") in GOOD and r.get("save_id")}
    out, adopted = [], []
    for r in records:
        if r.get("status") != "saving" or r.get("save_id") in finished:
            if r.get("status") != "saving":
                out.append(r)
            continue
        found = _read_svg(out_root / _svg_of(r))
        if found and found[1] == r.get("save_id") and r.get("saving_status") in GOOD:
            rec = {k: v for k, v in r.items() if k != "saving_status"} | {"status": r["saving_status"]}
            out.append(rec)
            adopted.append(rec)
        else:
            out.append(r)
    newest = {}
    for i, r in enumerate(out):
        if r.get("status") in GOOD:
            newest[r["source"]] = r
    return out, [a for a in adopted if newest.get(a["source"]) is a]   # only adoptions still each picture's newest


def _published(records: list[dict], on_disk=None) -> dict[str, dict]:
    """Each picture's last good record (from a settled history, see _settle). `on_disk(rec)`: is the
    record's SVG really there, its own and unchanged? If the newest's isn't, the newest that is wins."""
    stack: dict[str, list] = {}
    for r in records:
        if r.get("status") in GOOD:
            stack.setdefault(r["source"], []).append(r)
    out = {}
    for src, s in stack.items():
        out[src] = s[-1]
        if on_disk and s[-1].get("save_id") and not on_disk(s[-1]):
            out[src] = next((r for r in reversed(s) if r.get("save_id") and on_disk(r)), s[-1])
    return out


def _trusted(records: list[dict], out_root: Path):
    """Is a record's SVG really there, its own and unchanged? A record from this version: the file
    carries its save's stamp and the trace above the stamp is unchanged. A record from an older version
    (no stamp, and those versions could write two pictures to one SVG): the file is an unstamped SVG
    that no other picture ever recorded (counting linked folders as one)."""
    claimed: dict = {}
    for r in records:
        if r.get("status") in GOOD or r.get("status") == "saving":   # an unfinished save still claims its file
            claimed.setdefault(_ident(out_root, _svg_of(r)), set()).add(r["source"])
    seen: dict = {}

    def check(rec: dict) -> bool:
        if rec.get("status") not in GOOD:
            return False
        f = out_root / _svg_of(rec)
        if f not in seen:
            seen[f] = _read_svg(f)
        found = seen[f]
        if found is None:
            return False
        trace_sha, stamp, start = found
        if rec.get("save_id"):
            return stamp == rec["save_id"] and trace_sha == rec.get("trace_sha256")
        return (stamp is None and start.startswith((b"<svg", b"<?xml")) and b"<svg" in start
                and claimed.get(_ident(out_root, _svg_of(rec))) == {rec["source"]})
    return check


def _length(name: str) -> int:
    """A file name's length as the disk counts it: UTF-16 units on Windows, bytes elsewhere."""
    return len(name.encode("utf-16-le", "surrogatepass")) // 2 if os.name == "nt" else len(os.fsencode(name))


def _fit(name: str, rel: str) -> str:
    """Keep the file name within the 255 most disks allow (long names plus a.png.svg can pass it)."""
    folder, _, base = name.rpartition("/")
    if _length(base) <= 255:
        return name
    tag = "-" + hashlib.sha256(os.fsencode(rel)).hexdigest()[:10] + ".svg"
    stem = base[:-4]
    while _length(stem + tag) > 255:
        stem = stem[:-1]
    return (folder + "/" if folder else "") + stem + tag


def svg_names(rels: list[str], folders=(), allowed=None, place=None, prefer=None) -> dict[str, str]:
    """Where each input's SVG goes, relative to svg/. Usually a.png -> a.svg, but files that share a
    name apart from the extension (a.png and a.jpg) keep it, so neither replaces the other:
    a.png.svg and a.jpg.svg. If even that name is taken (a.png.webp would also want a.png.svg), or
    is the name of a folder, the next free one is used: a.png.webp.svg, then a.png.webp-2.svg, ...
    Names are compared ignoring case, as on Mac and Windows, so no two inputs ever share an SVG.
    `folders`: folders already in svg/ (from earlier runs), which no SVG may be named like.
    `allowed(input, name)`: whether that input may write that name (False for a file already there
    that isn't provably its own). `place(name)`: where the name really ends up on disk (through linked
    folders), so two names that land in one file count as one. `prefer`: names inputs already have
    and should keep if they still can (so a name doesn't wander once settled)."""
    allowed = allowed or (lambda r, name: True)
    place = place or _fold
    rels = sorted(rels)
    want = {r: Path(r).with_suffix(".svg").as_posix() for r in rels}
    stems = Counter(_fold(w) for w in want.values())
    want = {r: (r + ".svg") if stems[_fold(w)] > 1 else w for r, w in want.items()}
    blocked = {place(d.as_posix()) for r in rels for d in Path(r).parents if d.as_posix() != "."}
    blocked |= {place(f) for f in folders}
    out: dict[str, str] = {}
    taken: set[str] = set()

    def free(name, r):
        return place(name) not in taken and place(name) not in blocked and allowed(r, name)

    want = {r: _fit(w, r) for r, w in want.items()}
    for r in rels:                       # names inputs already have and can keep...
        name = (prefer or {}).get(r)
        if name and Path(name).parent == Path(r).parent and free(name, r):
            out[r] = name
            taken.add(place(name))
    for r in rels:                       # ...then every name that's free goes to the input that wants it...
        if r in out:
            continue
        if free(want[r], r):
            out[r] = want[r]
            taken.add(place(want[r]))
    for r in rels:                       # ...and only then the rest get the next free one, in name order
        if r in out:
            continue
        name, n = _fit(r + ".svg", r), 2
        while not free(name, r):
            name, n = _fit(f"{r}-{n}.svg", r + str(n)), n + 1
        out[r] = name
        taken.add(place(name))
    return out


def _svg_of(rec: dict) -> str:
    """The SVG written for a manifest record (older manifests may not say, so work it out as before)."""
    return rec.get("svg") or "svg/" + Path(rec["source"]).with_suffix(".svg").as_posix()


def _work(args: tuple) -> dict:
    src, rel, out_svg, pdict, prov, sha, svg_rel, tmp_dir = args
    p = Params(**pdict)
    rec = {
        "id": sha[:16], "sha256": sha, "source": rel, "source_abs": str(Path(src).resolve()),
        "source_bytes": os.path.getsize(src), "params": pdict, "params_key": p.key(),
        "provenance": prov, "provenance_declared": bool(prov.get("license")),
        "boottool_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        res = run_pipeline(src, p)
        trace = res.svg.encode("utf-8")
        score = res.scores["ssim_target"]                    # None when the self-check could not run
        rec.update(
            svg="svg/" + svg_rel, trace_sha256=hashlib.sha256(trace).hexdigest(), save_id=uuid.uuid4().hex,
            mode=res.mode, size=list(res.size), scale=res.scale,
            stats=res.stats, scores=res.scores,
            status="ok" if score is None or score >= p.min_score else "low_score",
        )
        Path(out_svg).parent.mkdir(parents=True, exist_ok=True)
        fd, part = tempfile.mkstemp(dir=tmp_dir, suffix=".svg")
        rec["_part"] = part                                  # discarded if anything below fails
        with os.fdopen(fd, "wb") as f:                       # moved into place after the "saving" note
            f.write(_stamped(trace, rec["save_id"]))
        os.chmod(part, 0o666 & ~_UMASK)                      # the usual permissions, not the temp file's
    except Exception as e:  # noqa: BLE001 - one bad file must not stop a batch
        rec.update(status="error", error=f"{type(e).__name__}: {e}")
    return rec


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option("--provenance", "prov_csv", type=click.Path(exists=True, dir_okay=False),
              help="CSV with columns: file, license, source, author, year.")
@click.option("--license", "default_license", default="",
              help="Fallback licence label for files not in the provenance CSV.")
@click.option("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), show_default=True)
@click.option("--force", is_flag=True, help="Reprocess files already in the manifest.")
@params_options
def run(input_dir, out_dir, prov_csv, default_license, workers, force, **kw):
    """Process every image under INPUT_DIR; write SVGs and manifest.jsonl to OUT_DIR."""
    p = make_params(kw)
    in_root, out_root = Path(input_dir), Path(out_dir)
    svg_root = out_root / "svg"
    manifest = Manifest(out_root / "manifest.jsonl")
    prov_map = load_provenance(prov_csv)
    staging = out_root / TEMP_DIR                    # only ever holds Boot Tool's own unsaved SVGs
    if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
        raise click.ClickException(f"{staging} must be a plain folder (Boot Tool keeps unsaved SVGs there); "
                                   "move it out of the way and run again")
    _clear_work_folders(staging, "run-*", "tmp*.svg")                  # left by a stopped run
    if svg_root.is_dir():
        for d in {p.parent for p in svg_root.rglob(".boottool-unsaved-*")}:
            _clear_work_folders(d, ".boottool-unsaved-*", "unsaved.svg")
    files = sorted(f for f in in_root.rglob("*") if f.suffix.lower() in IMG_EXTS and f.is_file())
    rels = [f.relative_to(in_root).as_posix() for f in files]
    old_folders = []                                 # folders already in svg/ (ones we can't look into are skipped)
    for top, dirs, _ in os.walk(svg_root):
        old_folders += [Path(top, d).relative_to(svg_root).as_posix() for d in dirs]
    records, adopted = _settle(manifest.load(), out_root)
    for rec in adopted:                              # saves a stopped run made but didn't write down
        manifest.append(rec)
    trusted = _trusted(records, out_root)
    current = set(rels)
    published = _published(records, trusted)         # each picture's last good, really saved record
    recorded: dict[str, set] = {}                    # every SVG a good record ever wrote -> who wrote it
    spellings: dict[str, set] = {}
    for r in records:
        if r.get("status") in GOOD or r.get("status") == "saving":
            recorded.setdefault(_fold(_svg_of(r)), set()).add(r["source"])
            spellings.setdefault(_fold(_svg_of(r)), set()).add(_svg_of(r))
    tidied = set()                                   # leftovers of stopped saves, also under linked folders
    for r in records:
        if r.get("svg"):
            parent = os.path.realpath(out_root / Path(r["svg"]).parent)
            if parent not in tidied:
                tidied.add(parent)
                _clear_work_folders(Path(parent), ".boottool-unsaved-*", "unsaved.svg")
    unreachable: dict[str, str] = {}                 # pictures whose place in svg/ can't even be looked at

    def present(path: Path) -> bool:                 # a file, folder or link (even a broken one) is there
        try:
            os.lstat(path)
            return True
        except (FileNotFoundError, NotADirectoryError):
            return False

    def there(name):                                 # is something at this name (in any spelling we recorded)?
        return any(present(out_root / sp) for sp in {"svg/" + name} | spellings.get(_fold("svg/" + name), set()))

    def allowed(rel, name):
        """Write over a file only if it is provably this picture's own, unchanged latest SVG."""
        try:
            if not there(name):
                return True
        except OSError as e:                         # e.g. a folder in svg/ we may not look into
            unreachable[rel] = f"can't look in svg/ for {name}: {type(e).__name__}: {e}"
            return True
        rec = published.get(rel)
        if rec is None or _fold(_svg_of(rec)) != _fold("svg/" + name):
            return False                             # someone's drawing, an edit, another picture's, an old one
        if rec.get("save_id"):
            return trusted(rec)
        return recorded.get(_fold(_svg_of(rec)), set()) <= current     # older versions: as long as no removed picture wrote it

    def place(name):                                 # where the name really lands (svg/ subfolders may be links)
        target = svg_root / name
        try:
            return _fold(os.path.join(os.path.realpath(target.parent), target.name))
        except OSError:
            return _fold(str(target))

    prefer = {rel: _svg_of(r)[len("svg/"):] for rel, r in published.items() if rel in current and trusted(r)}
    names = svg_names(rels, old_folders, allowed, place, prefer)
    spoken_for = set()                               # every SVG any record wrote or this run will write, and their folders
    for name in list(names.values()) + [_svg_of(r)[len("svg/"):] for r in records
                                        if r.get("status") in GOOD or r.get("status") == "saving"]:
        spoken_for.add(name)
        spoken_for.update(d.as_posix() for d in Path(name).parents if d.as_posix() != ".")

    def aside_free(dest: Path) -> bool:              # a kept file must not take a name that's spoken for
        # (worked out afresh each time: linked folders may have been pointed elsewhere since planning)
        return place(dest.relative_to(svg_root).as_posix()) not in {place(n) for n in spoken_for}
    # Resume: skip a picture only if its last good record is this same picture, with the same settings,
    # in the SVG it gets now, and that SVG can be trusted to still hold it (see _trusted).
    def already_done(rel, sha):                      # records from older versions have no fingerprint: redone once
        rec = None if force else published.get(rel)
        return (rec is not None and rec.get("save_id") and rec.get("sha256") == sha
                and rec.get("params_key") == p.key() and _svg_of(rec) == "svg/" + names[rel] and trusted(rec))

    jobs, skipped, blocked_here = [], 0, []
    for f, rel in zip(files, rels):
        if rel in unreachable:
            blocked_here.append(rel)
            continue
        sha = sha256_file(f)
        out_svg = svg_root / names[rel]
        if already_done(rel, sha):
            skipped += 1
            continue
        prov = dict(prov_map.get(rel, {}))
        if default_license and not prov.get("license"):
            prov["license"] = default_license
        before = published.get(rel, {})
        if not prov and before.get("sha256") == sha:  # redone (new settings, older record): keep its licence etc.
            prov = dict(before.get("provenance") or {})
        jobs.append((str(f), rel, str(out_svg), p.to_dict(), prov, sha, names[rel]))

    click.echo(f"{len(files)} images found, {skipped} skipped (already done), {len(jobs)} to process"
               + (f", {len(blocked_here)} can't be saved" if blocked_here else ""))
    for rel in blocked_here:
        manifest.append({"source": rel, "source_abs": str((in_root / rel).resolve()), "status": "error",
                         "error": unreachable[rel], "boottool_version": __version__,
                         "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        click.echo(f"{rel}  error  {unreachable[rel]}")
    tmp_dir = None
    if jobs:                                         # only now: a run with nothing to do needn't write anything
        staging.mkdir(parents=True, exist_ok=True)
        tmp_dir = _work_folder(staging, "run-")
        jobs = [j + (str(tmp_dir),) for j in jobs]
    counts: Counter = Counter()
    unchecked = [0]

    def handle(i, rec):
        part = rec.pop("_part", None)
        if not (part and rec["status"] in GOOD):
            manifest.append(rec)
        else:
            # Step 1: note where this save goes (see _settle). Step 2: move the stamped SVG into place.
            # Step 3: the finished record. If the run stops in between, the stamp tells the next run.
            manifest.append({**rec, "status": "saving", "saving_status": rec["status"]})
            target = svg_root / names[rec["source"]]
            near = None

            def bring_over():                        # put the SVG on the target's own disk first
                nonlocal near, part
                near = _work_folder(target.parent, ".boottool-unsaved-")
                shutil.copyfile(part, near / "unsaved.svg")
                shutil.copymode(part, near / "unsaved.svg")
                _discard(part)
                part = str(near / "unsaved.svg")

            def clear_the_way():
                # Last look before writing: anything there that isn't this picture's own unchanged SVG,
                # or an older version's identical trace of it (a file made meanwhile, an edit, a link,
                # another picture's), is moved aside, never destroyed. Looked at again after moving
                # something aside, in case something new appeared meanwhile.
                before = published.get(rec["source"])
                same_place = before is not None and _fold(_svg_of(before)) == _fold(_svg_of(rec))
                for _ in range(5):
                    try:
                        os.lstat(target)
                    except FileNotFoundError:
                        break
                    found = _read_svg(target)
                    own = False
                    if found and same_place:
                        trace_sha, stamp, _ = found
                        if before.get("save_id"):
                            own = stamp == before["save_id"] and trace_sha == before.get("trace_sha256")
                        elif stamp is None and before.get("sha256") == rec["sha256"]:
                            own = trace_sha == rec["trace_sha256"] or _reproduces(before, trace_sha, rec["source_abs"])
                            if own:              # that may have taken a while: still the same file?
                                again = _read_svg(target)
                                own = again is not None and again[:2] == (trace_sha, None)
                    if own:
                        try:
                            os.chmod(part, stat.S_IMODE(os.lstat(target).st_mode))   # keep its permissions
                        except OSError:
                            pass
                        break
                    moved = _put_aside(target, aside_free)
                    click.echo(f"  kept the file that was at svg/{names[rec['source']]}: now {moved.relative_to(out_root).as_posix()}")
                else:
                    raise OSError(f"svg/{names[rec['source']]} keeps changing")
            try:
                if not _same_disk(part, target.parent):
                    bring_over()
                clear_the_way()
                try:
                    os.replace(part, target)
                except OSError as e:
                    if e.errno != errno.EXDEV or near is not None:
                        raise
                    bring_over()                     # one disk to stat, but not to rename (a bind mount)
                    clear_the_way()
                    os.replace(part, target)
                part = None
            except OSError as e:                         # e.g. the SVG is open in another program on Windows
                rec = {k: v for k, v in rec.items() if k not in ("save_id", "trace_sha256")}
                rec.update(status="error", error=f"could not save the SVG: {type(e).__name__}: {e}")
                manifest.append(rec)
            else:
                try:
                    manifest.append(rec)
                except OSError as e:                     # saved; its stamp lets the next run write this down
                    click.echo(f"  saved, but couldn't add it to the manifest ({e}); the next run will", err=True)
            finally:
                if near is not None:
                    shutil.rmtree(near, ignore_errors=True)
        if part:
            _discard(part)
        counts[rec["status"]] += 1
        score = rec.get("scores", {}).get("ssim_target")
        if score is None:
            score = "-"
            unchecked[0] += rec["status"] != "error"
        click.echo(f"[{i}/{len(jobs)}] {rec['source']}  {rec['status']}  ssim={score}"
                   + (f"  {rec['error']}" if rec["status"] == "error" else ""))

    try:
        if workers <= 1 or len(jobs) <= 1:
            for i, job in enumerate(jobs, 1):
                handle(i, _work(job))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, rec in enumerate(ex.map(_work, jobs, chunksize=1), 1):
                    handle(i, rec)
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)  # this run's unsaved SVGs (a stopped run's go next time)

    click.echo("done: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) if counts
               else "nothing new to process")
    if unchecked[0]:
        click.echo(NO_CHECK, err=True)


# --------------------------------------------------------------------------- report

@main.command()
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False))
def report(out_dir):
    """Write report.html: source vs SVG side by side, worst scores first."""
    out_root = Path(out_dir)
    records = _settle(Manifest(out_root / "manifest.jsonl").load(), out_root)[0]
    trusted = _trusted(records, out_root)
    published = _published(records, trusted)
    latest = latest_by_source(records)
    failed = {r["source"]: r for r in latest if r["status"] == "error" and r["source"] in published}
    recs = [published.get(r["source"], r) for r in latest]           # a failed redo doesn't hide the saved trace
    recs.sort(key=lambda r: -1 if r.get("scores", {}).get("ssim_target") is None else r["scores"]["ssim_target"])
    rows = []
    for r in recs:
        sc = r.get("scores", {})
        svg_rel = _svg_of(r)
        src_uri = Path(r["source_abs"]).as_uri() if r.get("source_abs") else ""
        if r["status"] not in GOOD:
            shown = ""
        elif not trusted(r):
            shown = "This SVG may hold another picture's trace or be missing (older versions could mix them up); run again to redo it"
        else:
            shown = "<img src='" + html.escape(urllib.parse.quote(svg_rel)) + "' width=220>"   # a # or ? in a name stays a name
        rows.append(
            "<tr>"
            f"<td>{'<img src=' + chr(39) + html.escape(src_uri) + chr(39) + ' width=220>' if src_uri else ''}</td>"
            f"<td>{shown}</td>"
            f"<td>{html.escape(r['source'])}<br>status: <b>{r['status']}</b><br>"
            f"ssim_target: {'-' if sc.get('ssim_target') is None else sc['ssim_target']}<br>"
            f"paths: {r.get('stats', {}).get('path_count', '-')} "
            f"nodes: {r.get('stats', {}).get('node_count', '-')}<br>"
            f"licence: {html.escape(r.get('provenance', {}).get('license', '') or 'MISSING')}"
            f"{'<br>' + html.escape(r['error']) if r.get('error') else ''}"
            f"{'<br>last attempt failed: ' + html.escape(failed[r['source']]['error']) if r['source'] in failed else ''}</td></tr>"
        )
    page = ("<!doctype html><meta charset=utf-8><title>boottool report</title>"
            "<style>body{font:14px sans-serif}td{padding:6px;vertical-align:top;"
            "border-bottom:1px solid #ddd}img{background:#fff;border:1px solid #ccc}</style>"
            f"<h1>boottool report ({len(recs)} images)</h1><table>{''.join(rows)}</table>")
    (out_root / "report.html").write_text(page, encoding="utf-8")
    click.echo(f"wrote {out_root / 'report.html'}")


# --------------------------------------------------------------------------- dataset

@main.command()
@click.argument("out_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--min-score", type=float, default=0.90, show_default=True)
@click.option("--max-paths", type=int, default=400, show_default=True,
              help="Exclude SVGs too complex to be useful training sequences.")
@click.option("--val-frac", type=float, default=0.05, show_default=True)
@click.option("--require-provenance", is_flag=True,
              help="Exclude images with no declared licence.")
def dataset(out_dir, min_score, max_paths, val_frac, require_provenance):
    """Select clean, deduplicated SVGs and write a deterministic train/val split."""
    out_root = Path(out_dir)
    records = _settle(Manifest(out_root / "manifest.jsonl").load(), out_root)[0]
    trusted = _trusted(records, out_root)
    published = _published(records, trusted)
    recs = sorted((published.get(r["source"], r) for r in latest_by_source(records)), key=lambda r: r["source"])
    train, val, excluded, seen = [], [], Counter(), set()
    for r in recs:
        if r["status"] not in GOOD:                  # failed, or a save that never finished
            excluded["error"] += 1
        elif r["scores"]["ssim_target"] is None:
            excluded["unscored"] += 1
        elif r["scores"]["ssim_target"] < min_score:
            excluded["low_score"] += 1
        elif r["stats"]["path_count"] > max_paths:
            excluded["too_complex"] += 1
        elif require_provenance and not r["provenance_declared"]:
            excluded["no_provenance"] += 1
        elif not trusted(r):
            excluded["unverified"] += 1                  # its SVG may hold another picture's trace, or is missing
        elif r["sha256"] in seen or _ident(out_root, _svg_of(r)) in seen:
            excluded["duplicate"] += 1
        else:
            seen.update((r["sha256"], _ident(out_root, _svg_of(r))))
            svg = _svg_of(r)
            bucket = int(r["sha256"][:8], 16) / 0xFFFFFFFF
            (val if bucket < val_frac else train).append(svg)
    doc = {
        "criteria": {"min_score": min_score, "max_paths": max_paths, "val_frac": val_frac,
                     "require_provenance": require_provenance},
        "train": train, "val": val, "excluded": dict(excluded),
    }
    (out_root / "dataset.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    click.echo(f"train={len(train)} val={len(val)} excluded={dict(excluded)}")
