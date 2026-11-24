"""Print tab: one ink, CMYK or spot colour, as screen halftone or dither, with Levels.

Everything is laid out on a real page in millimetres and comes back as plates (a Job), the
same as the older halftone and separate builders. Spot colour has two image types:
  flat   flat-colour artwork split into one solid plate per ink (see separate.py)
  photo  each ink is halftoned (or dithered) on its own plate, with soft colour matching so
         gradients are shared between neighbouring inks
"""
from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass, fields
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from . import halftone as ht
from . import separate as sep
from .config import Params
from .page import Job, PageOpts, Plate, fit_artwork
from .preprocess import load_rgb
from .trace import mask_to_path, mask_to_polys, polygons_to_path, polys_to_rings, rings_to_mask

MAX_INKS = 10
MAX_DITHER_PX = 3_000_000       # dither pixels per plate
SPOT_WORK_PX = 2_500_000        # colour-matching resolution for spot photos
DITHERS = ("floyd", "atkinson", "stucki", "bayer", "blue", "serpentine", "random")
PATTERNS = ("mazes", "coral", "fingerprints", "loops", "spots")
DEFAULT_ANGLES = (15.0, 75.0, 0.0, 45.0, 22.5, 67.5, 30.0, 60.0, 7.5, 52.5)


@dataclass(frozen=True)
class PrintParams:
    method: str = "one"             # one | cmyk | spot
    kind: str = "flat"              # spot only: flat | photo
    render: str = "screen"          # screen | dither
    # levels (like Photoshop): black point, midtone gamma, white point
    lv_black: float = 0.0
    lv_gamma: float = 1.0
    lv_white: float = 255.0
    # screen halftone
    lpi: float = 30.0
    angle: float = 22.5             # one-ink angle
    shape: str = "round"
    ppi: float = 300.0
    edge: float = 0.5
    angles: tuple = ()              # per plate (CMYK / spot photo); negative = automatic
    # dither
    dither: str = "floyd"
    dither_size: float = 0.3        # mm per dither pixel
    pattern: str = "mazes"          # serpentine dither: mazes | coral | fingerprints | loops | spots
    pattern_size: float = 9.0       # serpentine dither: width of one stripe cycle, in grains
    # tone
    contrast: float = 1.0
    brightness: float = 0.0
    smoothing: float = 0.0
    invert: bool = False
    draft: bool = False             # preview only: coarser dots so it draws quickly
    # colour
    ink: str = "#000000"
    black_amount: float = 0.6
    cmyk_colors: tuple = ()         # 4 hex, "" = standard
    # spot colour
    inks: int = 4
    paper: bool = True
    trap: float = 0.2
    speckle: int = 8
    smooth: float = 0.6
    colors: tuple = ()              # hex per ink, "" = automatic (photo: used for the separation)
    names: tuple = ()
    picked: bool = False            # the colours are exactly the inks to use (eyedropper), in print order
    dark: bool = False              # printing on a dark garment / paper
    dark_bg: str = "#1b1b1d"        # its colour (preview only; never printed)
    dark_invert: bool = False       # swap the dark and empty areas so it looks the same on the garment
    outline_mm: float = 5.0         # ink margin grown round the design when inverted
    underbase: bool = False         # extra first plate (white) cut to the printed shape
    keep_solid: bool = False        # spot photo: flat areas of an ink stay solid instead of turning into dots
    solid_detect: float = 0.5       # 0 strict .. 1 loose: how readily an area counts as flat
    stamp: tuple = ()               # 256 ints for shape == "custom"


_BOUNDS = {
    "lv_black": (0.0, 253.0), "lv_gamma": (0.1, 9.99), "lv_white": (2.0, 255.0),
    "lpi": (5.0, 100.0), "angle": (0.0, 90.0), "ppi": (72.0, 1200.0), "edge": (0.0, 2.0),
    "dither_size": (0.1, 2.0), "pattern_size": (3.0, 40.0), "contrast": (0.3, 3.0), "brightness": (-0.5, 0.5),
    "smoothing": (0.0, 6.0), "black_amount": (0.0, 1.0), "inks": (1, MAX_INKS),
    "trap": (0.0, 1.0), "speckle": (0, 100), "smooth": (0.0, 2.0), "solid_detect": (0.0, 1.0), "outline_mm": (0.0, 30.0),
}
_CHOICES = {"method": {"one", "cmyk", "spot"}, "kind": {"flat", "photo"},
            "render": {"screen", "dither"}, "shape": set(ht.SHAPES), "dither": set(DITHERS),
            "pattern": set(PATTERNS)}


def _is_hex(s) -> bool:
    s = str(s)
    return len(s) == 7 and s[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in s[1:])


def coerce_print(raw: dict) -> PrintParams:
    d = PrintParams()
    out = {}
    raw = dict(raw)
    if raw.get("shape") == "dot":
        raw["shape"] = "round"
    for f in fields(PrintParams):
        if f.name not in raw:
            continue
        v, dv = raw[f.name], getattr(d, f.name)
        if f.name == "stamp":
            bits = [1 if x else 0 for x in list(v)[:256]]
            out["stamp"] = tuple(bits) if len(bits) == 256 else ()
        elif f.name == "angles":
            vals = []
            for x in list(v)[:MAX_INKS]:
                try:
                    a = float(x)
                except (TypeError, ValueError):
                    a = -1.0
                vals.append(min(a, 90.0) if a >= 0 else -1.0)
            out["angles"] = tuple(vals)
        elif f.name == "cmyk_colors":
            out["cmyk_colors"] = tuple(str(x) if _is_hex(x) else "" for x in list(v)[:4])
        elif f.name == "colors":
            out["colors"] = tuple(str(x) if _is_hex(x) else "" for x in list(v)[:MAX_INKS])
        elif f.name == "names":
            out["names"] = tuple(str(x)[:30] for x in list(v)[:MAX_INKS])
        elif f.name in ("ink", "dark_bg"):
            out[f.name] = str(v) if _is_hex(v) else dv
        elif isinstance(dv, bool):
            out[f.name] = bool(v)
        elif f.name in _CHOICES:
            out[f.name] = str(v) if str(v) in _CHOICES[f.name] else dv
        else:
            lo, hi = _BOUNDS[f.name]
            out[f.name] = type(dv)(round(min(max(float(v), lo), hi))) if isinstance(dv, int) \
                else min(max(float(v), lo), hi)
    p = PrintParams(**out)
    if p.lv_white - p.lv_black < 2:                       # keep the handles apart
        p = PrintParams(**{**out, "lv_black": min(p.lv_black, p.lv_white - 2)})
    return p


# --------------------------------------------------------------------------- levels

def auto_points(g: np.ndarray) -> tuple[int, int]:
    """Black and white points that stretch the picture's tones, ignoring pure paper."""
    body = g[g < 250]
    if body.size < 100:
        return 0, 255
    lo, hi = np.percentile(body, [1, 99])
    if hi - lo < 8:
        return 0, 255
    return int(lo), int(hi)


def analyse(path) -> dict:
    """Histogram, suggested levels and whether the picture has continuous tones."""
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    k = 900 / max(h, w)
    if k < 1:
        rgb = cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hist = np.bincount(g.ravel(), minlength=256)
    b, wh = auto_points(g)
    tonal = int((hist / g.size > 0.002).sum()) > 12
    return {"hist": [int(x) for x in hist], "auto": [b, 1.0, wh], "tonal": bool(tonal)}


def _lut(p: PrintParams) -> np.ndarray | None:
    if p.lv_black <= 0 and p.lv_white >= 255 and abs(p.lv_gamma - 1) < 1e-6:
        return None
    x = np.clip((np.arange(256, dtype=np.float32) - p.lv_black) / max(p.lv_white - p.lv_black, 1.0), 0, 1)
    return np.round(x ** (1.0 / p.lv_gamma) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- dithering

_KERNELS = {
    "floyd": [(1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)],
    "atkinson": [(1, 0, 1 / 8), (2, 0, 1 / 8), (-1, 1, 1 / 8), (0, 1, 1 / 8), (1, 1, 1 / 8), (0, 2, 1 / 8)],
    "stucki": [(1, 0, 8 / 42), (2, 0, 4 / 42), (-2, 1, 2 / 42), (-1, 1, 4 / 42), (0, 1, 8 / 42),
               (1, 1, 4 / 42), (2, 1, 2 / 42), (-2, 2, 1 / 42), (-1, 2, 2 / 42), (0, 2, 4 / 42),
               (1, 2, 2 / 42), (2, 2, 1 / 42)],
}


def _diffuse(tone: np.ndarray, kernel) -> np.ndarray:
    """Error diffusion, processed along diagonal wavefronts so whole fronts run in numpy.
    Pixels with the same k = x + 3y never feed each other, so the result is exact."""
    h, w = tone.shape
    pad = 2
    a = np.zeros((h + pad, w + 2 * pad), np.float32)
    a[:h, pad:pad + w] = tone
    out = np.zeros((h, w), bool)
    for k in range(w + 3 * (h - 1)):
        y0 = max(0, -(-(k - (w - 1)) // 3))
        y1 = min(h - 1, k // 3)
        if y1 < y0:
            continue
        ys = np.arange(y0, y1 + 1)
        xs = k - 3 * ys
        v = a[ys, xs + pad]
        on = v >= 0.5
        out[ys, xs] = on
        e = v - on
        for dx, dy, wt in kernel:
            a[ys + dy, xs + dx + pad] += e * wt
    return out


@lru_cache(maxsize=1)
def _bayer(n: int = 8) -> np.ndarray:
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (n * n)


@lru_cache(maxsize=1)
def _blue(n: int = 64) -> np.ndarray:
    """Blue-noise threshold tile by void filling: each new point goes in the emptiest spot."""
    yy, xx = np.mgrid[0:n, 0:n]
    dy, dx = np.minimum(yy, n - yy), np.minimum(xx, n - xx)
    kern = np.exp(-(dx * dx + dy * dy) / (2 * 1.5 ** 2)).astype(np.float32)
    rng = np.random.default_rng(7)
    energy = (rng.random((n, n)) * 1e-3).astype(np.float32)
    placed = np.zeros((n, n), bool)
    rank = np.zeros((n, n), np.float32)
    for i in range(n * n):
        y, x = divmod(int(np.argmin(np.where(placed, np.inf, energy))), n)
        rank[y, x] = i
        placed[y, x] = True
        energy += np.roll(np.roll(kern, y, 0), x, 1)
    return (rank + 0.5) / (n * n)


# name -> (bandwidth as a fraction of the stripe frequency, iterations, gain, stretch, seed)
_PATTERN_SPEC = {
    "mazes": (0.16, 45, 3.0, 1.0, 1),
    "coral": (0.45, 25, 2.2, 1.0, 2),
    "fingerprints": (0.13, 55, 3.2, 0.5, 3),
    "loops": (0.30, 30, 1.6, 1.0, 4),
    "spots": (0.20, 10, 1.1, 1.0, 5),
}


@lru_cache(maxsize=32)
def _serpentine(pattern: str = "mazes", size: float = 9.0, n: int = 256) -> np.ndarray:
    """Tileable maze-like threshold map (a Turing-style pattern). Random noise is filtered to one
    stripe width, then squashed and filtered again; repeating that lets stripes join into long
    serpentine mazes. The result is ranked, so a tone of t inks exactly the fraction t of the tile."""
    bw, iters, gain, stretch, seed = _PATTERN_SPEC[pattern]
    k0 = 1.0 / size
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    ring = np.exp(-((np.sqrt(fx ** 2 + (fy * stretch) ** 2) - k0) / (bw * k0)) ** 2 / 2)
    u = np.random.default_rng(seed).standard_normal((n, n))

    def filt(a):
        f = np.fft.ifft2(np.fft.fft2(a) * ring).real
        return f / (f.std() + 1e-9)
    for _ in range(iters):
        u = np.tanh(gain * filt(u))
    field = filt(u)
    rank = np.empty(n * n)
    rank[np.argsort(field.ravel(), kind="stable")] = np.arange(n * n)
    return ((rank + 0.5) / (n * n)).reshape(n, n).astype(np.float32)


def dither_mask(tone: np.ndarray, algo: str, plate: int = 0, tile: np.ndarray | None = None) -> np.ndarray:
    """tone 0 (no ink) .. 1 (full ink) -> boolean ink mask."""
    if algo in _KERNELS:
        return _diffuse(tone.astype(np.float32), _KERNELS[algo])
    h, w = tone.shape
    if algo == "random":
        return tone > np.random.default_rng(11 + plate).random((h, w), dtype=np.float32)
    if algo == "serpentine":
        tile = tile if tile is not None else _serpentine()
        off = (61, 97)
    else:
        tile = _bayer() if algo == "bayer" else _blue()
        off = (17, 29)
    th, tw = tile.shape
    ys = (np.arange(h) + plate * off[0]) % th
    xs = (np.arange(w) + plate * off[1]) % tw
    return tone > tile[ys[:, None], xs[None, :]]


def runs_path(mask: np.ndarray) -> tuple[str, int]:
    """Horizontal runs of ink as rectangles in one path (one pixel = one unit)."""
    d = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    ys, xs = np.nonzero(d == 1)
    _, xe = np.nonzero(d == -1)
    ln = xe - xs
    body = "".join(f"M{x} {y}h{l}v1h-{l}z" for x, y, l in zip(xs.tolist(), ys.tolist(), ln.tolist()))
    return body, int(len(xs))


# --------------------------------------------------------------------------- planes

def _hex_to_lab(h: str) -> np.ndarray:
    rgb = np.array([[[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]]], np.float32) / 255
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0, 0]


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb)


def _work(rgb: np.ndarray, W: int, H: int, max_px: int | None = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    tw, th = min(w, W), min(h, H)
    if max_px and tw * th > max_px:
        s = (max_px / (tw * th)) ** 0.5
        tw, th = max(1, int(tw * s)), max(1, int(th * s))
    if (tw, th) != (w, h):
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    return rgb


def _solid_masks(lab, cen, planes, detect: float):
    """Per ink: pixels that are flat (no texture around them) and sit on that ink's colour."""
    d = float(min(max(detect, 0.0), 1.0))
    k9 = np.ones((9, 9), np.uint8)
    rng = sum(cv2.dilate(lab[..., c], k9) - cv2.erode(lab[..., c], k9) for c in range(3))
    dist = np.stack([np.linalg.norm(lab - c, axis=2) for c in cen])
    win = dist.argmin(0)
    core = ((rng < 4 + 20 * d) & (dist.min(0) < 12 + 20 * d)).astype(np.uint8)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    grown = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    out = []
    for i, pl in enumerate(planes):
        m = (core.astype(bool) | (grown & (pl > 0.5))) & (win == i)
        out.append(m)
    return np.stack(out)


def _spot_planes(work: np.ndarray, p: PrintParams):
    """Soft colour matching: each ink's plane is how close each pixel is to that ink."""
    lab = cv2.cvtColor(work.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    paper = None
    if p.paper:
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        paper = np.median(border, axis=0)
        is_paper = np.linalg.norm(lab - paper, axis=2) < 8
        samples = lab[~is_paper]
    else:
        samples = lab.reshape(-1, 3)
    if samples.shape[0] < 50:
        samples = lab.reshape(-1, 3)
    samples = np.ascontiguousarray(samples[:: max(1, samples.shape[0] // 60000)], dtype=np.float32)
    picked = [c for c in p.colors if c] if p.picked else []
    if picked:                                                           # eyedropper: exactly these inks, in this order
        k = len(picked)
        cen = np.array([_hex_to_lab(c) for c in picked], np.float32)
        auto = cen.copy()
    else:
        k = int(min(p.inks, max(1, samples.shape[0] // 20)))
        cv2.setRNGSeed(20240)                                            # same picture, same inks, every time
        _, _, cen = cv2.kmeans(samples, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1),
                               3, cv2.KMEANS_PP_CENTERS)
        cen = cen[np.argsort(-cen[:, 0])]                                # light to dark = print order
        auto = cen.copy()
        for i in range(k):
            if i < len(p.colors) and p.colors[i]:
                cen[i] = _hex_to_lab(p.colors[i])
    if k > 1:
        near = [np.sort(np.linalg.norm(cen - c, axis=1))[1] for c in cen]
        sigma = max(6.0, 0.55 * float(np.mean(near)))
    else:
        sigma = 20.0
    cents = list(cen) + ([paper] if paper is not None else [])
    ws = [np.exp(-((lab - c) ** 2).sum(2) / (2 * sigma ** 2)).astype(np.float32) + 1e-6 for c in cents]
    tot = np.sum(ws, axis=0)
    planes = [w_ / tot for w_ in ws[:k]]
    if paper is not None:                       # pixels that are paper print nothing, however pale the lightest ink
        t = np.clip((np.linalg.norm(lab - paper, axis=2) - 3.0) / 6.0, 0, 1)
        keep = (t * t * (3 - 2 * t)).astype(np.float32)
        planes = [pl * keep for pl in planes]
    to_rgb = lambda c: np.clip(cv2.cvtColor(np.array([[c]], np.float32), cv2.COLOR_Lab2RGB)[0, 0], 0, 1) * 255
    final = [to_rgb(c) for c in cen]
    autos = [to_rgb(c) for c in auto]
    seen: dict[str, int] = {}
    pal, names = [], []
    for i in range(k):
        base = sep.color_name(autos[i])
        seen[base] = seen.get(base, 0) + 1
        auto_name = base if seen[base] == 1 else f"{base} {seen[base]}"
        name = p.names[i] if i < len(p.names) and p.names[i] else auto_name
        names.append(name)
        pal.append({"hex": _hex(final[i]), "auto_hex": _hex(autos[i]), "name": name, "auto_name": auto_name})
    solid = _solid_masks(lab, cen, planes, p.solid_detect) if p.keep_solid else None
    if solid is not None:
        any_solid = np.any(solid, axis=0)
        planes = [pl * (~any_solid) for pl in planes]                    # solid areas are drawn as vector, not dots
    return planes, pal, (_hex(to_rgb(paper)) if paper is not None else None), solid


def _angle(p: PrintParams, i: int, default: float) -> float:
    return p.angles[i] if i < len(p.angles) and p.angles[i] >= 0 else default


def _design_region(dark: np.ndarray, mm_px: float, outline_mm: float) -> np.ndarray:
    """dark: 0..1 ink amount of the artwork. Returns a 0/1 region: the design's outer shape (gaps closed,
    holes filled) grown outwards by the outline."""
    t = (dark > 0.08).astype(np.uint8)
    k = max(1, int(round(0.8 / mm_px)))
    sil = cv2.morphologyEx(t, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    h, w = sil.shape
    pad = cv2.copyMakeBorder((1 - sil) * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, flood, (0, 0), 128)
    filled = (pad[1:-1, 1:-1] != 128).astype(np.uint8)
    r = int(round(outline_mm / mm_px))
    if r > 0:
        filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return filled


def _disc(mm: float, mm_px: float) -> np.ndarray:
    r = max(1, int(round(mm / mm_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _gaps_closed(m: np.ndarray, mm: float, mm_px: float) -> np.ndarray:
    """The mask with gaps up to twice `mm` wide closed. The picture's edge is no part of the shape:
    a gap between the shape and the edge stays open."""
    disc = _disc(mm, mm_px)
    r = disc.shape[0] // 2 + 1
    return cv2.morphologyEx(np.pad(m, r), cv2.MORPH_CLOSE, disc)[r:-r, r:-r]


def _pulled_in(m: np.ndarray, mm_px: float) -> np.ndarray:
    """A quarter millimetre in from every edge of the shape, the picture's own edge included."""
    return cv2.erode(m, _disc(0.25, mm_px), borderType=cv2.BORDER_CONSTANT, borderValue=0)


def _underbase_mask(printed: np.ndarray, mm_px: float) -> np.ndarray:
    """Solid shape under everything that prints, pulled in a little so white never peeks out."""
    m = (printed > 0.03).astype(np.uint8)
    k = max(1, int(round(0.8 / mm_px)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    e = max(1, int(round(0.25 / mm_px)))
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))).astype(np.float32)


def _solid_core(ink: np.ndarray, grid_mm: float, lean: bool = False, grid_mm_y: float = 0.0) -> np.ndarray:
    """One ink: where white may always go. `ink` is the ink plate as it goes out (its dithered pixels,
    or what its traced outlines cover, on the plate's own pixels of `grid_mm`); this is that ink a
    quarter millimetre in from its edges (at least one pixel), so there is ink over every bit of it
    and around it.

    The underbase mask above is pulled in on the picture's own pixels, at least one whole pixel. In a
    picture with few pixels that is a millimetre or more, and a thin part is left with no white at
    all although its ink is solid. This gives such parts their white, and is drawn in with the mask."""
    # In whole pixels: rounded up, or (`lean`: the ink given is itself on the narrow side) to the nearest.
    # (`grid_mm_y`: the pixels' height, if they are not square.)
    rx, ry = (max(1, int(round(0.25 / g)) if lean else int(np.ceil(0.25 / g - 1e-6))) for g in (grid_mm, grid_mm_y or grid_mm))
    return cv2.erode(_bytes(ink), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1)),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)


def _bytes(m: np.ndarray) -> np.ndarray:
    """A mask as bytes of 0 and 1, without copying it if it need not be."""
    return m.view(np.uint8) if m.dtype == bool else m.astype(np.uint8, copy=False)


class _Rings:
    """Outlines (from polys_to_rings) that are drawn a piece of the plate at a time.

    A long outline is taken in runs of 128 points with a box round each. For one piece, a run that
    lies altogether beyond one of the piece's sides is left out but for its first point: the straight
    line that takes its place lies beyond that side too, so nothing changes within the piece. (Drawing
    the whole of a long outline again for every piece it crosses took most of the time on a large
    design: a stripe across a two-metre plate has a hundred thousand points.)"""
    RUN = 128

    def __init__(self, rings: list):
        self.rings = rings
        self.lo = np.array([r.min(axis=0) for r in rings]).reshape(-1, 2)
        self.hi = np.array([r.max(axis=0) for r in rings]).reshape(-1, 2)
        self.runs = {}
        for k, r in enumerate(rings):
            if len(r) > 4 * self.RUN:
                at = np.arange(0, len(r), self.RUN)
                nxt = r[np.append(at[1:], 0)]                               # (a run reaches to the first point of the next)
                self.runs[k] = (np.minimum(np.minimum.reduceat(r, at, axis=0), nxt), np.maximum(np.maximum.reduceat(r, at, axis=0), nxt))

    def near(self, xs: float, ys: float, xe: float, ye: float) -> list:
        """The outlines that can matter within that box (in the outlines' own units), long ones
        without their runs elsewhere."""
        out = []
        for k in np.flatnonzero((self.lo[:, 0] < xe) & (self.hi[:, 0] > xs) & (self.lo[:, 1] < ye) & (self.hi[:, 1] > ys)).tolist():
            r = self.rings[k]
            if k in self.runs:
                lo, hi = self.runs[k]
                far = (hi[:, 0] < xs - 1) | (lo[:, 0] > xe + 1) | (hi[:, 1] < ys - 1) | (lo[:, 1] > ye + 1)
                if far.any():
                    counts = np.where(far, 1, self.RUN)
                    if not far[-1]:
                        counts[-1] = len(r) - (len(far) - 1) * self.RUN     # (the last run is as long as it is)
                    first = np.cumsum(counts) - counts
                    r = r[np.repeat(np.arange(len(far)) * self.RUN - first, counts) + np.arange(int(counts.sum()))]
            out.append(r)
        return out


def _straight_outlines(mask: np.ndarray, dx: int = 0, dy: int = 0, halves: bool = True) -> str:
    """A mask as one path (fill-rule evenodd) of straight-sided outlines, in units of half a pixel
    counted from the middle of the mask's first half pixel (plus `dx`, `dy`, for a piece of a larger
    mask). Or with `halves` off, quicker and a little coarser: in whole pixels, counted from the
    middle of the first.

    For white that is worked out on fine pixels and lies under ink (under flat colours, and what is
    added under one ink): its edge is never seen, so it needs no smoothing and no curves. Tracing it
    as a plate is traced took most of the time on a large design, and drawing it pixel by pixel
    (a rectangle for every row of a slanting edge) made plates of many megabytes.

    Each outline runs through the middles of the mask's outermost half pixels (or pixels), so a
    quarter of a pixel (or half of one) inside the mask's own edge, and is simplified by up to one
    pixel: that much it takes to draw the pixel steps along a slanting edge as one line, whatever
    the slant (anything less leaves a point at every step). So the outline can cut up to three
    quarters (or half) of a pixel across a hollow in the mask's edge, a part one pixel thin (or two)
    is left out, and no specks are removed or holes filled: the plates have had that done, and the
    white follows them."""
    m = _bytes(np.ascontiguousarray(mask))
    if halves:
        m = cv2.resize(m, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        q = cv2.approxPolyDP(c, 2.0 if halves else 1.0, True).reshape(-1, 2) if len(c) >= 3 else ()
        if len(q) >= 3:
            out.append("M" + "L".join(f"{x + dx} {y + dy}" for x, y in q.tolist()) + "Z")
    return "".join(out)


_PIECE = 1024                                                              # fine pixels to a side of a piece of the flat underbase


def _flat_underbase(outlines: list, meant: np.ndarray, tp: Params, art: tuple) -> str:
    """Flat colours: the underbase as a plate body. It is what the colour plates draw, pulled in a
    quarter millimetre, and nothing else: no white under what the separation took for paper, under
    detail too fine for the plates, across a gap or in a cutout.

    `outlines`: each plate's outlines; `meant`: the areas they were traced from (both in working
    pixels). Where two colours touch, tracing leaves a hairline between their outlines. The white is
    not pulled back from those (only they are bridged, and only where `meant` says there is ink).

    It is worked out on pixels of an eighth of a millimetre or finer, so many to a working pixel that
    a quarter millimetre is (within a hundredth) a whole number of them; and a piece of the plate at
    a time, each with enough around it to get its edges right, so a design of any size gets the same.
    Each piece is outlined on its own (see _straight_outlines), lapping a little over its neighbours
    so nothing shows between. (A piece no outline comes near is all white or has none: those are not
    worked out, and the white ones are drawn as plain rectangles.)"""
    ox, oy, aw, ah = art
    h, w = meant.shape
    px = aw / w
    f = int(np.ceil(px / 0.125 - 1e-6))                                      # fine pixels to a working one
    f = next((k for k in range(f, f + 8) if abs(round(0.25 * k / px) * px / k - 0.25) <= 0.01), f)
    W, H, mm_px = w * f, h * f, px / f
    n, pad, lap = _PIECE, int(np.ceil(0.9 / mm_px)) + 4, 2                   # (0.9 mm: the seams' reach and the pull-in)
    plates = []
    for polys in outlines:
        rings = polys_to_rings(polys, tp.corner_deg, tp.precision, 0.25 / f)   # (to a quarter of a fine pixel)
        if rings:
            plates.append(_Rings(rings))
    # On the working pixels, roughly: the ones an outline passes through or beside, and what is filled.
    lines, filled = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    for plate in plates:
        rough = [np.rint((r - 0.5) * 8).astype(np.int32) for r in plate.rings]
        cv2.polylines(lines, rough, True, 1, lineType=cv2.LINE_8, shift=3)
        one = np.zeros((h, w), np.uint8)
        cv2.fillPoly(one, rough, 1, lineType=cv2.LINE_8, shift=3)
        filled |= one
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    edge = max(1, int(round(0.25 / mm_px)))                                  # (what _pulled_in takes off at the picture's sides)
    meant, parts, plain = meant.astype(np.uint8), [], []
    for y0 in range(0, H, n):
        for x0 in range(0, W, n):
            ys, ye, xs, xe = max(0, y0 - pad), min(H, y0 + n + pad), max(0, x0 - pad), min(W, x0 + n + pad)
            ty, by, lx, rx = max(ys, y0 - lap), min(ye, y0 + n + lap), max(xs, x0 - lap), min(xe, x0 + n + lap)
            rows, cols = slice(max(0, ys // f - 1), (ye - 1) // f + 2), slice(max(0, xs // f - 1), (xe - 1) // f + 2)
            if not lines[rows, cols].any():                                  # no outline near: all of it filled, or none
                if filled[(ys + ye) // 2 // f, (xs + xe) // 2 // f]:
                    plain.append((max(lx, edge), max(ty, edge), min(rx, W - edge), min(by, H - edge)))
                continue
            drawn = np.zeros((ye - ys, xe - xs), np.uint8)
            for plate in plates:
                near = plate.near(xs / f, ys / f, xe / f, ye / f)
                if near:
                    drawn |= rings_to_mask(near, drawn.shape, (f, f), (xs, ys)).astype(np.uint8)
            if not drawn.any():
                continue
            seams = _gaps_closed(drawn, 0.3, mm_px) & ~drawn                # what closing the gaps adds:
            if seams.any():                                                  # only where ink was meant
                r0, c0 = ys // f, xs // f
                want = np.repeat(np.repeat(meant[r0:(ye - 1) // f + 1, c0:(xe - 1) // f + 1], f, axis=0), f, axis=1)
                seams &= want[ys - r0 * f:ye - r0 * f, xs - c0 * f:xe - c0 * f]
            under = _pulled_in(drawn | seams, mm_px) & drawn
            d = _straight_outlines(under[ty - ys:by - ys, lx - xs:rx - xs])
            if d:                                                           # (in half fine pixels, from the middle of the first)
                parts.append(f'<g transform="translate({ox + (lx + 0.25) * aw / W:.3f} {oy + (ty + 0.25) * ah / H:.3f}) '
                             f'scale({aw / W / 2:.6f} {ah / H / 2:.6f})"><path fill-rule="evenodd" d="{d}"/></g>')
    # The plain pieces: those side by side as one rectangle, and rows of them that match as one.
    rects = []
    for x1, y1, x2, y2 in plain:
        if x2 <= x1 or y2 <= y1:
            continue
        if rects and rects[-1][1] == y1 and rects[-1][3] == y2 and rects[-1][2] >= x1:
            rects[-1][2] = x2
        else:
            rects.append([x1, y1, x2, y2])
    tall = {}
    for x1, y1, x2, y2 in rects:
        above = tall.pop((x1, x2, y1), None) or tall.pop((x1, x2, y1 + 2 * lap), None)
        tall[(x1, x2, y2)] = (above[0] if above else y1, y2)
    if tall:
        d = "".join(f"M{x1} {y1}h{x2 - x1}v{y2 - y1}h{x1 - x2}z" for (x1, x2, _), (y1, y2) in tall.items())
        parts.append(f'<g transform="translate({ox:.3f} {oy:.3f}) scale({aw / W:.6f} {ah / H:.6f})"><path d="{d}"/></g>')
    return "".join(parts)


def _added_white(ink: np.ndarray, has: np.ndarray, where: np.ndarray, mm: float, lean: bool = False, mm_y: float = 0.0) -> np.ndarray:
    """One ink: the white to add to an underbase. `ink` is the ink plate as it goes out, `has` the
    white that is there, and `where` the shape the underbase is for (before its pull-in), all as
    masks on the same pixels of `mm` millimetres (by `mm_y`, if not square). The answer is the solid
    core of the ink (see _solid_core) within that shape, but only what that adds to the white that is
    there, with a little overlap so that no hairline is left between the two. (Not the slivers one
    pixel thin that the two leave along a shared edge; a thin part on its own is no sliver.)"""
    # (All in bytes of 0 and 1, each step into the array of the one before: these are millions of pixels.)
    more = _solid_core(ink, mm, lean, mm_y)
    more &= _bytes(where)
    has = _bytes(has)
    added = np.bitwise_not(has)                                             # (of a 0 or 1 that is 255 or 254: its last bit is "not", and
    added &= more                                                           # the "and" keeps only that) what the core adds to the white there
    far = cv2.dilate(has, np.ones((5, 5), np.uint8))
    np.bitwise_not(far, out=far)
    far &= added                                                            # ...the part of that away from the white
    two = np.ones((2, 2), np.uint8)                                         # (opened with the square put back where it was taken from:
    gain = cv2.dilate(cv2.erode(added, two, anchor=(0, 0)), two, anchor=(1, 1))   # OpenCV's own opening shifts a 2 x 2 by a pixel)
    gain |= far                                                             # worth adding: more than a sliver, or on its own
    out = cv2.dilate(gain, np.ones((9, 9), np.uint8))
    out &= more
    inner = cv2.erode(has, np.ones((9, 9), np.uint8))
    np.bitwise_not(inner, out=inner)
    out &= inner                                                            # the core around that, lapping four pixels over the white
    return out.view(bool)


def _shape_at(shape: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, aw: float, ah: float) -> np.ndarray:
    """The underbase's shape (a mask over the artwork, aw x ah mm) at the points x_mm by y_mm, measured
    from the artwork's corner (both going up): the pixel each point falls in, nothing rounded up,
    nothing beyond it."""
    sh, sw = shape.shape
    xi, yi = np.floor(x_mm / aw * sw).astype(int), np.floor(y_mm / ah * sh).astype(int)
    xc, yc = np.clip(xi, 0, sw - 1), np.clip(yi, 0, sh - 1)
    if not len(xc) or not len(yc):
        return np.zeros((len(yc), len(xc)), bool)
    # (The points come in order, left to right and top to bottom: each pixel of the shape so many times over, in a block.)
    part = _bytes(np.ascontiguousarray(shape[yc[0]:yc[-1] + 1, xc[0]:xc[-1] + 1]))
    out = np.repeat(np.repeat(part, np.bincount(yc - yc[0]), axis=0), np.bincount(xc - xc[0]), axis=1)
    out[(yi < 0) | (yi >= sh), :] = 0
    out[:, (xi < 0) | (xi >= sw)] = 0
    return out.view(bool)


def _box_around(m: np.ndarray, pad: int) -> tuple | None:
    """The rows and columns of a mask that have anything in them, with `pad` more on every side:
    (first row, row after the last, first column, column after the last); None for an empty mask."""
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    if not len(rows):
        return None
    return int(rows[0]) - pad, int(rows[-1]) + 1 + pad, int(cols[0]) - pad, int(cols[-1]) + 1 + pad


def _piece_outlined(more: np.ndarray, f: int, ys: int, xs: int, ty: int, by: int, lx: int, rx: int) -> str:
    """One piece of added white as a path (see _straight_outlines; "" if there is none in it), in fine
    pixels of the whole plate, counted from the middle of the first. `more` is the white on fine pixels, `f` of them to a plate pixel, starting at
    the plate's pixel (ys, xs); the piece is its part for the plate's pixels [ty, by) x [lx, rx),
    and two fine pixels more on every side, so that it laps over the pieces beside it (their
    outlines run a little inside their masks, and a hairline would be left between them)."""
    y0, x0 = max(0, (ty - ys) * f - 2), max(0, (lx - xs) * f - 2)
    return _straight_outlines(more[y0:(by - ys) * f + 2, x0:(rx - xs) * f + 2], xs * f + x0, ys * f + y0, halves=False)


def _added_dither_white(ink: np.ndarray, has: np.ndarray, shape: np.ndarray, aw: float, ah: float) -> tuple[list, int]:
    """One ink, dithered: the white to add (see _added_white), as paths in units of 1/f of a dither
    square, one for each piece that has any (see _piece_outlined); f is returned with them. `ink`
    and `has` are the two plates' squares.

    Worked out on pixels of a sixteenth of a millimetre or finer (a whole number of them to a
    square), a piece of the plate at a time, so that a quarter millimetre is a quarter millimetre
    however large the design is."""
    H, W = ink.shape
    grain = max(aw / W, ah / H)
    f = max(1, int(np.ceil(grain / 0.0625 - 1e-6)))
    # ...and of the next few, the number that puts a quarter millimetre nearest to a whole number of
    # them from below (the pull-in is rounded up to whole pixels: 0.8 mm squares in 16 lose nothing).
    f = min(range(f, f + 8), key=lambda k: (round(np.ceil(0.25 * k / grain - 1e-6) * grain / k, 6), k))
    mx, my = aw / W / f, ah / H / f
    n = max(16, 2048 // f)                                                  # squares to a piece (about 4 million fine pixels)
    pad = int(np.ceil((0.25 / min(mx, my) + 9) / f)) + 1                    # squares around it that can still matter to it
    ink, has, parts = ink.astype(np.uint8), has.astype(np.uint8), []
    for y0 in range(0, H, n):
        for x0 in range(0, W, n):
            ys, ye, xs, xe = max(0, y0 - pad), min(H, y0 + n + pad), max(0, x0 - pad), min(W, x0 + n + pad)
            box = _box_around(ink[ys:ye, xs:xe] & ~has[ys:ye, xs:xe], pad)   # ink there that hasn't its white
            if box is None:                                                  # none: nothing to add
                continue
            # Only around those squares: what is added lies within a square of them.
            ty, by, lx, rx = max(y0, ys + box[0]), min(y0 + n, H, ys + box[1]), max(x0, xs + box[2]), min(x0 + n, W, xs + box[3])
            if ty >= by or lx >= rx:
                continue
            ys, ye, xs, xe = max(0, ty - pad), min(H, by + pad), max(0, lx - pad), min(W, rx + pad)

            def fine(m):
                return np.repeat(np.repeat(m[ys:ye, xs:xe], f, axis=0), f, axis=1)
            where = _shape_at(shape, (np.arange(xs * f, xe * f) + 0.5) * mx, (np.arange(ys * f, ye * f) + 0.5) * my, aw, ah)
            parts.append(_piece_outlined(_added_white(fine(ink), fine(has), where, mx, mm_y=my), f, ys, xs, ty, by, lx, rx))
    return [d for d in parts if d], f


def _added_screen_white(ink: list, has: list, tp: Params, shape: np.ndarray, size: tuple, px_mm: float,
                        aw: float, ah: float) -> tuple[list, int]:
    """One ink, screened: the white to add (see _added_white), as paths in units of 1/g of a plate
    pixel, one for each piece that has any (see _piece_outlined); g is returned with them. `ink` and
    `has` are the two plates' outlines as they go out (in plate pixels of `px_mm`, on a plate of
    `size` = (W, H) pixels; `tp`: how they are drawn).

    Worked out on pixels of a sixteenth of a millimetre or finer (a whole number of them to a plate
    pixel), a piece of the plate at a time, so that a quarter millimetre is a quarter millimetre
    however large the design is and however coarse the plate's own pixels are."""
    W, H = size
    g = max(1, int(np.ceil(px_mm / 0.0625 - 1e-6)))
    fine = px_mm / g
    n = max(16, 2048 // g)                                                  # plate pixels to a piece
    pad = int(np.ceil((0.25 / fine + 9) / g)) + 1
    # (The curves as they are written out, followed to a quarter of a fine pixel: a long curve drawn
    # as a few straight lines would put the white beside the ink.)
    ink, has = (polys_to_rings(q, tp.corner_deg, tp.precision, 0.25 / g) for q in (ink, has))
    if not ink:
        return [], g
    # First on the plate's own pixels, roughly: where is there ink at all that hasn't white right
    # under it? Only pieces with some of that are worked out finely.
    # (Ink taken generously and pulled in by less than a quarter millimetre; white only where a pixel
    # is all white and no edge of the white passes through it. So nothing that could get white is missed.)
    touched, edges = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    cv2.fillPoly(touched, [np.rint((r - 0.5) * 8).astype(np.int32) for r in ink], 1, lineType=cv2.LINE_8, shift=3)
    cv2.polylines(edges, [np.rint((r - 0.5) * 8).astype(np.int32) for r in has], True, 1, lineType=cv2.LINE_8, shift=3)
    k = int(0.25 / px_mm)
    core = cv2.erode(touched, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)), borderType=cv2.BORDER_CONSTANT, borderValue=0) if k else touched
    wanting = core.astype(bool) & ~(rings_to_mask(has, (H, W)) & ~edges.astype(bool))
    parts, ink, has = [], _Rings(ink), _Rings(has)
    for y0 in range(0, H, n):
        for x0 in range(0, W, n):
            ys, ye, xs, xe = max(0, y0 - pad), min(H, y0 + n + pad), max(0, x0 - pad), min(W, x0 + n + pad)
            box = _box_around(wanting[ys:ye, xs:xe], pad)
            if box is None:
                continue
            # Only around the pixels that want white: what is added lies within `pad` of them.
            ty, by, lx, rx = max(y0, ys + box[0]), min(y0 + n, H, ys + box[1]), max(x0, xs + box[2]), min(x0 + n, W, xs + box[3])
            if ty >= by or lx >= rx:
                continue
            ys, ye, xs, xe = max(0, ty - pad), min(H, by + pad), max(0, lx - pad), min(W, rx + pad)
            near = [rings.near(xs, ys, xe, ye) for rings in (ink, has)]
            there = ((ye - ys) * g, (xe - xs) * g)
            where = _shape_at(shape, (np.arange(xs * g, xe * g) + 0.5) * fine, (np.arange(ys * g, ye * g) + 0.5) * fine, aw, ah)
            more = _added_white(rings_to_mask(near[0], there, (g, g), (xs * g, ys * g)),
                                rings_to_mask(near[1], there, (g, g), (xs * g, ys * g)), where, fine, lean=True)
            parts.append(_piece_outlined(more, g, ys, xs, ty, by, lx, rx))
    return [d for d in parts if d], g


def _added_groups(paths: list, ax: float, ay: float, ux: float, uy: float) -> str:
    """The added white's paths (see _piece_outlined) on the plate, each a group of its own (they lap
    over each other, and each is filled by itself): the artwork's corner is at (ax, ay) mm and a
    fine pixel is ux by uy mm. (Their units are fine pixels, counted from the middle of the first.)"""
    place = f'<g transform="translate({ax + ux / 2:.4f} {ay + uy / 2:.4f}) scale({ux:.8f} {uy:.8f})">'
    return "".join(f'{place}<path fill-rule="evenodd" d="{d}"/></g>' for d in paths)


def _is_white(h: str) -> bool:
    return min(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) > 235


def _plate_specs(rgb: np.ndarray, p: PrintParams, W: int, H: int, aw: float = 0.0):
    """Returns (specs, plane(i), info): specs are (name, hex, angle); plane(i) is the ink amount
    (0..1) for plate i at working resolution."""
    lut = _lut(p)
    if p.method == "one":
        work = _work(rgb, W, H)
        g = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        if lut is not None:
            g = cv2.LUT(g, lut)
        plane = 1 - g.astype(np.float32) / 255
        if p.dark and aw:
            mm_px = aw / work.shape[1]              # the plane keeps the picture's own pixels when it has fewer than the plate
            if p.dark_invert:
                region = _design_region(plane, mm_px, p.outline_mm)
                plane = (1 - plane) * cv2.GaussianBlur(region.astype(np.float32), (0, 0), 0.6)
            if p.underbase and not _is_white(p.ink):
                # The white goes under what prints: the picture after Tone (brightness, contrast, invert),
                # which the ink plate is made from too (after smoothing it, if that is set). Taken from
                # the picture before Tone, the white lay under the wrong half of an inverted picture, and
                # under greys that Tone had turned to no ink.
                inked = ht._tone(np.clip(plane, 0, 1), p)
                ub = _underbase_mask(inked, mm_px)

                def planes(i):
                    return ub if i == 0 else plane
                planes.underbase = 0                                # plate 0 is the underbase made here, plate 1 its ink
                k = max(1, int(round(0.8 / mm_px)))                 # and the shape it is for, before the pull-in (as in _underbase_mask)
                planes.underbase_for = cv2.morphologyEx((inked > 0.03).astype(np.uint8), cv2.MORPH_CLOSE,
                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
                return [("White underbase", "#ffffff", 0.0), ("Ink", p.ink, p.angle)], planes, {"garment": p.dark_bg}
            return [("Ink", p.ink, p.angle)], (lambda i: plane), {"garment": p.dark_bg}
        return [("Ink", p.ink, p.angle)], (lambda i: plane), {}
    if p.method == "cmyk":
        work = _work(rgb, W, H)
        if lut is not None:
            work = cv2.LUT(work, lut)
        chans = ht.cmyk_split(work, p.black_amount)
        specs = []
        for i, (name, hx, ang) in enumerate(ht.CMYK_PLATES):
            col = p.cmyk_colors[i] if i < len(p.cmyk_colors) and p.cmyk_colors[i] else hx
            specs.append((name, col, _angle(p, i, ang)))
        return specs, (lambda i: chans[i]), {}
    work = _work(rgb, W, H, SPOT_WORK_PX)
    if lut is not None:
        work = cv2.LUT(work, lut)
    planes, pal, paper, solid = _spot_planes(work, p)
    specs = [(pal[i]["name"], pal[i]["hex"], _angle(p, i, DEFAULT_ANGLES[i % len(DEFAULT_ANGLES)]))
             for i in range(len(planes))]
    fn = lambda i: planes[i]
    fn.solid = (solid, work.shape[1], work.shape[0]) if solid is not None else None
    return specs, fn, {"palette": pal, "paper": paper}


def _to_grid(plane: np.ndarray, W: int, H: int, smoothing: float) -> np.ndarray:
    h, w = plane.shape
    if (w, h) != (W, H):
        plane = cv2.resize(plane, (W, H), interpolation=cv2.INTER_AREA if W < w else cv2.INTER_CUBIC)
    if smoothing > 0:
        plane = cv2.GaussianBlur(plane, (0, 0), smoothing)
    return plane


# --------------------------------------------------------------------------- build

def _flat_dark(rgb: np.ndarray, p: PrintParams, mm_px: float) -> np.ndarray:
    """Swap positives and negatives: dark parts of the design become garment, light parts (and the outline
    margin) print as white, other colours stay."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    dark = 1 - g
    region = _design_region(dark, mm_px, p.outline_mm).astype(bool)
    f = rgb.astype(np.float32)
    chroma = f.max(2) - f.min(2)
    is_dark = (g < 0.3) & (chroma < 60)
    is_light = (g > 0.85) & (chroma < 40)
    gar = np.array([int(p.dark_bg[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    out = np.empty_like(rgb); out[:] = gar
    keep = region & ~is_dark
    out[keep] = rgb[keep]
    out[region & is_light | (region & (dark < 0.08))] = (255, 255, 255)
    return out


def _flat(rgb: np.ndarray, p: PrintParams, page: PageOpts) -> Job:
    lut = _lut(p)
    if lut is not None:
        rgb = cv2.LUT(rgb, lut)
    mm_px = fit_artwork(page, rgb.shape[1], rgb.shape[0])[2] / rgb.shape[1]
    if p.dark and p.dark_invert:
        rgb = _flat_dark(rgb, p, mm_px)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "levels.png")
        Image.fromarray(rgb).save(f)
        picked = [c for c in p.colors if c] if p.picked else []
        sp = sep.SepParams(inks=len(picked) or p.inks, paper=p.paper, trap=p.trap, speckle=p.speckle,
                           smooth=p.smooth, colors=tuple(picked) if picked else p.colors, names=p.names, fixed=bool(picked))
        keep = {} if p.dark and p.underbase else None
        job = sep.build(f, sp, page, keep)
    if keep is not None:
        body = _flat_underbase(keep["outlines"], keep["meant"], keep["trace"], job.art)
        if body:
            job.plates.insert(0, Plate("White underbase", "#ffffff", body))
    job.info["kind"] = "flat"
    if p.dark:
        job.info["garment"] = p.dark_bg
        job.info["overprint"] = False
    return job


def _parallel(fn, n):
    """Run fn(0..n-1) on threads so the plates build side by side (screening, blurring and
    contour finding spend much of their time outside Python's lock)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(n, os.cpu_count() or 1)) as ex:
        return list(ex.map(fn, range(n)))


def build(path, p: PrintParams, page: PageOpts) -> Job:
    rgb = load_rgb(path)
    if p.method == "spot" and p.kind == "flat":
        return _flat(rgb, p, page)
    h, w = rgb.shape[:2]
    art = fit_artwork(page, w, h)
    ax, ay, aw, ah = art
    job = Job(opts=page, art=art)
    info: dict = {}

    if p.render == "screen":
        cell_mm = 25.4 / p.lpi
        if aw * ah / (cell_mm * cell_mm) > ht.MAX_DOTS:
            raise ValueError(f"About {int(aw * ah / (cell_mm * cell_mm)):,} dots per plate is too many. "
                             f"Lower the LPI or use a smaller page.")
        px_mm = 25.4 / p.ppi
        if p.draft:                 # draft: about 6 pixels across a cell instead of ~10
            px_mm = max(px_mm, cell_mm / 6)
        W, H = max(8, int(round(aw / px_mm))), max(8, int(round(ah / px_mm)))
        if W * H > ht.MAX_PIXELS:
            k = (ht.MAX_PIXELS / (W * H)) ** 0.5
            W, H = max(8, int(W * k)), max(8, int(H * k))
            px_mm = aw / W
        cell_px = cell_mm / px_mm
        if cell_px < 3:
            raise ValueError("Output resolution is too low for this LPI. Raise the pixels/inch or lower the LPI.")
        stamp = ht._stamp_field(p.stamp) if p.shape == "custom" else None
        table = ht._rank_table(p.shape, stamp)
        tp = Params(speckle=0, smooth_sigma=p.edge, epsilon=0.8, corner_deg=125.0, precision=1)
        info.update(kind="screen", spacing_mm=round(cell_mm, 3), lpi=p.lpi, ppi=round(25.4 / px_mm))
    else:
        size = p.dither_size
        W, H = max(8, int(round(aw / size))), max(8, int(round(ah / size)))
        if W * H > MAX_DITHER_PX:
            raise ValueError(f"A dither size of {size:.2f} mm makes {W * H:,} pixels on this page, which is too "
                             f"many. Make the dither size larger or use a smaller page.")
        info.update(kind="dither", size_mm=round(size, 3), dither_ppi=round(25.4 / size))
        if p.dither == "serpentine":
            info["pattern_mm"] = round(p.pattern_size * size, 2)

    serp = _serpentine(p.pattern, round(p.pattern_size, 2)) if (p.render == "dither" and p.dither == "serpentine") else None
    specs, plane, extra = _plate_specs(rgb, p, W, H, aw)
    info.update(extra)
    total = 0

    solid = getattr(plane, "solid", None)
    under = getattr(plane, "underbase", None)
    kept, keeping = {}, threading.Lock()

    def laid(i, t):
        """Tone t (0..1) as plate i's ink on the plate's own pixels: screened or dithered."""
        if p.render == "screen":
            return ht._screen(t, cell_px, specs[i][2], table, p.shape, stamp)
        return dither_mask(t, p.dither, i, serp)

    def finished(i):
        """Plate i as it goes out: its dithered pixels, or (a screen) the outlines traced from it.
        (The ink over an underbase is wanted twice, for its own plate and for the underbase, and is
        worked out once.)"""
        def make():
            mask = laid(i, ht._tone(np.clip(_to_grid(plane(i), W, H, p.smoothing), 0, 1), p))
            return mask_to_polys(mask, tp, fast=p.draft, edge=True) if p.render == "screen" else mask   # (ink at the picture's edge is outlined along it)
        if under is None or i != under + 1:
            return make()
        with keeping:
            if i not in kept:
                kept[i] = make()
            return kept[i]

    def solid_body(i):
        """Flat areas of ink i as clean vector shapes, in page millimetres."""
        if not solid:
            return "", 0
        masks, sw, sh = solid
        if not masks[i].any():
            return "", 0
        sd = mask_to_path(masks[i], Params(speckle=0, smooth_sigma=0.6, epsilon=0.8, corner_deg=125.0, precision=1), fast=p.draft, edge=True)
        return ((f'<g transform="translate({ax:.3f} {ay:.3f}) scale({aw / sw:.6f} {ah / sh:.6f})">'
                 f'<path fill-rule="evenodd" d="{sd}"/></g>') if sd else ""), sd.count("M")

    def one(i):
        name, ink, ang = specs[i]
        solid_white = name == "White underbase"                    # a solid, never screened
        t = np.where(_to_grid(plane(i), W, H, 0) > 0.5, 1.0, 0.0).astype(np.float32) if solid_white else None
        if p.render == "screen":
            # (Ink at the picture's edge is outlined along it: see mask_to_polys.)
            polys = mask_to_polys(laid(i, t), tp, fast=p.draft, edge=True) if solid_white else finished(i)
            d = polygons_to_path(polys, tp.corner_deg, tp.precision)
            body = (f'<g transform="translate({ax:.3f} {ay:.3f}) scale({px_mm:.6f})">'
                    f'<path fill-rule="evenodd" d="{d}"/></g>') if d else ""
            n = d.count("M")
            if i == under:                                          # the underbase made here (not a plate that is only named so)
                # ...also goes well inside its ink, wherever that is: inside the ink's outlines as they
                # go out, worked out on fine pixels and outlined along those pixels with straight lines
                # (it lies under ink, so its edge is never seen and needs no smoothing).
                md, g = _added_screen_white(finished(i + 1), polys, tp, plane.underbase_for, (W, H), px_mm, aw, ah)
                body += _added_groups(md, ax, ay, float(f"{px_mm:.6f}") / g, float(f"{px_mm:.6f}") / g)   # (on the ink plate's own scale, as written; not counted as dots)
            sb, sn = solid_body(i)
            return body + sb, n + sn
        mask = laid(i, t) if solid_white else finished(i)
        d, n = runs_path(mask)
        body = (f'<g transform="translate({ax:.3f} {ay:.3f}) scale({aw / W:.6f} {ah / H:.6f})">'
                f'<path d="{d}"/></g>') if d else ""
        if i == under:
            # Likewise for a dither: well inside the ink's own squares.
            md, f = _added_dither_white(finished(i + 1), mask, plane.underbase_for, aw, ah)
            body += _added_groups(md, ax, ay, float(f"{aw / W:.6f}") / f, float(f"{ah / H:.6f}") / f)   # (on the ink plate's own scale, as written)
        sb, sn = solid_body(i)
        return body + sb, n + sn

    results = _parallel(one, len(specs)) if (p.render == "screen" and len(specs) > 1) else [one(i) for i in range(len(specs))]
    for (name, ink, ang), (body, n) in zip(specs, results):
        job.plates.append(Plate(name, ink, body))
        total += n
    info.update(dots=total, overprint=p.method != "one" and not p.dark, draft=bool(p.draft and p.render == "screen"))
    job.info = info
    return job
