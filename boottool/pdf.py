"""Plates as PDF: one page per plate, pure black, at the page's real size.

Many print shops and film printers take PDF rather than SVG. A plate's film is already vector
shapes (page.py), so this writes those same shapes into a PDF: the artwork, the registration marks
and the plate's name, exactly where the SVG film has them. It is made from the SVG film itself, so
the two cannot drift apart, and it needs no extra library.

The film is read by film.py, which the picture export (raster.py) uses too.
"""
from __future__ import annotations

import io
import math
import zipfile
import zlib

from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path

from . import __version__, film
from .page import PT, Job, slug

_NARROW = 82                          # % of Helvetica's width: about as wide as the SVG's Arial Narrow


def _r(v: float) -> str:
    """A number as PDF wants it: plain decimals, never 1e-05."""
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("A plate has a shape with a broken number in it; it cannot be written as PDF.")
    s = repr(round(v, 4))
    if "e" in s:
        s = f"{v:.4f}"
    s = s[:-2] if s.endswith(".0") else s
    return "0" if s == "-0" else s


def _m(v: float) -> str:
    """A number of a transform: kept finer, because it multiplies every point drawn with it."""
    if not math.isfinite(v):
        raise ValueError("A plate has a transform with a broken number in it; it cannot be written as PDF.")
    return f"{v:.8f}".rstrip("0").rstrip(".") or "0"


class _PathPen(BasePen):
    """Takes a path's moves, lines and curves and writes them as PDF path operators."""

    def __init__(self, out: list):
        super().__init__(None)
        self.out = out
        self.back = ""                 # the line that would lead back to where this outline began

    def _moveTo(self, p):
        at = f"{_r(p[0])} {_r(p[1])}"
        self.out.append(at + " m")
        self.back = at + " l"

    def _lineTo(self, p):
        self.out.append(f"{_r(p[0])} {_r(p[1])} l")

    def _curveToOne(self, a, b, c):
        self.out.append(f"{_r(a[0])} {_r(a[1])} {_r(b[0])} {_r(b[1])} {_r(c[0])} {_r(c[1])} c")

    def _closePath(self):
        if self.out[-1] == self.back:  # closing draws that line anyway (the path reader spells it out)
            self.out.pop()
        self.out.append("h")


def _draw(shape: film.Shape) -> list[str]:
    """Fill it, outline it, or both, as the film says."""
    rule = "*" if shape.evenodd else ""
    if shape.stroke is None:
        return ["f" + rule] if shape.fill else ["n"]
    return [f"{_r(shape.stroke)} w", "B" + rule if shape.fill else "S"]


def _pdf_text(s: str) -> str:
    """Text inside a PDF's ( ). The built-in fonts know Western letters only: anything else becomes "?"."""
    raw = s.encode("cp1252", errors="replace")
    return "".join("\\" + chr(b) if b in b"()\\" else chr(b) if 32 <= b <= 126 else f"\\{b:03o}" for b in raw)


def _page_stream(job: Job, i: int) -> bytes:
    """Everything on plate i's film, as PDF drawing operators (in millimetres, y running down)."""
    pw, ph = job.page
    k = 1 / PT                                                             # points per millimetre
    out = ["q", f"{_m(k)} 0 0 {_m(-k)} 0 {_m(ph * k)} cm", "0 g", "0 G"]
    for it in film.read(job, i, "written as PDF"):
        moved = it.matrix != film.PLAIN
        if moved:
            out += ["q", " ".join(_m(x) for x in it.matrix) + " cm"]
        if isinstance(it, film.Text):
            # the page is drawn with y running down, as in the SVG; letters are turned upright again here
            out.append(f"BT /F1 {_r(it.size)} Tf {_r(it.gap * 100 / _NARROW)} Tc {_NARROW} Tz "
                       f"1 0 0 -1 {_r(it.x)} {_r(it.y)} Tm ({_pdf_text(it.words)}) Tj ET")
        else:
            if it.d is None:
                out.append(" ".join(_r(v) for v in it.box) + " re")
            else:
                parse_path(it.d, _PathPen(out))
            out += _draw(it)
        if moved:
            out.append("Q")
    out.append("Q")
    return "\n".join(out).encode("ascii")


def _hex_text(s: str) -> str:
    return "<FEFF" + s.encode("utf-16-be").hex().upper() + ">"


def plates_pdf(job: Job, only: int | None = None, title: str = "") -> bytes:
    """One PDF with a page per plate (or just plate `only`), each page the film as it prints."""
    which = list(range(len(job.plates))) if only is None else [only]
    if not which:
        raise ValueError("Nothing to download: no plates were produced.")
    pw, ph = job.page
    box = f"[0 0 {_m(pw / PT)} {_m(ph / PT)}]"
    objs: list[bytes] = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    kids = []
    for i in which:
        data = zlib.compress(_page_stream(job, i), 6)
        kids.append(len(objs) + 1)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox {box} /Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {len(objs) + 2} 0 R >>".encode("ascii"))
        objs.append(f"<< /Length {len(data)} /Filter /FlateDecode >>\nstream\n".encode("ascii") + data + b"\nendstream")
    objs[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[1] = f"<< /Type /Pages /Count {len(kids)} /Kids [{' '.join(f'{n} 0 R' for n in kids)}] >>".encode("ascii")
    name = title or " / ".join(job.plates[i].name for i in which)
    objs.append(f"<< /Title {_hex_text(name)} /Producer {_hex_text('Boot Tool ' + __version__)} >>".encode("ascii"))

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    at = []
    for n, body in enumerate(objs, 1):
        at.append(buf.tell())
        buf.write(f"{n} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    xref = buf.tell()
    buf.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for a in at:
        buf.write(f"{a:010d} 00000 n \n".encode("ascii"))
    buf.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info {len(objs)} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return buf.getvalue()


def plates_pdf_zip(job: Job, base: str = "plate") -> bytes:
    """A .zip with one single-page PDF per plate, named like the SVGs in the SVG .zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:                # the PDFs are compressed already
        for i, p in enumerate(job.plates):
            z.writestr(f"{base}-{i + 1:02d}-{slug(p.name)}.pdf", plates_pdf(job, i))
    return buf.getvalue()
