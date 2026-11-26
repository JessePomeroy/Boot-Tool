"""A plate's film as a plain list of what is on it.

page.py makes a film as SVG. The PDF export and the picture export (pdf.py, raster.py) both have
to draw exactly that film. Here it is read once, into shapes and texts with their place on the
page worked out, so that the two share one reading of it.

Only what films are made of is understood (groups, paths, rectangles and one line of text, in
black). Anything else stops with an error rather than being left out of a film without a word.
"""
from __future__ import annotations

import re
from typing import NamedTuple

from lxml import etree

from .page import Job, _film_group

_BLACK = {"#000", "#000000", "black"}
PLAIN = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)          # the transform that changes nothing


class Shape(NamedTuple):
    matrix: tuple            # a, b, c, d, e, f as SVG writes them: from the shape's own numbers to page millimetres
    d: str | None            # a path's data, or
    box: tuple | None        # a rectangle's x, y, width, height
    fill: bool
    stroke: float | None     # the width of its outline in its own numbers, or None for no outline
    evenodd: bool


class Text(NamedTuple):
    matrix: tuple
    x: float
    y: float
    size: float
    gap: float               # extra room between letters
    words: str


def times(m, n) -> tuple:
    """Transform m, then n."""
    return (m[0] * n[0] + m[1] * n[2], m[0] * n[1] + m[1] * n[3], m[2] * n[0] + m[3] * n[2], m[2] * n[1] + m[3] * n[3],
            m[4] * n[0] + m[5] * n[2] + n[4], m[4] * n[1] + m[5] * n[3] + n[5])


def _moved(text: str, m: tuple, cannot: str) -> tuple:
    """m with an SVG transform (translate, scale and matrix: what the films use) put before it."""
    rest = text.strip()
    while rest:
        got = re.match(r"(\w+)\s*\(([^)]*)\)\s*,?\s*", rest)
        v = [float(x) for x in re.split(r"[\s,]+", got.group(2).strip()) if x] if got else []
        kind = got.group(1) if got else ""
        if kind == "translate" and len(v) in (1, 2):
            step = (1, 0, 0, 1, v[0], v[1] if len(v) == 2 else 0)
        elif kind == "scale" and len(v) in (1, 2):
            step = (v[0], 0, 0, v[-1], 0, 0)
        elif kind == "matrix" and len(v) == 6:
            step = tuple(v)
        else:
            raise ValueError(f"A plate has a transform that cannot be {cannot}: {text!r}")
        m = times(step, m)
        rest = rest[got.end():]
    return m


def _walk(el, m: tuple, st: dict, out: list, cannot: str) -> None:
    tag = etree.QName(el).localname
    st = dict(st)                                       # fill, outline and line width: its own, or what its group gave it
    for k in ("fill", "stroke"):
        v = el.get(k)
        if v is not None:
            if v != "none" and v.lower() not in _BLACK:
                raise ValueError(f"A plate has a colour other than black ({v!r}); films are pure black.")
            st[k] = v != "none"
    if el.get("stroke-width") is not None:
        st["width"] = float(el.get("stroke-width"))
    if el.get("fill-rule") is not None:
        st["evenodd"] = el.get("fill-rule") == "evenodd"
    paint = (st["fill"], st["width"] if st["stroke"] else None, st["evenodd"])
    if tag == "g":
        m = _moved(el.get("transform") or "", m, cannot)
        for child in el:
            _walk(child, m, st, out, cannot)
    elif tag == "path":
        if el.get("transform"):
            raise ValueError(f"A plate has a path with its own transform; it cannot be {cannot}.")
        if (el.get("d") or "").strip():
            out.append(Shape(m, el.get("d"), None, *paint))
    elif tag == "rect":
        if el.get("transform") or el.get("rx") or el.get("ry"):
            raise ValueError(f"A plate has a rectangle that cannot be {cannot}.")
        out.append(Shape(m, None, tuple(float(el.get(k) or 0) for k in ("x", "y", "width", "height")), *paint))
    elif tag == "text":
        if len(el) or el.get("transform"):
            raise ValueError(f"A plate has text that cannot be {cannot}.")
        out.append(Text(m, float(el.get("x") or 0), float(el.get("y") or 0), float(el.get("font-size") or 4.2),
                        float(el.get("letter-spacing") or 0), el.text or ""))
    else:
        raise ValueError(f"A plate has something that cannot be {cannot} (<{tag}>).")


def read(job: Job, i: int, cannot: str) -> list:
    """What is on plate i's film, in the order it is drawn: Shapes and Texts, in page millimetres
    (y running down, as in the SVG). `cannot` finishes the error messages: "written as PDF"."""
    try:
        svg = etree.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{_film_group(job, i, ident=False)}</svg>'.encode("utf-8"),
                               etree.XMLParser(huge_tree=True))
    except etree.XMLSyntaxError:       # in practice: a character in an ink's name that no file can hold (pasted in, not typed)
        raise ValueError(f"Plate {i + 1} could not be {cannot}. If its ink name has unusual characters in it, "
                         f"type the name again and try once more.") from None
    out: list = []
    for el in svg:
        _walk(el, PLAIN, {"fill": True, "stroke": False, "width": 1.0, "evenodd": False}, out, cannot)
    return out
