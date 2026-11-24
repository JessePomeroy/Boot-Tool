import json
import re

import numpy as np
import pytest
from PIL import Image

from boottool import page as pg
from boottool import printing as pr

from test_picture_edge import plate_mask as _plate_mask
from test_plates import _post, ok, server  # noqa: F401 (server is a fixture)


def _naive_floyd(t):
    """Plain textbook Floyd-Steinberg, left to right, top to bottom."""
    a = t.astype(np.float64).copy()
    h, w = a.shape
    out = np.zeros((h, w), bool)
    for y in range(h):
        for x in range(w):
            on = a[y, x] >= 0.5
            out[y, x] = on
            e = a[y, x] - on
            for dx, dy, wt in pr._KERNELS["floyd"]:
                if 0 <= x + dx < w and 0 <= y + dy < h:
                    a[y + dy, x + dx] += e * wt
    return out


def test_error_diffusion_matches_textbook():
    t = np.random.default_rng(3).random((23, 31)).astype(np.float32)
    assert (pr.dither_mask(t, "floyd") == _naive_floyd(t)).all()


@pytest.mark.parametrize("algo", pr.DITHERS)
def test_dither_ink_matches_tone(algo):
    for tone in (0.15, 0.5, 0.85):
        m = pr.dither_mask(np.full((160, 160), tone, np.float32), algo)
        if algo == "atkinson" and tone != 0.5:        # Atkinson drops 1/4 of the error: crisp highlights and shadows
            assert (m.mean() < tone) if tone < 0.5 else (m.mean() > tone)
            continue
        assert abs(m.mean() - tone) < 0.05, (algo, tone, m.mean())
    assert not pr.dither_mask(np.zeros((40, 40), np.float32), algo).any()
    assert pr.dither_mask(np.ones((40, 40), np.float32), algo).all()


def test_runs_path_covers_exactly_the_ink():
    import re
    m = np.random.default_rng(4).random((30, 50)) > 0.6
    d, n = pr.runs_path(m)
    runs = re.findall(r"M(\d+) (\d+)h(\d+)v1h-\d+z", d)
    back = np.zeros_like(m)
    for x, y, l in runs:
        back[int(y), int(x):int(x) + int(l)] = True
    assert n == len(runs) and (back == m).all()


def test_levels_lut_and_coerce():
    assert pr._lut(pr.PrintParams()) is None
    lut = pr._lut(pr.PrintParams(lv_black=50, lv_white=200))
    assert lut[50] == 0 and lut[200] == 255 and lut[0] == 0 and lut[255] == 255
    lighter = pr._lut(pr.PrintParams(lv_gamma=1.5))
    assert lighter[128] > 128                                   # gamma above 1 lightens, like Photoshop
    p = pr.coerce_print({"lv_black": 250, "lv_white": 100, "lv_gamma": 99, "method": "x", "dither_size": 0})
    assert p.lv_white - p.lv_black >= 2 and p.lv_gamma == 9.99 and p.method == "one" and p.dither_size == 0.1
    assert pr.coerce_print({"shape": "dot"}).shape == "round"
    assert pr.coerce_print({"angles": [10, None, "x", 200]}).angles == (10.0, -1.0, -1.0, 90.0)


@pytest.fixture
def ramp(tmp_path):
    arr = np.tile(np.linspace(255, 0, 600, dtype=np.uint8), (300, 1))
    Image.fromarray(arr).save(tmp_path / "ramp.png")
    return tmp_path / "ramp.png"


def test_analyse_reports_histogram_and_auto_levels(ramp, lineart):
    a = pr.analyse(ramp)
    assert len(a["hist"]) == 256 and a["tonal"] and a["auto"][0] < 10 and a["auto"][2] > 240
    assert not pr.analyse(lineart)["tonal"] or len(pr.analyse(lineart)["hist"]) == 256


@pytest.mark.parametrize("kw,plates", [
    ({}, 1), ({"render": "dither"}, 1), ({"render": "dither", "dither": "blue"}, 1),
    ({"method": "cmyk"}, 4), ({"method": "cmyk", "render": "dither", "dither": "atkinson"}, 4),
    ({"method": "spot", "kind": "photo", "inks": 3}, 3),
    ({"method": "spot", "kind": "photo", "inks": 3, "render": "dither", "dither": "stucki"}, 3),
    ({"method": "spot", "kind": "flat", "inks": 3}, None),
])
def test_every_method_builds_valid_plates(color, kw, plates):
    job = pr.build(color, pr.coerce_print(kw), pg.PageOpts())
    if plates:
        assert len(job.plates) == plates
    assert job.plates
    ok(pg.compose(job, "all"))
    lay = pg.layered_svg(job)
    ok(lay)
    assert 'fill="#000"' in lay and "PLATE 1" in lay


def test_levels_change_the_ink_in_every_method(ramp):
    from boottool.preprocess import load_rgb
    rgb = load_rgb(ramp)

    def ink(**kw):
        p = pr.coerce_print(kw)
        specs, plane, _ = pr._plate_specs(rgb, p, 600, 300)
        return sum(float(plane(i).mean()) for i in range(len(specs)))
    for method in ("one", "cmyk", "spot"):
        extra = {"method": method, "kind": "photo", "inks": 3}
        base = ink(**extra)
        if method == "spot":                                # inks are picked again after Levels, so just check it acts
            assert ink(**extra, lv_white=128) != base and ink(**extra, lv_black=128) != base
            continue
        assert ink(**extra, lv_white=128) < base            # lighter half drops to paper
        assert ink(**extra, lv_black=128) > base            # darker half prints solid
        assert ink(**extra, lv_gamma=2.0) < base            # midtones lighter


def test_dither_size_controls_grain_and_has_a_cap(ramp):
    fine = pr.build(ramp, pr.coerce_print({"render": "dither", "dither_size": 0.2}), pg.PageOpts()).info
    coarse = pr.build(ramp, pr.coerce_print({"render": "dither", "dither_size": 0.8}), pg.PageOpts()).info
    assert fine["dots"] > coarse["dots"] * 4 and coarse["size_mm"] == 0.8
    with pytest.raises(ValueError, match="too many"):
        pr.build(ramp, pr.coerce_print({"render": "dither", "dither_size": 0.1}), pg.PageOpts(size="A3"))


def test_cmyk_colours_are_preview_only(color):
    std = pr.build(color, pr.coerce_print({"method": "cmyk"}), pg.PageOpts())
    custom = pr.build(color, pr.coerce_print({"method": "cmyk", "cmyk_colors": ["#112233", "", "", ""]}), pg.PageOpts())
    assert custom.plates[0].ink == "#112233" and std.plates[0].ink == "#00aeef"
    assert [p.body for p in std.plates] == [p.body for p in custom.plates]      # separation unchanged
    assert [p.name for p in custom.plates] == ["Cyan", "Magenta", "Yellow", "Black"]
    assert "mix-blend-mode" in pg.compose(custom, "all") and 'fill="#112233"' in pg.compose(custom, "all")


def test_spot_photo_uses_your_ink_colours_for_the_separation(color):
    auto = pr.build(color, pr.coerce_print({"method": "spot", "kind": "photo", "inks": 3}), pg.PageOpts())
    mine = pr.build(color, pr.coerce_print({"method": "spot", "kind": "photo", "inks": 3, "colors": ["#00ff00"]}), pg.PageOpts())
    again = pr.build(color, pr.coerce_print({"method": "spot", "kind": "photo", "inks": 3}), pg.PageOpts())
    assert again.info["palette"] == auto.info["palette"]                        # steady between redraws
    g = mine.info["palette"][0]["hex"]
    assert int(g[3:5], 16) > 230 and int(g[1:3], 16) < 30                        # your green, give or take rounding
    assert mine.info["palette"][0]["auto_hex"] == auto.info["palette"][0]["auto_hex"]
    assert mine.plates[0].body != auto.plates[0].body                           # separation really changed
    assert auto.info["paper"] and all(p["name"] for p in auto.info["palette"])


def test_print_endpoint_and_upload_histogram(server, color):
    up = json.loads(_post(server + "/api/upload", color.read_bytes(), "application/octet-stream")[1])
    assert len(up["hist"]) == 256 and len(up["auto"]) == 3 and "tonal" in up
    req = {"id": up["id"], "session": "p", "seq": 1, "tool": "print", "view": "all",
           "params": {"method": "cmyk", "render": "dither", "dither": "bayer", "dither_size": 0.5},
           "page": {"size": "A5", "orientation": "portrait"}}
    code, body, _ = _post(server + "/api/plates", req)
    res = json.loads(body)
    assert code == 200 and len(res["plates"]) == 4 and res["info"]["kind"] == "dither"
    ok(res["svg"])
    code, z, h = _post(server + "/api/download", {**req, "kind": "zip"})
    assert code == 200 and h["Content-Type"] == "application/zip"
    code, body, _ = _post(server + "/api/plates", {**req, "seq": 2, "params": {"shape": "custom", "stamp": []}})
    assert code == 400 and "Draw a shape" in json.loads(body)["error"]
    code, body, _ = _post(server + "/api/plates", {**req, "seq": 3, "params": {"render": "dither", "dither_size": 0.1}, "page": {"size": "A3"}})
    assert code == 400 and "too many" in json.loads(body)["error"]


def test_spot_photo_never_prints_on_paper(tmp_path):
    """A very pale ink next to white paper must not tint the paper."""
    arr = np.full((200, 300, 3), 255, np.uint8)
    arr[60:140, 80:220] = (200, 120, 60)
    Image.fromarray(arr).save(tmp_path / "blob.png")
    job = pr.build(tmp_path / "blob.png", pr.coerce_print(
        {"method": "spot", "kind": "photo", "inks": 2, "colors": ["#fff0d8", ""], "render": "dither", "dither": "bayer"}), pg.PageOpts())
    import re
    ax, ay, aw, ah = job.art
    for plate in job.plates:
        xs = [int(m) for m in re.findall(r"M(\d+) \d+h", plate.body)]
        ys = [int(m) for m in re.findall(r"M\d+ (\d+)h", plate.body)]
        W = max(xs) + 1
        assert min(ys) > 0.2 * (max(ys) + 1) and min(xs) > 0.15 * W       # nothing in the paper margin


def test_serpentine_tile_is_a_long_maze_and_inks_the_right_amount():
    from scipy import ndimage
    for name in pr.PATTERNS:
        tile = pr._serpentine(name, 9.0)
        assert tile.shape == (256, 256) and abs(float(tile.mean()) - 0.5) < 0.01          # ranked: fills evenly
        assert np.array_equal(tile, pr._serpentine(name, 9.0))                             # same every time
        for tone in (0.25, 0.5, 0.75):
            m = pr.dither_mask(np.full((200, 200), tone, np.float32), "serpentine", 0, tile)
            assert abs(m.mean() - tone) < 0.03
    mazes = pr.dither_mask(np.full((256, 256), 0.5, np.float32), "serpentine", 0, pr._serpentine("mazes", 9.0))
    noise = pr.dither_mask(np.full((256, 256), 0.5, np.float32), "random")
    big = lambda m: ndimage.label(m)[1]
    assert big(mazes) * 8 < big(noise)                                                       # few, long strokes


def test_serpentine_pattern_size_and_presets_change_the_result(color):
    a, b = pr._serpentine("mazes", 6.0), pr._serpentine("mazes", 16.0)
    assert not np.array_equal(a, b) and not np.array_equal(pr._serpentine("coral", 9.0), pr._serpentine("mazes", 9.0))
    base = {"render": "dither", "dither": "serpentine", "dither_size": 0.4}
    j1 = pr.build(color, pr.coerce_print({**base, "pattern": "mazes"}), pg.PageOpts())
    j2 = pr.build(color, pr.coerce_print({**base, "pattern": "fingerprints", "pattern_size": 20}), pg.PageOpts())
    assert j1.plates[0].body != j2.plates[0].body and j1.info["pattern_mm"] == 3.6
    ok(pg.compose(j1, "all"))
    cm = pr.build(color, pr.coerce_print({**base, "method": "cmyk"}), pg.PageOpts())
    assert len({p.body for p in cm.plates}) == 4                                            # plates are offset, not identical
    assert pr.coerce_print({"pattern": "nope", "pattern_size": 999}).pattern == "mazes"
    assert pr.coerce_print({"pattern_size": 999}).pattern_size == 40.0


def test_draft_preview_is_coarser_but_downloads_stay_full(color):
    import json
    from boottool import page as pg
    from boottool import printing as pr
    from boottool.ui import Engine, download_response
    full = pr.build(color, pr.coerce_print({"method": "cmyk"}), pg.PageOpts())
    draft = pr.build(color, pr.coerce_print({"method": "cmyk", "draft": True}), pg.PageOpts())
    assert draft.info["draft"] and not full.info["draft"]
    assert draft.info["ppi"] < full.info["ppi"] and [p.name for p in draft.plates] == [p.name for p in full.plates]
    assert pr.coerce_print({"draft": "x"}).draft is True
    # dither has no draft mode
    assert not pr.build(color, pr.coerce_print({"render": "dither", "draft": True}), pg.PageOpts()).info["draft"]


def _mix_image(path):
    a = np.full((240, 360, 3), 255, np.uint8)
    a[40:200, 30:150] = (0, 0, 0)                                    # flat black
    a[60:180, 200:330] = np.linspace(40, 215, 130, dtype=np.uint8)[None, :, None]   # grey ramp
    Image.fromarray(a).save(path)
    return str(path)


def test_picked_colours_are_exactly_the_inks(tmp_path):
    f = _mix_image(tmp_path / "m.png")
    cols = ("#808080", "#101010")
    for kind in ("flat", "photo"):
        job = pr.build(f, pr.PrintParams(method="spot", kind=kind, inks=5, picked=True, colors=cols), pg.PageOpts())
        assert [p.ink for p in job.plates] == list(cols) or {p.ink for p in job.plates} == set(cols)
        assert len(job.plates) == 2


def test_keep_solid_prints_flat_areas_as_vector(tmp_path):
    f = _mix_image(tmp_path / "m.png")
    base = pr.PrintParams(method="spot", kind="photo", inks=3)
    dots = pr.build(f, base, pg.PageOpts())
    solid = pr.build(f, pr.PrintParams(method="spot", kind="photo", inks=3, keep_solid=True), pg.PageOpts())
    assert sum(len(p.body) for p in solid.plates) < sum(len(p.body) for p in dots.plates)


def _bolt(path):
    a = np.full((200, 400, 3), 255, np.uint8)
    a[60:140, 80:320] = 0                                             # a dark bar
    a[90:110, 120:280] = 255                                          # with a light slot in it
    Image.fromarray(a).save(path)
    return str(path)


def test_dark_invert_swaps_positives_and_adds_an_outline(tmp_path):
    f = _bolt(tmp_path / "b.png")
    page = pg.PageOpts()
    light = pr.build(f, pr.PrintParams(method="one", ink="#000000", render="dither"), page)
    dark = pr.build(f, pr.PrintParams(method="one", ink="#ffffff", render="dither", dark=True, dark_invert=True, outline_mm=5.0), page)
    assert dark.info["garment"] == "#1b1b1d" and len(dark.plates) == 1
    assert dark.plates[0].body and dark.plates[0].body != light.plates[0].body
    none = pr.build(f, pr.PrintParams(method="one", ink="#ffffff", render="dither", dark=True, dark_invert=True, outline_mm=0.0), page)
    assert len(dark.plates[0].body) > len(none.plates[0].body)        # the margin ring adds ink


def test_dark_underbase_is_a_first_plate_for_coloured_ink_only(tmp_path):
    f = _bolt(tmp_path / "b.png")
    page = pg.PageOpts()
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", render="dither", dark=True, underbase=True), page)
    assert [p.name for p in j.plates] == ["White underbase", "Ink"] and j.plates[0].ink == "#ffffff"
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", inks=2, dark=True, dark_invert=True, underbase=True), page)
    assert j.plates[0].name == "White underbase"
    assert "#1b1b1d" in pg.compose(j, "all")                           # the preview shows the garment colour


def _ink_box(plate):
    """Where a plate's ink sits on the page, in mm: (left, top, right, bottom). (A plate can be drawn
    in several pieces; this is the box around all of them.)"""
    boxes = []
    for tx, ty, sx, sy, d in re.findall(r'translate\(([\d.-]+) ([\d.-]+)\) scale\(([\d.]+) ?([\d.]*)\)"><path[^>]* d="([^"]+)"', plate.body):
        tx, ty, sx, sy = float(tx), float(ty), float(sx), float(sy or sx)
        runs = np.array(re.findall(r"M(\d+) (\d+)h(\d+)v(\d+)", d), float)      # dither, plain pieces of white: one rectangle per run
        if len(runs):
            x0, y0, x1, y1 = runs[:, 0].min(), runs[:, 1].min(), (runs[:, 0] + runs[:, 2]).max(), (runs[:, 1] + runs[:, 3]).max()
        else:                                                                    # screen, flat colours: traced outlines
            xy = np.array(re.findall(r"(-?[\d.]+) (-?[\d.]+)", d), float)
            x0, y0, x1, y1 = xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()
        boxes.append((tx + x0 * sx, ty + y0 * sy, tx + x1 * sx, ty + y1 * sy))
    boxes = np.array(boxes)
    return boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()


@pytest.mark.parametrize("render", ["dither", "screen"])
def test_dark_underbase_sits_under_the_ink_when_the_picture_is_small(tmp_path, render):
    f = _bolt(tmp_path / "b.png")                                      # 400 px wide: fewer pixels than the plate is drawn with
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", render=render, dark=True, underbase=True), pg.PageOpts())
    under, ink = _ink_box(j.plates[0]), _ink_box(j.plates[1])
    assert all(abs(u - i) < 1.5 for u, i in zip(under, ink)), (under, ink)   # same place, same size (it is pulled in 0.25 mm)


def _ink_area(plate):
    """Roughly how much a traced plate covers, in mm² (its outlines taken as straight-sided; holes come off)."""
    tx, ty, sx, sy = re.search(r"translate\(([\d.-]+) ([\d.-]+)\) scale\(([\d.]+) ?([\d.]*)\)", plate.body).groups()
    sx, sy = float(sx), float(sy or sx)
    d = re.search(r' d="([^"]+)"', plate.body).group(1)
    areas = []
    for sub in d.split("M")[1:]:
        ends = [re.findall(r"-?[\d.]+", seg)[-2:] for seg in re.findall(r"[MLC]?([^LCZ]+)", sub) if seg.strip()]
        pts = np.array(ends, float)
        x, y = pts[:, 0] * sx, pts[:, 1] * sy
        areas.append(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return max(areas) - (sum(areas) - max(areas))


def _white_and_ink(job, per_mm=20):
    """(mm² of underbase with no ink over it, the share of the ink that has white under it). The white
    is taken a touch too wide and each ink a touch too narrow, so bare white is not missed."""
    white = _plate_mask(job, job.plates[0], per_mm)
    ink = np.zeros_like(white)
    for p in job.plates[1:]:
        ink |= _plate_mask(job, p, per_mm, lean=True)
    return (white & ~ink).sum() / per_mm ** 2, (white & ink).sum() / max(1, ink.sum())


def _added_outlines(plate):
    """The outlines of the white added to a white plate (its groups of straight-sided outlines), in
    mm: (width, height, area) of each."""
    out = []
    for sx, sy, d in re.findall(r'scale\(([\d.]+)(?: ([\d.]+))?\)"><path fill-rule="evenodd" d="(M[^"C]+)"', plate.body):
        sx, sy = float(sx), float(sy or sx)
        for sub in d.split("M")[1:]:
            q = np.array([float(v) for v in re.findall(r"-?\d+", sub)]).reshape(-1, 2) * (sx, sy)
            x, y = q[:, 0], q[:, 1]
            out.append((np.ptp(x), np.ptp(y), 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))
    return out


def _box_of(plates):
    """The box around several plates' ink together."""
    boxes = np.array([_ink_box(p) for p in plates])
    return boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()


@pytest.mark.parametrize("how", ["dither", "screen", "flat"])
def test_thin_artwork_in_a_small_picture_keeps_its_underbase(tmp_path, how):
    """A picture 100 px wide with a line 2 px thick: on the page that line is some 3 mm thick. The
    underbase used to be pulled in by a whole pixel of the picture (1.5 mm here) on each side, which
    left nothing of it. It is pulled in a quarter millimetre from where the ink is solid (with one
    ink the line's edges are soft, so that is its middle half)."""
    a = np.full((100, 100, 3), 255, np.uint8)
    a[50:52, 10:90] = 0
    f = tmp_path / "line.png"
    Image.fromarray(a).save(f)
    kind = dict(method="spot", kind="flat", inks=2) if how == "flat" else dict(method="one", ink="#e436a0", render=how)
    j = pr.build(f, pr.PrintParams(dark=True, underbase=True, **kind), pg.PageOpts())
    assert j.plates[0].name == "White underbase" and j.plates[0].body      # (flat used to leave the plate out, the others left it empty)
    assert len(j.plates) > 1 and all(p.body for p in j.plates[1:])
    under, ink = _ink_box(j.plates[0]), _box_of(j.plates[1:])
    pulled_in = (under[0] - ink[0], under[1] - ink[1], ink[2] - under[2], ink[3] - under[3])
    assert all(0.05 < v < 1.8 for v in pulled_in), pulled_in               # inside the ink on every side
    mm = j.art[2] / 100                                                     # one pixel of the picture on the page
    assert under[3] - under[1] > (2 * mm - 0.8 if how == "flat" else 0.4 * 2 * mm)   # a good part of the line's thickness,
    assert under[2] - under[0] > 80 * mm - 3                                # and nearly all its length
    bare, covered = _white_and_ink(j, 10)
    assert bare < 0.5, (bare, covered)                                      # and none of it where the ink leaves the cloth bare


@pytest.mark.parametrize("settings", [
    dict(render="dither"), dict(render="screen"),                           # a 1 px line (1.5 mm) with the usual settings
    dict(render="dither", dither="blue", dither_size=1.0, contrast=0.996),  # ink that is very nearly solid, on a coarse dither
    dict(render="screen", shape="square", lpi=15, ppi=150, angle=45),       # a coarse screen that leaves gaps in solid ink
    dict(render="screen", lpi=20, ppi=72, angle=30, edge=2.0, thick=1),     # few pixels to the inch, edges smoothed a lot
    dict(render="screen", lpi=30, ppi=120, angle=0, edge=0.5, thick=1, art_w=70.0),   # a small print: the line is 0.7 mm
    dict(render="screen", shape="line", lpi=20, ppi=105, angle=0, edge=1.0, thick=1, art_w=80.0),   # 0.85 mm of ink: room for a third of a millimetre of white
])
def test_one_ink_white_is_only_ever_put_inside_the_finished_ink(tmp_path, settings):
    """The white a thin line gets is taken from the ink plate as it comes out (dithered or screened),
    a quarter millimetre in. So there is none where that plate leaves the cloth bare, whatever the
    settings; and a line of one pixel has some."""
    settings = dict(settings)
    a = np.full((100, 100, 3), 255, np.uint8)
    a[50:50 + settings.pop("thick", 1 if len(settings) == 1 else 2), 10:90] = 0
    f = tmp_path / "line.png"
    Image.fromarray(a).save(f)
    page = pg.PageOpts(art_w=settings.pop("art_w", 0.0))
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", dark=True, underbase=True, **settings), page)
    assert [p.name for p in j.plates] == ["White underbase", "Ink"] and j.plates[0].body
    white, ink = _plate_mask(j, j.plates[0], 20), _plate_mask(j, j.plates[1], 20)
    assert (white & ~ink).sum() / 400 < 0.5                                 # mm² of white with no ink over it
    assert (white & ink).sum() > 0.1 * ink.sum()                            # and white under a fair part of the line


def test_one_ink_white_goes_by_the_ink_as_it_is_drawn(tmp_path):
    """With the tones turned round the ink covers the picture up to its edge, and with the screen's
    edges smoothed a lot the outline that is drawn for it leaves large parts out. The white must not
    be put where the ink plate "ought" to have ink, only where its drawing has."""
    a = np.full((60, 60, 3), 255, np.uint8)
    a[15:45, 15:45] = 0
    f = tmp_path / "square.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", render="screen", lpi=30, ppi=300, angle=22.5, edge=2.0,
                                   invert=True, dark=True, underbase=True), pg.PageOpts())
    white, ink = _plate_mask(j, j.plates[0], 10), _plate_mask(j, j.plates[1], 10)
    assert (white & ~ink).sum() / 100 < 2.0                                 # mm² of white with no ink over it
    assert (white & ink).sum() > 0.93 * ink.sum()                           # and the ink has its white (0.88 without what is added)


def test_one_ink_white_sits_on_the_ink_plates_own_scale(tmp_path):
    """A very flat picture: its screen plate is given a few rows more than its height, so one pixel of
    the plate is not the picture's height divided by the rows. The added white has to use the ink
    plate's own scale (so many fine pixels to one of the plate's, the same both ways), or it slides
    out from under the ink."""
    a = np.full((3, 100, 3), 255, np.uint8)
    a[:, 10:90] = 0
    f = tmp_path / "flat.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", render="screen", shape="line", lpi=15, ppi=72, angle=0, edge=0.5,
                                   dark=True, underbase=True), pg.PageOpts(art_w=60.0))
    ink = float(re.search(r"scale\(([\d.]+)\)", j.plates[1].body).group(1))
    scales = re.findall(r"scale\(([\d.]+)(?: ([\d.]+))?\)", j.plates[0].body)
    assert len(scales) >= 2, "no white was added to this plate"
    for across, down in scales[1:]:                                         # (each piece of the added white)
        x, y = float(across), float(down or across)
        assert abs(x - y) < 1e-6 and abs(ink / x - round(ink / x)) < 1e-3, (ink, x, y)


@pytest.mark.parametrize("render", ["dither", "screen"])
def test_added_white_stays_within_the_shape_it_is_for(render):
    """The white that is added under the ink is only for the shape the underbase was made for. Here
    the ink plate is solid all over, there is no white yet, and that shape is a square in the middle:
    the added white stops at the square's edge, not a pixel of the picture further."""
    from boottool import trace as tr
    from boottool.config import Params
    shape = np.zeros((20, 20), np.uint8)
    shape[5:15, 5:15] = 1                                                   # on a picture of 20 x 20 pixels, 100 mm wide: 25 to 75 mm
    if render == "dither":
        paths, f = pr._added_dither_white(np.ones((100, 100), bool), np.zeros((100, 100), bool), shape, 100.0, 100.0)   # 1 mm squares
        unit = 1.0 / f
    else:
        tp = Params(speckle=0, smooth_sigma=0.5, epsilon=0.8, corner_deg=125.0, precision=1)
        ink = tr.mask_to_polys(np.ones((400, 400), bool), tp, edge=True)    # a plate of 400 x 400 pixels of 0.25 mm
        paths, g = pr._added_screen_white(ink, [], tp, shape, (400, 400), 0.25, 100.0, 100.0)
        unit = 0.25 / g
    assert paths
    points = np.concatenate([np.array([float(v) for v in re.findall(r"-?\d+", d)]).reshape(-1, 2) for d in paths])
    box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]) * unit + unit / 2   # mm (points are pixel middles)
    assert np.all(np.abs(box - (25, 25, 75, 75)) < 0.15), box               # the square (a hair inside it), and nothing beyond it


def test_a_spot_plate_that_is_only_named_white_underbase_is_left_alone(tmp_path):
    """Plate names can be typed. A red plate called "White underbase" gets none of the white that is
    added to the underbase Boot Tool makes itself: nothing of it lies under the blue ink beside it.
    (A guard: only the underbase made here is given that white. Such a plate was, and still is,
    printed solid because of its name; that is a separate matter.)"""
    a = np.full((100, 100, 3), 255, np.uint8)
    a[20:80, 20:49] = (0xd7, 0x2d, 0x2d)
    a[20:80, 49:80] = (0x2d, 0x2d, 0xd7)
    f = tmp_path / "two.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="spot", kind="photo", picked=True, colors=("#d72d2d", "#2d2dd7"),
                                   names=("White underbase", "Blue"), render="dither", contrast=1.5), pg.PageOpts())
    assert [p.name for p in j.plates] == ["White underbase", "Blue"]
    mm = j.art[2] / 100
    red = _plate_mask(j, j.plates[0], 10)
    inside_blue = red[int((2 + 25 * mm) * 10): int((2 + 75 * mm) * 10), int((2 + 55 * mm) * 10): int((2 + 75 * mm) * 10)]
    assert inside_blue.sum() / 100 < 1.0                                    # mm² of red in the middle of the blue block


@pytest.mark.parametrize("render", ["dither", "screen"])
def test_a_thin_arm_keeps_its_underbase_right_up_to_the_body(tmp_path, render):
    """A block with a thin arm (2 px, 3 mm on the page). The arm had no white; it has now, and the
    white runs on into the block's without a break."""
    a = np.full((100, 100, 3), 255, np.uint8)
    a[25:75, 12:40] = 0
    a[50:52, 40:87] = 0
    f = tmp_path / "arm.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", render=render, dark=True, underbase=True), pg.PageOpts())
    white, ink = _plate_mask(j, j.plates[0], 10), _plate_mask(j, j.plates[1], 10)
    mm = j.art[2] / 100
    along = white[int((2 + 51 * mm) * 10), int((2 + 30 * mm) * 10): int((2 + 85 * mm) * 10)]   # the arm's centre line, from inside the block
    assert along.all()
    assert (white & ~ink).sum() / 100 < 2.0                                 # mm² of white with no ink over it


def test_flat_underbase_is_only_under_what_prints(tmp_path):
    """A faint stroke that the colour separation takes for paper gets no ink, so it gets no white
    under it either."""
    a = np.full((100, 100, 3), 255, np.uint8)
    a[20:45, 20:80] = 0                                                     # a black block: printed
    a[70:72, 20:80] = 240                                                   # a very light grey stroke: not printed
    f = tmp_path / "faint.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", inks=2, dark=True, underbase=True), pg.PageOpts())
    assert j.plates[0].name == "White underbase" and len(j.plates) > 1
    under, ink = _ink_box(j.plates[0]), _box_of(j.plates[1:])
    assert under[0] > ink[0] and under[1] > ink[1] and under[2] < ink[2] and under[3] < ink[3], (under, ink)
    mm = j.art[2] / 100
    assert under[3] < j.art[1] + 50 * mm                                    # nothing down where the faint stroke is
    bare, covered = _white_and_ink(j)
    assert bare < 0.5 and covered > 0.9, (bare, covered)


@pytest.mark.parametrize("detail", ["hairlines", "dashes"])
def test_flat_underbase_leaves_out_detail_the_plates_drop(tmp_path, detail):
    """Fine marks close together are too fine for the colour plate, which leaves them out. The
    underbase must leave them out too, not fill them in as one white patch."""
    a = np.full((1600, 1600, 3), 255, np.uint8)
    a[300:700, 300:1300] = 0                                                # a black block: printed
    if detail == "hairlines":
        for y in range(1000, 1060, 8):
            a[y:y + 2, 300:1300] = 0                                        # 2 px every 8 px: not printed
    else:
        for y in range(1000, 1140, 14):
            for x in range(300, 1300, 9):
                a[y:y + 11, x:x + 3] = 0                                    # dashes 3 px wide: not printed
    f = tmp_path / "hair.png"
    Image.fromarray(a).save(f)
    colour = dict(picked=True, colors=("#000000",)) if detail == "dashes" else {}   # (a picked colour keeps more of them)
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", inks=1, dark=True, underbase=True, **colour), pg.PageOpts())
    assert [p.name for p in j.plates][0] == "White underbase" and len(j.plates) == 2
    under, ink = _ink_box(j.plates[0]), _ink_box(j.plates[1])
    mm = j.art[2] / 1600
    assert ink[3] < j.art[1] + 800 * mm                                     # (the plate has the block only)
    assert under[1] > ink[1] and under[3] < ink[3], (under, ink)            # and so has the underbase (it used to reach the hairlines)
    bare, covered = _white_and_ink(j)
    assert bare < 0.5 and covered > 0.9, (bare, covered)


def test_flat_underbase_with_the_background_printed(tmp_path):
    """With "paper" off the background is printed as an ink, so the underbase goes under it as well:
    the whole picture, a little inside its edge."""
    a = np.full((300, 300, 3), 255, np.uint8)
    a[100:200, 60:240] = (200, 30, 30)
    f = tmp_path / "block.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", inks=2, paper=False, dark=True, underbase=True), pg.PageOpts())
    assert [p.name for p in j.plates] == ["White underbase", "White", "Red"]
    ox, oy, aw, ah = j.art
    left, top, right, bottom = _ink_box(j.plates[0])
    pulled_in = (left - ox, top - oy, ox + aw - right, oy + ah - bottom)
    assert all(0.05 < v < 0.8 for v in pulled_in), pulled_in
    # The background's own plate is the background (all of the picture but the block). It used to
    # come out as the block alone, because its edge along the picture's edge was lost.
    assert all(abs(a_ - b_) < 0.3 for a_, b_ in zip(_ink_box(j.plates[1]), (ox, oy, ox + aw, oy + ah)))
    assert abs(_ink_area(j.plates[1]) - (aw * ah - _ink_area(j.plates[2]))) < 0.02 * aw * ah
    bare, covered = _white_and_ink(j)
    assert bare < 0.5 and covered > 0.95, (bare, covered)


def test_flat_colour_in_a_corner_of_the_picture(tmp_path):
    """A block of colour in a corner reaches two edges of the picture. Its plate is the whole block
    (it used to be cut off slantwise across the corner), and the underbase lies inside it."""
    a = np.full((100, 100, 3), 255, np.uint8)
    a[:40, :40] = (200, 30, 30)
    f = tmp_path / "corner.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", inks=1, dark=True, underbase=True), pg.PageOpts())
    assert [p.name for p in j.plates] == ["White underbase", "Red"]
    mm = j.art[2] / 100
    block = (40 * mm) ** 2
    red, white = _ink_area(j.plates[1]), _plate_mask(j, j.plates[0], 10).sum() / 100   # (the white may be in several pieces)
    assert abs(red - block) < 0.05 * block                                  # the whole block
    assert 0.9 * red < white < red                                          # white under nearly all of it, and no more
    bare, covered = _white_and_ink(j)
    assert bare < 0.5 and covered > 0.9, (bare, covered)
    under, ink = _ink_box(j.plates[0]), _ink_box(j.plates[1])
    assert under[0] > ink[0] and under[1] > ink[1] and under[2] < ink[2] and under[3] < ink[3], (under, ink)



def _red(a, y0, y1, x0, x1):
    a[y0:y1, x0:x1] = (200, 30, 30)


def _flat_red(tmp_path, a, page=None, **more):
    f = tmp_path / "red.png"
    Image.fromarray(a).save(f)
    return pr.build(f, pr.PrintParams(method="spot", kind="flat", inks=1, picked=True, colors=("#c81e1e",), dark=True,
                                      underbase=True, **more), page or pg.PageOpts())


@pytest.mark.parametrize("what", ["gap", "narrow gap", "cutout"])
def test_flat_underbase_keeps_out_of_gaps_and_cutouts(tmp_path, what):
    """White only where there is ink over it: not across a narrow gap between two blocks (it used to
    be closed up first), and not into a cutout (the plate's outline is curved; the white was laid out
    along straight lines between its points)."""
    import cv2
    a = np.full((1500, 1500, 3), 255, np.uint8)
    page = None
    if what == "gap":
        _red(a, 300, 1200, 200, 743)
        _red(a, 300, 1200, 757, 1300)                                       # two blocks 14 px apart: 1.4 mm on the page
    elif what == "narrow gap":
        page = pg.PageOpts(size="A3")
        _red(a, 300, 1200, 300, 750)
        _red(a, 300, 1200, 754, 1200)                                       # 4 px apart: two thirds of a millimetre on A3
    else:
        _red(a, 150, 1350, 150, 1350)
        hole = np.array([[750 + 450 * np.cos(t), 750 + 450 * np.sin(t)] for t in np.arange(8) * np.pi / 4])
        cv2.fillPoly(a, [np.rint(hole).astype(np.int32)], (255, 255, 255))  # an eight-sided cutout
    j = _flat_red(tmp_path, a, page, trap=0.0)
    bare, covered = _white_and_ink(j)
    assert bare < 1.0 and covered > 0.95, (bare, covered)                   # used to be about 100 mm² of bare white or more


def test_a_flat_stripe_on_a_design_two_metres_wide_gets_its_white(tmp_path):
    """On a very large design the flat-colour underbase used to be worked out on pixels of half a
    millimetre (there was a limit on their number), and a stripe two millimetres thick got no white.
    It is worked out a piece at a time now, as finely as on a small design: the stripe has its white
    all along, a little inside the ink on both sides."""
    a = np.full((100, 3000, 3), 255, np.uint8)
    _red(a, 10, 14, 150, 2850)
    j = _flat_red(tmp_path, a, pg.PageOpts(size="Custom", custom_w=2000, custom_h=200, art_w=2000.0))
    assert [p.name for p in j.plates] == ["White underbase", "Red"]
    white, ink = _plate_mask(j, j.plates[0], 10), _plate_mask(j, j.plates[1], 10)
    span = slice((2 + 300) * 10, (2 + 1700) * 10)                            # the middle 1.4 m of the stripe
    assert white[:, span].any(axis=0).all()                                 # white in every column: no break between pieces
    assert (white & ~ink).sum() / 100 < 0.5                                 # mm² of white with no ink over it
    column = (2 + 1000) * 10
    w, i = np.flatnonzero(white[:, column]), np.flatnonzero(ink[:, column])
    assert w[0] - i[0] >= 1 and i[-1] - w[-1] >= 1                          # at least 0.1 mm inside the ink, above and below
    assert len(w) >= 3                                                      # and at least 0.3 mm of it


@pytest.mark.parametrize("width", [511.9, 512.1, 2000.0])
def test_a_large_flat_design_keeps_small_parts_and_cutouts_in_its_white(tmp_path, width):
    """A large design's white is worked out a piece at a time. A small stroke or a small cutout that
    lies across the border of two pieces must not be taken for a speck there: the stroke keeps its
    white, and the cutout stays open, at widths either side of where the pieces change and at two
    metres. (A guard for working in pieces: upstream, which does not, passes it too.)"""
    if width < 1000:                                                        # a short stroke, with "remove specks" set high
        a = np.full((270, 3000, 3), 255, np.uint8)
        _red(a, 55, 61, 1009, 1039)
        j = _flat_red(tmp_path, a, pg.PageOpts(size="Custom", custom_w=600, custom_h=200, art_w=width), speckle=60, trap=0.0)
        assert [p.name for p in j.plates] == ["White underbase", "Red"]
        white, ink = _plate_mask(j, j.plates[0], 20), _plate_mask(j, j.plates[1], 20)
        assert (white & ink).sum() > 0.3 * ink.sum()
    else:                                                                   # a block with a small square cut out of it
        a = np.full((700, 3000, 3), 255, np.uint8)
        _red(a, 180, 580, 160, 560)
        a[338:344, 338:344] = 255
        j = _flat_red(tmp_path, a, pg.PageOpts(size="Custom", custom_w=2000, custom_h=600, art_w=width), trap=0.0)
        assert [p.name for p in j.plates] == ["White underbase", "Red"]
        white, ink = _plate_mask(j, j.plates[0], 10), _plate_mask(j, j.plates[1], 10)
        middle = int((2 + 341 * width / 3000) * 10)
        assert not ink[middle, middle] and not white[middle, middle]        # the cutout is in the colour plate, and stays out of the white
    assert (white & ~ink).sum() / (white.shape[1] / (j.art[2] + 4)) ** 2 < 0.5   # mm² of white with no ink over it


@pytest.mark.parametrize("joined", [False, True])
def test_a_stroke_just_inside_the_edge_of_the_picture_is_kept(tmp_path, joined):
    """A thin stroke one pixel away from the picture's edge is traced like any other shape, also when
    a single pixel joins it to the edge. Its white lies inside it."""
    a = np.full((180, 3000, 3), 255, np.uint8)
    _red(a, 1, 4, 1000, 1140)
    if joined:
        _red(a, 0, 1, 1000, 1001)
    j = _flat_red(tmp_path, a, pg.PageOpts(size="Custom", custom_w=2000, custom_h=200, art_w=2000.0), trap=0.0)
    assert [p.name for p in j.plates] == ["White underbase", "Red"]
    white, ink = _plate_mask(j, j.plates[0], 20), _plate_mask(j, j.plates[1], 20)
    assert 60 < ink.sum() / 400 < 100                                       # mm²: the stroke, 93 mm long and nearly a millimetre thick
    assert (white & ~ink).sum() / 400 < 0.5 and (white & ink).sum() > 0.25 * ink.sum()


def _vee(k, colour):
    """A block 1300k x 380k px with a wide, shallow V cut into its top: long, gently curved outlines."""
    yy, xx = np.mgrid[0:500 * k, 0:1500 * k]
    a = np.full((500 * k, 1500 * k, 3), 255, np.uint8)
    a[(xx >= 100 * k) & (xx < 1400 * k) & (yy > 20 * k + np.maximum(0, 150 * k - np.abs(xx - 750 * k) / 4)) & (yy < 400 * k)] = colour
    return a


def test_the_white_follows_long_curves_of_the_ink(tmp_path):
    """To pull the white in, the plates' curves are turned into short straight lines. On a large
    design one curve can be half a metre long; too few lines would cut across its bend and put the
    white beside the ink. They keep within a quarter of a fine pixel of the curve as it is written
    out, so there is no white outside the ink along the long, shallow curves of this shape. (A guard
    for the way the white is worked out here: upstream's white has no such lines.)"""
    f = tmp_path / "vee.png"
    Image.fromarray(_vee(1, (200, 30, 30))).save(f)
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", picked=True, inks=1, colors=("#c81e1e",), trap=0.0, speckle=0,
                                   dark=True, underbase=True), pg.PageOpts(size="Custom", custom_w=2000, custom_h=700, art_w=2000.0))
    bare, covered = _white_and_ink(j, 10)
    assert bare < 1.0 and covered > 0.99, (bare, covered)
    f = tmp_path / "vee3.png"
    Image.fromarray(_vee(3, (0, 0, 0))).save(f)
    j = pr.build(f, pr.PrintParams(method="one", render="screen", ink="#c81e1e", ppi=72, lpi=5, edge=0.0, contrast=3.0, dark=True,
                                   underbase=True), pg.PageOpts(size="Custom", custom_w=2000, custom_h=700, art_w=1587.5))
    bare, covered = _white_and_ink(j, 10)
    assert bare < 1.0 and covered > 0.99, (bare, covered)


def test_rings_keep_to_the_curve():
    """polys_to_rings: the lines between its points are within the tolerance asked of the curve drawn,
    for a curve of any length, and the curve is the one written out (rounded to the plate's decimals)."""
    from boottool import trace as tr
    poly = np.array([(0, 0), (900, 40), (1800, 0), (1830, 400), (900, 430), (-20, 380)], np.float32) + 0.123
    for prec, tol in ((None, 0.02), (1, 0.02), (1, 0.5)):
        ring = tr.polys_to_rings([poly], 125.0, prec, tol)[0]
        pts, c1, c2, p3, _straight, _s, _e = tr._curves([poly], 125.0)
        if prec is not None:
            pts, c1, c2, p3 = (np.rint(a * 10 ** prec) / 10 ** prec for a in (pts, c1, c2, p3))
        t = np.linspace(0, 1, 2001)[:, None, None]
        curve = ((1 - t) ** 3 * pts + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3).transpose(1, 0, 2).reshape(-1, 2)
        a, b = ring, np.roll(ring, -1, axis=0)                              # distance of every point of the curve to the ring's lines
        ab = b - a
        far = 0.0
        for chunk in np.array_split(curve, 40):
            ap = chunk[:, None, :] - a[None]
            along = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
            far = max(far, float(np.sqrt(((ap - along[..., None] * ab) ** 2).sum(-1)).min(axis=1).max()))
        assert far <= tol * 1.05, (prec, tol, far)
        assert len(ring) < 4000 * (0.02 / tol) ** 0.5


@pytest.mark.parametrize("what", ["two blocks", "block with a hole"])
def test_large_plain_areas_get_their_white_as_everything_else(tmp_path, what):
    """Two metres square: two blocks of colour side by side, or one with a hole, in a picture whose
    size is no multiple of anything. Pieces of the underbase that no outline comes near are not
    worked out finely but drawn as plain rectangles. The white is as it is everywhere else: all
    under the ink, a quarter millimetre in from its edges, none missing between the pieces or where
    the colours meet. (A guard for that shortcut.)"""
    import cv2
    if what == "two blocks":
        a = np.full((300, 300, 3), 255, np.uint8)
        a[30:270, 30:150] = (200, 30, 30)
        a[30:270, 150:270] = (30, 30, 200)
        size, edges = 2000, 2 * (1600 + 1600)
    else:
        a = np.full((1503, 1503, 3), 255, np.uint8)
        a[120:1383, 120:1383] = (200, 30, 30)
        a[690:813, 690:813] = 255
        size, edges = 1997, 4 * 1263 * 1997 / 1503 + 4 * 123 * 1997 / 1503
    f = tmp_path / "plain.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", picked=True, inks=2, colors=("#c81e1e", "#1e1ec8"), trap=0.0, dark=True,
                                   underbase=True), pg.PageOpts(size="Custom", custom_w=size, custom_h=size, art_w=float(size)))
    assert j.plates[0].name == "White underbase" and len(j.plates) == (3 if what == "two blocks" else 2)
    assert re.search(r'<path d="(M\d+ \d+h\d+v\d+h-\d+z)+"/>', j.plates[0].body)   # the plain pieces
    white = _plate_mask(j, j.plates[0], 4, lean=True)                       # (4 px to the mm: the pull-in is one of them)
    ink = np.zeros_like(white)
    for plate in j.plates[1:]:
        ink |= _plate_mask(j, plate, 4)
    assert not (white & ~ink).any()                                        # none outside the ink
    well_inside = cv2.erode(ink.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)   # 3/4 mm in
    assert not (well_inside & ~white).any()                                 # and none missing: not between pieces, not at the join
    assert white.sum() < ink.sum() - 0.15 * 4 * edges * 4                   # pulled in all round (0.25 mm, less the measuring)


def test_straight_outlines_keep_to_the_mask():
    """The white under flat colours is outlined with straight lines: at most three quarters of a
    pixel outside the mask it is made from (across a hollow), at most a pixel and a quarter inside
    it, holes and all; and with few points, whatever the slant of an edge."""
    import cv2
    yy, xx = np.mgrid[0:120, 0:160]
    m = ((xx - 60) ** 2 + (yy - 60) ** 2 < 50 ** 2) & ~((xx - 50) ** 2 + (yy - 55) ** 2 < 15 ** 2)   # a disc with a round hole
    m |= (np.abs((xx - 120) - (yy - 60)) < 4) & (yy > 10) & (yy < 110)       # a slanting bar across it
    m[2:6, 100:150] = True                                                  # a thin bar
    m[110:118, 0:40] = True                                                 # a block at the corner
    d = pr._straight_outlines(m.astype(np.uint8))
    rings = [np.array([float(v) for v in re.findall(r"-?\d+", sub)], np.float64).reshape(-1, 2) for sub in d.split("M")[1:]]
    assert len(rings) == 4 and all(len(r) >= 3 for r in rings)              # disc with bar, hole, thin bar, block
    k = 4                                                                   # drawn 4 times the size of its half pixels
    got = np.zeros((240 * k, 320 * k), np.uint8)
    cv2.fillPoly(got, [np.rint(((r + 0.5) * k - 0.5) * 16).astype(np.int32) for r in rings], 1, lineType=cv2.LINE_8, shift=4)
    got = got.astype(bool)
    full = np.repeat(np.repeat(m, 2 * k, axis=0), 2 * k, axis=1).astype(np.uint8)

    def grown(by):                                                          # the mask, `by` eighths of a pixel larger (or smaller)
        box = np.ones((2 * abs(by) + 1, 2 * abs(by) + 1), np.uint8)
        return (cv2.dilate(full, box) if by > 0 else cv2.erode(full, box, borderType=cv2.BORDER_CONSTANT, borderValue=0)).astype(bool)
    assert not (got & ~grown(7)).any()                                      # nowhere more than 3/4 px outside it (and the drawing's own 1/8)
    assert not (grown(-10) & ~got).any()                                    # and all of it but a rim of 1 1/4 px
    assert (got & ~full.astype(bool)).sum() < 0.01 * full.sum()             # (outside it only here and there)
    assert len(d) < 700                                                     # few points: slants and arcs are not drawn step by step
    for slope in (0.05, 0.2, 1 / 3, 0.4, 0.5, 0.75, 1.0, 1.5, 3.0, 7.3):    # a straight edge at any slant: a handful of points
        edge = (yy > 5 + slope * xx) if slope <= 1 else (xx > 5 + yy / slope)
        edge &= (xx > 2) & (xx < 157) & (yy > 2) & (yy < 117)
        assert pr._straight_outlines(edge.astype(np.uint8)).count("L") <= 8, slope


@pytest.mark.parametrize("side", ["left", "right", "above", "below"])
def test_added_white_laps_over_the_white_that_is_there(side):
    """A square of ink with white under one half of it. The white that is added for the other half
    laps four fine pixels over the white that is there, whichever side that is on: the added white's
    outline runs a little inside its own edge, and with less of a lap a hairline could be left."""
    ink, has = np.zeros((200, 200), np.uint8), np.zeros((200, 200), np.uint8)
    ink[40:160, 40:160] = 1
    has[{"left": np.s_[40:160, 40:100], "right": np.s_[40:160, 100:160], "above": np.s_[40:100, 40:160], "below": np.s_[100:160, 40:160]}[side]] = 1
    more = pr._added_white(ink, has, np.ones((200, 200), np.uint8), 0.0625)
    both = more & has.astype(bool)
    lap = both.any(axis=0).sum() if side in ("left", "right") else both.any(axis=1).sum()
    assert lap == 4
    assert (more | has.astype(bool))[44:156, 44:156].all() and not more[:44].any() and not more[:, :44].any()   # the core is all white now, a quarter millimetre in


@pytest.mark.parametrize("render", ["dither", "screen"])
def test_added_white_is_the_same_worked_out_only_where_it_is_wanted(tmp_path, monkeypatch, render):
    """One ink on a large design: the white that is added is worked out only around the ink that
    lacks it, not over whole pieces of the plate, and pieces with none of that are skipped. What
    comes out is what working out every piece in full gives."""
    a = np.full((240, 240, 3), 255, np.uint8)
    a[30:210, 30:200] = 0
    a[100:140, 200:236] = 0
    yy, xx = np.mgrid[0:240, 0:240]
    a[(xx - 115) ** 2 + (yy - 120) ** 2 < 30 ** 2] = 255
    f = tmp_path / "shape.png"
    Image.fromarray(a).save(f)
    par = pr.PrintParams(method="one", ink="#e436a0", render=render, dither_size=1.2, ppi=72, lpi=6, dark=True, underbase=True)
    page = pg.PageOpts(size="Custom", custom_w=1200, custom_h=1200, art_w=1100.0)
    near = pr.build(f, par, page)
    calls = []
    monkeypatch.setattr(pr, "_box_around", lambda m, pad: (calls.append(1), (-10 ** 6, 10 ** 6, -10 ** 6, 10 ** 6))[1])   # every piece, all of it
    everywhere = pr.build(f, par, page)
    assert calls and near.plates[0].body.count("<g ") >= 2                  # (there is added white, and it went that way)
    assert (_plate_mask(near, near.plates[0], 8) == _plate_mask(everywhere, everywhere.plates[0], 8)).all()


@pytest.mark.parametrize("how", ["flat", "screen"])
def test_long_outlines_are_drawn_the_same_a_piece_at_a_time(tmp_path, monkeypatch, how):
    """Diagonal stripes right across a large design: each outline has thousands of points and crosses
    many pieces of the plate. For one piece, the runs of an outline that lie beyond the piece are
    left out (drawing all of it every time took most of the time). The plates are the same, byte
    for byte, as with nothing left out."""
    size = 600 if how == "flat" else 300
    yy, xx = np.mgrid[0:size, 0:size]
    a = np.full((size, size, 3), 255, np.uint8)
    a[(xx + yy) % (size // 10) < size // 20] = 0
    f = tmp_path / "stripes.png"
    Image.fromarray(a).save(f)
    if how == "flat":
        par = pr.PrintParams(method="spot", kind="flat", inks=1, trap=0.0, dark=True, underbase=True)
        page = pg.PageOpts(size="Custom", custom_w=1300, custom_h=1300, art_w=1200.0)
    else:
        par = pr.PrintParams(method="one", ink="#e436a0", render="screen", lpi=8, ppi=72, dark=True, underbase=True)
        page = pg.PageOpts(size="Custom", custom_w=700, custom_h=700, art_w=600.0)
    left_out = []
    near = pr._Rings.near
    monkeypatch.setattr(pr._Rings, "near", lambda self, *box: (lambda got: (left_out.append(sum(map(len, self.rings)) - sum(map(len, got))), got)[1])(near(self, *box)))
    short = pr.build(f, par, page)
    assert max(left_out) > 1000                                             # (points were left out, by the thousand)
    monkeypatch.setattr(pr._Rings, "near", near)
    monkeypatch.setattr(pr._Rings, "RUN", 10 ** 9)                          # no outline is long enough for it now
    whole = pr.build(f, par, page)
    assert [p.name for p in short.plates] == [p.name for p in whole.plates] and short.plates[0].name == "White underbase"
    assert all(x.body == y.body for x, y in zip(short.plates, whole.plates))


def test_flat_white_is_pulled_in_a_quarter_millimetre_whatever_the_size(tmp_path):
    """The pull-in is done in whole fine pixels, so their size is chosen to make a quarter millimetre
    come out (nearly) whole. A slanting stroke 0.9 mm thick keeps about 0.4 mm of white, at 599.9 mm
    wide as at 600 mm (where the number of fine pixels to a working pixel changes)."""
    yy, xx = np.mgrid[0:220, 0:3000]
    a = np.full((220, 3000, 3), 255, np.uint8)
    a[(np.abs((xx - 1315) - (yy - 110)) <= 3) & (np.abs(xx - 1315) < 75) & (np.abs(yy - 110) < 77)] = (200, 30, 30)
    got = []
    for width in (599.9, 600.0):
        j = _flat_red(tmp_path, a, pg.PageOpts(size="Custom", custom_w=700, custom_h=200, art_w=width), speckle=0, trap=0.0)
        assert [p.name for p in j.plates] == ["White underbase", "Red"]
        white, ink = _plate_mask(j, j.plates[0], 40), _plate_mask(j, j.plates[1], 40)
        middle = slice(int((2 + 1315 * width / 3000 - 5) * 40), int((2 + 1315 * width / 3000 + 5) * 40))   # 10 mm across its middle
        along = 10 * 2 ** 0.5                                               # (it runs at 45 degrees)
        thick, under = ink[:, middle].sum() / 1600 / along, white[:, middle].sum() / 1600 / along
        assert 0.85 < thick < 1.0 and thick - 0.57 < under < thick - 0.45, (width, thick, under)
        assert not (white & ~ink).any()
        got.append(under)
    assert abs(got[0] - got[1]) < 0.03, got


@pytest.mark.parametrize("top, thick", [(0, 3), (0, 5), (100, 5)])
def test_a_thin_stripe_keeps_its_ink_and_its_white(tmp_path, top, thick):
    """A stripe a few pixels thick, along the top edge of the picture or further in, stays on its
    colour plate. One thick enough to have white under it (5 px on A3 is 0.8 mm) gets its white."""
    a = np.full((1500, 1500, 3), 255, np.uint8)
    _red(a, 500, 900, 300, 1200)
    _red(a, top, top + thick, 150, 1350)
    j = _flat_red(tmp_path, a, pg.PageOpts(size="A3"))
    white, ink = _plate_mask(j, j.plates[0]), _plate_mask(j, j.plates[1])
    mm = j.art[2] / 1500
    stripe = slice(int((2 + top * mm) * 20), int((2 + (top + thick) * mm) * 20))   # (the masks start 2 mm above the artwork)
    assert ink[stripe].sum() > 0.6 * ink[stripe].shape[0] * 1200 * mm * 20   # the stripe is there, most of it
    if thick == 5:
        assert (white[stripe] & ink[stripe]).sum() > 0.25 * ink[stripe].sum()   # with white under a good part of it (there was none)
    bare, covered = _white_and_ink(j)
    assert bare < 0.5 and covered > 0.9, (bare, covered)


@pytest.mark.parametrize("shape, stroke, width, size", [
    ((40, 80), (20, 21, 8, 72), 80.0, 1.0),                                 # 1 mm squares, a stroke one square wide
    ((40, 80), (20, 21, 8, 72), 160.0, 2.0),                                # 2 mm squares
    ((40, 900), (20, 21, 100, 800), 738.0, 0.82),                           # a banner: many squares, still fine enough
    ((80, 10), (32, 36, 0, 10), 5.0, 2.0),                                  # so narrow that its squares are not square
    ((1100, 2500), (550, 551, 250, 2250), 2000.0, 0.8),                     # two metres wide
])
def test_a_stroke_one_dither_square_wide_gets_its_white(tmp_path, shape, stroke, width, size):
    """A solid stroke exactly one dither square wide. There is room for white under it, a quarter
    millimetre in from each side; it used to get none, because the white was cut to whole squares."""
    a = np.full(shape + (3,), 255, np.uint8)
    a[stroke[0]:stroke[1], stroke[2]:stroke[3]] = 0
    f = tmp_path / "grain.png"
    Image.fromarray(a).save(f)
    page = pg.PageOpts(size="Custom", custom_w=800, custom_h=120, art_w=width) if width > 250 else pg.PageOpts(art_w=width)
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", render="dither", dither="floyd", dither_size=size, dark=True,
                                   underbase=True), page)
    assert [p.name for p in j.plates] == ["White underbase", "Ink"] and j.plates[0].body
    per = 4 if width > 1000 else 10 if width > 250 else 20
    white, ink = _plate_mask(j, j.plates[0], per), _plate_mask(j, j.plates[1], per)
    assert (white & ~ink).sum() / per ** 2 < 0.5                            # mm² of white with no ink over it
    # The white itself, from its outlines: a strip along the stroke (every stroke here lies across
    # the picture and is one square thick), about as thick as the room there is and nearly as long.
    strips = _added_outlines(j.plates[0])
    room = size - 0.5                                                       # a quarter millimetre in from each side
    length = (stroke[3] - stroke[2]) * width / shape[1]
    assert room - 0.2 < max(h for _w, h, _a in strips) < room + 0.01        # (an outline runs half a fine pixel inside its white)
    assert sum(a for _w, _h, a in strips) > 0.8 * (room - 0.2) * (length - 0.5)


@pytest.mark.parametrize("size", [1.0, 2.0])
def test_one_ink_underbase_on_a_coarse_dither_stays_under_the_ink(tmp_path, size):
    """With dither squares of a millimetre or two the white is cut to the same squares as the ink. A
    thin stroke must not get squares of white that the ink leaves bare."""
    a = np.full((240, 240, 3), 255, np.uint8)
    a[36:204, 16:19] = 0                                                    # a stroke 3 px wide: 1.9 mm on the page
    f = tmp_path / "stroke.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", render="dither", dither="floyd", dither_size=size, dark=True,
                                   underbase=True), pg.PageOpts())
    assert [p.name for p in j.plates] == ["White underbase", "Ink"] and j.plates[0].body and j.plates[1].body
    white, ink = _plate_mask(j, j.plates[0], 10), _plate_mask(j, j.plates[1], 10)
    assert (white & ~ink).sum() / 100 < 2.0                                 # mm² of white with no ink over it
    assert (white & ink).sum() > 0.3 * ink.sum()                            # and white under a good part of the stroke
