"""The white underbase under one ink on a dark garment."""
import numpy as np
import pytest
from PIL import Image

from boottool import page as pg
from boottool import printing as pr

from test_picture_edge import plate_mask


def _white_and_ink(tmp_path, a, **settings):
    """The picture `a` printed in one ink on a dark garment with a white underbase: the two plates
    as masks, 10 px to the mm (the white a touch too wide, the ink a touch too narrow), and the job."""
    f = tmp_path / "picture.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="one", ink="#e436a0", dark=True, underbase=True, **settings), pg.PageOpts())
    assert [p.name for p in j.plates] == ["White underbase", "Ink"]
    return plate_mask(j, j.plates[0], 10), plate_mask(j, j.plates[1], 10, lean=True), j


@pytest.mark.parametrize("render", ["dither", "screen"])
def test_with_the_tones_inverted_the_white_is_under_the_ink(tmp_path, render):
    """Tone "Invert" turns a black square on white into ink all round a bare square. The white
    underbase was still made for the picture as it was before: a white square with no ink over it
    (5,000 mm² of it here), and none under the ink. It goes by the ink as Tone leaves it."""
    a = np.full((60, 60, 3), 255, np.uint8)
    a[15:45, 15:45] = 0
    white, ink, j = _white_and_ink(tmp_path, a, render=render, invert=True)
    mm = j.art[2] / 60
    middle = slice(int((2 + 20 * mm) * 10), int((2 + 40 * mm) * 10))
    assert not white[middle, middle].any() and not ink[middle, middle].any()   # the square: bare cloth, no white
    assert (white & ink).sum() > 0.9 * ink.sum()                            # the ink has its white
    assert (white & ~ink).sum() / 100 < 150                                 # mm² of white with no ink over it (a hair along the picture's own edge)


@pytest.mark.parametrize("render", ["dither", "screen"])
def test_grey_that_tone_turns_to_no_ink_gets_no_white(tmp_path, render):
    """Half the block is black, half light grey. With Contrast and Brightness as high as they go,
    Tone leaves the grey half without any ink; it must not keep its white."""
    a = np.full((100, 100, 3), 255, np.uint8)
    a[20:80, 20:50] = 0
    a[20:80, 50:80] = 220
    white, ink, j = _white_and_ink(tmp_path, a, render=render, contrast=3.0, brightness=0.5)
    mm = j.art[2] / 100
    grey = (slice(int((2 + 25 * mm) * 10), int((2 + 75 * mm) * 10)), slice(int((2 + 55 * mm) * 10), int((2 + 75 * mm) * 10)))
    black = (grey[0], slice(int((2 + 25 * mm) * 10), int((2 + 45 * mm) * 10)))
    assert not ink[grey].any() and ink[black].all()                         # (Tone has done that)
    assert not white[grey].any() and white[black].all()
    assert (white & ~ink).sum() / 100 < 20                                  # mm² of white with no ink over it


def test_with_tone_left_alone_the_white_is_made_from_the_same_picture(tmp_path, monkeypatch):
    """Two small things the fix rests on. With no Tone setting touched, Tone gives back exactly what
    it is given: so the underbase, which now goes by the picture after Tone, is then made from what
    it was always made from. And with Invert on, what the underbase is made from is the inverted
    picture."""
    from boottool import halftone as ht
    t = np.random.default_rng(4).random((50, 70)).astype(np.float32)
    assert np.array_equal(ht._tone(t, pr.PrintParams()), t)
    a = np.full((100, 100, 3), 255, np.uint8)
    a[20:80, 20:50] = 0
    a[20:80, 50:80] = 120
    f = tmp_path / "picture.png"
    Image.fromarray(a).save(f)
    given, toned = [], []
    real_mask, real_tone = pr._underbase_mask, ht._tone
    monkeypatch.setattr(pr, "_underbase_mask", lambda printed, mm_px: (given.append(printed), real_mask(printed, mm_px))[1])
    monkeypatch.setattr(ht, "_tone", lambda tone, p: (lambda out: (toned.append(out), out)[1])(real_tone(tone, p)))
    pr.build(f, pr.PrintParams(method="one", ink="#e436a0", dark=True, underbase=True, render="dither", invert=True), pg.PageOpts())
    assert len(given) == 1 and any(out.shape == given[0].shape and np.array_equal(out, given[0]) for out in toned)
    assert given[0][50, 35] == 0 and given[0][5, 5] == 1                    # the black block: no ink now; the paper round it: ink


def _thin_line():
    """A small picture (100 x 100) with a line two pixels thick. Fitted to the page one pixel is about
    a millimetre and a half, so the line is 3 mm thick in print: plenty for white under it."""
    a = np.full((100, 100, 3), 255, np.uint8)
    a[49:51, 20:80] = 0
    return a


@pytest.mark.parametrize("render", ["dither", "screen"])
def test_a_thin_line_in_a_small_picture_keeps_its_white(tmp_path, render):
    """The white is pulled in a quarter of a millimetre. That was done in whole pixels of the picture,
    never less than one: here a millimetre and a half from each side, and the plate came out empty."""
    white, ink, j = _white_and_ink(tmp_path, _thin_line(), render=render)
    mm = j.art[2] / 100
    line_mm2 = (60 * mm) * (2 * mm)
    assert 0.6 * line_mm2 < white.sum() / 100 < line_mm2                   # most of the line has white under it (it was 0)
    rows = np.nonzero(white.any(axis=1))[0]
    assert (rows[-1] - rows[0] + 1) / 10 == pytest.approx(2 * mm - 0.5, abs=0.3)   # a quarter of a millimetre in from each side


def test_a_thin_line_in_a_small_picture_keeps_its_white_with_flat_colours(tmp_path):
    f = tmp_path / "picture.png"
    Image.fromarray(_thin_line()).save(f)
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", dark=True, underbase=True), pg.PageOpts())
    assert j.plates[0].name == "White underbase" and len(j.plates) > 1
    mm = j.art[2] / 100
    white, ink = plate_mask(j, j.plates[0], 10), np.any([plate_mask(j, p, 10) for p in j.plates[1:]], axis=0)
    assert white.sum() / 100 > 0.6 * (60 * mm) * (2 * mm)
    assert (white & ~ink).sum() / 100 < 3                                   # and it lies under the ink (it was white over the whole picture)


def test_artwork_too_thin_for_white_gets_none_with_flat_colours(tmp_path):
    """A line under half a millimetre has nothing left once the white is pulled in. The white plate
    then covered the whole picture, a white block on the shirt; now there is no white plate."""
    a = np.full((400, 400, 3), 255, np.uint8)
    a[200, 80:320] = 0                                                      # one pixel: about 0.4 mm in print
    f = tmp_path / "picture.png"
    Image.fromarray(a).save(f)
    j = pr.build(f, pr.PrintParams(method="spot", kind="flat", dark=True, underbase=True), pg.PageOpts())
    assert j.plates and "White underbase" not in [p.name for p in j.plates]


def test_the_white_is_worked_out_in_millimetres_whatever_the_pictures_size():
    """Two blocks 3 mm apart: the gap stays free of white (gaps under a millimetre and a half are
    bridged, as before). With pixels of a millimetre and a half the gap is two pixels, and it was
    bridged because "less than one pixel" could not be told from "less than 0.8 mm"."""
    printed = np.zeros((40, 40), np.float32)
    printed[10:30, 5:19] = 1
    printed[10:30, 21:35] = 1
    coarse = pr._underbase_mask(printed, 1.5)
    k = coarse.shape[1] / 40                                               # how much finer it was drawn
    assert k >= 8 and not coarse[int(15 * k):int(25 * k), int(19.3 * k):int(20.7 * k)].any()   # the gap is open
    assert coarse[int(20 * k), int(12 * k)] and coarse[int(20 * k), int(28 * k)]
    left = np.nonzero(coarse[int(20 * k)])[0][0] / k                        # where the white begins, in picture pixels
    assert (left - 5) * 1.5 == pytest.approx(0.25, abs=0.15)               # a quarter of a millimetre in (it was 1.5 mm)
    fine = pr._underbase_mask(np.kron(printed, np.ones((12, 12), np.float32)), 0.125)
    assert fine.shape == (480, 480)                                        # a picture that is fine enough is used as it is
    assert pr._underbase_mask(printed, 0.3002).shape == (40, 40)           # and so is the usual dither (0.3 mm, rounded to the page)
