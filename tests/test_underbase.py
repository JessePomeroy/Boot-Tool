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
