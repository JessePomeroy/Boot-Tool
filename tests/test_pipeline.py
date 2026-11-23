import json
import re
import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from lxml import etree

from boottool import Params, run_pipeline
from boottool.cli import main

from conftest import make_color, make_lineart


def _valid_svg(text):
    root = etree.fromstring(text.encode())
    assert root.tag.endswith("svg")
    return root


def test_lineart(lineart):
    res = run_pipeline(lineart)
    _valid_svg(res.svg)
    assert res.mode == "lineart"
    assert res.stats["path_count"] >= 1
    assert res.scores["ssim_target"] >= 0.90


def test_lineart_inverted_polarity(lineart_inverted):
    res = run_pipeline(lineart_inverted)
    assert res.mode == "lineart"
    assert res.scores["ssim_target"] >= 0.90


def test_color(color):
    res = run_pipeline(color)
    _valid_svg(res.svg)
    assert res.mode == "color"
    assert res.stats["color_count"] >= 3
    assert res.scores["ssim_target"] >= 0.90


def test_no_corner_overshoot(lineart):
    """Regression: curve handles used to spike past corners. SSIM is too coarse to see it,
    so check the rendered ink never extends beyond the source ink's bounding box."""
    import numpy as np
    from boottool.preprocess import prepare
    from boottool.score import rasterize

    p = Params()
    prep = prepare(lineart, p)
    res = run_pipeline(lineart, p)
    ink = rasterize(res.svg, *prep.size).min(axis=2) < 128
    (sy, sx), (oy, ox) = np.nonzero(prep.mask), np.nonzero(ink)
    for src, out in ((sx, ox), (sy, oy)):
        assert out.min() >= src.min() - 2 and out.max() <= src.max() + 2


def test_fewer_nodes_with_higher_epsilon(lineart):
    tight = run_pipeline(lineart, Params(epsilon=0.5))
    loose = run_pipeline(lineart, Params(epsilon=3.0))
    assert loose.stats["node_count"] < tight.stats["node_count"]


def test_params_key_ignores_min_score():
    assert Params(min_score=0.5).key() == Params(min_score=0.99).key()
    assert Params(epsilon=1.0).key() != Params().key()


def test_cli_run_resume_report_dataset(tmp_path):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    make_lineart(src / "a.png")
    make_color(src / "sub" / "b.png")
    (src / "broken.png").write_bytes(b"not an image")
    prov = tmp_path / "prov.csv"
    prov.write_text("file,license,source\na.png,PD-US pre-1929,example archive\n")
    out = tmp_path / "out"
    runner = CliRunner()

    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1",
                             "--provenance", str(prov)])
    assert r.exit_code == 0, r.output
    lines = (out / "manifest.jsonl").read_text().splitlines()
    recs = {json.loads(line)["source"]: json.loads(line) for line in lines}
    assert recs["a.png"]["status"] == "ok" and recs["a.png"]["provenance_declared"]
    assert recs["sub/b.png"]["status"] == "ok" and not recs["sub/b.png"]["provenance_declared"]
    assert recs["broken.png"]["status"] == "error"
    assert (out / "svg" / "a.svg").exists() and (out / "svg" / "sub" / "b.svg").exists()

    # resume: nothing to redo except the broken file
    r2 = runner.invoke(main, ["run", str(src), str(out), "--workers", "1",
                              "--provenance", str(prov)])
    assert "2 skipped" in r2.output

    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    assert (out / "report.html").exists()

    r3 = runner.invoke(main, ["dataset", str(out), "--require-provenance", "--val-frac", "0"])
    assert r3.exit_code == 0, r3.output
    ds = json.loads((out / "dataset.json").read_text())
    assert ds["train"] == ["svg/a.svg"]
    assert ds["excluded"]["no_provenance"] == 1 and ds["excluded"]["error"] == 1


def test_cli_run_keeps_files_that_differ_only_by_extension(tmp_path):
    """a.png and a.jpg used to both write svg/a.svg, so one silently replaced the other."""
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "A.jpg")      # same name apart from case and type
    make_lineart(src / "b.png", invert=True)
    runner = CliRunner()
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0, r.output
    recs = {json.loads(line)["source"]: json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()}
    assert recs["a.png"]["svg"] == "svg/a.png.svg" and recs["A.jpg"]["svg"] == "svg/A.jpg.svg"
    assert recs["b.png"]["svg"] == "svg/b.svg"                                         # names that do not clash are unchanged
    assert recs["a.png"]["mode"] == "lineart" and recs["A.jpg"]["mode"] == "color"
    for rec in recs.values():
        _valid_svg((out / rec["svg"]).read_text())
    assert (out / "svg" / "a.png.svg").read_text() != (out / "svg" / "A.jpg.svg").read_text()
    assert "3 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    page = (out / "report.html").read_text()
    assert "svg/a.png.svg" in page and "svg/A.jpg.svg" in page
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    assert sorted(json.loads((out / "dataset.json").read_text())["train"]) == ["svg/A.jpg.svg", "svg/a.png.svg", "svg/b.svg"]


def test_cli_run_never_writes_two_pictures_to_one_svg(tmp_path):
    """a.png clashes with a.jpg and becomes a.png.svg, which is also what a.png.webp would get."""
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    Image.open(make_lineart(tmp_path / "w.png", invert=True)).save(src / "a.png.webp")
    runner = CliRunner()
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0, r.output
    recs = {json.loads(line)["source"]: json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()}
    assert {k: v["svg"] for k, v in recs.items()} == {
        "a.jpg": "svg/a.jpg.svg", "a.png": "svg/a.png.svg", "a.png.webp": "svg/a.png.webp.svg"}
    svgs = [(out / rec["svg"]).read_text() for rec in recs.values()]
    assert len(set(svgs)) == 3                                                         # three different traces
    assert "3 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output


def _old_svg_names(rels):
    """svg_names as released in v0.9.0, to check that names that were fine then are unchanged."""
    from collections import Counter
    key = lambda r: Path(r).with_suffix("").as_posix().lower()
    count = Counter(key(r) for r in rels)
    return {r: r + ".svg" if count[key(r)] > 1 else Path(r).with_suffix(".svg").as_posix() for r in rels}


def test_svg_names_are_unique_and_keep_plain_names():
    import random
    import unicodedata
    from boottool.cli import svg_names
    assert svg_names(["a.png", "b.jpg", "sub/c.gif"]) == {"a.png": "a.svg", "b.jpg": "b.svg", "sub/c.gif": "sub/c.svg"}
    assert svg_names(["a.png", "a.svg/b.png"]) == {"a.png": "a.png.svg", "a.svg/b.png": "a.svg/b.svg"}   # not a folder's name
    assert svg_names(["a.png"], folders=["A.svg"]) == {"a.png": "a.png.svg"}           # a folder left in svg/ earlier
    assert svg_names(["Sub/a.png", "sub/a.png"]) == {"Sub/a.png": "Sub/a.png.svg", "sub/a.png": "sub/a.png-2.svg"}
    for a, b in (("καφές", "ΚΑΦΈΣ"), ("ας", "ασ"), ("AΣ", "Aς"), ("ß", "ss")):                       # one name on a Mac or Windows disk
        assert len({n.lower() for n in svg_names([a + ".png", b + ".jpg"]).values()}) == 2
        assert svg_names([a + ".png", b + ".jpg"]) == {a + ".png": a + ".png.svg", b + ".jpg": b + ".jpg.svg"}
    assert svg_names(["a.png"], allowed=lambda r, n: n != "a.svg") == {"a.png": "a.png.svg"}   # a.svg is someone's
    same = lambda n: "same" if n in ("x/a.svg", "y/a.svg") else n                      # x/ and y/ are one folder
    assert len({same(n) for n in svg_names(["x/a.png", "y/a.png"], place=same).values()}) == 2
    assert len(set(svg_names(["normal.png", "odd\udcff.png", "odd\udcff.jpg"]).values())) == 3   # not UTF-8 (Linux)
    assert svg_names(["a.png", "a\u200c.jpg"]) == {"a.png": "a.png.svg", "a\u200c.jpg": "a\u200c.jpg.svg"}  # older Mac disks skip U+200C
    long = "x" * 250
    names = svg_names([long + ".png", long + ".jpg"])
    assert len(set(names.values())) == 2 and all(len(n.encode()) <= 255 for n in names.values())
    composed, decomposed = unicodedata.normalize("NFC", "café"), unicodedata.normalize("NFD", "café")
    names = svg_names([composed + ".png", decomposed + ".jpg"])                        # one name on a Mac
    assert len({unicodedata.normalize("NFC", n).lower() for n in names.values()}) == 2
    rng = random.Random(7)
    words = ["a", "A", "a.png", "a.png-2", "b", "a.svg", "a.png.svg", "x-2", "é", unicodedata.normalize("NFD", "é"),
             "ς", "Σ", "ß", "ss"]
    from boottool.cli import _fold as fold
    for _ in range(3000):
        rels = set()
        for _ in range(rng.randint(1, 8)):
            folders = [rng.choice(words) for _ in range(rng.randint(0, 2))]
            rels.add("/".join(folders + [rng.choice(words) + rng.choice([".png", ".jpg", ".PNG", ".webp"])]))
        rels = sorted(rels)
        dirs = {fold(p.as_posix()) for r in rels for p in Path(r).parents if p.as_posix() != "."}
        if any(fold(r) in dirs for r in rels) or len({fold(r) for r in rels}) < len(rels):
            continue                                       # can't all exist side by side on a Mac or Windows disk
        names = svg_names(rels)
        shuffled = rels[:]
        rng.shuffle(shuffled)
        assert svg_names(shuffled) == names                                            # same names in any order
        folded = [fold(n) for n in names.values()]
        assert len(set(folded)) == len(rels), (rels, names)                            # never two on one SVG
        assert not set(folded) & dirs, (rels, names)                                   # never a folder's name
        old = _old_svg_names(rels)
        old_count = __import__("collections").Counter(fold(n) for n in old.values())
        stem = lambda r: Path(r).with_suffix("").as_posix()
        for r, n in names.items():
            assert Path(n).parent == Path(r).parent and n.endswith(".svg"), (r, n)     # stays in its own folder
            accent_twin = any(fold(stem(o)) == fold(stem(r)) and stem(o).lower() != stem(r).lower() for o in rels)
            accent_twin |= any(fold(o + ".svg") == fold(old[r]) and (o + ".svg").lower() != old[r].lower() for o in rels)
            if old_count[fold(old[r])] == 1 and fold(old[r]) not in dirs and not accent_twin:
                assert n == old[r], (rels, names, old)                                 # names that were fine are unchanged


def _kept(folder, name="a"):
    """Files Boot Tool kept aside in a folder (renamed to <name>.replaced-N.svg)."""
    return sorted(Path(folder).glob(f"{name}.replaced-*.svg"))


def _finished(manifest):
    """The finished records (not the "saving" notes written just before each SVG is moved in)."""
    return [r for r in map(json.loads, Path(manifest).read_text().splitlines()) if r["status"] != "saving"]


def _unstamped(svg_text):
    """The trace without the stamp `run` adds at the end of each SVG."""
    return re.sub(r"\n?<!-- boottool-run save [0-9a-f]{32} -->\s*\Z", "", svg_text)


def _as_if_made_by_v090(out):
    """As v0.9.0 and earlier wrote it: no stamps in the SVGs, no save ids or fingerprints, no notes."""
    m = out / "manifest.jsonl"
    recs = _finished(m)
    for r in recs:
        f = out / (r.get("svg") or "")
        if r.get("svg") and f.is_file():
            f.write_text(_unstamped(f.read_text()))
    m.write_text("".join(json.dumps({k: v for k, v in r.items() if k not in ("save_id", "trace_sha256", "svg_sha256")},
                                    sort_keys=True) + "\n" for r in recs))


def _traces_match_their_pictures(src, out):
    """Every manifest record points at an SVG traced from its own picture."""
    from boottool.manifest import latest_by_source
    recs = [r for r in latest_by_source([json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()])
            if r["status"] != "error"]
    for rec in recs:
        if (src / rec["source"]).exists():
            fresh = run_pipeline(src / rec["source"], Params()).svg
            assert _unstamped((out / rec["svg"]).read_text()) == fresh, rec["source"]
    return {r["source"]: r["svg"] for r in recs}


def test_resume_repairs_an_output_folder_where_two_pictures_shared_an_svg(tmp_path, monkeypatch):
    """Output folders made before this fix can have a.png and a.png.webp both pointing at a.png.svg."""
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    Image.open(make_lineart(tmp_path / "w.png", invert=True)).save(src / "a.png.webp")
    runner = CliRunner()
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))             # the old naming
        assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    _as_if_made_by_v090(out)
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "3 to process" in r.output, r.output                   # older records: redone once
    assert "3 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    assert _traces_match_their_pictures(src, out) == {
        "a.jpg": "svg/a.jpg.svg", "a.png": "svg/a.png.svg", "a.png.webp": "svg/a.png.webp.svg"}


def test_resume_when_pictures_are_added_and_removed(tmp_path):
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    make_lineart(src / "a.png")
    assert go().exit_code == 0                                          # a.png -> a.svg
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    r = go()
    assert "1 to process" in r.output, r.output                         # a.png keeps a.svg; a.jpg -> a.jpg.svg
    (src / "a.jpg").unlink()
    Image.open(make_lineart(tmp_path / "w.png", invert=True)).save(src / "a.png.webp")
    assert "1 to process" in go().output
    names = _traces_match_their_pictures(src, out)
    assert names["a.png"] == "svg/a.svg" and names["a.png.webp"] == "svg/a.png.svg"
    (src / "a.png.webp").unlink()
    shutil.copy(src / "a.png", src / "z.png")                           # a byte copy is still its own picture
    r = go()
    assert "1 to process" in r.output, r.output
    assert _traces_match_their_pictures(src, out)["z.png"] == "svg/z.svg"


def test_run_leaves_the_svg_of_a_removed_picture_alone(tmp_path):
    """A picture that is no longer in the input keeps its record and its SVG; a new picture that would
    get the same name is put elsewhere instead of overwriting it."""
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "a.svg").mkdir(parents=True)
    old = make_lineart(tmp_path / "old.png", invert=True)
    Image.open(old).save(src / "a.png.webp")
    make_color(src / "a.svg" / "b.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0                                          # a.png.webp -> a.png.svg
    kept = (out / "svg" / "a.png.svg").read_text()
    (src / "a.png.webp").unlink()
    make_lineart(src / "a.png")                                         # wants a.svg (a folder), then a.png.svg
    assert go().exit_code == 0
    assert (out / "svg" / "a.png.svg").read_text() == kept              # the removed picture's SVG is untouched
    names = _traces_match_their_pictures(src, out)
    assert names == {"a.png": "svg/a.png-2.svg", "a.png.webp": "svg/a.png.svg", "a.svg/b.png": "svg/a.svg/b.svg"}
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    listed = json.loads((out / "dataset.json").read_text())["train"]
    assert sorted(listed) == ["svg/a.png-2.svg", "svg/a.png.svg", "svg/a.svg/b.svg"]   # each SVG once, the removed one too


def test_old_output_folder_where_a_replaced_picture_overwrote_another(tmp_path, monkeypatch):
    """Older versions: a.png -> a.svg, then a.png replaced by a different a.jpg, also -> a.svg. The removed
    a.png's record now shows a.jpg's trace; it must not stay in the dataset or the report as a.png's."""
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))
        make_lineart(src / "a.png")
        assert go().exit_code == 0
        (src / "a.png").unlink()
        Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
        assert go().exit_code == 0                                      # a.jpg traced over a.svg
    _as_if_made_by_v090(out)
    r = go()
    assert r.exit_code == 0 and "1 to process" in r.output, r.output    # can't be sure a.svg is a.jpg's: redone
    assert _traces_match_their_pictures(src, out)["a.jpg"] == "svg/a.jpg.svg"      # a.svg is left as it is
    assert "1 skipped" in go().output                                   # and from now on it can be checked
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    doc = json.loads((out / "dataset.json").read_text())
    assert doc["train"] == ["svg/a.jpg.svg"] and doc["excluded"] == {"unverified": 1}   # a.png's record is left out
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    assert "may hold another picture" in (out / "report.html").read_text()


def test_old_output_folder_with_a_long_history(tmp_path, monkeypatch):
    """v0.9.0: a.png -> a.svg; replaced by a different a.jpg -> a.svg; a.jpeg added, so a.jpg -> a.jpg.svg.
    Then both JPEGs are removed and a.png comes back: a.svg still holds the JPEG's trace."""
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    keep = make_lineart(tmp_path / "keep.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))
        shutil.copy(keep, src / "a.png")
        assert go().exit_code == 0
        (src / "a.png").unlink()
        Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
        assert go().exit_code == 0
        Image.open(make_lineart(tmp_path / "w.png", invert=True)).convert("RGB").save(src / "a.jpeg")
        assert go().exit_code == 0
    _as_if_made_by_v090(out)
    (src / "a.jpg").unlink()
    (src / "a.jpeg").unlink()
    shutil.copy(keep, src / "a.png")
    r = go()
    assert "1 to process" in r.output, r.output
    assert _traces_match_their_pictures(src, out)["a.png"] == "svg/a.png.svg"     # a.svg holds the JPEG's: kept


def test_fingerprints_tell_whose_trace_an_svg_holds(tmp_path, monkeypatch):
    """With fingerprints, the same history needs no retrace: a.svg is known to hold a.jpg."""
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))
        make_lineart(src / "a.png")
        assert go().exit_code == 0
        (src / "a.png").unlink()
        Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
        assert go().exit_code == 0
    assert "1 skipped" in go().output
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    doc = json.loads((out / "dataset.json").read_text())
    assert doc["train"] == ["svg/a.svg"] and doc["excluded"] == {"unverified": 1}


def test_resume_redoes_a_picture_when_it_its_settings_or_its_svg_change(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_lineart(src / "b.png", invert=True)
    runner = CliRunner()
    go = lambda *extra: runner.invoke(main, ["run", str(src), str(out), "--workers", "1", *extra]).output
    assert "2 to process" in go()
    assert "0 to process" in go()
    make_color(src / "a.png")                                           # the picture changed
    assert "1 to process" in go()
    assert "2 to process" in go("--colors", "4")                        # a setting changed
    (out / "svg" / "b.svg").unlink()                                    # an SVG was deleted
    assert "1 to process" in go("--colors", "4")
    (out / "svg" / "a.svg").write_text("")                              # an SVG was emptied
    assert "1 to process" in go("--colors", "4")
    assert "2 to process" in go("--colors", "4", "--force")


def test_renaming_only_the_case_is_redone_and_listed_once(tmp_path):
    """Like any rename, a.PNG -> a.png is a new picture; the old name's SVG is left alone, and the
    dataset lists the picture once (same content)."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.PNG")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    (src / "a.PNG").rename(src / "a.png")
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert "1 to process" in r.output, r.output
    assert _traces_match_their_pictures(src, out)["a.png"] == "svg/a.png.svg"
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    doc = json.loads((out / "dataset.json").read_text())
    assert len(doc["train"]) == 1 and doc["excluded"] == {"duplicate": 1}


@pytest.mark.parametrize("stop_at", ["before the record", "after the record"])
def test_a_run_stopped_halfway_never_leaves_a_wrong_trace_trusted(tmp_path, monkeypatch, stop_at):
    """v0.9.0 folder with a.png.webp -> a.png.svg; the upgraded run (adding a.png and a.jpg, so a.png
    now gets a.png.svg) is stopped right after a.png's record is saved."""
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    Image.open(make_lineart(tmp_path / "w.png", invert=True)).save(src / "a.png.webp")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))
        assert go().exit_code == 0
    _as_if_made_by_v090(out)
    make_lineart(src / "a.png")
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    from boottool.manifest import Manifest
    calls = []

    real_append, real_replace = Manifest.append, cli.os.replace

    def append(self, rec):                                              # a.png traced, its record not saved yet
        if rec["source"] == "a.png":
            calls.append(rec)
            raise KeyboardInterrupt
        real_append(self, rec)

    def replace(a, b):                                                  # a.png's record saved, SVG not moved yet
        if str(b).endswith("a.png.svg"):
            calls.append(b)
            raise KeyboardInterrupt
        real_replace(a, b)
    with monkeypatch.context() as m:
        if stop_at == "before the record":
            m.setattr(Manifest, "append", append)
        else:
            m.setattr(cli.os, "replace", replace)
        assert go().exit_code != 0 and calls
    (src / "a.jpg").unlink()                                            # a.png.webp wants a.png.svg again
    assert go().exit_code == 0
    _traces_match_their_pictures(src, out)
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    listed = json.loads((out / "dataset.json").read_text())["train"]
    for svg in listed:                                                  # every listed SVG is its picture's
        assert svg in _traces_match_their_pictures(src, out).values()


def test_an_older_output_folder_is_redone_once_then_resumes(tmp_path):
    """Records from v0.9.0 have no fingerprint, so nothing proves their SVG is theirs: redo them once."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_lineart(src / "b.png", invert=True)
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    go()
    _as_if_made_by_v090(out)
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    assert len(json.loads((out / "dataset.json").read_text())["train"]) == 2   # still usable before that
    (out / "svg" / "b.svg").write_text("")
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    assert json.loads((out / "dataset.json").read_text())["excluded"] == {"unverified": 1}   # not an SVG
    assert "2 to process" in go()
    assert "2 skipped" in go()


def test_a_failed_save_is_recorded_and_the_run_goes_on(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_lineart(src / "b.png", invert=True)
    real = cli.os.replace

    def replace(a, b):
        if str(b).endswith("a.svg"):
            raise PermissionError("in use")                                # as when it's open on Windows
        real(a, b)
    runner = CliRunner()
    with monkeypatch.context() as m:
        m.setattr(cli.os, "replace", replace)
        r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "could not save the SVG" in r.output, r.output
    assert (out / "svg" / "b.svg").exists() and not (out / "svg" / "a.svg").exists()
    assert not list((out / ".boottool-unsaved").iterdir())                    # nothing left behind
    assert "1 to process" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output


def test_leftovers_and_odd_names_in_the_output(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    (out / "svg" / "a.svg.part").mkdir(parents=True)                        # someone else's things in svg/
    (out / "svg" / ".boottool-sketch.tmp").write_text("mine")
    from boottool.cli import MARK_TEXT
    stopped = out / ".boottool-unsaved" / "run-old"                       # left by a stopped run (marked)
    stopped.mkdir(parents=True)
    (stopped / "made-by-boottool").write_text(MARK_TEXT)
    (stopped / "tmpx.svg").write_text("unsaved")
    lookalike = out / ".boottool-unsaved" / "run-drawings"               # marked, but holds something else
    lookalike.mkdir()
    (lookalike / "made-by-boottool").write_text(MARK_TEXT)
    (lookalike / "original.svg").write_text("mine")
    fake = out / ".boottool-unsaved" / "run-notes"                        # a marker file with other words
    fake.mkdir()
    (fake / "made-by-boottool").write_text("my notes")
    (fake / "tmp1.svg").write_text("mine too")
    (out / ".boottool-unsaved" / "unsaved-draft.svg").write_text("not Boot Tool's")
    r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "error" not in r.output, r.output
    assert (out / "svg" / "a.svg.part").is_dir() and (out / "svg" / ".boottool-sketch.tmp").read_text() == "mine"
    assert sorted(t.name for t in (out / ".boottool-unsaved").iterdir()) == ["run-drawings", "run-notes", "unsaved-draft.svg"]
    assert (lookalike / "original.svg").read_text() == "mine" and (fake / "tmp1.svg").read_text() == "mine too"
    _traces_match_their_pictures(src, out)
    if os.name != "nt":                                                     # a name as long as disks allow
        long = "x" * 251 + ".png"
        shutil.copy(src / "a.png", src / long)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
        assert r.exit_code == 0 and "error" not in r.output, r.output


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs Unix permissions")
def test_an_unreadable_svg_does_not_stop_everything(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_lineart(src / "b.png", invert=True)
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    (out / "svg" / "a.svg").chmod(0)
    try:
        r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
        assert r.exit_code == 0 and "1 skipped" in r.output, r.output   # b is fine; a is tried again
        for cmd in (["report", str(out)], ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]):
            assert runner.invoke(main, cmd).exit_code == 0
    finally:
        (out / "svg" / "a.svg").chmod(0o644)


def test_a_manifest_line_cut_off_by_a_stopped_run(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_lineart(src / "b.png", invert=True)
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    m = out / "manifest.jsonl"
    lines = m.read_text().splitlines()
    m.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2])   # b's last line was being written
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "2 skipped" in r.output, r.output        # b's SVG was in place: its note says so
    m.write_text("\n".join(lines[:-2]) + "\n" + lines[-2][: len(lines[-2]) // 2])   # stopped while noting b down
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "1 to process" in r.output, r.output     # b's SVG can't be vouched for: redone
    assert [r["source"] for r in _finished(m)] == ["a.png", "b.png"]
    assert "2 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    m.write_text(m.read_text().rstrip("\n"))                             # a whole last record, just no newline
    assert "2 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    make_lineart(src / "c.png")
    assert "1 to process" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    assert [r["source"] for r in _finished(m)] == ["a.png", "b.png", "c.png"]
    m.write_text("not json\n" + m.read_text())                          # damage in the middle is reported
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code != 0


def test_a_hand_edited_svg_of_a_removed_picture_is_kept(tmp_path):
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0
    (out / "svg" / "a.svg").write_text((out / "svg" / "a.svg").read_text().replace("<path", "<path data-edited='1'", 1))
    edited = (out / "svg" / "a.svg").read_text()
    (src / "a.png").unlink()
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    assert go().exit_code == 0
    assert (out / "svg" / "a.svg").read_text() == edited
    assert _traces_match_their_pictures(src, out)["a.jpg"] == "svg/a.jpg.svg"


def test_redoing_a_picture_keeps_its_licence(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--license", "CC0"]).exit_code == 0
    _as_if_made_by_v090(out)                                            # redone once, without --license
    assert "1 to process" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    assert "1 to process" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--colors", "4"]).output
    rec = json.loads((out / "manifest.jsonl").read_text().splitlines()[-1])
    assert rec["provenance"] == {"license": "CC0"} and rec["provenance_declared"]


@pytest.mark.skipif(os.name == "nt", reason="Unix permissions")
def test_saved_svgs_get_the_usual_permissions(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    assert CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    umask = os.umask(0)
    os.umask(umask)
    assert (out / "svg" / "a.svg").stat().st_mode & 0o777 == 0o666 & ~umask


def test_report_links_survive_odd_characters(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "a.svg#b").mkdir(parents=True) if os.name != "nt" else src.mkdir(parents=True)
    make_lineart(src / "c.png")
    if os.name != "nt":
        make_lineart(src / "a.svg#b" / "d.png", invert=True)
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    page = (out / "report.html").read_text()
    assert "src='svg/c.svg'" in page
    if os.name != "nt":
        assert "svg/a.svg%23b/d.svg" in page and "svg/a.svg#b" not in page


def _edit(svg):
    svg.write_text(svg.read_text().replace("<path", "<path data-edited='1'", 1))
    return svg.read_text()


def test_an_edited_svg_survives_a_failed_redo_and_a_new_picture(tmp_path, monkeypatch):
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda *x: runner.invoke(main, ["run", str(src), str(out), "--workers", "1", *x])
    assert go().exit_code == 0
    edited = _edit(out / "svg" / "a.svg")
    with monkeypatch.context() as m:                                    # redoing it fails this time
        m.setattr(cli, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert "boom" in go("--colors", "4").output
    (src / "a.png").unlink()
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    assert go().exit_code == 0
    assert (out / "svg" / "a.svg").read_text() == edited
    assert _traces_match_their_pictures(src, out)["a.jpg"] == "svg/a.jpg.svg"


def test_an_edited_svg_survives_its_picture_moving_then_going(tmp_path):
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0
    edited = _edit(out / "svg" / "a.svg")
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    assert go().exit_code == 0                                          # a.png moves to a.png.svg
    (src / "a.png").unlink()
    assert go().exit_code == 0                                          # a.jpg would like a.svg now
    assert (out / "svg" / "a.svg").read_text() == edited
    assert _traces_match_their_pictures(src, out)["a.jpg"] == "svg/a.jpg.svg"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_a_linked_work_folder_is_refused(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    make_lineart(src / "a.png")
    elsewhere = tmp_path / "notes"
    elsewhere.mkdir()
    (elsewhere / "unsaved-1.svg").write_text("mine")
    (out / ".boottool-unsaved").symlink_to(elsewhere)
    r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code != 0 and "plain folder" in r.output, r.output
    assert (elsewhere / "unsaved-1.svg").read_text() == "mine"


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs Unix permissions")
def test_an_unreadable_old_folder_does_not_stop_the_run(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "private").mkdir(parents=True)
    make_lineart(src / "private" / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    shutil.rmtree(src / "private")
    make_lineart(src / "b.png", invert=True)
    (out / "svg" / "private").chmod(0)
    try:
        r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
        assert r.exit_code == 0 and "1 to process" in r.output, r.output
    finally:
        (out / "svg" / "private").chmod(0o755)
    assert (out / "svg" / "b.svg").exists()


def test_the_dataset_lists_each_svg_once(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    m = out / "manifest.jsonl"
    rec = _finished(m)[0]
    m.write_text(m.read_text() + json.dumps({**rec, "source": "b.png", "sha256": "f" * 64}) + "\n")   # claims the same SVG
    for frac in ("0", "1", "0.5"):
        assert runner.invoke(main, ["dataset", str(out), "--val-frac", frac, "--min-score", "0"]).exit_code == 0
        doc = json.loads((out / "dataset.json").read_text())
        assert doc["train"] + doc["val"] == ["svg/a.svg"]


def test_svg_folder_on_another_disk(tmp_path, monkeypatch):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    (out / "svg").mkdir(parents=True)
    (out / "svg" / "a.svg").write_text("<svg>my drawing</svg>")
    with monkeypatch.context() as m:
        _replace_on_another_disk(m, out)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "error" not in r.output, r.output
    _traces_match_their_pictures(src, out)
    assert (out / "svg" / "a.svg").read_text() == "<svg>my drawing</svg>"
    assert not list((out / ".boottool-unsaved").iterdir()) and not list((out / "svg").glob(".boottool-*"))


def test_a_failed_write_leaves_nothing_behind(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")

    def full(*a, **k):
        raise OSError(28, "No space left on device")
    with monkeypatch.context() as m:
        m.setattr(cli.os, "chmod", full)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "No space left" in r.output, r.output
    assert not list((out / ".boottool-unsaved").iterdir())


def test_a_drawing_already_in_the_output_and_an_edited_trace_are_kept(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_lineart(src / "b.png", invert=True)
    (out / "svg").mkdir(parents=True)
    (out / "svg" / "a.svg").write_text("<svg>my drawing</svg>")          # there before Boot Tool ran
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    go()
    assert (out / "svg" / "a.svg").read_text() == "<svg>my drawing</svg>"
    edited = _edit(out / "svg" / "b.svg")                               # its own trace, edited by hand
    assert "1 to process" in go()
    assert (out / "svg" / "b.svg").read_text() == edited
    names = _traces_match_their_pictures(src, out)
    assert names == {"a.png": "svg/a.png.svg", "b.png": "svg/b.png.svg"}
    assert "2 skipped" in go()                                          # and it settles there


def test_names_stay_put_when_a_redo_fails_now_and_then(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force"])
    for _ in range(3):
        assert go().exit_code == 0
        with monkeypatch.context() as m:
            m.setattr(cli, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
            assert "boom" in go().output
    assert sorted(p.name for p in (out / "svg").iterdir()) == ["a.svg"]
    assert "1 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output


def _case_sensitive(folder):
    (folder / "Cc").write_text("")
    try:
        return not (folder / "cc").exists()
    finally:
        (folder / "Cc").unlink()


def test_every_spelling_of_an_old_name_is_checked(tmp_path, monkeypatch):
    """v0.9.0 on Linux: A.png -> A.svg and a.jpg -> a.svg. A.svg deleted, a.jpg removed, a.png added."""
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    if not _case_sensitive(src):
        pytest.skip("needs a disk that tells A and a apart")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))
        make_lineart(src / "A.png")
        assert go().exit_code == 0
        (src / "A.png").unlink()
        Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
        assert go().exit_code == 0
    _as_if_made_by_v090(out)
    edited = _edit(out / "svg" / "a.svg")
    (out / "svg" / "A.svg").unlink()
    (src / "a.jpg").unlink()
    make_lineart(src / "a.png", invert=True)
    assert go().exit_code == 0
    assert (out / "svg" / "a.svg").read_text() == edited


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_linked_output_folders_never_share_a_file(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "left").mkdir(parents=True)
    (src / "right").mkdir()
    make_lineart(src / "left" / "a.png")
    make_color(src / "right" / "a.png")
    (out / "svg" / "right").mkdir(parents=True)
    (out / "svg" / "left").symlink_to(out / "svg" / "right")
    runner = CliRunner()
    for _ in range(2):
        assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
        _traces_match_their_pictures(src, out)
    assert "2 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output


def test_a_file_called_manifest_jsonl_that_isnt_one_is_left_alone(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    make_lineart(src / "a.png")
    for notes in ("my notes, no newline", '{"title": "my notes"}\n', '{"title": "my notes',
                  '{"source": "camera", "status": "queued"}\n{"source": "camera", "sta'):
        (out / "manifest.jsonl").write_text(notes)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
        assert r.exit_code != 0, notes
        assert (out / "manifest.jsonl").read_text() == notes
    (out / "manifest.jsonl").write_text('{"boottool_version": "0.9.0", "crea')  # the very first record, cut off
    r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "1 to process" in r.output, r.output
    assert [r["source"] for r in _finished(out / "manifest.jsonl")] == ["a.png"]


def test_leftovers_from_a_save_to_another_disk_are_cleared(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    from boottool.cli import MARK_TEXT
    left = out / "svg" / ".boottool-unsaved-old"
    left.mkdir(parents=True)
    (left / "made-by-boottool").write_text(MARK_TEXT)
    (left / "unsaved.svg").write_text("half saved")
    mine = out / "svg" / ".boottool-unsaved-mine"                       # not marked: not Boot Tool's
    mine.mkdir()
    assert CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    assert not left.exists() and mine.is_dir()


def test_upgrading_keeps_a_hand_edited_svg(tmp_path):
    """v0.9.0 traced a.png -> a.svg, then a.svg was edited. The upgraded run redoes a.png (no fingerprint)
    and must not destroy the edit: an unedited old trace is simply replaced, an edited one is kept aside."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_lineart(src / "b.png", invert=True)
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0
    _as_if_made_by_v090(out)
    edited = _edit(out / "svg" / "a.svg")
    r = go()
    assert r.exit_code == 0 and "2 to process" in r.output, r.output
    assert [p.read_text() for p in _kept(out / "svg")] == [edited]            # the edit is kept
    assert not _kept(out / "svg", "b")                                       # b.svg wasn't edited: just replaced
    _traces_match_their_pictures(src, out)


def test_a_file_made_while_tracing_is_kept(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    real = cli.run_pipeline

    def slow(path, p):
        (out / "svg").mkdir(parents=True, exist_ok=True)
        (out / "svg" / "a.svg").write_text("<svg>drawn meanwhile</svg>")     # someone saves a drawing right now
        return real(path, p)
    with monkeypatch.context() as m:
        m.setattr(cli, "run_pipeline", slow)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0, r.output
    assert [p.read_text() for p in _kept(out / "svg")] == ["<svg>drawn meanwhile</svg>"]
    _traces_match_their_pictures(src, out)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_a_link_where_an_svg_goes_is_not_followed_or_lost(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    (out / "svg").mkdir(parents=True)
    (out / "svg" / "a.svg").symlink_to(tmp_path / "nowhere.svg")           # a broken link
    r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0, r.output
    assert (out / "svg" / "a.svg").is_symlink() and not (tmp_path / "nowhere.svg").exists()
    assert _traces_match_their_pictures(src, out)["a.png"] == "svg/a.png.svg"


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs Unix permissions")
def test_a_folder_that_cant_be_looked_into_fails_only_its_pictures(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "private").mkdir(parents=True)
    make_lineart(src / "private" / "a.png")
    make_lineart(src / "b.png", invert=True)
    (out / "svg" / "private").mkdir(parents=True)
    (out / "svg" / "private").chmod(0)
    try:
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "2"])
    finally:
        (out / "svg" / "private").chmod(0o755)
    assert r.exit_code == 0 and "1 can't be saved" in r.output and "1 to process" in r.output, r.output
    assert (out / "svg" / "b.svg").exists()


def test_a_save_that_keeps_failing_doesnt_pile_up_names(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    real = cli.os.replace

    def stuck(a, b):
        if str(b).endswith("a.svg"):
            raise PermissionError("in use")
        real(a, b)
    for ink in ("#ff0000", "#00ff00"):
        with monkeypatch.context() as m:
            m.setattr(cli.os, "replace", stuck)
            assert "could not save" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", ink]).output
    assert sorted(p.name for p in (out / "svg").iterdir()) == ["a.svg"]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_a_linked_folder_and_a_folder_the_run_needs(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "left").mkdir(parents=True)
    (src / "right" / "a.svg").mkdir(parents=True)
    make_lineart(src / "left" / "a.png")
    make_color(src / "right" / "a.svg" / "b.png")
    (out / "svg" / "right").mkdir(parents=True)
    (out / "svg" / "left").symlink_to(out / "svg" / "right")
    r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "error" not in r.output, r.output
    _traces_match_their_pictures(src, out)


def test_after_a_failed_redo_report_and_dataset_still_use_the_saved_trace(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    with monkeypatch.context() as m:
        m.setattr(cli, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert "boom" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force"]).output
    assert "1 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    assert json.loads((out / "dataset.json").read_text())["train"] == ["svg/a.svg"]
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    page = (out / "report.html").read_text()
    assert "src='svg/a.svg'" in page and "last attempt failed" in page


def test_a_name_stays_once_settled(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    (out / "svg").mkdir(parents=True)
    (out / "svg" / "a.svg").write_text("<svg>my drawing</svg>")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).output
    go()                                                                # a.png -> a.png.svg
    (out / "svg" / "a.svg").rename(tmp_path / "drawing.svg")            # the drawing is moved away
    assert "1 skipped" in go()
    assert sorted(p.name for p in (out / "svg").iterdir()) == ["a.png.svg"]


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs Unix permissions")
def test_nothing_to_do_works_in_a_read_only_folder(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    for d in [out, *[p for p in out.rglob("*") if p.is_dir()]]:
        d.chmod(0o555)
    try:
        r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
        assert r.exit_code == 0 and "1 skipped" in r.output, r.output
    finally:
        for d in [out, *[p for p in out.rglob("*") if p.is_dir()]]:
            d.chmod(0o755)


def test_upgrading_keeps_names_when_a_picture_was_removed(tmp_path, monkeypatch):
    """v0.9.0 traced a.png and a.jpg to a.png.svg and a.jpg.svg; a.jpg is removed, then the upgrade runs."""
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    runner = CliRunner()
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))
        assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    _as_if_made_by_v090(out)
    (src / "a.jpg").unlink()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    assert sorted(p.name for p in (out / "svg").iterdir()) == ["a.jpg.svg", "a.png.svg"]
    assert not list((out / "svg").rglob("*.replaced-*.svg"))


def test_an_interrupted_redo_keeps_the_name(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0

    def stop(*a):
        raise KeyboardInterrupt
    with monkeypatch.context() as m:                                    # stopped after the record, before the move
        m.setattr(cli.os, "replace", stop)
        assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#ff0000"]).exit_code != 0
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#ff0000"])
    assert r.exit_code == 0 and "1 to process" in r.output, r.output
    assert sorted(p.name for p in (out / "svg").iterdir()) == ["a.svg"]
    assert not list((out / "svg").rglob("*.replaced-*.svg"))


def test_upgrading_with_other_settings_keeps_no_needless_copy(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#ff0000"]).exit_code == 0
    _as_if_made_by_v090(out)
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#00ff00", "--force"])
    assert r.exit_code == 0 and "1 to process" in r.output, r.output
    assert not list((out / "svg").rglob("*.replaced-*.svg"))                              # the old red trace wasn't edited


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs Unix permissions")
def test_report_after_a_picture_that_couldnt_be_saved(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "private").mkdir(parents=True)
    make_lineart(src / "private" / "a.png")
    make_lineart(src / "b.png", invert=True)
    (out / "svg" / "private").mkdir(parents=True)
    (out / "svg" / "private").chmod(0)
    runner = CliRunner()
    try:
        assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
        for cmd in (["report", str(out)], ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]):
            r = runner.invoke(main, cmd)
            assert r.exit_code == 0, r.output
    finally:
        (out / "svg" / "private").chmod(0o755)
    assert "look in svg/ for private/a.svg" in (out / "report.html").read_text()


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs Unix permissions")
def test_a_linked_folder_that_cant_be_reached(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "private").mkdir(parents=True)
    make_lineart(src / "private" / "a.png")
    make_lineart(src / "b.png", invert=True)
    locked = tmp_path / "locked"
    (locked / "inside").mkdir(parents=True)
    (out / "svg").mkdir(parents=True)
    (out / "svg" / "private").symlink_to(locked / "inside")
    locked.chmod(0)
    try:
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    finally:
        locked.chmod(0o755)
    assert r.exit_code == 0 and (out / "svg" / "b.svg").exists(), r.output


@pytest.mark.skipif(os.name == "nt", reason="long names behave differently on Windows")
def test_a_kept_copy_with_a_very_long_name(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    long = "x" * 251
    make_lineart(src / (long + ".png"))
    runner = CliRunner()
    go = lambda *x: runner.invoke(main, ["run", str(src), str(out), "--workers", "1", *x])
    assert go().exit_code == 0
    _as_if_made_by_v090(out)
    edited = _edit(out / "svg" / (long + ".svg"))
    r = go()
    assert r.exit_code == 0 and "error" not in r.output, r.output
    kept = [p for p in (out / "svg").iterdir() if p.name.endswith(".replaced-1.svg")]
    assert len(kept) == 1 and len(kept[0].name.encode()) <= 255 and kept[0].read_text() == edited


def test_an_edit_saved_while_an_old_trace_is_checked_is_kept(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#ff0000"]).exit_code == 0
    _as_if_made_by_v090(out)
    real = cli._reproduces

    def slow(rec, sha, picture):
        ok = real(rec, sha, picture)
        (out / "svg" / "a.svg").write_text("<svg>edited meanwhile</svg>")  # saved while it was being checked
        return ok
    with monkeypatch.context() as m:
        m.setattr(cli, "_reproduces", slow)
        r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#00ff00", "--force"])
    assert r.exit_code == 0, r.output
    assert [p.read_text() for p in _kept(out / "svg")] == ["<svg>edited meanwhile</svg>"]


def test_an_interrupted_first_upgrade_keeps_the_name(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    _as_if_made_by_v090(out)

    def stop(*a):
        raise KeyboardInterrupt
    with monkeypatch.context() as m:
        m.setattr(cli.os, "replace", stop)
        assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#ff0000"]).exit_code != 0
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    assert json.loads((out / "dataset.json").read_text())["train"] == ["svg/a.svg"]       # the intact old trace
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#ff0000"]).exit_code == 0
    assert sorted(p.name for p in (out / "svg").iterdir()) == ["a.svg"]
    assert not list((out / "svg").rglob("*.replaced-*.svg"))


def test_an_old_trace_is_recognised_after_the_pictures_moved(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#ff0000"]).exit_code == 0
    _as_if_made_by_v090(out)
    moved = tmp_path / "moved"
    src.rename(moved)
    r = runner.invoke(main, ["run", str(moved), str(out), "--workers", "1", "--ink", "#00ff00", "--force"])
    assert r.exit_code == 0, r.output
    assert not list((out / "svg").rglob("*.replaced-*.svg"))


def test_text_after_a_manifest_stops_before_any_work(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    m = out / "manifest.jsonl"
    m.write_text(m.read_text() + "my notes")
    make_lineart(src / "b.png", invert=True)
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code != 0 and "to process" not in r.output, r.output
    assert m.read_text().endswith("my notes") and not (out / "svg" / "b.svg").exists()


def test_an_svg_copied_over_another_after_the_upgrade_is_not_trusted(tmp_path, monkeypatch):
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    runner = CliRunner()
    go = lambda *x: runner.invoke(main, ["run", str(src), str(out), "--workers", "1", *x])
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))
        assert go("--license", "old").exit_code == 0
    _as_if_made_by_v090(out)
    assert go("--license", "new").exit_code == 0                       # the upgrade
    shutil.copy(out / "svg" / "a.jpg.svg", out / "svg" / "a.png.svg")   # someone copies one over the other
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    assert "src='svg/a.png.svg'" not in (out / "report.html").read_text()
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0", "--require-provenance"]).exit_code == 0
    assert json.loads((out / "dataset.json").read_text())["train"] == ["svg/a.jpg.svg"]
    assert go().exit_code == 0
    rec = [r for r in _finished(out / "manifest.jsonl") if r["source"] == "a.png"][-1]
    assert rec["provenance"] == {"license": "new"}


def test_a_broken_tail_that_isnt_a_cut_record_stops_the_run(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    m = out / "manifest.jsonl"
    m.write_text(m.read_text() + '{"boottool_version": !my notes')
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force"])
    assert r.exit_code != 0 and "to process" not in r.output, r.output
    assert m.read_text().endswith("!my notes")


def _stop_after_the_move(monkeypatch):
    """Stop the run right after an SVG is moved in, before its finished record is written."""
    from boottool.manifest import Manifest
    real = Manifest.append

    def append(self, rec):
        if rec["status"] in ("ok", "low_score"):                         # a finished record (notes say "saving")
            raise KeyboardInterrupt
        real(self, rec)
    monkeypatch.setattr(Manifest, "append", append)


def _stop_before_the_move(monkeypatch):
    import boottool.cli as cli
    monkeypatch.setattr(cli.os, "replace", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))


@pytest.mark.parametrize("how", ["stopped", "failed"])
def test_a_licence_isnt_published_by_a_save_that_didnt_happen(tmp_path, monkeypatch, how):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    with monkeypatch.context() as m:                                    # same trace, new licence, not saved
        if how == "stopped":
            _stop_before_the_move(m)
        else:
            m.setattr(cli.os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError("in use")))
        runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force", "--license", "new"])
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    assert "licence: MISSING" in (out / "report.html").read_text()
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0", "--require-provenance"]).exit_code == 0
    assert json.loads((out / "dataset.json").read_text())["train"] == []


def test_a_save_that_happened_survives_a_later_stop(tmp_path, monkeypatch):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda *x: runner.invoke(main, ["run", str(src), str(out), "--workers", "1", *x])
    assert go("--ink", "#ff0000").exit_code == 0
    with monkeypatch.context() as m:
        _stop_after_the_move(m)
        go("--ink", "#00ff00", "--license", "green")                    # saved, but its record never written
    with monkeypatch.context() as m:
        _stop_before_the_move(m)
        go("--ink", "#0000ff")                                          # not saved
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    assert "licence: green" in (out / "report.html").read_text()
    r = go("--ink", "#00ff00")
    assert "1 skipped" in r.output, r.output
    assert sorted(p.name for p in (out / "svg").iterdir()) == ["a.svg"]


def test_a_recovered_save_owns_its_svg(tmp_path, monkeypatch):
    import boottool.cli as cli
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0
    _as_if_made_by_v090(out)
    (src / "a.png").unlink()
    (out / "svg" / "a.svg").unlink()
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.webp")
    with monkeypatch.context() as m:
        _stop_after_the_move(m)
        go()                                                            # a.webp -> a.svg, record not written
    assert "1 skipped" in go().output
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    doc = json.loads((out / "dataset.json").read_text())
    assert doc["train"] == ["svg/a.svg"] and doc["excluded"] == {"unverified": 1}
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    assert (out / "report.html").read_text().count("src='svg/a.svg'") == 1


def test_an_impossible_tail_stops_the_run(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    m = out / "manifest.jsonl"
    m.write_text(m.read_text() + '{"boottool_version": "0.8.8", "created_at": !my notes')
    r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force"])
    assert r.exit_code != 0 and "to process" not in r.output, r.output
    assert m.read_text().endswith("!my notes")


def test_a_failed_manifest_write_after_a_save_doesnt_undo_it(tmp_path, monkeypatch):
    from boottool.manifest import Manifest
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    real = Manifest.append

    def append(self, rec):
        if rec["status"] in ("ok", "low_score"):
            raise OSError(5, "Input/output error")                         # the finished record can't be written
        real(self, rec)
    with monkeypatch.context() as m:
        m.setattr(Manifest, "append", append)
        r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force", "--ink", "#ff0000", "--license", "new"])
    assert r.exit_code == 0, r.output
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    assert "licence: new" in (out / "report.html").read_text()
    assert "1 skipped" in runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--ink", "#ff0000"]).output
    assert sorted(p.name for p in (out / "svg").iterdir()) == ["a.svg"]


def _replace_on_another_disk(monkeypatch, out):
    """As if svg/ (and replaced/) were on a different disk from the rest of the output folder."""
    import boottool.cli as cli
    monkeypatch.setattr(cli, "_same_disk", lambda a, b: False)


def test_a_save_to_another_disk_that_was_stopped_is_recognised(tmp_path, monkeypatch):
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda *x: runner.invoke(main, ["run", str(src), str(out), "--workers", "1", *x])
    assert go("--license", "old").exit_code == 0
    _as_if_made_by_v090(out)
    (src / "a.png").unlink()
    (out / "svg" / "a.svg").unlink()
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    with monkeypatch.context() as m:
        _replace_on_another_disk(m, out)
        _stop_after_the_move(m)
        go("--license", "new")
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0", "--require-provenance"]).exit_code == 0
    assert json.loads((out / "dataset.json").read_text())["train"] == ["svg/a.svg"]
    assert runner.invoke(main, ["report", str(out)]).exit_code == 0
    page = (out / "report.html").read_text()
    assert page.count("src='svg/a.svg'") == 1 and "licence: new" in page
    assert "1 skipped" in go("--license", "new").output


def test_editing_a_recovered_svg_keeps_who_it_belongs_to(tmp_path, monkeypatch):
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0
    _as_if_made_by_v090(out)
    (src / "a.png").unlink()
    (out / "svg" / "a.svg").unlink()
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    with monkeypatch.context() as m:
        _stop_after_the_move(m)
        go()
    edited = _edit(out / "svg" / "a.svg")                               # edited before the next run
    for cmd in (["report", str(out)], ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]):
        assert runner.invoke(main, cmd).exit_code == 0
    assert "src='svg/a.svg'" not in (out / "report.html").read_text()   # a.png doesn't get it back
    assert json.loads((out / "dataset.json").read_text())["train"] == []
    assert go().exit_code == 0
    assert (out / "svg" / "a.svg").read_text() == edited or [p.read_text() for p in _kept(out / "svg")] == [edited]
    jpg = [r for r in _finished(out / "manifest.jsonl") if r["source"] == "a.jpg"]
    assert jpg, "the stopped save of a.jpg was written down"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_an_old_folder_where_linked_folders_shared_a_file(tmp_path, monkeypatch):
    """v0.9.0 with svg/right linked to svg/left wrote left/a.png and right/a.png to one file."""
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "left").mkdir(parents=True)
    (src / "right").mkdir()
    make_lineart(src / "left" / "a.png")
    make_color(src / "right" / "a.png")
    (out / "svg" / "left").mkdir(parents=True)
    (out / "svg" / "right").symlink_to(out / "svg" / "left")
    runner = CliRunner()
    with monkeypatch.context() as m:
        m.setattr(cli, "svg_names", lambda rels, *a: _old_svg_names(rels))
        assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    _as_if_made_by_v090(out)
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    doc = json.loads((out / "dataset.json").read_text())
    assert doc["train"] == [] and doc["excluded"] == {"unverified": 2}
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    _traces_match_their_pictures(src, out)


def test_a_file_made_during_a_copy_to_another_disk_is_kept(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    real = cli.shutil.copyfile

    def copyfile(a, b, *k, **kw):
        (out / "svg" / "a.svg").write_text("<svg>saved during the copy</svg>")
        return real(a, b, *k, **kw)
    with monkeypatch.context() as m:
        _replace_on_another_disk(m, out)
        m.setattr(cli.shutil, "copyfile", copyfile)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0, r.output
    assert [p.read_text() for p in _kept(out / "svg")] == ["<svg>saved during the copy</svg>"]
    _traces_match_their_pictures(src, out)


def test_an_edit_made_while_a_file_is_kept_aside_is_kept_too(tmp_path, monkeypatch):
    """An editor still has the file open and saves after Boot Tool kept it aside: the edit must land in
    the kept file (it's the very same file, renamed), even with svg/ on another disk."""
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    real_trace = cli.run_pipeline
    handle = {}

    def trace(path, p):                                                 # a drawing is opened while tracing
        (out / "svg").mkdir(parents=True, exist_ok=True)
        (out / "svg" / "a.svg").write_text("<svg>first</svg>")
        handle["f"] = open(out / "svg" / "a.svg", "r+")
        return real_trace(path, p)
    with monkeypatch.context() as m:
        _replace_on_another_disk(m, out)
        m.setattr(cli, "run_pipeline", trace)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0, r.output
    with handle["f"] as f:                                              # the editor saves now
        f.seek(0)
        f.write("<svg>second</svg>")
        f.truncate()
    if os.name == "nt":                                                 # Windows won't rename a file that's open:
        assert "could not save the SVG" in r.output                     # the picture fails, the drawing stays put
        assert (out / "svg" / "a.svg").read_text() == "<svg>second</svg>"
    else:
        assert [p.read_text() for p in _kept(out / "svg")] == ["<svg>second</svg>"]
        _traces_match_their_pictures(src, out)


def test_a_stripped_stamp_doesnt_give_an_svg_to_an_old_record(tmp_path, monkeypatch):
    from PIL import Image
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda *x: runner.invoke(main, ["run", str(src), str(out), "--workers", "1", *x])
    assert go("--license", "old").exit_code == 0
    _as_if_made_by_v090(out)
    (src / "a.png").unlink()
    (out / "svg" / "a.svg").unlink()
    Image.open(make_color(tmp_path / "c.png")).convert("RGB").save(src / "a.jpg")
    with monkeypatch.context() as m:
        _stop_after_the_move(m)
        go("--license", "new")
    f = out / "svg" / "a.svg"
    f.write_text(_unstamped(f.read_text()))                             # an editor drops the stamp
    for _ in range(2):
        assert runner.invoke(main, ["report", str(out)]).exit_code == 0
        assert "src='svg/a.svg'" not in (out / "report.html").read_text()
        assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0", "--require-provenance"]).exit_code == 0
        assert "svg/a.svg" not in json.loads((out / "dataset.json").read_text())["train"]
        assert go("--license", "new").exit_code == 0                    # and after a retry too


def test_stamps_hold_only_the_save_id(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--license", "secret-client"]).exit_code == 0
    text = (out / "svg" / "a.svg").read_text()
    assert re.search(r"<!-- boottool-run save [0-9a-f]{32} -->\s*$", text)
    assert str(tmp_path) not in text and "secret-client" not in text
    (out / "svg" / "a.svg").write_text(_unstamped(text) + "\n<!-- boottool-run WzFd -->\n")   # a lookalike
    for cmd in (["run", str(src), str(out), "--workers", "1"], ["report", str(out)],
                ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]):
        assert runner.invoke(main, cmd).exit_code == 0


def test_manifest_tails_are_judged_the_same_when_reading_and_writing(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    m = out / "manifest.jsonl"
    whole = m.read_text()
    m.write_text(whole + "   ")                                          # just spaces, no newline
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force"]).exit_code == 0
    for bad in ('{"boottool_version": false, "source": 42, ', '{"boottool_version": "0.9.0", "source":42, '):
        m.write_text(whole + bad)                                       # can't be one of ours
        r = runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force"])
        assert r.exit_code != 0 and m.read_text().endswith(bad), bad
    m.write_bytes((whole + "\u00a0").encode("utf-8"))                  # a no-break space, no newline
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1", "--force"]).exit_code == 0


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_leftovers_under_a_linked_folder_are_cleared(tmp_path):
    from boottool.cli import MARK_TEXT
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "linked").mkdir(parents=True)
    make_lineart(src / "linked" / "a.png")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (out / "svg").mkdir(parents=True)
    (out / "svg" / "linked").symlink_to(elsewhere)
    runner = CliRunner()
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    left = elsewhere / ".boottool-unsaved-old"
    left.mkdir()
    (left / "made-by-boottool").write_text(MARK_TEXT)
    (left / "unsaved.svg").write_text("half saved")
    assert runner.invoke(main, ["run", str(src), str(out), "--workers", "1"]).exit_code == 0
    assert not left.exists()


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_links_and_folders_that_turn_up_are_kept_as_they_are(tmp_path, monkeypatch):
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_lineart(src / "b.png", invert=True)
    real_trace = cli.run_pipeline

    def trace(path, p):
        svg = out / "svg"
        svg.mkdir(parents=True, exist_ok=True)
        (svg / "reference.svg").write_text("<svg>reference</svg>")
        if Path(path).name == "a.png" and not os.path.lexists(svg / "a.svg"):
            (svg / "a.svg").symlink_to("reference.svg")                # a relative link turns up
        if Path(path).name == "b.png" and not os.path.lexists(svg / "b.svg"):
            (svg / "b.svg").mkdir()                                     # a folder turns up
        return real_trace(path, p)
    with monkeypatch.context() as m:
        _replace_on_another_disk(m, out)
        m.setattr(cli, "run_pipeline", trace)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "error" not in r.output, r.output
    link = _kept(out / "svg")[0]
    assert link.is_symlink() and link.read_text() == "<svg>reference</svg>"   # still points where it did
    assert _kept(out / "svg", "b")[0].is_dir()
    _traces_match_their_pictures(src, out)


def test_a_rename_that_says_cross_device_on_one_disk(tmp_path, monkeypatch):
    """A bind-mounted svg/: same disk by stat, but renaming into it fails as if across disks."""
    import errno
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    real = cli.os.replace
    staging = str(out / ".boottool-unsaved") + os.sep

    def replace(a, b):
        if str(a).startswith(staging):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real(a, b)
    with monkeypatch.context() as m:
        m.setattr(cli.os, "replace", replace)
        r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "error" not in r.output, r.output
    _traces_match_their_pictures(src, out)
    assert not list((out / "svg").glob(".boottool-*"))


def test_a_kept_file_never_takes_a_name_thats_spoken_for(tmp_path, monkeypatch):
    """v0.9.0 traced a.png and a.replaced-1.png. The second picture and its SVG go, a.svg is edited,
    and a folder a.replaced-2.svg/ with a picture arrives: the edit must not land on either name."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    make_color(src / "a.replaced-1.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0
    _as_if_made_by_v090(out)
    (src / "a.replaced-1.png").unlink()
    (out / "svg" / "a.replaced-1.svg").unlink()
    edited = _edit(out / "svg" / "a.svg")
    (src / "a.replaced-2.svg").mkdir()
    make_color(src / "a.replaced-2.svg" / "b.png")
    r = go()
    assert r.exit_code == 0 and "error" not in r.output, r.output
    assert (out / "svg" / "a.replaced-3.svg").read_text() == edited
    assert not (out / "svg" / "a.replaced-1.svg").exists() and (out / "svg" / "a.replaced-2.svg").is_dir()
    _traces_match_their_pictures(src, out)
    assert runner.invoke(main, ["dataset", str(out), "--val-frac", "0", "--min-score", "0"]).exit_code == 0
    assert "svg/a.replaced-1.svg" not in json.loads((out / "dataset.json").read_text())["train"]
    assert "2 skipped" in go().output


@pytest.mark.skipif(os.name == "nt", reason="symlinks need extra rights on Windows")
def test_a_kept_file_avoids_spoken_for_names_after_a_link_is_repointed(tmp_path, monkeypatch):
    """svg/archive/a.replaced-1.svg belongs to an old record (its file is gone). While a run traces,
    archive is re-pointed at live/: the edit kept from live/a.svg must not become that record's file."""
    import boottool.cli as cli
    src, out = tmp_path / "in", tmp_path / "out"
    (src / "live").mkdir(parents=True)
    (src / "archive").mkdir()
    make_lineart(src / "live" / "a.png")
    make_color(src / "archive" / "a.replaced-1.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0
    _as_if_made_by_v090(out)
    shutil.rmtree(src / "archive")
    shutil.rmtree(out / "svg" / "archive")
    edited = _edit(out / "svg" / "live" / "a.svg")
    real = cli.run_pipeline

    def trace(path, p):
        if not (out / "svg" / "archive").exists():
            (out / "svg" / "archive").symlink_to(out / "svg" / "live")   # re-pointed after planning
        return real(path, p)
    with monkeypatch.context() as m:
        m.setattr(cli, "run_pipeline", trace)
        assert go().exit_code == 0
    assert (out / "svg" / "live" / "a.replaced-2.svg").read_text() == edited
    assert not (out / "svg" / "live" / "a.replaced-1.svg").exists()


def test_a_kept_file_and_a_picture_with_that_name_in_the_same_run(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    runner = CliRunner()
    go = lambda: runner.invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert go().exit_code == 0
    _as_if_made_by_v090(out)
    edited = _edit(out / "svg" / "a.svg")
    make_color(src / "a.replaced-1.png")                                # wants svg/a.replaced-1.svg itself
    r = go()
    assert r.exit_code == 0 and "error" not in r.output, r.output
    assert (out / "svg" / "a.replaced-2.svg").read_text() == edited
    assert _traces_match_their_pictures(src, out)["a.replaced-1.png"] == "svg/a.replaced-1.svg"


def test_run_avoids_folders_left_in_the_output(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    (out / "svg" / "a.svg").mkdir(parents=True)                         # from an earlier run with a folder a.svg/
    r = CliRunner().invoke(main, ["run", str(src), str(out), "--workers", "1"])
    assert r.exit_code == 0 and "error" not in r.output, r.output
    assert _traces_match_their_pictures(src, out) == {"a.png": "svg/a.png.svg"}


def test_cli_trace(lineart, tmp_path):
    out = tmp_path / "o.svg"
    r = CliRunner().invoke(main, ["trace", str(lineart), str(out)])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["scores"]["ssim_target"] >= 0.90
    _valid_svg(out.read_text())


def test_cli_works_without_an_svg_rasteriser(lineart, tmp_path, monkeypatch):
    """The self-check is optional on the command line too: no rsvg/cairo means no scores, not a crash."""
    import boottool.score as sc
    monkeypatch.setattr(sc.shutil, "which", lambda name: None)
    monkeypatch.setitem(__import__("sys").modules, "cairosvg", None)
    runner = CliRunner()

    out = tmp_path / "o.svg"
    r = runner.invoke(main, ["trace", str(lineart), str(out)])
    assert r.exit_code == 0, r.output
    assert "self-check was skipped" in r.output
    assert json.loads(r.output[r.output.index("{"):])["scores"]["ssim_target"] is None
    _valid_svg(out.read_text())

    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_lineart(src / "a.png")
    r = runner.invoke(main, ["run", str(src), str(dst), "--workers", "1"])
    assert r.exit_code == 0 and "self-check was skipped" in r.output, r.output
    rec = _finished(dst / "manifest.jsonl")[0]
    assert rec["status"] == "ok" and rec["scores"]["ssim_target"] is None
    _valid_svg((dst / "svg" / "a.svg").read_text())
    assert "1 skipped" in runner.invoke(main, ["run", str(src), str(dst), "--workers", "1"]).output   # resume still works

    assert runner.invoke(main, ["report", str(dst)]).exit_code == 0
    r = runner.invoke(main, ["dataset", str(dst)])
    assert r.exit_code == 0, r.output
    assert json.loads((dst / "dataset.json").read_text())["excluded"] == {"unscored": 1}   # quality unknown, so left out


def test_upscale_and_level(lineart):
    from boottool.preprocess import prepare
    base = prepare(lineart, Params(min_side=1))
    up = prepare(lineart, Params(min_side=1, upscale=3.0))
    assert up.size == (base.size[0] * 3, base.size[1] * 3)
    # a very low manual threshold keeps less ink than a high one
    lo = prepare(lineart, Params(level=40)).mask.sum()
    hi = prepare(lineart, Params(level=200)).mask.sum()
    assert lo < hi


def test_sepia_scan_is_lineart(tmp_path):
    import numpy as np
    from PIL import Image
    from boottool.preprocess import detect_mode
    grey = np.asarray(Image.open(make_lineart(tmp_path / "g.png")).convert("L"), dtype=np.float32) / 255
    sepia = np.stack([grey * 200, grey * 175, grey * 120], axis=2).astype(np.uint8)
    assert detect_mode(sepia) == "lineart"


def test_downscale_keeps_declared_size(lineart):
    res = run_pipeline(lineart, Params(upscale=0.25, min_side=768))
    root = _valid_svg(res.svg)
    assert res.size[0] < 200                              # worked small, not forced up to 768
    assert int(root.get("width")) == 600 and int(root.get("height")) == 400   # opens at source size
    assert root.get("viewBox") == f"0 0 {res.size[0]} {res.size[1]}"
