"""Print library, Design size and the Simulate tab."""
import base64
import io
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from boottool import page as pg
from boottool import prints
from boottool.ui import make_server


@pytest.fixture
def server():
    srv = make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _post(url, data, raw=False):
    body = data if raw else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _png(w=40, h=20):
    b = io.BytesIO()
    Image.new("RGBA", (w, h), (255, 0, 0, 255)).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def test_print_library_roundtrip():
    n = prints.save("My design!", _png(), 280, 140, "#112233")
    assert n == "My design_.png"
    assert prints.save("My design!", _png(), 100, 50) == "My design_-2.png"
    items = {i["name"]: i for i in prints.listing()}
    assert items[n]["width_mm"] == 280 and items[n]["garment"] == "#112233"
    assert prints.read(n)[:4] == b"\x89PNG"
    prints.delete(n)
    assert n not in [i["name"] for i in prints.listing()]
    with pytest.raises(ValueError):
        prints.save("x", "data:image/png;base64," + base64.b64encode(b"nope").decode(), 10, 10)
    with pytest.raises(ValueError):
        prints.read("../secret.png")


def test_set_size_prints_exact_width_centred():
    o = pg.coerce_page({"art_w": 100})
    x, y, w, h = pg.fit_artwork(o, 400, 200)
    pw, ph = o.dims()
    assert abs(w - 100) < 1e-6 and abs(h - 50) < 1e-6
    assert abs(x - (pw - 100) / 2) < 1e-6 and abs(y - (ph - 50) / 2) < 1e-6
    assert pg.fit_artwork(pg.coerce_page({}), 400, 200)[2] > 100      # 0 still means fit to page


def test_design_endpoint_and_library_http(server, lineart):
    code, up = _post(server + "/api/upload", lineart.read_bytes(), raw=True)
    assert code == 200
    req = {"id": up["id"], "tool": "print", "params": {"method": "one"}, "page": {"art_w": 120}}
    code, j = _post(server + "/api/design", req)
    assert code == 200 and j["svg"].startswith("<svg")
    assert abs(j["width_mm"] - 120) < 0.01 and j["width_mm"] / j["height_mm"] == pytest.approx(600 / 400, rel=0.02)
    assert "mm" in j["svg"][:200]
    code, k = _post(server + "/api/prints/save", {"name": "bolt", "png": _png(), "width_mm": j["width_mm"], "height_mm": j["height_mm"]})
    assert code == 200 and k["name"] == "bolt.png"
    assert json.loads(urllib.request.urlopen(server + "/api/prints").read())["items"][0]["name"] == "bolt.png"
    assert urllib.request.urlopen(server + "/api/prints/file?name=bolt.png").read()[:4] == b"\x89PNG"
    assert _post(server + "/api/prints/save", {"name": "bad", "png": "AAAA", "width_mm": 5, "height_mm": 5})[0] == 400
    assert _post(server + "/api/prints/delete", {"name": "bolt.png"})[0] == 200
    assert json.loads(urllib.request.urlopen(server + "/api/prints").read())["items"] == []
    assert urllib.request.urlopen(server + "/static/simulate.js").status == 200


def test_an_out_of_date_list_cant_use_or_delete_another_print():
    """A print is deleted elsewhere and a different one saved under its name."""
    prints.save("bolt", _png(300, 150), 280, 140)
    listed = prints.listing()[0]["id"]
    assert prints.read("bolt.png", listed)
    prints.delete("bolt.png", listed)
    prints.save("bolt", _png(100, 100), 100, 100)                         # another print, same name
    for act in (lambda: prints.read("bolt.png", listed), lambda: prints.delete("bolt.png", listed)):
        with pytest.raises(KeyError):
            act()
    assert [i["name"] for i in prints.listing()] == ["bolt.png"]         # still there
    prints.save("same", _png(300, 150), 280, 140)
    a = {i["name"]: i["id"] for i in prints.listing()}["same.png"]
    prints.delete("same.png")
    prints.save("same", _png(300, 150), 100, 50)                          # the same picture at another size
    with pytest.raises(KeyError):
        prints.read("same.png", a)
    prints.delete("same.png")
    ids = lambda: {i["name"]: i["id"] for i in prints.listing()}         # (by name: the list's order is by date)
    first = prints.save("twin", _png(300, 150), 280, 140)
    a = ids()["twin.png"]
    prints.delete(first)
    prints.save("twin", _png(300, 150), 280, 140)                         # the very same again, straight away
    assert ids()["twin.png"] != a                                         # is still another print
    with pytest.raises(KeyError):
        prints.delete("twin.png", a)
    listed = {i["name"]: i for i in prints.listing()}["twin.png"]
    assert (listed["width_mm"], listed["height_mm"]) == (280, 140)
    import os
    f = prints.folder() / "twin.png"
    st = f.stat()
    data = bytearray(f.read_bytes())
    data[-1] ^= 1                                                         # other bytes, same length...
    f.write_bytes(bytes(data))
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))                      # ...and the same date
    with pytest.raises(KeyError):
        prints.read("twin.png", listed["id"])
    assert prints.read("twin.png", ids()["twin.png"]) == bytes(data)      # a refreshed list works again
    import shutil
    listed = ids()["twin.png"]
    shutil.copy(f, prints.folder() / "copy.tmp")                          # an identical copy put in its place
    f.unlink()
    (prints.folder() / "copy.tmp").rename(f)
    if f.stat().st_ino:
        with pytest.raises(KeyError):
            prints.delete("twin.png", listed)
        assert f.exists()
    prints.delete("twin.png")
    prints.save("same", _png(300, 150), 100, 50)
    (prints.folder() / "odd.png").mkdir()                                 # a folder named like a print
    assert sorted(i["name"] for i in prints.listing()) == ["bolt.png", "same.png"]


def test_print_library_changes_take_turns_with_a_second_boot_tool():
    import subprocess
    import sys
    import time
    hold = ("import sys, time; from boottool import prints, filelock\n"
            "with filelock.Turns().held(prints.folder() / prints.LOCK):\n"
            "    print('held', flush=True); time.sleep(1.5)\n")
    env = {**__import__("os").environ, "BOOTTOOL_PRINTS": str(prints.folder())}
    other = subprocess.Popen([sys.executable, "-c", hold], env=env, stdout=subprocess.PIPE, text=True)
    assert other.stdout.readline().strip() == "held"
    t0 = time.monotonic()
    assert prints.save("bolt", _png(300, 150), 280, 140) == "bolt.png"    # waits for the other one, then goes
    assert time.monotonic() - t0 > 1.0
    other.wait(10)
    assert [i["name"] for i in prints.listing()] == ["bolt.png"]


def test_print_thumbnails_waiting_for_another_boot_tool_wait_side_by_side(monkeypatch):
    import subprocess
    import sys
    import time
    from boottool import filelock
    for n in range(4):
        prints.save(f"p{n}", _png(30 + n, 15), 280, 140)
    listed = prints.listing()
    hold = ("import sys, time; from boottool import prints, filelock\n"
            "with filelock.Turns().held(prints.folder() / prints.LOCK):\n"
            "    print('held', flush=True); time.sleep(30)\n")
    env = {**__import__("os").environ, "BOOTTOOL_PRINTS": str(prints.folder())}
    other = subprocess.Popen([sys.executable, "-c", hold], env=env, stdout=subprocess.PIPE, text=True)
    assert other.stdout.readline().strip() == "held"
    monkeypatch.setattr(filelock, "WAIT", 1.5)
    try:
        got = []
        readers = [threading.Thread(target=lambda i=i: got.append(prints.read(i["name"], i["id"]))) for i in listed]
        t0 = time.monotonic()
        for t in readers:
            t.start()
        for t in readers:
            t.join(30)
        assert len(got) == 4 and time.monotonic() - t0 < 3.5            # together (1.5 s), not 4 x 1.5 s
    finally:
        other.kill()
        other.wait(10)


@pytest.mark.parametrize("at", ["the size note", "the picture"])
def test_a_print_save_cut_short_never_leaves_a_picture_without_its_size(monkeypatch, at):
    from pathlib import Path
    real_replace, real_write = Path.replace, Path.write_text

    def replace(self, target):
        if at == "the picture" and str(target).endswith(".png"):
            raise SystemExit("killed")
        return real_replace(self, target)

    def write_text(self, *a, **k):
        if at == "the size note" and ".json" in self.name:
            raise SystemExit("killed")
        return real_write(self, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(Path, "replace", replace)
        m.setattr(Path, "write_text", write_text)
        with pytest.raises(SystemExit):
            prints.save("bolt", _png(300, 150), 280, 140)
    assert all(i["width_mm"] and i["height_mm"] for i in prints.listing())   # no picture without a size
    assert prints.save("bolt", _png(300, 150), 280, 140).startswith("bolt")  # and saving again works
    assert all((i["width_mm"], i["height_mm"]) == (280, 140) for i in prints.listing())


def test_simulate_tile_from_an_out_of_date_list(server):
    """The Convert-library tile still says badge.svg, but that item was renamed and another graphic
    took the name: clicking the old tile must not put the other graphic on the shirt."""
    from boottool import library
    sync_api = pytest.importorskip("playwright.sync_api")
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10"><rect width="5" height="5"/></svg>'
    library.save("badge", svg)
    prints.save("bolt", _png(300, 150), 280, 140)
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        pg_ = b.new_page(viewport={"width": 1400, "height": 850})
        errors = []
        pg_.on("pageerror", lambda e: errors.append(str(e)))
        pg_.goto(server)
        pg_.click('#nav a[data-t="simulate"]')
        pg_.wait_for_selector("#ssvg path")
        pg_.hover("#slib .ltab")
        pg_.click("[data-lt=convert]")
        pg_.wait_for_selector("#slibStrip .lt", state="attached")
        library.rename("badge.svg", "moved")
        library.save("badge", svg.replace("5", "8"))
        pg_.locator("#slibStrip .lt").first.dispatch_event("click")       # the old tile
        pg_.wait_for_function("document.querySelectorAll('#slibStrip .lt').length === 2", timeout=10000)   # list refreshed
        assert pg_.evaluate("Simulate._S.designs.front.length") == 0      # nothing was put on the shirt
        pg_.locator('#slibStrip .lt:has-text("moved")').dispatch_event("click")
        pg_.wait_for_function("Simulate._S.designs.front.length === 1", timeout=10000)   # the refreshed tile works

        pg_.hover("#slib .ltab")                                          # the same for the Print library
        pg_.click("[data-lt=print]")
        pg_.wait_for_selector("#slibStrip .lt", state="attached")
        shown = pg_.evaluate("Simulate._S.prints[0].id")
        prints.delete("bolt.png")
        prints.save("bolt", _png(100, 100), 100, 100)
        pg_.locator("#slibStrip .lt .ltx").first.dispatch_event("click")   # the old tile's ×
        pg_.wait_for_function("document.body.innerText.includes('Nothing was deleted')", timeout=10000)
        assert [i["name"] for i in prints.listing()] == ["bolt.png"]
        pg_.wait_for_function("Simulate._S.prints.length === 1 && Simulate._S.prints[0].id !== " + json.dumps(shown), timeout=10000)   # list refreshed
        pg_.locator("#slibStrip .lt").first.dispatch_event("click")       # the refreshed tile works
        pg_.wait_for_function("Simulate._S.designs.front.length === 2", timeout=10000)
        assert not errors, errors
        b.close()


def test_simulate_in_the_browser(server, lineart, tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    prints.save("bolt", _png(300, 150), 280, 140)
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        pg_ = b.new_page(viewport={"width": 1400, "height": 850}, accept_downloads=True)
        errors = []
        pg_.on("pageerror", lambda e: errors.append(str(e)))
        pg_.goto(server)
        assert [a.text_content() for a in pg_.query_selector_all("#nav a")] == ["Convert", "Compose", "Print", "Simulate", "Help"]
        pg_.click('#nav a[data-t="simulate"]')
        pg_.wait_for_selector("#ssvg path")
        # the shirt is to scale: L is 56 cm wide, so its body (not sleeves) spans 56 viewBox units
        assert pg_.evaluate("Simulate._api.geo().chest") == 56
        pg_.evaluate("Simulate._api.addDesign('print','bolt.png',28)")
        pg_.wait_for_selector("#ssvg image")
        assert pg_.evaluate("Simulate._S.designs.front[0].w") == 28
        assert pg_.evaluate("Simulate._S.designs.front[0].h") == 14
        # size XXL is bigger than XS
        pg_.click("#side .cells span:text-is('XS')")
        xs = pg_.evaluate("Simulate._api.geo().chest")
        pg_.click("#side .cells span:text-is('XXL')")
        assert pg_.evaluate("Simulate._api.geo().chest") > xs
        # typing a width keeps proportions
        i = pg_.locator("#side .row", has_text="Width").locator("input")
        i.fill("20"); i.press("Enter")
        assert pg_.evaluate("Simulate._S.designs.front[0].h") == 10
        # front and back keep their own designs
        pg_.click("#side .cells span:text-is('Back')")
        assert pg_.evaluate("Simulate._S.designs.back.length") == 0
        pg_.click("#side .cells span:text-is('Front')")
        # drag snaps to the centre line
        box = pg_.locator("#ssvg image").bounding_box()
        pg_.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        pg_.mouse.down(); pg_.mouse.move(box["x"] + box["width"] / 2 + 3, box["y"] + box["height"] / 2 + 20, steps=4); pg_.mouse.up()
        assert pg_.evaluate("Simulate._S.designs.front[0].x") == 0
        # rotate by typing an angle, crop is on by default and can be turned off
        r = pg_.locator("#side .row", has_text="Rotate").locator("input")
        r.fill("30"); r.press("Enter")
        assert pg_.evaluate("Simulate._S.designs.front[0].rot") == 30
        assert pg_.is_checked("#cropCk") and pg_.locator("#ssvg g[clip-path]").count() == 1
        pg_.uncheck("#cropCk")
        assert pg_.locator("#ssvg g[clip-path]").count() == 0
        pg_.check("#cropCk")
        # From collar follows the highest point of the rotated design
        top = pg_.evaluate("(() => { const d = Simulate._S.designs.front[0]; const a = d.rot * Math.PI / 180; return [d.y + d.h / 2 - (Math.abs(d.w * Math.sin(a)) + Math.abs(d.h * Math.cos(a))) / 2, d.y]; })()")
        assert top[0] < top[1] - 1
        fc = pg_.locator("#side .row", has_text="From collar").locator("input")
        assert abs(float(fc.input_value()) - top[0]) < 0.06
        fc.fill("12"); fc.press("Enter")
        assert abs(pg_.evaluate("(() => { const d = Simulate._S.designs.front[0]; const a = d.rot * Math.PI / 180; return d.y + d.h / 2 - (Math.abs(d.w * Math.sin(a)) + Math.abs(d.h * Math.cos(a))) / 2; })()") - 12) < 0.01
        r.fill("0"); r.press("Enter")
        # drag a tile from the Print library onto the shirt
        pg_.hover("#slib .ltab"); pg_.click("[data-lt=print]"); pg_.wait_for_selector("#slibStrip .lt")
        w = pg_.locator("#swrap").bounding_box()
        pg_.locator("#slibStrip .lt").first.drag_to(pg_.locator("#swrap"), target_position={"x": w["width"] / 2 + 60, "y": 250})
        pg_.wait_for_function("Simulate._S.designs.front.length === 2")
        assert pg_.evaluate("Simulate._S.designs.front[1].x") != 0
        pg_.evaluate("Simulate._S.designs.front.pop(); Simulate._S.sel = Simulate._S.designs.front[0].id; Simulate._api.draw()")
        pg_.mouse.move(10, 10)
        # Real size scrolls the shirt, not the library bar
        pg_.mouse.move(700, 300); pg_.click("#sreal"); pg_.mouse.move(700, 300); pg_.wait_for_timeout(400)
        top0 = pg_.evaluate("document.querySelector('#slib').getBoundingClientRect().top")
        pg_.evaluate("document.querySelector('#sscroll').scrollTop = 400; document.querySelector('#sscroll').scrollLeft = 300")
        assert pg_.evaluate("document.querySelector('#sscroll').scrollTop") > 0
        assert abs(pg_.evaluate("document.querySelector('#slib').getBoundingClientRect().top") - top0) < 1
        pg_.click("#sfit")
        # number boxes line up on the right
        xs = {round(v) for v in pg_.evaluate("[...document.querySelectorAll('#side input[type=text]')].filter(i => i.style.width === '84px').map(i => i.getBoundingClientRect().right)")}
        assert len(xs) == 1, xs
        # JPG
        assert not pg_.is_checked("#jpgLines")
        with pg_.expect_download() as d:
            pg_.click("#saveJpg")
        out = tmp_path / "m.jpg"; d.value.save_as(out)
        im = Image.open(out)
        assert im.format == "JPEG" and im.size[0] > 800
        # Print tab: Design size and Save to Print library
        pg_.click('#nav a[data-t="print"]')
        pg_.set_input_files("#file", str(lineart))
        pg_.wait_for_selector("#stage img#svgimg", state="attached")
        pg_.wait_for_function("lastRes && lastRes.plates.length")
        pg_.click("#side .cells span:text-is('Set size')")
        w = pg_.locator("#side .row", has_text="Width").locator("input")
        w.fill("350"); w.press("Enter")
        pg_.wait_for_function("lastRes && Math.abs(lastRes.art[2]-350) < 0.5")
        assert pg_.evaluate("PG.size") == "Custom"           # an A4 page cannot hold 350 mm, so it grew
        pg_.click("text=Save to Print library")
        pg_.wait_for_function("document.querySelector('#toast') ? true : true")
        pg_.wait_for_timeout(2500)
        items = json.loads(urllib.request.urlopen(server + "/api/prints").read())["items"]
        assert any(abs(i["width_mm"] - 350) < 0.5 for i in items)
        assert not errors, errors
        b.close()


def test_opening_an_image_after_viewing_a_batch_item(server, lineart):
    """Viewing an item from the batch list used to wipe the file-name box, so the next image failed with 'Cannot set properties of null'."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        pg_ = b.new_page(viewport={"width": 1400, "height": 850})
        errors = []
        pg_.on("pageerror", lambda e: errors.append(str(e)))
        pg_.goto(server)
        pg_.set_input_files("#file", str(lineart))
        pg_.wait_for_function("imageId && lastShown")
        pg_.evaluate("BATCH.items = [{id: imageId, name: 'a.png', w: 600, h: 400, url: '', own: null}]; viewItem(0)")
        assert pg_.locator("#nmEdit").count() == 1
        pg_.set_input_files("#file", str(lineart))
        pg_.wait_for_function("document.querySelector('#nmEdit').textContent.length > 0")
        assert not errors, errors
        b.close()
