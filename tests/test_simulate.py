"""Print library, Design size and the Simulate tab."""
import base64
import io
import json
import urllib.request

import pytest
from PIL import Image

from boottool import page as pg
from boottool import prints
from conftest import call_json as _post


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
    code, up = _post(server + "/api/upload", lineart.read_bytes())
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


def test_simulate_in_the_browser(server, lineart, tmp_path, page):
    prints.save("bolt", _png(300, 150), 280, 140)
    page.set_viewport_size({"width": 1400, "height": 850})
    page.goto(server)
    assert [a.text_content() for a in page.query_selector_all("#nav a")] == ["Convert", "Compose", "Print", "Simulate", "Help"]
    page.click('#nav a[data-t="simulate"]')
    page.wait_for_selector("#ssvg path")
    # the shirt is to scale: L is 56 cm wide, so its body (not sleeves) spans 56 viewBox units
    assert page.evaluate("Simulate._api.geo().chest") == 56
    page.evaluate("Simulate._api.addDesign('print','bolt.png',28)")
    page.wait_for_selector("#ssvg image")
    assert page.evaluate("Simulate._S.designs.front[0].w") == 28
    assert page.evaluate("Simulate._S.designs.front[0].h") == 14
    # size XXL is bigger than XS
    page.click("#side .cells span:text-is('XS')")
    xs = page.evaluate("Simulate._api.geo().chest")
    page.click("#side .cells span:text-is('XXL')")
    assert page.evaluate("Simulate._api.geo().chest") > xs
    # typing a width keeps proportions
    i = page.locator("#side .row", has_text="Width").locator("input")
    i.fill("20"); i.press("Enter")
    assert page.evaluate("Simulate._S.designs.front[0].h") == 10
    # front and back keep their own designs
    page.click("#side .cells span:text-is('Back')")
    assert page.evaluate("Simulate._S.designs.back.length") == 0
    page.click("#side .cells span:text-is('Front')")
    # drag snaps to the centre line
    box = page.locator("#ssvg image").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down(); page.mouse.move(box["x"] + box["width"] / 2 + 3, box["y"] + box["height"] / 2 + 20, steps=4); page.mouse.up()
    assert page.evaluate("Simulate._S.designs.front[0].x") == 0
    # rotate by typing an angle, crop is on by default and can be turned off
    r = page.locator("#side .row", has_text="Rotate").locator("input")
    r.fill("30"); r.press("Enter")
    assert page.evaluate("Simulate._S.designs.front[0].rot") == 30
    assert page.is_checked("#cropCk") and page.locator("#ssvg g[clip-path]").count() == 1
    page.uncheck("#cropCk")
    assert page.locator("#ssvg g[clip-path]").count() == 0
    page.check("#cropCk")
    # From collar follows the highest point of the rotated design
    top = page.evaluate("(() => { const d = Simulate._S.designs.front[0]; const a = d.rot * Math.PI / 180; return [d.y + d.h / 2 - (Math.abs(d.w * Math.sin(a)) + Math.abs(d.h * Math.cos(a))) / 2, d.y]; })()")
    assert top[0] < top[1] - 1
    fc = page.locator("#side .row", has_text="From collar").locator("input")
    assert abs(float(fc.input_value()) - top[0]) < 0.06
    fc.fill("12"); fc.press("Enter")
    assert abs(page.evaluate("(() => { const d = Simulate._S.designs.front[0]; const a = d.rot * Math.PI / 180; return d.y + d.h / 2 - (Math.abs(d.w * Math.sin(a)) + Math.abs(d.h * Math.cos(a))) / 2; })()") - 12) < 0.01
    r.fill("0"); r.press("Enter")
    # drag a tile from the Print library onto the shirt
    page.hover("#slib .ltab"); page.click("[data-lt=print]"); page.wait_for_selector("#slibStrip .lt")
    w = page.locator("#swrap").bounding_box()
    page.locator("#slibStrip .lt").first.drag_to(page.locator("#swrap"), target_position={"x": w["width"] / 2 + 60, "y": 250})
    page.wait_for_function("Simulate._S.designs.front.length === 2")
    assert page.evaluate("Simulate._S.designs.front[1].x") != 0
    page.evaluate("Simulate._S.designs.front.pop(); Simulate._S.sel = Simulate._S.designs.front[0].id; Simulate._api.draw()")
    page.mouse.move(10, 10)
    # Real size scrolls the shirt, not the library bar
    page.mouse.move(700, 300); page.click("#sreal"); page.mouse.move(700, 300); page.wait_for_timeout(400)
    top0 = page.evaluate("document.querySelector('#slib').getBoundingClientRect().top")
    page.evaluate("document.querySelector('#sscroll').scrollTop = 400; document.querySelector('#sscroll').scrollLeft = 300")
    assert page.evaluate("document.querySelector('#sscroll').scrollTop") > 0
    assert abs(page.evaluate("document.querySelector('#slib').getBoundingClientRect().top") - top0) < 1
    page.click("#sfit")
    # number boxes line up on the right
    xs = {round(v) for v in page.evaluate("[...document.querySelectorAll('#side input[type=text]')].filter(i => i.style.width === '84px').map(i => i.getBoundingClientRect().right)")}
    assert len(xs) == 1, xs
    # JPG
    assert not page.is_checked("#jpgLines")
    with page.expect_download() as d:
        page.click("#saveJpg")
    out = tmp_path / "m.jpg"; d.value.save_as(out)
    im = Image.open(out)
    assert im.format == "JPEG" and im.size[0] > 800
    # Print tab: Design size and Save to Print library
    page.click('#nav a[data-t="print"]')
    page.set_input_files("#file", str(lineart))
    page.wait_for_selector("#stage img#svgimg", state="attached")
    page.wait_for_function("lastRes && lastRes.plates.length")
    page.click("#side .cells span:text-is('Set size')")
    w = page.locator("#side .row", has_text="Width").locator("input")
    w.fill("350"); w.press("Enter")
    page.wait_for_function("lastRes && Math.abs(lastRes.art[2]-350) < 0.5")
    assert page.evaluate("PG.size") == "Custom"           # an A4 page cannot hold 350 mm, so it grew
    page.click("text=Save to Print library")
    page.wait_for_function("document.querySelector('#toast') ? true : true")
    page.wait_for_timeout(2500)
    items = json.loads(urllib.request.urlopen(server + "/api/prints").read())["items"]
    assert any(abs(i["width_mm"] - 350) < 0.5 for i in items)


def test_opening_an_image_after_viewing_a_batch_item(server, lineart, page):
    """Viewing an item from the batch list used to wipe the file-name box, so the next image failed with 'Cannot set properties of null'."""
    page.set_viewport_size({"width": 1400, "height": 850})
    page.goto(server)
    page.set_input_files("#file", str(lineart))
    page.wait_for_function("imageId && lastShown")
    page.evaluate("BATCH.items = [{id: imageId, name: 'a.png', w: 600, h: 400, url: '', own: null}]; viewItem(0)")
    assert page.locator("#nmEdit").count() == 1
    page.set_input_files("#file", str(lineart))
    page.wait_for_function("document.querySelector('#nmEdit').textContent.length > 0")
