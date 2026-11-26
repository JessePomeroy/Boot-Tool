import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image, ImageDraw


def _noisy(arr, rng, sigma=6.0):
    h, w = arr.shape[:2]
    grad = np.linspace(-25, 10, w, dtype=np.float32)[None, :]
    if arr.ndim == 3:
        grad = grad[..., None]
    out = arr.astype(np.float32) + grad + rng.normal(0, sigma, arr.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_lineart(path, invert=False):
    rng = np.random.default_rng(1)
    img = Image.new("L", (600, 400), 235)
    d = ImageDraw.Draw(img)
    d.ellipse((80, 60, 300, 280), outline=20, width=6)
    d.rectangle((340, 100, 540, 300), outline=20, width=5)
    d.line((60, 350, 540, 350), fill=20, width=4)
    arr = _noisy(np.asarray(img), rng)
    if invert:
        arr = 255 - arr
    Image.fromarray(arr).save(path)
    return path


def make_color(path):
    rng = np.random.default_rng(2)
    img = Image.new("RGB", (600, 400), (250, 240, 220))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 60, 260, 260), fill=(40, 80, 200), outline=(0, 0, 0), width=4)
    d.rectangle((320, 80, 540, 240), fill=(200, 40, 40), outline=(0, 0, 0), width=4)
    d.polygon([(150, 380), (300, 280), (450, 380)], fill=(40, 160, 70), outline=(0, 0, 0))
    Image.fromarray(_noisy(np.asarray(img), rng, sigma=3.0)).save(path)
    return path


@pytest.fixture
def lineart(tmp_path):
    return make_lineart(tmp_path / "line.png")


@pytest.fixture
def lineart_inverted(tmp_path):
    return make_lineart(tmp_path / "inv.png", invert=True)


@pytest.fixture
def color(tmp_path):
    return make_color(tmp_path / "color.png")


@pytest.fixture(autouse=True)
def _private_library(tmp_path_factory, monkeypatch):
    """Tests never touch the real ~/BootTool folders."""
    base = tmp_path_factory.mktemp("boot-home")
    monkeypatch.setenv("BOOTTOOL_LIBRARY", str(base / "Library"))
    monkeypatch.setenv("BOOTTOOL_FONTS", str(base / "Fonts"))
    monkeypatch.setenv("BOOTTOOL_PRINTS", str(base / "Prints"))


@pytest.fixture
def server():
    """Boot Tool running on a free port: its address."""
    from boottool.ui import make_server
    srv = make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture(scope="session")
def browser():
    """One browser for the whole run (the tests that need one are skipped without Playwright)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    """A fresh browser page. Afterwards it is closed, also when the test failed, and it must not
    have reported any error of the page's own scripts."""
    pg_ = browser.new_page(viewport={"width": 1400, "height": 900}, accept_downloads=True)
    errors = []
    pg_.on("pageerror", lambda e: errors.append(str(e)))
    yield pg_
    pg_.close()
    assert not errors, errors


def call(url, data=None, ctype="application/json", headers=None):
    """GET, or POST when there is data (bytes as they are, anything else as JSON): the answer as
    (status, body, headers), also when it is an error."""
    if data is not None and not isinstance(data, bytes):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, headers={**({"Content-Type": ctype} if data is not None else {}), **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def call_json(url, data=None):
    """The same, for answers in JSON: (status, what it says)."""
    status, body, _ = call(url, data)
    return status, json.loads(body)
