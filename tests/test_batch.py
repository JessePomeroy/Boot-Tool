import io
import json
import time
import urllib.error
import urllib.request
import zipfile

import pytest

from conftest import call, call as _post


def _get(url, headers=None):
    return call(url, headers=headers)


def _upload(server, path):
    return json.loads(_post(server + "/api/upload", path.read_bytes(), "application/octet-stream")[1])["id"]


def _wait(server, job, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        code, body, _ = _get(f"{server}/api/batch/status?job={job}")
        st = json.loads(body)
        if st["finished"]:
            return st
        time.sleep(0.2)
    raise AssertionError("batch did not finish")


def _start(server, items, **kw):
    code, body, _ = _post(server + "/api/batch/start", {"items": items, **kw})
    return code, json.loads(body)


def test_batch_saves_into_one_folder_with_own_settings(server, lineart, color, tmp_path):
    a, b = _upload(server, lineart), _upload(server, color)
    out = tmp_path / "svg out"
    code, res = _start(server, [
        {"id": a, "name": "my scan.png", "params": {}},
        {"id": a, "name": "my scan.jpg", "params": {"speckle": 0, "epsilon": 0.25, "smooth_sigma": 0}},   # same stem
        {"id": b, "name": "../evil/logo.png", "params": {"mode": "color", "colors": 4}},
    ], out_dir=str(out))
    assert code == 200 and res["out_dir"] == str(out)
    st = _wait(server, res["job"])
    assert st["counts"]["done"] == 3 and st["counts"]["error"] == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["logo.svg", "my scan-2.svg", "my scan.svg"]            # one flat folder, safe names, no clash
    assert not (tmp_path / "evil").exists()
    sizes = {i["fname"]: i["node_count"] for i in st["items"]}
    assert sizes["my scan-2.svg"] != sizes["my scan.svg"]                   # each image used its own settings
    assert all((out / n).read_text().lstrip().startswith("<svg") or "<svg" in (out / n).read_text() for n in names)
    code, svg, h = _get(f"{server}/api/batch/svg?job={res['job']}&i=0")
    assert code == 200 and h["Content-Type"] == "image/svg+xml" and b"<svg" in svg
    code, z, h = _get(f"{server}/api/batch/zip?job={res['job']}")
    assert code == 200 and sorted(zipfile.ZipFile(io.BytesIO(z)).namelist()) == names


def test_batch_never_overwrites_existing_files(server, lineart, tmp_path):
    a = _upload(server, lineart)
    out = tmp_path / "o"
    out.mkdir()
    (out / "x.svg").write_text("KEEP ME")
    _, res = _start(server, [{"id": a, "name": "x.png", "params": {}}], out_dir=str(out))
    st = _wait(server, res["job"])
    assert (out / "x.svg").read_text() == "KEEP ME" and (out / "x-2.svg").exists()
    assert st["items"][0]["saved_as"] == "x-2.svg"


def test_removing_a_row_after_a_batch_keeps_the_other_results(server, lineart, color, page):
    """Taking one picture off the list after a batch used to throw every result away: the other rows
    went back to "ready" and the .zip could not be downloaded any more. So did adding a picture."""
    rows = "[...document.querySelectorAll('.brow')].map(r => [r.querySelector('div[style*=ellipsis]').textContent, r.querySelector('.bchip:last-of-type').textContent, r.querySelectorAll('.bth img').length])"
    zip_off = "[...document.querySelectorAll('#side button')].find(b => /zip/.test(b.textContent)).disabled"
    page.goto(server + "/")
    page.wait_for_function("document.querySelector('#ver').textContent.length > 1")
    page.click("#toolbar [data-m=batch]")
    add = "#side input[type=file][multiple]:not([webkitdirectory])"
    page.set_input_files(add, [str(color), str(lineart)])
    page.wait_for_function("BATCH.items.length === 2 && !BATCH.adding", timeout=60000)
    page.click("#side button:text-is('Convert')")
    page.wait_for_function("BATCH.status && BATCH.status.finished", timeout=180000)
    assert page.evaluate(rows) == [["color.png", "DONE", 2], ["line.png", "DONE", 2]]   # each with its picture and its SVG

    def names_in_the_zip():
        with page.expect_download(timeout=60000) as d:
            page.click("#side button:text-is('Download .zip of the SVGs')")
        return zipfile.ZipFile(d.value.path()).namelist()
    assert sorted(names_in_the_zip()) == ["color.svg", "line.svg"]

    page.click(".brow:first-child [data-rm]")                          # the first picture is taken off the list
    page.wait_for_function("BATCH.items.length === 1")
    assert page.evaluate(rows) == [["line.png", "DONE", 2]]            # used to go back to READY, without its SVG
    thumb = "document.querySelector('.brow .bth:nth-of-type(3) img')"
    page.wait_for_function(thumb + ".naturalWidth > 0")
    shown = page.evaluate(f"fetch({thumb}.src).then(r => r.text())")    # and it is the line art's SVG that shows, not the removed picture's
    assert shown == zipfile.ZipFile(io.BytesIO(_get(f"{server}/api/batch/zip?job={page.evaluate('BATCH.job')}")[1])).read("line.svg").decode()
    assert "1 of 1 done" in page.evaluate("document.querySelector('#batchHead').textContent")
    assert "Done" in page.evaluate("document.querySelector('#stats').textContent") and page.evaluate("BATCH.status.counts.done") == 1
    assert not page.evaluate(zip_off) and names_in_the_zip() == ["line.svg"]   # the .zip has what is still on the list

    page.set_input_files(add, [str(color)])                            # a picture added now is new; the result stays
    page.wait_for_function("BATCH.items.length === 2 && !BATCH.adding", timeout=60000)
    assert page.evaluate(rows) == [["line.png", "DONE", 2], ["color.png", "READY", 1]]

    page.click(".brow:first-child [data-rm]")                          # the last result goes: nothing left to download
    page.wait_for_function("BATCH.items.length === 1")
    assert page.evaluate(rows) == [["color.png", "READY", 1]] and page.evaluate(zip_off)
    before = page.evaluate("BATCH.job")
    page.click("#side button:text-is('Convert')")                      # and a new batch starts afresh
    page.wait_for_function("j => BATCH.job !== j && BATCH.status && BATCH.status.finished", arg=before, timeout=180000)
    assert page.evaluate(rows) == [["color.png", "DONE", 2]] and names_in_the_zip() == ["color.svg"]
    job = page.evaluate("BATCH.job")
    for odd in (None, "0", 7, -1, 0.0, False):                               # nonsense about which picture changes nothing
        assert _post(server + "/api/batch/forget", {"job": job, "i": odd})[0] == 200
    assert _post(server + "/api/batch/forget", {"job": "no such batch", "i": 0})[0] == 200 and names_in_the_zip() == ["color.svg"]


def test_batch_keeps_going_after_a_failure_and_can_stop(server, lineart, tmp_path):
    a = _upload(server, lineart)
    _, res = _start(server, [{"id": "nope", "name": "bad.png", "params": {}}, {"id": a, "name": "ok.png", "params": {}}],
                    out_dir=str(tmp_path / "o"), keep_going=True)
    st = _wait(server, res["job"])
    assert [i["state"] for i in st["items"]] == ["error", "done"] and "unknown image" in st["items"][0]["error"]
    _, res = _start(server, [{"id": "nope", "name": "bad.png", "params": {}}, {"id": a, "name": "ok.png", "params": {}}],
                    keep_going=False)
    st = _wait(server, res["job"])
    assert [i["state"] for i in st["items"]] == ["error", "stopped"]
    assert _post(server + "/api/batch/stop", {"job": res["job"]})[0] == 200


def test_a_status_check_that_fails_does_not_leave_the_batch_view_stuck(server, lineart, color, page):
    """The page asks how a batch is getting on every half second or so. If one of those questions got
    no answer, it stopped asking: the list stayed on "queued" and Convert and Clear list stayed
    greyed out until the page was reloaded, though the batch had long finished."""
    buttons = "Object.fromEntries([...document.querySelectorAll('#side button')].map(b => [b.textContent, !b.disabled]))"
    page.goto(server + "/")
    page.wait_for_function("document.querySelector('#ver').textContent.length > 1")
    page.click("#toolbar [data-m=batch]")
    page.set_input_files("#side input[type=file][multiple]:not([webkitdirectory])", [str(lineart), str(color)])
    page.wait_for_function("BATCH.items.length === 2 && !BATCH.adding", timeout=60000)

    asked = []
    page.route("**/api/batch/status*", lambda route: (asked.append(1), route.abort() if len(asked) in (1, 2) else route.continue_()))
    page.click("#side button:text-is('Convert')")                      # the first two questions get no answer
    page.wait_for_function("BATCH.status && BATCH.status.finished", timeout=60000)
    assert len(asked) >= 3 and page.evaluate("BATCH.status.counts.done") == 2
    assert page.evaluate(buttons) == {"Convert": True, "Download .zip of the SVGs": True, "Stop": False, "Clear list": True}
    page.unroute("**/api/batch/status*")

    page.route("**/api/batch/status*", lambda route: route.abort())      # no answer ever again (Boot Tool was closed), and the
    page.evaluate("BATCH.status.finished = false; pollBatch(25)")       # page has asked for a quarter of a minute: it gives up
    page.wait_for_function("/stopped answering/.test(document.querySelector('#toast').textContent)", timeout=60000)
    assert page.evaluate("[BATCH.job, BATCH.status]") == [None, None]
    assert page.evaluate(buttons) == {"Convert": True, "Download .zip of the SVGs": False, "Stop": False, "Clear list": True}
    page.unroute("**/api/batch/status*")

    page.route("**/api/batch/status*", lambda route: route.fulfill(status=404, content_type="application/json", body='{"error": "unknown batch; run it again"}'))
    page.click("#side button:text-is('Convert')")                      # Boot Tool has forgotten the batch (it was restarted, say)
    page.wait_for_function("/unknown batch/.test(document.querySelector('#toast').textContent)", timeout=60000)
    assert page.evaluate("[BATCH.job, BATCH.status]") == [None, None]
    assert page.evaluate(buttons) == {"Convert": True, "Download .zip of the SVGs": False, "Stop": False, "Clear list": True}


def test_batch_rejects_bad_requests(server, lineart, tmp_path):
    a = _upload(server, lineart)
    assert _start(server, [])[0] == 400
    code, res = _start(server, [{"id": a, "name": "a.png"}], out_dir="relative/folder")
    assert code == 400 and "full folder path" in res["error"]
    assert _get(f"{server}/api/batch/status?job=nope")[0] == 404
    blocker = tmp_path / "file"
    blocker.write_text("x")
    code, res = _start(server, [{"id": a, "name": "a.png"}], out_dir=str(blocker / "sub"))
    assert code == 400 and "folder" in res["error"].lower()


def test_a_page_of_another_program_on_this_computer_may_not_use_the_api(server):
    """A page loaded from another port of this computer (the web page of some other program) is not
    Boot Tool's own page. It used to be let in, and could save into the library and delete from it."""
    from boottool import library
    port = server.rsplit(":", 1)[1]

    def post(path, data, origin, host=None):
        headers = {"Content-Type": "text/plain", **({} if origin is None else {"Origin": origin}), **({"Host": host} if host else {})}
        req = urllib.request.Request(server + path, data=json.dumps(data).encode(), method="POST", headers=headers)   # (as a page can send without asking first)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def saved():
        return [i["name"] for i in library.listing()]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M0 0h5v5z"/></svg>'
    own = f"http://127.0.0.1:{port}"
    assert post("/api/library/save", {"name": "logo", "svg": svg}, own) == 200 and saved() == ["logo.svg"]
    for other in (f"http://localhost:{int(port) + 1}", f"http://127.0.0.1:{int(port) + 1}", "http://127.0.0.1", f"http://localhost:{port}",
                  f"https://127.0.0.1:{port}", f"http://127.0.0.1:{port}x", "null", ""):
        assert post("/api/library/delete", {"name": "logo.svg"}, other) == 403, other
        assert post("/api/library/save", {"name": "planted", "svg": svg}, other) == 403, other
        assert saved() == ["logo.svg"], other
    assert post("/api/library/delete", {"name": "logo.svg"}, own, host="127.0.0.1:abc") == 403 and saved() == ["logo.svg"]   # an address that cannot be read
    assert post("/api/library/save", {"name": "second", "svg": svg}, f"http://localhost:{port}", host=f"localhost:{port}") == 200   # the page opened as "localhost"
    assert post("/api/library/delete", {"name": "second.svg"}, None) == 200           # no browser at all (the command line, the tests)
    assert post("/api/library/delete", {"name": "logo.svg"}, own) == 200 and saved() == []


def test_only_this_computers_own_page_may_use_the_api(server, lineart):
    port = server.rsplit(":", 1)[1]
    assert _get(server + "/api/defaults")[0] == 200
    assert _get(server + "/api/defaults", {"Host": "evil.example"})[0] == 403
    assert _get(server + "/api/defaults", {"Host": f"localhost:{port}"})[0] == 200
    req = urllib.request.Request(server + "/api/batch/start", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json", "Origin": "https://evil.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403


@pytest.mark.parametrize("terminal", ["there", "gone"])
def test_closing_boot_tool_during_a_batch_lets_the_picture_in_hand_finish(lineart, tmp_path, monkeypatch, terminal):
    """Closing Boot Tool deletes its session folder. A batch that was still running went on writing
    into that folder while it was deleted (which could leave the folder behind), and could be cut
    off halfway through an SVG in its output folder. The batch is stopped first now and the picture
    it has in hand finished: that one is saved whole, the rest are not started, and the folder goes.
    A batch in another Boot Tool is left alone. And it is the same when Boot Tool is closed by
    closing its terminal window, so that there is nowhere left to print to."""
    import threading

    from boottool import batch as bt
    from boottool.ui import make_server

    real_pipeline, real_run, jobs = bt.run_pipeline, bt.Job.run, []

    def slow(path, params, score=False):                                    # every picture takes two seconds
        time.sleep(2.0)
        return real_pipeline(path, params, score=score)
    monkeypatch.setattr(bt, "run_pipeline", slow)
    monkeypatch.setattr(bt.Job, "run", lambda self, *a: (jobs.append(self), real_run(self, *a))[1])
    if terminal == "gone":
        def no_terminal(*a, **kw):
            raise OSError(5, "Input/output error")
        monkeypatch.setattr(bt, "print", no_terminal, raising=False)

    servers = [make_server("127.0.0.1", 0) for _ in range(2)]
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    one, other = (f"http://127.0.0.1:{srv.server_address[1]}" for srv in servers)
    try:
        code, there = _start(other, [{"id": _upload(other, lineart), "name": f"o{i}.png", "params": {}} for i in range(2)])
        assert code == 200
        out = tmp_path / "out"
        image = _upload(one, lineart)
        code, here = _start(one, [{"id": image, "name": f"p{i}.png", "params": {}} for i in range(4)], out_dir=str(out))
        assert code == 200
        for _ in range(600):                                                # until both batches are at work on their first picture
            if len(jobs) == 2 and all(j.items[0]["state"] == "working" for j in jobs):   # (it takes two seconds; closing, below,
                break                                                       # takes half a second at most to begin)
            time.sleep(0.05)
        assert len(jobs) == 2
        job = next(j for j in jobs if j.id == here["job"])
        folder = job.dir.parent
        assert folder.is_dir()

        servers[0].shutdown()
        servers[0].server_close()                                           # Boot Tool is closed

        assert not job.thread.is_alive() and job.finished
        assert [it["state"] for it in job.items] == ["done", "stopped", "stopped", "stopped"]
        assert [p.name for p in out.iterdir()] == ["p0.svg"]
        assert (out / "p0.svg").read_text(encoding="utf-8").rstrip().endswith("</svg>")
        assert not folder.exists()
        st = _wait(other, there["job"])                                     # the other Boot Tool's batch runs on to its end
        assert st["counts"]["done"] == 2
    finally:
        for srv in servers:                                                 # (closing the first again does no harm)
            srv.shutdown()
            srv.server_close()


def test_a_picture_that_takes_too_long_is_not_waited_for_for_ever(lineart, tmp_path, monkeypatch, capsys):
    """The wait for the picture in hand has an end. When the time is up, closing goes ahead, and
    says that this picture was not saved."""
    import threading

    from boottool import batch as bt
    from boottool.ui import make_server

    real_pipeline, real_run, jobs, at_work = bt.run_pipeline, bt.Job.run, [], threading.Event()

    def slow(path, params, score=False):
        at_work.set()
        time.sleep(1.5)
        return real_pipeline(path, params, score=score)
    monkeypatch.setattr(bt, "run_pipeline", slow)
    monkeypatch.setattr(bt.Job, "run", lambda self, *a: (jobs.append(self), real_run(self, *a))[1])
    srv = make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        code, res = _start(url, [{"id": _upload(url, lineart), "name": f"p{i}.png", "params": {}} for i in range(2)])
        assert code == 200 and at_work.wait(30)
        job = jobs[0]
        assert bt.stop_all(job.dir.parent, wait=0.2) is False               # not done in a fifth of a second
        said = capsys.readouterr().out
        assert "Finishing the picture" in said and "not finished in time: that picture was not saved" in said
        assert job.stop.is_set()
        job.thread.join(30)                                                 # it ends by itself after its picture, starting no other
        assert [it["state"] for it in job.items] == ["done", "stopped"]
        assert bt.stop_all(job.dir.parent) is True                          # and then there is nothing to wait for
        assert capsys.readouterr().out == ""
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGHUP"), reason="Mac and Linux only")
@pytest.mark.parametrize("terminal", ["there", "gone"])
def test_stopping_the_real_tool_during_a_batch(lineart, tmp_path, terminal):
    """The same with Boot Tool run for real and asked to stop (as closing its window does) while a
    batch of thirty is under way: it says what it is waiting for, ends cleanly, leaves only whole
    SVGs in the output folder and no session folder. With nothing left to print to ("gone": what it
    prints to is closed first), it cannot say so, and still tidies up. (A check of the whole way
    through; the tests above are the ones that pin the behaviour down.)"""
    import os
    import signal
    import subprocess
    import sys
    from pathlib import Path
    tmp, out = tmp_path / "tmp", tmp_path / "out"
    tmp.mkdir()
    env = {**os.environ, "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp), "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    proc = subprocess.Popen([sys.executable, "-u", "-m", "boottool", "ui", "--no-browser", "--port", "0"],
                            stdout=subprocess.PIPE, text=True, env=env)
    try:
        url = proc.stdout.readline().split("at ", 1)[1].strip().rstrip("/")
        image = _upload(url, lineart)
        (folder,) = Path(tmp).glob("boottool-ui-*")
        code, res = _start(url, [{"id": image, "name": f"p{i}.png", "params": {}} for i in range(30)], out_dir=str(out))
        assert code == 200
        for _ in range(600):                                                # until the first picture is done
            st = json.loads(_get(f"{url}/api/batch/status?job={res['job']}")[1])
            if st["counts"]["done"] >= 1:
                break
            time.sleep(0.05)
        assert 1 <= st["counts"]["done"] < 30 and not st["finished"]
        if terminal == "gone":
            proc.stdout.close()
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=90)                                           # (its exit code is not 0 then: Python's own last flush fails)
        else:
            proc.send_signal(signal.SIGTERM)
            said = proc.communicate(timeout=90)[0]
            assert proc.returncode == 0 and "Finishing the picture" in said
    finally:
        if proc.poll() is None:
            proc.kill()
    saved = sorted(out.iterdir())
    assert 1 <= len(saved) < 30 and all(p.read_text(encoding="utf-8").rstrip().endswith("</svg>") for p in saved)
    assert not folder.exists()
