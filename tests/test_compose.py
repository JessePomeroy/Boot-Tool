"""Library, fonts, text outlines and custom page size (the server side of Compose), plus a browser smoke test."""
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

from boottool import fonts, library
from boottool import page as pg
from boottool.ui import make_server

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10"><rect width="5" height="5"/></svg>'


@pytest.fixture
def server():
    srv = make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def test_library_keys_overwrite_and_names_are_unique():
    a = library.save("My Star!", SVG, key="s1:img1")
    assert a == "My Star_.svg" and library.save("My Star!", SVG.replace("5", "6"), key="s1:img1") == a
    assert b"6" in library.read(a)
    assert library.save("My Star!", SVG) == "My Star_-2.svg"                       # no key: never overwrites
    assert {i["name"] for i in library.listing()} == {"My Star_.svg", "My Star_-2.svg"}
    new = library.rename(a, "Badge")
    assert new == "Badge.svg" and library.save("x", SVG, key="s1:img1") == "Badge.svg"   # key follows the rename
    library.delete(new)
    assert "Badge.svg" not in {i["name"] for i in library.listing()}


def test_a_picture_can_only_rename_its_own_library_item():
    mine = library.save("line", SVG, key="s1:a")
    library.delete(mine)                                                # removed in Compose's library panel...
    other = library.save("line", SVG.replace("5", "7"), key="s1:b")      # ...and another picture takes the name
    assert other == mine == "line.svg"
    with pytest.raises(KeyError):
        library.rename("line.svg", "mine again", key="s1:a")             # picture a has no item any more
    assert library.rename("", "theirs", key="s1:b") == "theirs.svg"      # by key: whatever it is called now
    assert library.rename("theirs.svg", "drawer") == "drawer.svg"        # renamed in Compose's library panel...
    assert library.rename("theirs.svg", "header", key="s1:b") == "header.svg"   # ...the header still finds it
    assert library.rename("", "Header", key="s1:b") == "Header.svg"      # only the case: no -2
    assert [i["name"] for i in library.listing()] == ["Header.svg"]
    library.save("Taken", SVG)
    assert library.rename("Header.svg", "Taken") == "Taken-2.svg"        # a name another item has
    assert library.save("renamed", SVG, key="s1:b", rename=True) == "renamed.svg"   # a save can carry the new name
    assert library.save("renamed", SVG, key="s1:b", rename=True) == "renamed.svg"   # ...and doesn't drift to -2
    assert library.save("Taken", SVG, key="s1:b") == "renamed.svg"       # without a rename the item keeps its name
    for _ in range(3):                                                   # a name another item has: -2, and it stays -2
        assert library.save("Taken", SVG, key="s1:b", rename=True) == "Taken-2.svg"


def test_an_out_of_date_list_cant_delete_or_rename_another_item():
    """Tab 1 still lists a.svg. In tab 2 it is renamed and a different picture takes the name a.svg."""
    library.save("a", SVG, key="s1:a")
    listed = {i["name"]: i["id"] for i in library.listing()}["a.svg"]   # what tab 1's tile holds
    library.rename("a.svg", "original")
    library.save("a", SVG.replace("5", "7"), key="s2:b")                 # another picture, same name
    for act in (lambda: library.delete("a.svg", listed), lambda: library.rename("a.svg", "mine", item=listed)):
        with pytest.raises(KeyError):
            act()
    now = {i["name"]: i["id"] for i in library.listing()}
    assert sorted(now) == ["a.svg", "original.svg"] and now["original.svg"] == listed != now["a.svg"]
    library.delete("original.svg", listed)                               # the item itself, under its new name
    library.delete("a.svg", now["a.svg"])
    assert library.listing() == []


def test_a_file_swapped_outside_boot_tool_is_a_different_item():
    """art.svg is listed; in the file manager it is moved aside and another art.svg put in its place."""
    import os
    library.save("art", SVG, key="s1:a")
    listed = library.listing()[0]["id"]
    f = library.folder() / "art.svg"
    f.rename(library.folder() / "art (old).svg")
    f.write_text(SVG.replace("5", "9"), encoding="utf-8")                # someone else's drawing, same name
    os.utime(f, ns=(1, 1))                                               # (whatever its date)
    for act in (lambda: library.delete("art.svg", listed), lambda: library.rename("art.svg", "mine", item=listed),
                lambda: library.read("art.svg", listed)):
        with pytest.raises(KeyError):
            act()
    assert b"9" in library.read("art.svg")                               # still there, untouched
    assert library.save("art", SVG, key="s1:a") == "art-2.svg"           # the open picture's next save doesn't overwrite it
    assert b"9" in library.read("art.svg")
    now = {i["name"]: i["id"] for i in library.listing()}
    library.delete("art.svg", now["art.svg"])                            # an up-to-date list can
    assert sorted(i["name"] for i in library.listing()) == ["art (old).svg", "art-2.svg"]


def test_an_old_picture_cant_take_over_a_new_pictures_item():
    """Tab A saved art.svg. The file is removed in the file manager, and tab B's different picture is
    saved under the now-free name. Tab A must not rename or overwrite it."""
    assert library.save("art", SVG, key="tabA:1") == "art.svg"
    (library.folder() / "art.svg").unlink()
    assert library.save("art", SVG.replace("5", "7"), key="tabB:1") == "art.svg"
    with pytest.raises(KeyError):
        library.rename("", "mine", key="tabA:1")
    assert library.save("art", SVG.replace("5", "3"), key="tabA:1") == "art-2.svg"   # A gets a new item
    assert b"7" in library.read("art.svg")                               # B's is untouched
    assert library.save("art", SVG.replace("5", "4"), key="tabB:1") == "art.svg"     # and B still owns it
    assert library.rename("", "theirs", key="tabB:1") == "theirs.svg"


def test_the_list_survives_odd_things_in_the_folder_and_doesnt_hold_up_new_results():
    import threading
    library.save("a", SVG)
    (library.folder() / "archive.svg").mkdir()                           # a folder named like a graphic
    assert [i["name"] for i in library.listing()] == ["a.svg"]
    listed = library.listing()[0]["id"]
    library.rename("a.svg", "b")
    with pytest.raises(KeyError):
        library.delete("a.svg", listed)                                  # an old tile whose name is now free: say so
    library.delete("a.svg")                                              # (without an id, deleting nothing is fine)
    got = []
    with library._busy(change=False):                                    # while the library is busy (a long listing)...
        t = threading.Thread(target=lambda: got.append(library.tick()))
        t.start()
        t.join(5)
    assert got, "a new Convert result had to wait for the library"


def test_an_import_hands_over_only_the_bytes_it_checked(monkeypatch):
    """The file is swapped for another graphic just as it is read, and swapped back straight after."""
    library.save("a", SVG)
    listed = library.listing()[0]["id"]
    other = SVG.replace("5", "9").encode()
    import builtins

    class Swapped:                                                       # the right file, but other bytes come out
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

        def fileno(self):
            return self.f.fileno()

        def read(self):
            return other

    def opener(path, mode="r", *a, **k):
        f = builtins.open(path, mode, *a, **k)
        return Swapped(f) if str(path).endswith("a.svg") and "b" in mode else f
    with monkeypatch.context() as m:
        m.setattr(library, "open", opener, raising=False)
        with pytest.raises(KeyError):
            library.read("a.svg", listed)                                # not the item the tile showed
    assert library.listing()[0]["id"] == listed                          # it is still the same item
    assert library.read("a.svg", listed) == SVG.encode()


@pytest.mark.skipif(not hasattr(__import__("os"), "link"), reason="needs hard links")
def test_renaming_onto_a_hard_link_of_itself_is_still_another_name():
    import os
    library.save("a", SVG, key="s1:a")
    try:
        os.link(library.folder() / "a.svg", library.folder() / "b.svg")
    except OSError:
        pytest.skip("this disk has no hard links")
    assert library.rename("a.svg", "b") == "b-2.svg"                     # b.svg is someone else's entry
    assert library.save("x", SVG.replace("5", "7"), key="s1:a") == "b-2.svg"
    assert library.read("b.svg") == SVG.encode()                         # untouched (a save replaces the file, not its twin)
    library.save("art", SVG, key="s1:c")
    try:
        os.link(library.folder() / "art.svg", library.folder() / "Art.svg")   # a twin that differs only in case
    except OSError:
        return                                                           # Mac, Windows: that is the same name
    assert library.rename("", "Art", key="s1:c") == "Art-2.svg"
    assert library.save("x", SVG.replace("5", "3"), key="s1:c") == "Art-2.svg"
    assert library.read("Art.svg") == SVG.encode()


def _restart():
    """As if Boot Tool had been stopped and started again: it knows only what it wrote down."""
    library._loaded = None


def test_the_library_remembers_across_a_restart():
    assert library.save("current", SVG, key="s1:a") == "current.svg"
    listed = library.listing()[0]["id"]
    _restart()
    assert library.listing()[0]["id"] == listed                          # the same item, the same id
    assert library.rename("", "after restart", key="s1:a") == "after restart.svg"   # the open picture still has it
    _restart()
    newer = SVG.replace("5", "7")
    assert library.save("x", newer, key="s1:a") == "after restart.svg"   # and its next save carries on in it
    assert [i["name"] for i in library.listing()] == ["after restart.svg"]
    library.rename("after restart.svg", "moved in compose")              # renamed in Compose, then a restart
    _restart()
    assert library.rename("", "header", key="s1:a") == "header.svg"      # the header finds it; no copy
    old = library.tick()                                                 # a result on screen...
    library.delete("header.svg")                                         # ...its item deleted, then a restart
    _restart()
    with pytest.raises(KeyError):
        library.rename("", "back again", key="s1:a")
    with pytest.raises(KeyError):
        library.save("back again", newer, key="s1:a", rev=old)           # the old result doesn't bring it back
    assert library.listing() == []
    assert library.save("fresh", newer, key="s1:a", rev=library.tick()) == "fresh.svg"   # a new result is saved
    (library.folder() / library.STATE).write_text("not what Boot Tool wrote")
    _restart()
    assert [i["name"] for i in library.listing()] == ["fresh.svg"]       # a spoilt notes file: start afresh, lose nothing


def test_an_identical_copy_put_in_its_place_is_another_item():
    """art.svg is replaced in the file manager by a copy with the very same contents; then a restart."""
    import shutil
    library.save("art", SVG, key="tabA:1")
    f = library.folder() / "art.svg"
    shutil.copy(f, library.folder() / "copy.tmp")
    f.unlink()
    (library.folder() / "copy.tmp").rename(f)                           # the same bytes, another file
    if f.stat().st_ino == 0:
        pytest.skip("this disk doesn't tell files apart")
    _restart()
    with pytest.raises(KeyError):
        library.rename("", "mine", key="tabA:1")
    assert library.save("art", SVG.replace("5", "3"), key="tabA:1", rev=library.tick()) == "art-2.svg"
    assert library.read("art.svg") == SVG.encode()


def test_a_deletion_is_written_down_before_the_file_goes(monkeypatch):
    import os
    library.save("art", SVG, key="s1:a")
    old = library.tick()
    real = os.replace

    def stuck(a, b):
        if str(b).endswith(library.STATE):
            raise OSError(28, "No space left on device")
        real(a, b)
    with monkeypatch.context() as m:
        m.setattr(library.os, "replace", stuck)
        with pytest.raises(OSError):
            library.delete("art.svg")                                    # can't be noted down: not deleted either
    assert [i["name"] for i in library.listing()] == ["art.svg"]
    assert not list(library.folder().glob(".boottool-library-*.tmp"))    # nothing left lying around
    library.delete("art.svg")
    _restart()
    with pytest.raises(KeyError):
        library.save("art", SVG, key="s1:a", rev=old)                   # stays deleted across the restart
    assert library.listing() == []


def test_a_delete_that_fails_changes_nothing(monkeypatch):
    """The folder can't be changed just now (read-only, or the file is in use): the item stays, and so
    does the open picture's hold on it."""
    library.save("art", SVG, key="s1:a")
    pending = library.tick()                                            # a result waiting to be saved
    real = Path.unlink

    def stuck(self, *a, **k):
        if self.name == "art.svg":
            raise PermissionError("in use")
        return real(self, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", stuck)
        with pytest.raises(OSError):
            library.delete("art.svg")
    assert library.save("art", SVG.replace("5", "7"), key="s1:a", rev=pending) == "art.svg"   # still its item, still saved
    assert [i["name"] for i in library.listing()] == ["art.svg"]
    _restart()
    assert library.rename("", "after", key="s1:a") == "after.svg"       # and that is what was written down


def _other_boot_tool(code, *args):
    """Run a bit of library code in another process, as a second Boot Tool on the same folders would."""
    import subprocess
    import sys
    env = {**__import__("os").environ, "BOOTTOOL_LIBRARY": str(library.folder())}
    return subprocess.Popen([sys.executable, "-c", code, *args], env=env, stdout=subprocess.PIPE, text=True)


HOLD = ("import sys, time; from boottool import library, filelock\n"
        "with filelock.Turns().held(library.folder() / library.LOCK):\n"
        "    print('held', flush=True); time.sleep(float(sys.argv[1]))\n")


def test_changes_take_turns_with_a_second_boot_tool(monkeypatch):
    import time
    from boottool import filelock
    other = _other_boot_tool(HOLD, "1.5")                               # another Boot Tool is in the middle of a change
    assert other.stdout.readline().strip() == "held"
    t0 = time.monotonic()
    assert library.save("a", SVG) == "a.svg"                             # we wait for it, then go
    assert time.monotonic() - t0 > 1.0
    other.wait(10)

    other = _other_boot_tool(HOLD, "30")                                 # one that is killed in the middle...
    assert other.stdout.readline().strip() == "held"
    other.kill()
    other.wait(10)
    t0 = time.monotonic()
    assert library.save("b", SVG) == "b.svg"                             # ...leaves nothing locked
    assert time.monotonic() - t0 < 5

    other = _other_boot_tool(HOLD, "6")                                  # one that keeps the folder too long:
    assert other.stdout.readline().strip() == "held"
    monkeypatch.setattr(filelock, "WAIT", 0.5)
    with pytest.raises(filelock.Busy):
        library.save("c", SVG)                                           # a change says so instead of going ahead
    assert sorted(i["name"] for i in library.listing()) == ["a.svg", "b.svg"]   # reading still works
    other.kill()
    other.wait(10)


def test_just_looking_never_tidies_up_for_a_boot_tool_that_is_still_at_it(monkeypatch):
    """Another Boot Tool wrote a delete down and holds the folder while it removes the file. A list that
    gives up waiting must not finish that delete for it (nor write anything)."""
    from boottool import filelock
    library.save("art", SVG, key="s1:a")
    state = library.folder() / library.STATE
    data = json.loads(state.read_text())
    item = next(iter(data["items"]))
    f = library.folder() / "art.svg"
    st = f.stat()
    data["intent"] = {"do": "delete", "item": item, "name": "art.svg", "to": "art.svg", "digest": data["items"][item]["digest"],
                      "file": [st.st_dev, st.st_ino], "key": None}
    state.write_text(json.dumps(data))
    written = state.read_bytes()
    other = _other_boot_tool(HOLD, "4")
    assert other.stdout.readline().strip() == "held"
    monkeypatch.setattr(filelock, "WAIT", 0.3)
    assert [i["name"] for i in library.listing()] == ["art.svg"]         # looked, touched nothing
    assert f.exists() and state.read_bytes() == written
    other.kill()
    other.wait(10)
    monkeypatch.setattr(filelock, "WAIT", 10.0)
    assert library.listing() == [] and not f.exists()                    # with the folder free, it is finished


def test_a_change_needs_the_folders_lock(monkeypatch):
    from boottool import filelock
    library.save("a", SVG, key="s1:a")
    real_open = filelock.os.open                                        # the lock file can't be opened:
    monkeypatch.setattr(filelock.os, "open", lambda p, *a, **k: (_ for _ in ()).throw(PermissionError("lock file")) if str(p).endswith(library.LOCK) else real_open(p, *a, **k))
    for change in (lambda: library.save("b", SVG), lambda: library.rename("a.svg", "c"), lambda: library.delete("a.svg")):
        with pytest.raises(OSError):
            change()
    assert [i["name"] for i in library.listing()] == ["a.svg"]           # looking still works
    assert library.read("a.svg") == SVG.encode()


def test_changes_cut_short_are_sorted_out_the_next_time(monkeypatch):
    """Boot Tool is killed between changing a file and writing that down."""
    assert library.save("art", SVG, key="s1:a") == "art.svg"
    real_store = library._store

    def killed_before_second_note(must=False):
        if library._intent is None:                                     # the note after the change
            raise SystemExit("killed")
        real_store(must)
    newer = SVG.replace("5", "7")
    with monkeypatch.context() as m:
        m.setattr(library, "_store", killed_before_second_note)
        with pytest.raises(SystemExit):
            library.save("art", newer, key="s1:a")                       # the new SVG is in place, not written down
    _restart()
    assert library.save("art", SVG.replace("5", "8"), key="s1:a") == "art.svg"   # still the picture's item: no art-2
    with monkeypatch.context() as m:
        m.setattr(library, "_store", killed_before_second_note)
        with pytest.raises(SystemExit):
            library.rename("", "renamed", key="s1:a")                   # renamed on disk, not written down
    _restart()
    assert [i["name"] for i in library.listing()] == ["renamed.svg"]
    assert library.save("x", newer, key="s1:a") == "renamed.svg"         # and still the picture's item

    real_replace = Path.replace

    def killed_before_the_file(self, target):
        if str(target).endswith("renamed.svg"):
            raise SystemExit("killed")
        return real_replace(self, target)
    with monkeypatch.context() as m:
        m.setattr(Path, "replace", killed_before_the_file)
        with pytest.raises(SystemExit):
            library.save("x", SVG, key="s1:a")                          # written down, the file not replaced
    _restart()
    assert library.read("renamed.svg") == newer.encode()
    assert library.save("x", SVG, key="s1:a") == "renamed.svg"           # the old file is still the picture's item


def test_a_cut_short_delete_that_cant_be_finished_yet_is_tried_again(monkeypatch):
    library.save("crash", SVG, key="s1:a")
    real = Path.unlink

    def killed(self, *a, **k):
        if self.name == "crash.svg":
            raise SystemExit("killed")
        return real(self, *a, **k)

    def in_use(self, *a, **k):
        if self.name == "crash.svg":
            raise PermissionError("in use")
        return real(self, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", killed)
        with pytest.raises(SystemExit):
            library.delete("crash.svg")
    _restart()
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", in_use)
        library.listing()                                               # can't be finished just now
    assert "intent" in json.loads((library.folder() / library.STATE).read_text())   # still written down
    assert library.listing() == [] and not (library.folder() / "crash.svg").exists()   # finished the next time


def test_a_delete_cut_short_is_finished_the_next_time(monkeypatch):
    """Boot Tool is killed after writing the deletion down and before removing the file."""
    library.save("crash", SVG, key="s1:a")
    library.save("other", SVG.replace("5", "7"), key="s1:b")
    real = Path.unlink

    def killed(self, *a, **k):
        if self.name == "crash.svg":
            raise SystemExit("killed")
        return real(self, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", killed)
        with pytest.raises(SystemExit):
            library.delete("crash.svg")
    assert (library.folder() / "crash.svg").exists()
    assert [i["name"] for i in library.listing()] == ["other.svg"]      # the delete is finished at the next use
    _restart()
    assert [i["name"] for i in library.listing()] == ["other.svg"]
    with pytest.raises(KeyError):
        library.rename("", "back", key="s1:a")
    assert library.rename("", "still mine", key="s1:b") == "still mine.svg"
    assert "deleting" not in json.loads((library.folder() / library.STATE).read_text())


def test_nothing_else_is_changed_until_a_cut_short_change_is_sorted_out(monkeypatch):
    """A delete was written down, Boot Tool was killed, and the file can't be removed just now. Saving
    something else meanwhile must not make Boot Tool forget the unfinished delete."""
    library.save("crash", SVG, key="s1:a")
    real = Path.unlink

    def killed(self, *a, **k):
        if self.name == "crash.svg":
            raise SystemExit("killed")
        return real(self, *a, **k)

    def in_use(self, *a, **k):
        if self.name == "crash.svg":
            raise PermissionError("in use")
        return real(self, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", killed)
        with pytest.raises(SystemExit):
            library.delete("crash.svg")
    _restart()
    other = SVG.replace("5", "7")
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", in_use)
        with pytest.raises(OSError, match="crash.svg"):
            library.save("other", other, key="s1:b")                    # refused, and says which file is in the way
        assert library.read("crash.svg") == SVG.encode()                # looking still works
    assert "intent" in json.loads((library.folder() / library.STATE).read_text())   # still written down
    assert sorted(f.name for f in library.folder().iterdir() if not f.name.startswith(".")) == ["crash.svg"]
    assert library.save("other", other, key="s1:b") == "other.svg"       # the delete is finished first, then the save
    assert [i["name"] for i in library.listing()] == ["other.svg"]


def test_no_change_is_made_that_cant_be_written_down_first(monkeypatch):
    """If Boot Tool can't note a save or a rename down (a full disk, say), it doesn't make it: otherwise
    a restart would no longer know whose item that is."""
    library.save("art", SVG, key="s1:a")
    real = library.os.replace
    newer = SVG.replace("5", "7")

    def full(src, dst, *a, **k):
        if str(dst).endswith(library.STATE):
            raise OSError(28, "No space left on device")
        return real(src, dst, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(library.os, "replace", full)
        for change in (lambda: library.rename("", "after", key="s1:a"), lambda: library.rename("art.svg", "after"),
                       lambda: library.save("art", newer, key="s1:a"), lambda: library.save("new", newer, key="s1:b")):
            with pytest.raises(OSError):
                change()
    assert sorted(f.name for f in library.folder().iterdir() if not f.name.startswith(".")) == ["art.svg"]
    assert library.read("art.svg") == SVG.encode()
    _restart()
    assert library.rename("", "after", key="s1:a") == "after.svg"       # still the picture's item
    assert library.save("x", newer, key="s1:a") == "after.svg"


def test_a_cut_short_rename_is_recognised_whatever_is_at_the_old_name(monkeypatch):
    real_store = library._store

    def killed_before_second_note(must=False):
        if library._intent is None:                                     # the note after the change
            raise SystemExit("killed")
        real_store(must)
    library.save("mine", SVG, key="s1:a")
    with monkeypatch.context() as m:
        m.setattr(library, "_store", killed_before_second_note)
        with pytest.raises(SystemExit):
            library.rename("", "moved", key="s1:a")                     # renamed on disk, not written down
    other = SVG.replace("5", "7")
    (library.folder() / "mine.svg").write_text(other)                    # and something else takes the old name
    _restart()
    assert library.rename("", "again", key="s1:a") == "again.svg"       # the picture still has its item
    assert library.read("again.svg") == SVG.encode() and library.read("mine.svg") == other.encode()
    (library.folder() / "mine.svg").unlink()

    # Only the case changes: on Mac and Windows the old spelling still opens the file afterwards.
    with monkeypatch.context() as m:
        m.setattr(library, "_store", killed_before_second_note)
        with pytest.raises(SystemExit):
            library.rename("", "Again", key="s1:a")
    _restart()
    assert [i["name"] for i in library.listing()] == ["Again.svg"]
    assert library.rename("", "once more", key="s1:a") == "once more.svg"

    real_rename = Path.rename

    def killed_before_the_rename(self, target):
        if self.name == "once more.svg":
            raise SystemExit("killed")
        return real_rename(self, target)
    with monkeypatch.context() as m:
        m.setattr(Path, "rename", killed_before_the_rename)
        with pytest.raises(SystemExit):
            library.rename("", "Once More", key="s1:a")                 # written down, not renamed
    _restart()
    assert [i["name"] for i in library.listing()] == ["once more.svg"]  # it has its old name, and is still the picture's
    assert library.save("x", other, key="s1:a") == "once more.svg"


def test_notes_that_cant_be_read_just_now_are_not_thrown_away(monkeypatch):
    """The library's own notes are there but can't be read for a moment (in use, not allowed). That
    is not the same as having none: nothing may be written over them."""
    library.save("art", SVG, key="s1:a")
    state = library.folder() / library.STATE
    written = state.read_bytes()
    _restart()
    real = Path.read_text
    newer = SVG.replace("5", "7")

    def in_use(self, *a, **k):
        if self.name == library.STATE:
            raise PermissionError("in use")
        return real(self, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(Path, "read_text", in_use)
        listed = library.listing()                                      # looking still works
        assert [i["name"] for i in listed] == ["art.svg"]
        assert library.read("art.svg", listed[0]["id"]) == SVG.encode()
        for change in (lambda: library.save("art", newer, key="s1:a"), lambda: library.rename("art.svg", "b"),
                       lambda: library.delete("art.svg")):
            with pytest.raises(OSError):
                change()                                                # changes wait until it can be read
    assert state.read_bytes() == written
    assert sorted(f.name for f in library.folder().iterdir() if not f.name.startswith(".")) == ["art.svg"]
    assert library.save("art", newer, key="s1:a") == "art.svg"           # readable again: still the picture's item


def test_requests_waiting_for_another_boot_tool_wait_side_by_side(monkeypatch):
    """Another Boot Tool keeps the folder for too long. The list and the thumbnails each give up after
    one wait, and together, not one after the other."""
    import time
    from boottool import filelock
    for n in range(5):
        library.save(f"g{n}", SVG.replace("5", str(n + 1)))
    other = _other_boot_tool(HOLD, "30")
    assert other.stdout.readline().strip() == "held"
    monkeypatch.setattr(filelock, "WAIT", 1.5)
    try:
        t0 = time.monotonic()
        listed = library.listing()
        assert len(listed) == 5 and time.monotonic() - t0 < 2.5         # one wait (1.5 s), not two
        got = []
        readers = [threading.Thread(target=lambda i=i: got.append(library.read(i["name"], i["id"]))) for i in listed]
        t0 = time.monotonic()
        for t in readers:
            t.start()
        for t in readers:
            t.join(30)
        assert len(got) == 5 and time.monotonic() - t0 < 4              # together (1.5 s), not 5 x 1.5 s
    finally:
        other.kill()
        other.wait(10)


def test_requests_of_one_boot_tool_go_in_the_order_they_came(monkeypatch):
    """A slow request of our own is not "another Boot Tool": a save behind it waits its turn, however
    long, and is not overtaken by requests that came after it."""
    import time
    from boottool import filelock
    monkeypatch.setattr(filelock, "WAIT", 0.3)
    library.save("a", SVG)
    order, inside = [], threading.Event()

    def slow():
        with library._busy(change=False):
            inside.set()
            time.sleep(1.0)                                              # (longer than the wait for another Boot Tool)

    def then(what, do):
        do()
        order.append(what)
    first = threading.Thread(target=slow)
    first.start()
    assert inside.wait(10)
    later = [threading.Thread(target=then, args=("save", lambda: library.save("b", SVG)))]
    later += [threading.Thread(target=then, args=("read", lambda: library.read("a.svg"))) for _ in range(3)]
    for t in later:
        t.start()
        time.sleep(0.05)
    for t in [first, *later]:
        t.join(30)
    assert order == ["save", "read", "read", "read"]
    assert sorted(i["name"] for i in library.listing()) == ["a.svg", "b.svg"]


def test_a_folder_that_cant_be_locked_still_works(monkeypatch):
    """Some network drives have no file locks. Boot Tool then carries on as the only one there."""
    import errno
    from boottool import filelock
    monkeypatch.setattr(filelock, "_take", lambda fd: (_ for _ in ()).throw(OSError(errno.ENOLCK, "No locks available")))
    assert library.save("a", SVG, key="s1:a") == "a.svg"
    _restart()
    assert library.rename("", "b", key="s1:a") == "b.svg"               # and it is written down as usual
    library.delete("b.svg")
    assert library.listing() == []
    # Any other trouble with the lock is not "no locks here": a change doesn't go ahead without it.
    library.save("c", SVG)
    monkeypatch.setattr(filelock, "_take", lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "Input/output error")))
    with pytest.raises(OSError):
        library.save("d", SVG)
    assert [i["name"] for i in library.listing()] == ["c.svg"]           # looking still works


def test_a_cut_short_rename_never_locks_the_library(monkeypatch):
    """Written down, Boot Tool killed before the file was renamed, and the file can't be read any more.
    That must not stop everything else from being saved."""
    library.save("mine", SVG, key="s1:a")
    real_rename, real_read = Path.rename, Path.read_bytes

    def killed(self, target):
        raise SystemExit("killed")

    def unreadable(self):
        if self.name == "mine.svg":
            raise PermissionError("not allowed")
        return real_read(self)
    with monkeypatch.context() as m:
        m.setattr(Path, "rename", killed)
        with pytest.raises(SystemExit):
            library.rename("", "moved", key="s1:a")
    _restart()
    with monkeypatch.context() as m:
        m.setattr(Path, "read_bytes", unreadable)
        assert library.save("other", SVG.replace("5", "7"), key="s1:b") == "other.svg"
    assert sorted(i["name"] for i in library.listing()) == ["mine.svg", "other.svg"]
    assert library.save("x", SVG.replace("5", "8"), key="s1:a") == "mine.svg"   # and the picture still has its item


def _notes_cant_be_written(monkeypatch, after=0):
    """The library's notes can't be written any more (a full disk), after `after` more times."""
    real, left = library.os.replace, [after]

    def full(src, dst, *a, **k):
        if str(dst).endswith(library.STATE):
            if left[0] <= 0:
                raise OSError(28, "No space left on device")
            left[0] -= 1
        return real(src, dst, *a, **k)
    monkeypatch.setattr(library.os, "replace", full)


def test_a_delete_that_fails_is_never_finished_behind_your_back(monkeypatch):
    library.save("keep", SVG, key="s1:a")
    real = Path.unlink

    def in_use(self, *a, **k):
        if self.name == "keep.svg":
            raise PermissionError("in use")
        return real(self, *a, **k)
    with monkeypatch.context() as m:                                     # the file can't be removed:
        m.setattr(Path, "unlink", in_use)
        with pytest.raises(PermissionError):
            library.delete("keep.svg")
    _restart()
    assert [i["name"] for i in library.listing()] == ["keep.svg"]       # it stays, also after a restart,
    assert library.save("x", SVG.replace("5", "7"), key="s1:a") == "keep.svg"   # and is still the picture's item

    with monkeypatch.context() as m:                                     # ...and that can't be written down either:
        m.setattr(Path, "unlink", in_use)
        _notes_cant_be_written(m, after=1)
        with pytest.raises(OSError, match="as soon as it can"):         # then it says the delete is still to come
            library.delete("keep.svg")
    assert (library.folder() / "keep.svg").exists()
    _restart()
    assert library.listing() == []                                      # and it comes


def test_a_save_that_fails_doesnt_rename_the_item_either(monkeypatch):
    """The picture was given a new name, and its next result can't be saved. Its item keeps its name."""
    library.save("art", SVG, key="s1:a")
    newer = SVG.replace("5", "7")
    real_replace, real_rename = Path.replace, Path.rename

    def no_room(self, target):
        if str(target).endswith(".svg"):
            raise OSError(28, "No space left on device")
        return real_replace(self, target)

    def files():
        return sorted(f.name for f in library.folder().iterdir() if not f.name.startswith("."))
    with monkeypatch.context() as m:
        m.setattr(Path, "replace", no_room)
        with pytest.raises(OSError):
            library.save("new name", newer, key="s1:a", rename=True)
    assert files() == ["art.svg"] and library.read("art.svg") == SVG.encode()
    with monkeypatch.context() as m:
        _notes_cant_be_written(m)
        with pytest.raises(OSError):
            library.save("new name", newer, key="s1:a", rename=True)
    assert files() == ["art.svg"] and library.read("art.svg") == SVG.encode()

    def in_use(self, target):
        raise PermissionError("in use")
    with monkeypatch.context() as m:                                     # saved, but the item can't be renamed:
        m.setattr(Path, "rename", in_use)
        assert library.save("new name", newer, key="s1:a", rename=True) == "art.svg"   # it says the name it still has
        library.save("Badge-2026", SVG, key="s1:b")                      # (an old name that looks like a numbered one)
        assert library.save_named("Badge", newer, key="s1:b", rename=True) == ("Badge-2026.svg", False)   # and that it isn't renamed
    assert files() == ["Badge-2026.svg", "art.svg"] and library.read("art.svg") == newer.encode()
    assert library.save_named("new name", newer, key="s1:a", rename=True) == ("new name.svg", True)   # the next save renames it
    assert library.save_named("Badge", newer, key="s1:b", rename=True) == ("Badge.svg", True)
    assert library.save_named("Badge", newer, key="s1:c") == ("Badge-2.svg", True)   # (a new item, the name taken)
    assert files() == ["Badge-2.svg", "Badge.svg", "new name.svg"]


def test_a_save_cut_short_leaves_nothing_lying_about(monkeypatch):
    library.save("art", SVG, key="s1:a")
    real_replace = Path.replace

    def killed(self, target):
        if str(target).endswith("art.svg"):
            raise SystemExit("killed")
        return real_replace(self, target)
    with monkeypatch.context() as m:
        m.setattr(Path, "replace", killed)
        with pytest.raises(SystemExit):
            library.save("art", SVG.replace("5", "7"), key="s1:a")
    assert (library.folder() / "art.svg.tmp").exists()
    _restart()
    library.listing()
    assert sorted(f.name for f in library.folder().iterdir() if not f.name.startswith(".")) == ["art.svg"]
    assert library.read("art.svg") == SVG.encode()


def test_a_second_boot_tool_on_the_same_library(tmp_path):
    """Two Boot Tool windows share the folder: what one writes down, the other reads before it writes."""
    import subprocess
    import sys
    library.save("a", SVG, key="one:a")
    other = ("import sys; from boottool import library;"
             "print(library.save('b', sys.argv[1], key='two:b'))")
    env = {**__import__("os").environ, "BOOTTOOL_LIBRARY": str(library.folder())}
    assert subprocess.run([sys.executable, "-c", other, SVG], env=env, capture_output=True, text=True).stdout.strip() == "b.svg"
    assert library.save("a", SVG.replace("5", "7"), key="one:a") == "a.svg"   # the first window saves again
    again = ("import sys; from boottool import library;"
             "print(library.save('b', sys.argv[1], key='two:b'))")
    out = subprocess.run([sys.executable, "-c", again, SVG.replace("5", "9")], env=env, capture_output=True, text=True).stdout.strip()
    assert out == "b.svg"                                               # the second window's item is still its own
    assert sorted(i["name"] for i in library.listing()) == ["a.svg", "b.svg"]


def test_a_spoilt_notes_file_never_stops_the_library():
    library.save("a", SVG, key="s1:a")
    state = library.folder() / library.STATE
    data = json.loads(state.read_text())
    item = next(iter(data["items"]))
    huge = json.dumps({**data, "keys": {"s1:a": [item, 0]}}).replace('[\"%s\", 0]' % item, '["%s", 1e309]' % item).replace('["%s", 0]' % item, '["%s", 1e309]' % item)
    assert "1e309" in huge
    for bad in (huge, '{"items": {"i": {"name": "../x.svg", "digest": "d"}}, "keys": {}, "gone": {}}', "[]", "", '{"items": 3}'):
        state.write_text(bad)
        _restart()
        assert [i["name"] for i in library.listing()] == ["a.svg"]


def test_two_tabs_with_the_same_picture_after_a_restart():
    """Tabs A and B opened the same picture: line.svg and line-2.svg. A's item is deleted and B's renamed
    to line.svg; then Boot Tool restarts. A must not rename or overwrite B's item."""
    assert library.save("line", SVG, key="tabA:1") == "line.svg"
    assert library.save("line", SVG, key="tabB:1") == "line-2.svg"
    library.delete("line.svg")
    assert library.rename("line-2.svg", "line") == "line.svg"
    _restart()
    with pytest.raises(KeyError):
        library.rename("", "mine", key="tabA:1")
    assert library.save("line", SVG.replace("5", "3"), key="tabA:1", rev=library.tick()) == "line-2.svg"
    assert library.read("line.svg") == SVG.encode()                      # B's item is untouched
    assert library.rename("", "theirs", key="tabB:1") == "theirs.svg"    # and still B's


def test_an_item_is_its_contents_not_its_date():
    import os
    library.save("art", SVG, key="s1:a")
    listed = library.listing()[0]["id"]
    f = library.folder() / "art.svg"
    for _ in range(3):                                                   # only touched (a backup or sync tool): same item
        os.utime(f, ns=(5, 5 + _))
        assert library.listing()[0]["id"] == listed
        assert library.save("art", SVG, key="s1:a") == "art.svg"         # and no art-2, art-3...
    st = f.stat()
    f.write_text(SVG.replace("5", "9"), encoding="utf-8")                # edited in place, same length...
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))                     # ...and the same date
    assert f.stat().st_size == st.st_size
    for act in (lambda: library.delete("art.svg", listed), lambda: library.rename("art.svg", "mine", item=listed),
                lambda: library.read("art.svg", listed)):
        with pytest.raises(KeyError):
            act()
    assert library.save("art", SVG, key="s1:a") == "art-2.svg"           # the edit is not overwritten
    assert b"9" in library.read("art.svg")


def test_deleting_by_a_name_in_another_case_leaves_no_stale_link():
    """On Mac and Windows line.svg and Line.svg are one file: deleting by the old spelling must also
    forget which picture owned it, or that picture could later rename someone else's line.svg."""
    library.save("line", SVG, key="s1:a")
    assert library.rename("", "Line", key="s1:a") == "Line.svg"
    library.delete("line.svg")                                          # a stale tile in Compose
    if "Line.svg" in [i["name"] for i in library.listing()]:
        return                                                          # this disk tells them apart: nothing was deleted
    assert library.save("line", SVG.replace("5", "7"), key="s1:b") == "line.svg"
    with pytest.raises(KeyError):
        library.rename("", "mine", key="s1:a")
    assert [i["name"] for i in library.listing()] == ["line.svg"]


def test_library_rejects_bad_names_and_non_svg():
    with pytest.raises(ValueError):
        library.save("x", "hello")
    for bad in ("../x.svg", "a/b.svg", ".hidden.svg", "x.txt", ""):
        with pytest.raises(ValueError):
            library.read(bad)
    with pytest.raises(KeyError):
        library.read("missing.svg")


def test_library_endpoints(server):
    code, j = _post(server + "/api/library/save", {"name": "Wave", "svg": SVG, "key": "k"})
    assert code == 200 and j["name"] == "Wave.svg"
    code, body, h = _get(server + "/api/library")
    assert [i["name"] for i in json.loads(body)["items"]] == ["Wave.svg"]
    code, body, h = _get(server + "/api/library/file?name=Wave.svg")
    assert code == 200 and h["Content-Type"] == "image/svg+xml" and b"<rect" in body
    assert _get(server + "/api/library/file?name=../../etc/passwd")[0] == 404
    assert _post(server + "/api/library/rename", {"name": "Wave.svg", "to": "Surf"})[1]["name"] == "Surf.svg"
    assert _post(server + "/api/library/save", {"name": "x", "svg": "nope"})[0] == 400
    assert _post(server + "/api/library/delete", {"name": "Surf.svg"})[0] == 200
    assert json.loads(_get(server + "/api/library")[1])["items"] == []


def test_static_files_are_limited_to_the_known_scripts(server):
    assert _get(server + "/static/compose.js")[0] == 200
    code, body, _ = _get(server + "/static/vendor/paper-core.js")
    assert code == 200 and b"Paper.js" in body[:400]
    assert _get(server + "/static/../ui.py")[0] == 404 and _get(server + "/static/index.html")[0] == 404


def _make_font(path, family="Test Sans", advance=600):
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A", "space"])
    fb.setupCharacterMap({65: "A", 32: "space"})
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0)); pen.lineTo((100, 700)); pen.lineTo((500, 700)); pen.lineTo((500, 0)); pen.closePath()
    glyphs = {".notdef": TTGlyphPen(None).glyph(), "A": pen.glyph(), "space": TTGlyphPen(None).glyph()}
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({".notdef": (500, 0), "A": (advance, 100), "space": (300, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200)
    fb.setupPost()
    fb.save(str(path))


def test_fonts_are_found_and_text_becomes_outlines(tmp_path):
    _make_font(fonts.user_folder() / "TestSans.ttf")
    lst = fonts.listing(refresh=True)
    f = next(x for x in lst if x["family"] == "Test Sans")
    one = fonts.outline(f["id"], "A", 10)
    assert one["d"].startswith("M") and one["d"].endswith("Z")
    assert abs(one["w"] - 6.0) < 0.01 and abs(one["h"] - 10.0) < 0.01          # 600/1000 em wide, ascent+descent tall
    two = fonts.outline(f["id"], "A A", 10, spacing=100)
    assert two["w"] > one["w"] * 2.5
    lines = fonts.outline(f["id"], "A\nA", 10)
    assert lines["h"] > one["h"] * 1.5
    with pytest.raises(ValueError, match="Type some text"):
        fonts.outline(f["id"], "  ", 10)
    with pytest.raises(KeyError):
        fonts.outline(99999, "A", 10)


def test_text_endpoint(server):
    _make_font(fonts.user_folder() / "TestSans.ttf")
    fid = next(x for x in fonts.listing(refresh=True) if x["family"] == "Test Sans")["id"]
    code, j = _post(server + "/api/text", {"font": fid, "text": "AA", "size": 20, "align": "center"})
    assert code == 200 and j["d"].count("M") == 2 and j["w"] > 20
    assert _post(server + "/api/text", {"font": fid, "text": "", "size": 20})[0] == 400


def test_edited_text_keeps_its_own_font(server):
    """Picking a font for new text must not redraw existing text in it; picking a font for selected text does."""
    _make_font(fonts.user_folder() / "TestSans.ttf")
    _make_font(fonts.user_folder() / "TestWide.ttf", family="Test Wide", advance=1200)
    fonts.listing(refresh=True)
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.click('a[data-t="compose"]')
        page.wait_for_function("typeof Compose !== 'undefined' && Compose._S.ready")
        w = page.evaluate("""async () => {
          const A = Compose._api, S = Compose._S, L = A.getL(), wait = () => new Promise(r => setTimeout(r, 1200));
          const j = await (await fetch('/api/fonts')).json(); S.fonts = j.fonts;
          const narrow = S.fonts.find(f => f.family === 'Test Sans'), wide = S.fonts.find(f => f.family === 'Test Wide');
          S.font = narrow; S.text.text = 'AAAA';
          await A.addText(new paper.Point(20, 20));
          const t = L.art.lastChild, own = t.bounds.width;
          S.font = wide;                                   // the font chosen for the NEXT text
          A.retext(t, {spacing: 0}); await wait();         // then the first text is edited without touching its font
          const kept = L.art.lastChild.bounds.width;
          A.select([L.art.lastChild]);
          return {own, kept, font: L.art.lastChild.data.text.font, narrow: narrow.id, wide: wide.id};
        }""")
        assert abs(w["kept"] - w["own"]) < 0.01 and w["font"] == w["narrow"], w      # used to be redrawn in Test Wide
        page.select_option("#side select", str(w["wide"]))                         # choosing a font for the selected text still works
        page.wait_for_timeout(1200)
        wider = page.evaluate("[Compose._api.getL().art.lastChild.bounds.width, Compose._api.getL().art.lastChild.data.text.font]")
        assert wider[0] > w["own"] * 1.5 and wider[1] == w["wide"], wider
        assert not errs, errs
        b.close()


def test_custom_page_size():
    o = pg.coerce_page({"size": "Custom", "custom_w": 353, "custom_h": 266})
    assert o.dims() == (353.0, 266.0)
    assert pg.coerce_page({"size": "Custom", "custom_w": 9999}).custom_w == 2000
    assert pg.coerce_page({"size": "A4"}).dims() == (297.0, 210.0)
    assert len(pg.mark_positions(o)) == 6


def test_batch_results_also_land_in_the_library(server, lineart):
    iid = json.loads(urllib.request.urlopen(urllib.request.Request(server + "/api/upload", data=lineart.read_bytes(), method="POST")).read())["id"]
    code, j = _post(server + "/api/batch/start", {"items": [{"id": iid, "name": "plate one.png", "params": {}}], "keep_going": True})
    assert code == 200
    import time
    for _ in range(100):
        s = json.loads(_get(server + f"/api/batch/status?job={j['job']}")[1])
        if s.get("finished"):
            break
        time.sleep(0.2)
    assert any(i["name"].startswith("plate one") for i in library.listing())


@pytest.mark.parametrize("target", ["compose", "print", "simulate", "help"])
def test_leaving_the_batch_view_for_another_tab(server, lineart, target):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        page.wait_for_function("/Paths/i.test(document.querySelector('#stats').textContent)", timeout=60000)
        page.click("#toolbar [data-m=batch]")
        page.click(f'a[data-t="{target}"]')
        page.wait_for_timeout(1000)
        assert not errs, errs
        assert page.evaluate("document.querySelector('#side').innerText.trim()")          # the new tab drew its panel
        page.click('a[data-t="convert"]')                                                    # and Convert still works afterwards
        page.wait_for_function("/Paths/i.test(document.querySelector('#stats').textContent)", timeout=60000)
        assert not errs, errs
        b.close()


def test_compose_in_the_browser(server):
    sync_api = pytest.importorskip("playwright.sync_api")
    library.save("badge", '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">'
                          '<circle cx="50" cy="50" r="45" fill="#c8102e"/><rect x="30" y="30" width="40" height="40" fill="#1d3f8f"/></svg>')
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.click('a[data-t="compose"]')
        page.wait_for_function("typeof Compose !== 'undefined' && Compose._S.ready")
        page.hover("#clib .ltab")
        page.click("#clibStrip .lt")
        page.wait_for_function("Compose._api.getL().art.children.length === 1")
        # recolour the circle, then cut the badge in two with a line
        page.evaluate("Compose._api.select([Compose._api.getL().art.children[0]])")
        page.evaluate("Compose._api.applyColour('fill', '#00a651')")
        assert page.evaluate("Compose._api.leaves(Compose._S.sel).map(l => l.fillColor.toCSS(true))[0]") == "#00a651"
        page.evaluate("Compose._api.sendToPrint()")
        page.wait_for_function("document.querySelector('#nav a.on').dataset.t === 'print'", timeout=60000)
        assert page.evaluate("[PG.size, PG.custom_w, PG.custom_h]") == ["Custom", 353, 266]
        assert not errs
        b.close()


def test_compose_send_rename_and_eyedropper(server):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.click('a[data-t="compose"]')
        page.wait_for_function("typeof Compose !== 'undefined' && Compose._S.ready")
        assert page.query_selector('[data-ct="cut"]') is None
        page.evaluate("""(() => { const L = Compose._api.getL();
          L.art.addChild(new paper.Path.Rectangle({point: [40, 40], size: [100, 100], fillColor: '#c8102e'}));
          L.art.addChild(new paper.Path.Circle({center: [220, 100], radius: 40, fillColor: '#000000'})); Compose._api.commit(); })()""")
        page.evaluate("Compose._api.sendToPrint()")
        page.wait_for_function("document.querySelector('#nav a.on').dataset.t === 'print'", timeout=60000)
        page.wait_for_function("lastPalette.length > 0", timeout=120000)
        page.click("#nmEdit")
        page.keyboard.type("my badge")
        page.keyboard.press("Enter")
        assert page.evaluate("fileName") == "my badge"
        page.evaluate("eyeStart(0)")
        assert page.evaluate("EYE.on && P.print.picked")
        page.keyboard.press("Escape")
        assert page.evaluate("!EYE.on")
        assert not errs, errs
        b.close()


def test_a_result_is_saved_even_when_its_item_cant_take_the_new_name(server, lineart, monkeypatch):
    """The picture is renamed, but the library file can't be renamed just now (in use). New results are
    still saved, in the item under its old name, and the rename is done as soon as it can be."""
    sync_api = pytest.importorskip("playwright.sync_api")
    real = Path.rename

    def in_use(self, target):
        if self.parent == library.folder() and self.suffix == ".svg":
            raise PermissionError("in use")
        return real(self, target)
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        page.wait_for_function("libName !== ''", timeout=60000)
        first = page.evaluate("libName")
        saves = []
        page.on("response", lambda r: saves.append(r.json()) if r.url.endswith("/api/library/save") else None)
        with monkeypatch.context() as m:
            m.setattr(Path, "rename", in_use)
            page.click("#nmEdit")
            page.keyboard.type("my badge")
            page.keyboard.press("Enter")
            page.wait_for_function("document.querySelector('#libcell').textContent.startsWith('Not renamed')", timeout=60000)
            page.evaluate("lastShown = null; run()")                    # a new result, with the rename still to do
            page.wait_for_function("lastShown !== null", timeout=120000)
            page.wait_for_timeout(3000)                                  # longer than the delayed save and the rename after it
            assert [(j.get("name"), j.get("renamed")) for j in saves] == [(first, False)]   # saved, under the old name
            assert page.evaluate("LIB.failed[imageId] === undefined && LIB.renamed[imageId] === true")
            assert page.inner_text("#libcell").startswith("Not renamed")
            assert [i["name"] for i in library.listing()] == [first]
        real_replace = Path.replace

        def no_room(self, target):
            if Path(target).parent == library.folder() and str(target).endswith(".svg"):
                raise OSError(28, "No space left on device")
            return real_replace(self, target)
        with monkeypatch.context() as m:                                 # the file is free again, but now saving fails:
            m.setattr(Path, "replace", no_room)
            page.evaluate("lastShown = null; run()")
            page.wait_for_function("document.querySelector('#libcell').textContent.startsWith('Not saved')", timeout=60000)
            page.wait_for_timeout(1500)
            assert [i["name"] for i in library.listing()] == [first]    # a result that isn't saved doesn't rename the item
            assert page.inner_text("#libcell").startswith("Not saved")
        page.evaluate("lastShown = null; run()")                        # all is well again: the next result renames it
        page.wait_for_function("libName === 'my badge.svg'", timeout=60000)
        page.wait_for_function("document.querySelector('#libcell').textContent === 'Saved as my badge.svg'", timeout=60000)
        assert [i["name"] for i in library.listing()] == ["my badge.svg"]
        assert not errs, errs
        b.close()


def test_renaming_in_the_header_renames_the_library_item(server, lineart):
    """The header name names the library item. Renaming renames the file and leaves the graphic alone,
    even on Print (where it used to save the whole print page over the graphic)."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        page.wait_for_function("libName !== ''", timeout=60000)
        traced = library.read(page.evaluate("libName"))

        def rename(to):
            page.click("#nmEdit")                                         # clicking selects the whole name
            if to is None:
                page.keyboard.press("Escape")                              # looked at the name, changed nothing
            else:
                page.keyboard.type(to)
                page.keyboard.press("Enter")
            page.wait_for_timeout(2500)                                    # longer than the delayed save

        rename("my badge")
        assert [i["name"] for i in library.listing()] == ["my badge.svg"]  # used to stay line.svg
        page.click('a[data-t="print"]')
        page.wait_for_function("lastRes && lastRes.plates && lastRes.plates.length > 0", timeout=120000)
        rename(None)
        rename("final badge")
        assert [i["name"] for i in library.listing()] == ["final badge.svg"]
        assert library.read("final badge.svg") == traced                   # still the Convert graphic, not the print page
        assert not errs, errs
        b.close()


def test_a_result_made_before_its_item_was_deleted_doesnt_bring_it_back():
    r1, r2 = library.tick(), library.tick()                             # two Convert results, in any tabs
    a = library.save("line", SVG, key="s1:a", rev=r1)
    library.delete(a)
    with pytest.raises(KeyError):
        library.save("line", SVG, key="s1:a", rev=r2)                   # made before the deletion, sent after
    assert library.listing() == []
    r3 = library.tick()
    assert library.save("line", SVG, key="s1:a", rev=r3) == "line.svg"  # a new result is saved again
    assert library.save("line", SVG, key="s1:a", rev=library.tick()) == "line.svg"   # and keeps its one item
    assert [i["name"] for i in library.listing()] == ["line.svg"]


def _hold(page, pattern):
    """Hold requests matching `pattern` until release() is called (returns the held list and release)."""
    held = []
    page.route(pattern, lambda route: held.append(route))

    def release():
        waiting = list(held)
        page.unroute(pattern)
        for route in waiting:
            try:
                route.continue_()
            except Exception:                                           # the page gave up on that request meanwhile
                pass
    return held, release


def test_deleting_in_compose_while_a_save_is_on_its_way(server, lineart, color):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        page.wait_for_function("libName !== ''", timeout=60000)
        held, release = _hold(page, "**/api/library/save")
        page.evaluate("lastShown = null; run()")                        # a new result...
        page.wait_for_function("lastShown !== null", timeout=60000)
        page.wait_for_timeout(2500)
        assert len(held) == 1                                           # ...whose save is on its way
        page.click('a[data-t="compose"]')
        page.hover("#clib .ltab")
        page.wait_for_selector("#clibStrip .lt", state="attached", timeout=10000)
        page.locator("#clibStrip .lt .ltx").dispatch_event("click")     # deleted in Compose's library panel (the × shows on hover)
        page.wait_for_function("document.querySelectorAll('#clibStrip .lt').length === 0", timeout=10000)
        release()
        page.wait_for_timeout(1500)
        assert library.listing() == []                                   # the save didn't bring it back
        page.click('a[data-t="convert"]')
        page.evaluate("lastShown = null; run()")                        # but the next new result is kept again
        page.wait_for_function("libName === 'line.svg'", timeout=60000)
        assert [i["name"] for i in library.listing()] == ["line.svg"]
        page.set_input_files("#file", str(color))                        # and gone pictures are forgotten at once
        page.wait_for_function("Object.keys(LIB.names).length === 1", timeout=5000)
        assert not errs, errs
        b.close()


def _shape(path, size=(600, 400)):
    """A picture that traces differently from the other fixtures (a ring and a bar)."""
    from PIL import Image, ImageDraw
    im = Image.new("L", size, 255)
    d = ImageDraw.Draw(im)
    w, h = size
    d.ellipse((w * 0.1, h * 0.15, w * 0.55, h * 0.85), outline=0, width=max(6, w // 40))
    d.rectangle((w * 0.65, h * 0.2, w * 0.9, h * 0.8), fill=0)
    im.save(path)
    return path


def _shown(page):
    """The SVG on screen, once the current picture's own result is in."""
    page.wait_for_function("lastShown !== null", timeout=120000)
    return page.evaluate("svgText").encode()


def test_each_picture_keeps_its_own_library_item(server, lineart, color, tmp_path):
    """Renaming a picture opened from Batch before it was saved used to rename the previous picture's
    library file; and switching pictures quickly dropped the first one's save."""
    sync_api = pytest.importorskip("playwright.sync_api")
    ring = _shape(tmp_path / "ring.png")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        page.wait_for_function("libName !== ''", timeout=60000)
        first = library.read("line.svg")

        def open_from_batch(name):
            page.click("#toolbar [data-m=batch]")
            page.click(f'.brow:has-text("{name}")')

        def rename(to):
            page.click("#nmEdit")
            page.keyboard.type(to)
            page.keyboard.press("Enter")

        page.click("#toolbar [data-m=batch]")
        page.set_input_files("#addImgs", [str(color), str(ring)])
        page.wait_for_function("BATCH.items.length >= 2 && BATCH.adding === 0", timeout=60000)
        open_from_batch("color.png")
        rename("renamed colour")                                         # before its first save
        colour = _shown(page)
        page.wait_for_function("libName === 'renamed colour.svg'", timeout=60000)
        assert library.read("line.svg") == first                         # the first picture's item is untouched
        assert library.read("renamed colour.svg") == colour != first
        assert sorted(i["name"] for i in library.listing()) == ["line.svg", "renamed colour.svg"]

        open_from_batch("ring.png")                                      # shown, then left before its save
        traced_ring = _shown(page)
        assert traced_ring not in (colour, first)
        open_from_batch("color.png")
        assert page.inner_text("#nmEdit") == "renamed colour"            # the name given earlier is still shown
        page.wait_for_timeout(4000)
        assert sorted(i["name"] for i in library.listing()) == ["line.svg", "renamed colour.svg", "ring.svg"]
        assert library.read("ring.svg") == traced_ring and library.read("renamed colour.svg") == colour

        rename("final colour")                                           # an ordinary rename of a saved picture
        page.wait_for_function("libName === 'final colour.svg'", timeout=60000)
        page.wait_for_timeout(2500)
        assert sorted(i["name"] for i in library.listing()) == ["final colour.svg", "line.svg", "ring.svg"]
        assert library.read("final colour.svg") == colour
        assert not errs, errs
        b.close()


def test_leaving_a_picture_before_its_trace_is_done_saves_nothing_wrong(server, lineart, color, tmp_path):
    """Opening a picture re-showed the previous picture's result for a moment, which was saved to the
    library under the new picture's name (and could be downloaded as it) if you moved on before its
    own trace came back."""
    sync_api = pytest.importorskip("playwright.sync_api")
    ring = _shape(tmp_path / "ring.png")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        page.wait_for_function("libName !== ''", timeout=60000)
        first = library.read("line.svg")
        page.click("#toolbar [data-m=batch]")
        page.set_input_files("#addImgs", [str(ring), str(color)])
        page.wait_for_function("BATCH.items.length >= 2 && BATCH.adding === 0", timeout=60000)
        held, release = _hold(page, "**/api/trace")
        page.click('.brow:has-text("ring.png")')                         # its trace doesn't come back...
        page.wait_for_function("fileName === 'ring'", timeout=10000)
        page.wait_for_timeout(500)
        assert page.is_disabled("#dl")                                   # ...so there is nothing of it to download yet,
        assert page.evaluate("!document.querySelector('#svgimg').getAttribute('src') && svgText === ''")   # and line's SVG isn't shown as ring's
        assert "Paths" not in page.inner_text("#stats")
        release()                                                        # (let later traces through)
        page.click("#toolbar [data-m=batch]")
        page.click('.brow:has-text("color.png")')                        # ...and we move on
        colour = _shown(page)
        page.wait_for_function("libName === 'color.svg'", timeout=60000)
        assert page.is_enabled("#dl")
        page.wait_for_timeout(3000)
        names = sorted(i["name"] for i in library.listing())
        assert "line.svg" in names and "color.svg" in names
        assert library.read("color.svg") == colour and library.read("line.svg") == first
        assert all(library.read(n) != first for n in names if n != "line.svg")   # nothing else holds line's graphic
        assert not errs, errs
        b.close()


def test_download_gives_the_convert_result_not_the_print_page(server, lineart, tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900}, accept_downloads=True)
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        traced = _shown(page)
        page.click('a[data-t="print"]')
        page.wait_for_function("lastRes && lastRes.plates && lastRes.plates.length > 0", timeout=120000)
        held, release = _hold(page, "**/api/trace")
        page.click('a[data-t="convert"]')                                # back on Convert, its trace not in yet
        page.wait_for_timeout(500)
        assert page.is_disabled("#dl")                                   # the print page is not offered as the graphic
        release()
        page.evaluate("run()")
        assert _shown(page) == traced
        with page.expect_download() as d:
            page.click("#dl")
        assert Path(d.value.path()).read_bytes() == traced
        assert not errs, errs
        b.close()


def test_compose_library_tiles(server):
    """Double-clicking a tile's name renames it without adding copies to the page, and a tile from an
    out-of-date list doesn't add a different graphic that has since taken its name."""
    sync_api = pytest.importorskip("playwright.sync_api")
    library.save("badge", SVG)
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.click('a[data-t="compose"]')
        page.hover("#clib .ltab")
        page.wait_for_selector("#clibStrip .lt", state="attached", timeout=10000)
        count = "paper.project.getItems({match: i => i.data && i.data.name}).length"
        start = page.evaluate(count)
        page.click("#clibStrip .lt .ltn")                                 # one click on the name adds it, once
        page.wait_for_function(count + " > " + str(start), timeout=10000)
        before = page.evaluate(count)
        page.wait_for_timeout(800)
        assert page.evaluate(count) == before
        page.hover("#clib .ltab")
        page.dblclick("#clibStrip .lt .ltn")                              # a double-click renames...
        page.keyboard.type("crest")
        page.keyboard.press("Enter")
        page.wait_for_function("document.querySelector('#clibStrip .ltn').textContent === 'crest'", timeout=10000)
        assert [i["name"] for i in library.listing()] == ["crest.svg"]
        page.wait_for_timeout(800)
        assert page.evaluate(count) == before                             # ...and adds nothing to the page

        library.rename("crest.svg", "moved")                              # meanwhile, elsewhere: renamed,
        library.save("crest", SVG.replace("5", "8"))                      # and another graphic takes the name
        page.hover("#clib .ltab")
        page.click("#clibStrip .lt .lth")                                 # the old tile
        page.wait_for_function("document.querySelectorAll('#clibStrip .lt').length === 2", timeout=10000)   # list refreshed
        assert page.evaluate(count) == before                             # the other graphic was not added as this one
        page.click('#clibStrip .lt:has-text("moved") .lth')               # the refreshed tile works
        page.wait_for_function(count + " > " + str(before), timeout=10000)

        library.rename("moved.svg", "moved again")                        # again behind this tab's back
        library.save("moved", SVG.replace("5", "6"))
        page.hover("#clib .ltab")
        page.locator('#clibStrip .lt:has-text("moved") .ltx').first.dispatch_event("click")   # the old tile's ×
        page.wait_for_function("document.querySelectorAll('#clibStrip .lt').length === 3", timeout=10000)
        assert sorted(i["name"] for i in library.listing()) == ["crest.svg", "moved again.svg", "moved.svg"]   # nothing deleted
        assert "Nothing was deleted" in page.inner_text("body")
        assert not errs, errs
        b.close()


def test_save_to_print_library_keeps_the_name_it_was_started_with(server, lineart, color):
    from boottool import prints
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        _shown(page)
        page.click('a[data-t="print"]')
        page.wait_for_function("lastRes && lastRes.plates && lastRes.plates.length > 0", timeout=120000)
        held, release = _hold(page, "**/api/design")
        page.click('button:has-text("Save to Print library")')            # the design is being prepared...
        page.wait_for_timeout(500)
        assert len(held) == 1
        page.set_input_files("#file", str(color))                         # ...and another picture is opened
        page.wait_for_function("fileName === 'color'", timeout=60000)
        release()
        page.wait_for_function("document.body.innerText.includes('Saved as')", timeout=60000)
        assert [i["name"] for i in prints.listing()] == ["line.png"]      # used to be color.png
        assert not errs, errs
        b.close()


def test_a_rename_after_a_failed_save_still_reaches_the_item(server, lineart):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.set_input_files("#file", str(lineart))
        page.wait_for_function("libName === 'line.svg'", timeout=60000)
        page.route("**/api/library/save", lambda route: route.fulfill(status=500, content_type="application/json", body='{"error": "disk trouble"}'))
        page.evaluate("lastShown = null; run()")                        # this result's save fails
        page.wait_for_function("document.querySelector('#libcell') && document.querySelector('#libcell').textContent.startsWith('Not saved')", timeout=60000)
        page.unroute("**/api/library/save")
        page.click("#nmEdit")
        page.keyboard.type("badge")
        page.keyboard.press("Enter")
        page.wait_for_function("libName === 'badge.svg'", timeout=60000)
        assert [i["name"] for i in library.listing()] == ["badge.svg"]    # the item that was there took the name
        assert page.inner_text("#libcell").startswith("Not saved")       # and it still says the latest result isn't in it
        held, release = _hold(page, "**/api/library/save")
        page.evaluate("lastShown = null; run()")                        # the next result: while its save is on its way...
        page.wait_for_function("lastShown !== null", timeout=60000)
        page.wait_for_timeout(2500)
        assert len(held) == 1 and page.inner_text("#libcell").startswith("Not saved")   # ...it doesn't claim to be saved
        release()
        page.wait_for_function("document.querySelector('#libcell').textContent === 'Saved as badge.svg'", timeout=60000)

        library.rename("badge.svg", "moved")                              # renamed in Compose's panel, then...
        _restart()                                                       # ...Boot Tool restarted, the page still open
        page.click("#nmEdit")
        page.keyboard.type("after restart")
        page.keyboard.press("Enter")
        page.wait_for_function("libName === 'after restart.svg'", timeout=60000)
        page.wait_for_timeout(2500)
        assert [i["name"] for i in library.listing()] == ["after restart.svg"]   # renamed, not copied

        library.delete("after restart.svg")                              # deleted, restarted, renamed in the header:
        _restart()
        page.click("#nmEdit")
        page.keyboard.type("back again")
        page.keyboard.press("Enter")
        page.wait_for_function("document.querySelector('#libcell').textContent.startsWith('Not in the library')", timeout=60000)
        page.wait_for_timeout(2500)
        assert library.listing() == []                                   # the old result doesn't come back
        page.evaluate("lastShown = null; run()")                        # a new result is saved, under the new name
        page.wait_for_function("libName === 'back again.svg'", timeout=60000)
        assert not errs, errs
        b.close()


def test_type_on_the_page_layer_thumbnails_and_shortcuts(server):
    sync_api = pytest.importorskip("playwright.sync_api")
    _make_font(fonts.user_folder() / "TestSans.ttf")
    fonts.listing(refresh=True)
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.click('a[data-t="compose"]')
        page.wait_for_function("typeof Compose !== 'undefined' && Compose._S.ready")
        assert page.inner_text('[data-ct="pen"] i') == "P"
        page.click('[data-ct="text"]')
        page.wait_for_function("Compose._S.fonts && Compose._S.fonts.length")
        box = page.eval_on_selector("#ccanvas", "e => { const r = e.getBoundingClientRect(); return [r.x + 200, r.y + 150]; }")
        page.mouse.click(*box)
        page.wait_for_selector("#cwrap textarea")
        page.wait_for_function("document.activeElement && document.activeElement.tagName === 'TEXTAREA'")
        page.keyboard.type("AA")
        page.keyboard.press("Escape")
        page.wait_for_function("Compose._api.getL().art.children.length === 1", timeout=20000)
        assert page.evaluate("Compose._api.getL().art.children[0].data.text.text") == "AA"
        # layers show a thumbnail and the selected row is highlighted
        page.wait_for_selector("#clayers .lyr.on .th svg")
        # double-click the text with the select tool to edit it in place
        page.click('[data-ct="select"]')
        c = page.evaluate("(() => { const it = Compose._api.getL().art.children[0]; const p = paper.view.projectToView(it.bounds.leftCenter.add([it.bounds.width * 0.1, 0])); const r = document.querySelector('#ccanvas').getBoundingClientRect(); return [r.x + p.x, r.y + p.y]; })()")
        page.mouse.dblclick(*c)
        page.wait_for_selector("#cwrap textarea")
        page.wait_for_function("document.activeElement && document.activeElement.tagName === 'TEXTAREA'")
        page.keyboard.type("A")
        page.keyboard.press("Escape")
        page.wait_for_function("Compose._api.getL().art.children[0].data.text.text.length === 3 || Compose._api.getL().art.children[0].data.text.text.includes('AAA')", timeout=20000)
        assert not errs, errs
        b.close()


def test_names_with_html_are_shown_as_text(server):
    """A graphic's name comes from its file name; it must be shown as written, never run as part of the page."""
    sync_api = pytest.importorskip("playwright.sync_api")
    name = "<img src=x onerror=window.__ran=1>"     # not a valid Windows file name, so it is handed to the browser directly
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.click('a[data-t="compose"]')
        page.wait_for_function("typeof Compose !== 'undefined' && Compose._S.ready")
        page.set_input_files("#cfile", {"name": name + ".svg", "mimeType": "image/svg+xml", "buffer": SVG.encode()})
        page.wait_for_function("Compose._S.sel.length === 1")
        page.wait_for_timeout(300)
        assert not page.evaluate("window.__ran === 1")
        assert name in page.evaluate("[...document.querySelectorAll('#side .bh')].map(e => e.textContent)")
        assert not errs, errs
        b.close()


def test_groups_and_fill_tool_in_the_browser(server):
    sync_api = pytest.importorskip("playwright.sync_api")
    ring = '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60"><path fill-rule="evenodd" d="M5 5H55V55H5ZM15 15V45H45V15Z"/></svg>'
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        pg_ = b.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        pg_.on("pageerror", lambda e: errors.append(str(e)))
        pg_.goto(server); pg_.click('#nav a[data-t="compose"]'); pg_.wait_for_timeout(500)
        pg_.evaluate("s => { Compose._api.addSvg(s, 'One'); Compose._api.addSvg(s, 'Two'); }", ring)
        # group two graphics: one new row with a caret, children listed, then ungroup
        pg_.evaluate("() => { const L = Compose._api.getL(); Compose._api.select(L.art.children.slice()); Compose._api.group(); }")
        assert pg_.evaluate("Compose._api.getL().art.children.length") == 1
        assert pg_.locator("#clayers .lyr").count() == 3          # the group and its two children
        pg_.keyboard.press("Control+Shift+G")
        assert pg_.evaluate("Compose._api.getL().art.children.length") == 2
        pg_.evaluate("() => { const L = Compose._api.getL(); Compose._api.select(L.art.children.slice()); }")
        pg_.keyboard.press("Control+G")
        assert pg_.evaluate("Compose._api.getL().art.children.length") == 1
        pg_.keyboard.press("Control+Shift+G")
        pg_.evaluate("() => { const L = Compose._api.getL(); L.art.children[1].remove(); Compose._api.commit(); Compose._api.select([]); }")
        # Fill: click the hollow middle of the ring
        pg_.keyboard.press("f")
        assert pg_.locator("[data-ct=fill].on").count() == 1
        c = pg_.evaluate("() => { const it = Compose._api.getL().art.children[0], b = it.bounds; return [b.center.x, b.center.y]; }")
        pg_.evaluate("([x, y]) => Compose._api.fillAt(new paper.Point(x, y), false)", c)
        pg_.wait_for_function("Compose._api.getL().art.children.length === 2")
        info = pg_.evaluate("() => { const L = Compose._api.getL(), f = L.art.children[0], r = L.art.children[1]; return [f.data.kind, f.fillColor.toCSS(true), f.index, r.index, f.bounds.width, r.bounds.width]; }")
        assert info[0] == "fill" and info[1] == "#ffffff" and info[2] == 0 and info[3] == 1      # behind the ring
        assert info[5] * 0.6 < info[4] < info[5] * 0.6 + 1.0                                           # the hole is half the ring wide, grown a little
        # clicking the fill again recolours it; clicking outside the ring says it is open
        pg_.evaluate("([x, y]) => Compose._api.fillAt(new paper.Point(x, y), false)", c)
        assert pg_.evaluate("Compose._api.getL().art.children.length") == 2
        pg_.evaluate("() => Compose._api.fillAt(new paper.Point(4, 4), false)")
        pg_.wait_for_function("document.querySelector('#toast') && /open to the outside/.test(document.querySelector('#toast').textContent)")
        assert not errors, errors
        b.close()


def test_position_and_size_boxes(server):
    """Typing X then Y moves to both; a flat line survives the boxes; the boxes follow a drag."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1400, "height": 900})
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(server + "/")
        page.click('a[data-t="compose"]')
        page.wait_for_function("typeof Compose !== 'undefined' && Compose._S.ready")
        bounds = "(() => { const b = Compose._api.getL().art.lastChild.bounds; return [b.x, b.y, b.width, b.height].map(v => Math.round(v * 10) / 10); })()"
        def type_in(label, value):
            n = page.evaluate("l => [...document.querySelectorAll('.cell input')].findLastIndex(i => (i.closest('label,div') || {}).textContent.trim() === l)", label)   # the artboard has its own Width / Height above
            assert n >= 0, label
            i = page.locator(".cell input").nth(n)
            i.fill(str(value))
            i.press("Enter")

        page.evaluate("""(() => { const A = Compose._api, L = A.getL();
          const r = new paper.Path.Rectangle({point: [10, 10], size: [40, 30], fillColor: '#000'}); L.art.addChild(r); A.commit(); A.select([r]); })()""")
        units = page.evaluate("[...document.querySelectorAll('.cell input')].findLast(i => (i.closest('label,div') || {}).textContent.trim() === 'X').value")
        k = float(units) / 10                                              # the boxes may show cm or mm
        type_in("X", 100 * k)
        type_in("Y", 80 * k)
        assert page.evaluate(bounds)[:2] == [100, 80]                       # typing Y used to put X back
        type_in("Width", 60 * k)
        assert page.evaluate(bounds) == [100, 80, 60, 45]                   # proportions kept, position kept

        page.evaluate("""(() => { const A = Compose._api, L = A.getL(); L.art.removeChildren();
          const l = new paper.Path.Line({from: [20, 50], to: [120, 50], strokeColor: '#000', strokeWidth: 1}); L.art.addChild(l); A.commit(); A.select([l]); })()""")
        type_in("X", 30 * k)
        x, y, w, h = page.evaluate(bounds)
        assert (x, y, w, h) == (30, 50, 100, 0)                             # a flat line used to vanish (its points became NaN)
        assert "NaN" not in page.evaluate("Compose._api.svgString()")
        assert not errs, errs
        b.close()
