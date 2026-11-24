"""Taking turns over a folder when more than one Boot Tool is open on it.

The lock is the operating system's own (flock on Mac and Linux, a locked byte on Windows), held on a
small file in the folder for the moment a change takes. The system lets go of it when a program
ends, however it ends, so nothing is ever left locked.
"""
from __future__ import annotations

import collections
import contextlib
import errno
import os
import threading
import time

try:
    import fcntl
except ImportError:                      # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:                      # Mac, Linux
    msvcrt = None

WAIT = 10.0                              # how long (seconds) a request waits for another Boot Tool


class Busy(OSError):
    """Another Boot Tool kept the folder for too long."""


def _take(fd: int) -> None:
    if fcntl:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif msvcrt:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)


def _give_back(fd: int) -> None:
    try:
        if fcntl:
            fcntl.flock(fd, fcntl.LOCK_UN)
        elif msvcrt:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def _taken(e: OSError) -> bool:
    """Did taking the lock fail because someone else has it?"""
    return isinstance(e, BlockingIOError) or e.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK)


def _no_locks(e: OSError) -> bool:
    """Did it fail because this folder has no file locks at all (some network drives)?"""
    return e.errno in (errno.ENOLCK, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS)


class Turns:
    """Taking turns over one folder.

    Requests of this Boot Tool go one at a time, in the order they came, so none is overtaken. The one
    whose turn it is takes the folder's lock, waiting for another Boot Tool if need be. That waiting is
    counted once for everyone in the queue: requests that stood behind one that gave up have waited
    with it, and don't each wait all over again. Time spent behind this Boot Tool's own requests is
    not counted as waiting for another one."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._line: collections.deque = collections.deque()
        self._waited = 0.0               # seconds spent waiting for another Boot Tool, all told

    @contextlib.contextmanager
    def held(self, path, change: bool = True, wait: float | None = None):
        """Have the turn, and the folder's lock (the file at `path`), for the body. Yields True with
        the lock. A change raises Busy if another Boot Tool keeps the lock for longer than `wait`
        seconds (WAIT if not given), and the error if the lock file can't be opened (a folder that
        can't be written) or the lock can't be taken for another reason. Just looking (`change`
        False) goes ahead without the lock in those cases, and yields False. In a folder that has
        no file locks at all, everything goes ahead as if this were the only Boot Tool."""
        wait = WAIT if wait is None else wait
        me = object()
        with self._cond:
            self._line.append(me)
            since = self._waited
            try:
                while self._line[0] is not me:
                    self._cond.wait()
            except BaseException:
                self._line.remove(me)
                self._cond.notify_all()
                raise
        fd = None
        try:
            try:
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            except OSError:
                if change:
                    raise
            got = locked = False
            pause = 0.001
            while fd is not None:
                try:
                    _take(fd)
                    got = locked = True
                    break
                except OSError as e:
                    if _no_locks(e):
                        got = True       # carry on as the only Boot Tool
                        break
                    if not _taken(e):    # something is wrong with the lock itself
                        if change:
                            raise
                        break
                    if self._waited - since >= wait:
                        if change:
                            raise Busy("another Boot Tool is changing this folder; try again in a moment") from None
                        break
                    t = time.monotonic()
                    time.sleep(pause)
                    self._waited += time.monotonic() - t
                    pause = min(pause * 2, 0.02)
            try:
                yield got
            finally:
                if locked:
                    _give_back(fd)
        finally:
            if fd is not None:
                os.close(fd)
            with self._cond:
                self._line.remove(me)
                self._cond.notify_all()
