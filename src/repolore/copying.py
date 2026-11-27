import ctypes
import fnmatch
import os
import shutil
import sqlite3
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, closing, nullcontext
from dataclasses import dataclass
from pathlib import Path

from repolore.paths import CONFIG_FILENAME, QUESTIONS_FILENAME, AppPaths
from repolore.refresh import REQUEST_FILE, no_refresh

DATABASE_SUFFIX = ".db"
# A database's journal and WAL are part of it only to SQLite, which the backup asks; a lock, an unfinished download, a
# half-written status, the shared server's address and token, and the relays' locks belong to the processes of the
# home being copied.
LEFT_BEHIND = ("*-wal", "*-shm", "*-journal", "*.lock", "*.tmp", "*.partial", "server-*", REQUEST_FILE)
# The settings come from where they're read, which needn't be the data dir; a copy there that isn't read is left out.
SETTINGS = (CONFIG_FILENAME, QUESTIONS_FILENAME)
CLIENTS_FOLDER = "clients"
# From <sys/ioctl.h> on Linux: _IOW(0x94, 9, int).
FICLONE = 0x40049409


class CopyError(Exception):
    pass


@dataclass(frozen=True)
class Copied:
    target: Path
    files: int
    cloned: int


def copy_home(paths: AppPaths, target: Path) -> Copied:
    """Copy the config and the data into an empty folder that REPOLORE_HOME can then name, holding the refresh lock so
    no refresh writes meanwhile. The databases are copied with SQLite's backup API, which a reader holding them open
    doesn't disturb, and the rest is cloned where the file system can. A failed copy removes what it copied; no file of
    the home being copied changes."""
    target = target.expanduser().resolve()
    _check_target(target, paths.data_dir)
    created = not target.exists()
    counter = _Counter()
    with _refreshes_held(paths.data_dir):
        try:
            target.mkdir(parents=True, exist_ok=True)
            if paths.data_dir.is_dir():
                shutil.copytree(
                    paths.data_dir, target, ignore=_left_behind, copy_function=counter.copy, dirs_exist_ok=True
                )
            _copy_setting(paths.config_file, target / CONFIG_FILENAME, counter)
            _copy_setting(paths.questions_file, target / "evals" / QUESTIONS_FILENAME, counter)
        except BaseException:
            _remove_copy(target, created)
            raise
    return Copied(target, counter.files, counter.cloned)


def _check_target(target: Path, data_dir: Path) -> None:
    if target == data_dir or target.is_relative_to(data_dir):
        raise CopyError(f"{target} is inside the home it would copy, {data_dir}.")
    if target.exists() and not target.is_dir():
        raise CopyError(f"{target} isn't a folder.")
    if target.exists() and any(target.iterdir()):
        raise CopyError(f"{target} isn't empty; copy into a new or empty folder.")


def _refreshes_held(data_dir: Path) -> AbstractContextManager[None]:
    # Without a data dir there's nothing a refresh could be writing, and taking the lock would create the folder.
    return no_refresh(data_dir) if data_dir.is_dir() else nullcontext()


def _left_behind(folder: str, names: list[str]) -> set[str]:
    left = {
        name for name in names if name in SETTINGS or any(fnmatch.fnmatch(name, pattern) for pattern in LEFT_BEHIND)
    }
    # The relays' locks are in clients/<db>/, which only a relay of the copied home should create.
    if CLIENTS_FOLDER in names and (Path(folder) / CLIENTS_FOLDER).is_dir():
        left.add(CLIENTS_FOLDER)
    return left


class _Counter:
    def __init__(self) -> None:
        self.files = 0
        self.cloned = 0

    def copy(self, source: str, target: str) -> str:
        if source.endswith(DATABASE_SUFFIX):
            back_up(Path(source), Path(target))
        elif clone_or_copy(Path(source), Path(target)):
            self.cloned += 1
        self.files += 1
        return target


def _copy_setting(source: Path, target: Path, counter: _Counter) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        counter.copy(str(source), str(target))


def back_up(source: Path, target: Path) -> None:
    """A consistent copy of a database, even while another process reads or writes it; read-only, so a hot journal
    left in the source is never rolled back there."""
    with (
        closing(sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)) as old,
        closing(sqlite3.connect(target)) as new,
    ):
        old.backup(new)


def clone_or_copy(source: Path, target: Path) -> bool:
    """Clone a file where the file system can (APFS, Btrfs, XFS), so the copy takes no space until one side changes,
    and copy it where it can't; True when it was cloned."""
    try:
        _clone(source, target)
    except OSError:
        shutil.copy2(source, target)
        return False
    shutil.copystat(source, target)
    return True


def _clone(source: Path, target: Path) -> None:
    if sys.platform == "darwin":
        _clonefile(source, target)
    elif sys.platform == "linux":
        _ficlone(source, target)
    else:
        raise OSError(f"no file clones on {sys.platform}")


def _clonefile(source: Path, target: Path) -> None:
    clonefile: Callable[[bytes, bytes, int], int] = ctypes.CDLL(None, use_errno=True).clonefile
    if clonefile(os.fsencode(source), os.fsencode(target), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(source))


def _ficlone(source: Path, target: Path) -> None:
    import fcntl

    with source.open("rb") as original, target.open("wb") as clone:
        fcntl.ioctl(clone.fileno(), FICLONE, original.fileno())


def _remove_copy(target: Path, created: bool) -> None:
    if created:
        shutil.rmtree(target, ignore_errors=True)
        return
    for child in target.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
