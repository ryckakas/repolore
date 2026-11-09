import time
from pathlib import Path

from filelock import FileLock, Timeout

REPLACE_ATTEMPTS = 20
REPLACE_WAIT_SECONDS = 0.05


def hold(lock_file: Path, timeout: float) -> FileLock:
    """An OS lock, so a process that crashed or was killed never leaves a stale lock behind."""
    return FileLock(lock_file, timeout=timeout, fallback_to_soft=False)


def is_locked(lock_file: Path) -> bool:
    if not lock_file.exists():
        return False
    try:
        with FileLock(lock_file, blocking=False, fallback_to_soft=False):
            return False
    except Timeout:
        return True


def replace_when_free(partial: Path, target: Path) -> None:
    """Move a finished file over one that readers poll."""
    # Windows refuses to replace a file another process has open, which a polling reader does for a moment.
    for _ in range(REPLACE_ATTEMPTS - 1):
        try:
            partial.replace(target)
            return
        except PermissionError:
            time.sleep(REPLACE_WAIT_SECONDS)
    partial.replace(target)
