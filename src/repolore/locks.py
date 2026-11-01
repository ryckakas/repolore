from pathlib import Path

from filelock import FileLock, Timeout


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
