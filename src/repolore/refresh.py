import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from pydantic import BaseModel, ValidationError

LOCK_FILE = "refresh.lock"
STATUS_FILE = "refresh.json"
LOG_FILE = "refresh.log"
REQUEST_FILE = "refresh.requested"
# How long a started refresh counts as running before its process holds the lock: Python and torch take seconds to load.
STARTUP_SECONDS = 120
LOG_TAIL_LINES = 5
# The detached run writes to a log file, where progress bars would only add carriage-return noise.
QUIET = {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1"}


class RefreshRunning(Exception):
    pass


class RefreshStatus(BaseModel):
    running: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    ok: bool | None = None
    error: str | None = None
    log_tail: list[str] = []


@contextmanager
def refreshing(data_dir: Path) -> Generator[None]:
    """Hold the refresh lock and record the run's start, end and outcome, so only one refresh runs at a time."""
    data_dir.mkdir(parents=True, exist_ok=True)
    # An OS lock, so a refresh that crashed or was killed never leaves a stale lock behind.
    lock = FileLock(data_dir / LOCK_FILE, timeout=1, fallback_to_soft=False)
    try:
        lock.acquire()
    except Timeout:
        raise RefreshRunning("A refresh is already running.") from None
    try:
        (data_dir / REQUEST_FILE).unlink(missing_ok=True)
        started = _now()
        _save(data_dir, RefreshStatus(started_at=started))
        try:
            yield
        except BaseException as error:
            reason = str(error.__cause__ or error) or type(error).__name__
            _save(data_dir, RefreshStatus(started_at=started, finished_at=_now(), ok=False, error=reason))
            raise
        _save(data_dir, RefreshStatus(started_at=started, finished_at=_now(), ok=True))
    finally:
        lock.release()


def refresh_status(data_dir: Path) -> RefreshStatus:
    try:
        status = RefreshStatus.model_validate_json((data_dir / STATUS_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValidationError):
        # Missing before the first refresh; empty or cut short if read mid-write, or if a run was killed writing it.
        status = RefreshStatus()
    return status.model_copy(update={"running": _is_running(data_dir), "log_tail": _log_tail(data_dir, status)})


def start_refresh(data_dir: Path, db: str) -> bool:
    """Start `repolore refresh` in a detached process that outlives this one; False if one is already running."""
    if _is_running(data_dir):
        return False
    data_dir.mkdir(parents=True, exist_ok=True)
    request = data_dir / REQUEST_FILE
    request.touch()
    command = [sys.executable, "-m", "repolore", "--data-dir", str(data_dir), "refresh", "--db", db]
    try:
        with (data_dir / LOG_FILE).open("w", encoding="utf-8") as log:
            _spawn_detached(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=os.environ | QUIET
            )
    except OSError:
        request.unlink()
        raise
    return True


def _spawn_detached(command: list[str], **options: Any) -> None:
    if sys.platform != "win32":
        subprocess.Popen(command, start_new_session=True, **options)
        return
    # No console window for the child or for the `gh auth token` it runs, and no Ctrl+C from the server's group.
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        # Out of the client's job object, whose closing would otherwise kill the refresh along with the server.
        subprocess.Popen(command, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **options)
    except PermissionError:
        # A job that forbids breaking away, as the Python MCP SDK's client makes, keeps the refresh in it.
        subprocess.Popen(command, creationflags=flags, **options)


def _is_running(data_dir: Path) -> bool:
    return _is_starting(data_dir) or _is_locked(data_dir)


def _is_starting(data_dir: Path) -> bool:
    request = data_dir / REQUEST_FILE
    return request.exists() and datetime.now().timestamp() - request.stat().st_mtime < STARTUP_SECONDS


def _is_locked(data_dir: Path) -> bool:
    lock_file = data_dir / LOCK_FILE
    if not lock_file.exists():
        return False
    try:
        with FileLock(lock_file, blocking=False, fallback_to_soft=False):
            return False
    except Timeout:
        return True


def _log_tail(data_dir: Path, status: RefreshStatus) -> list[str]:
    log = data_dir / LOG_FILE
    if status.started_at is None or not log.exists():
        return []
    # A refresh run from a terminal leaves the log of an earlier detached run, which isn't about this run.
    written = datetime.fromtimestamp(log.stat().st_mtime, UTC)
    if written < datetime.fromisoformat(status.started_at):
        return []
    lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-LOG_TAIL_LINES:]


def _save(data_dir: Path, status: RefreshStatus) -> None:
    (data_dir / STATUS_FILE).write_text(status.model_dump_json(exclude={"running", "log_tail"}), encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
