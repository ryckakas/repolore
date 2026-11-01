import logging
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from filelock import Timeout
from pydantic import BaseModel, Field, ValidationError

from repolore.locks import hold, is_locked
from repolore.processes import SAFE_PATH, spawn_detached
from repolore.progress import Phase, Progress, Report

LOCK_FILE = "refresh.lock"
STATUS_FILE = "refresh.json"
LOG_FILE = "refresh.log"
REQUEST_FILE = "refresh.requested"
# How long a started refresh counts as running before its process holds the lock: Python and torch take seconds to load.
STARTUP_SECONDS = 120
LOG_TAIL_LINES = 5
PROGRESS_SECONDS = 2.0
REPLACE_ATTEMPTS = 20
REPLACE_WAIT_SECONDS = 0.05
# The detached run writes to a log file, where progress bars would only add carriage-return noise.
QUIET = {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TQDM_DISABLE": "1"}

log = logging.getLogger(__name__)


class RefreshRunning(Exception):
    pass


class RefreshStatus(BaseModel):
    running: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    ok: bool | None = None
    error: str | None = None
    progress: Progress | None = Field(default=None, description="How far the run has got, written every few seconds.")
    log_tail: list[str] = []


@contextmanager
def refreshing(data_dir: Path, interval: float = PROGRESS_SECONDS) -> Generator[Report]:
    """Hold the refresh lock and record the run's start, progress, end and outcome, so only one refresh runs at a
    time."""
    data_dir.mkdir(parents=True, exist_ok=True)
    lock = hold(data_dir / LOCK_FILE, timeout=1)
    try:
        lock.acquire()
    except Timeout:
        raise RefreshRunning("A refresh is already running.") from None
    try:
        (data_dir / REQUEST_FILE).unlink(missing_ok=True)
        started = _now()
        _save(data_dir, RefreshStatus(started_at=started))
        try:
            yield _Recorder(data_dir, started, interval)
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
    except (OSError, ValidationError):
        # Missing before the first refresh, locked on Windows for the moment it is replaced, or garbled by a run of an
        # older version that wrote it in place.
        status = RefreshStatus()
    return status.model_copy(update={"running": _is_running(data_dir), "log_tail": _log_tail(data_dir, status)})


def start_refresh(data_dir: Path, db: str) -> bool:
    """Start `repolore refresh` in a detached process that outlives this one; False if one is already running."""
    if _is_running(data_dir):
        return False
    data_dir.mkdir(parents=True, exist_ok=True)
    request = data_dir / REQUEST_FILE
    request.touch()
    command = [sys.executable, *SAFE_PATH, "-m", "repolore", "--data-dir", str(data_dir), "refresh", "--db", db]
    try:
        with (data_dir / LOG_FILE).open("w", encoding="utf-8") as log:
            spawn_detached(command, stdout=log, env=os.environ | QUIET, cwd=data_dir)
    except OSError:
        request.unlink()
        raise
    return True


def _is_running(data_dir: Path) -> bool:
    return _is_starting(data_dir) or is_locked(data_dir / LOCK_FILE)


def _is_starting(data_dir: Path) -> bool:
    request = data_dir / REQUEST_FILE
    return request.exists() and datetime.now().timestamp() - request.stat().st_mtime < STARTUP_SECONDS


def _log_tail(data_dir: Path, status: RefreshStatus) -> list[str]:
    log_file = data_dir / LOG_FILE
    if status.started_at is None or not log_file.exists():
        return []
    # A refresh run from a terminal leaves the log of an earlier detached run, which isn't about this run.
    written = datetime.fromtimestamp(log_file.stat().st_mtime, UTC)
    if written < datetime.fromisoformat(status.started_at):
        return []
    lines = [line for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return lines[-LOG_TAIL_LINES:]


class _Recorder:
    """Saves the progress it is given when the phase changes or ends, or when the last save is interval seconds old.
    Progress is only a hint, so failing to save it doesn't fail the refresh."""

    def __init__(self, data_dir: Path, started: str, interval: float) -> None:
        self._data_dir = data_dir
        self._started = started
        self._interval = interval
        self._phase: Phase | None = None
        self._saved_at = 0.0

    def __call__(self, progress: Progress) -> None:
        now = time.monotonic()
        finished = progress.done == progress.total
        if progress.phase == self._phase and not finished and now - self._saved_at < self._interval:
            return
        self._phase, self._saved_at = progress.phase, now
        try:
            _save(self._data_dir, RefreshStatus(started_at=self._started, progress=progress))
        except OSError as error:
            log.warning("progress not saved: %s", error)


def _save(data_dir: Path, status: RefreshStatus) -> None:
    partial = data_dir / f"{STATUS_FILE}.tmp"
    partial.write_text(status.model_dump_json(exclude={"running", "log_tail"}), encoding="utf-8")
    # Windows refuses to replace a file another process has open, which a reader polling the status does for a moment.
    for _ in range(REPLACE_ATTEMPTS - 1):
        try:
            partial.replace(data_dir / STATUS_FILE)
            return
        except PermissionError:
            time.sleep(REPLACE_WAIT_SECONDS)
    partial.replace(data_dir / STATUS_FILE)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
