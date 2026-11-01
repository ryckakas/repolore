import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO

import pytest

from repolore import refresh
from repolore.processes import SpawnError
from repolore.progress import Phase, Progress
from repolore.refresh import (
    LOCK_FILE,
    LOG_FILE,
    REPLACE_ATTEMPTS,
    REQUEST_FILE,
    STARTUP_SECONDS,
    RefreshRunning,
    RefreshStatus,
    refresh_status,
    refreshing,
    start_refresh,
)


def test_a_refresh_records_when_it_ran_and_that_it_worked(tmp_path: Path) -> None:
    with refreshing(tmp_path):
        during = refresh_status(tmp_path)

    after = refresh_status(tmp_path)
    assert during.running
    assert during.started_at is not None
    assert during.finished_at is None
    assert not after.running
    assert after.started_at == during.started_at
    assert after.finished_at is not None
    assert (after.ok, after.error) == (True, None)


def test_without_a_refresh_there_is_no_status(tmp_path: Path) -> None:
    assert refresh_status(tmp_path) == RefreshStatus()


def test_a_second_refresh_is_refused_while_one_runs(tmp_path: Path) -> None:
    with refreshing(tmp_path), pytest.raises(RefreshRunning), refreshing(tmp_path):
        pass

    assert refresh_status(tmp_path).ok is True


def test_a_failed_refresh_records_why_and_frees_the_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="boom"), refreshing(tmp_path):
        raise ValueError("boom")

    failed = refresh_status(tmp_path)
    assert (failed.running, failed.ok, failed.error) == (False, False, "boom")
    with refreshing(tmp_path):
        pass
    assert refresh_status(tmp_path).ok is True


def test_an_error_without_a_message_is_named_by_its_type(tmp_path: Path) -> None:
    with pytest.raises(KeyboardInterrupt), refreshing(tmp_path):
        raise KeyboardInterrupt

    assert refresh_status(tmp_path).error == "KeyboardInterrupt"


def test_a_lock_file_left_by_a_killed_refresh_does_not_block(tmp_path: Path) -> None:
    (tmp_path / LOCK_FILE).write_text("", encoding="utf-8")

    assert not refresh_status(tmp_path).running
    with refreshing(tmp_path):
        assert refresh_status(tmp_path).running


def test_the_log_of_the_last_detached_run_is_shown(tmp_path: Path) -> None:
    with refreshing(tmp_path):
        lines = [f"line {number}" for number in range(8)]
        (tmp_path / LOG_FILE).write_text("\n".join(lines[:4]) + "\n\n" + "\n".join(lines[4:]), encoding="utf-8")

    assert refresh_status(tmp_path).log_tail == ["line 3", "line 4", "line 5", "line 6", "line 7"]


def test_a_log_older_than_the_last_refresh_is_not_shown(tmp_path: Path) -> None:
    log = tmp_path / LOG_FILE
    log.write_text("an earlier detached run\n", encoding="utf-8")
    an_hour_ago = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(log, (an_hour_ago, an_hour_ago))

    with refreshing(tmp_path):
        pass

    assert refresh_status(tmp_path).log_tail == []


def test_a_log_without_a_recorded_refresh_is_not_shown(tmp_path: Path) -> None:
    (tmp_path / LOG_FILE).write_text("output\n", encoding="utf-8")

    assert refresh_status(tmp_path).log_tail == []


Spawned = list[tuple[list[str], IO[str], Mapping[str, str], Path]]


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> Spawned:
    calls: Spawned = []

    def spawn(command: list[str], *, stdout: IO[str], env: Mapping[str, str], cwd: Path) -> None:
        calls.append((command, stdout, env, cwd))

    monkeypatch.setattr(refresh, "spawn_detached", spawn)
    return calls


def test_a_refresh_is_started_as_a_quiet_detached_process(tmp_path: Path, spawned: Spawned) -> None:
    data_dir = tmp_path / "data"

    assert start_refresh(data_dir, "experiment")

    [(command, stdout, env, cwd)] = spawned
    assert command == [
        *[sys.executable, "-P", "-m", "repolore", "--data-dir", str(data_dir)],
        *["refresh", "--db", "experiment"],
    ]
    assert cwd == data_dir
    assert stdout.name == str(data_dir / LOG_FILE)
    assert env["TQDM_DISABLE"] == "1"
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert (data_dir / LOG_FILE).exists()


def test_no_second_refresh_is_started_while_one_runs(tmp_path: Path, spawned: Spawned) -> None:
    with refreshing(tmp_path):
        assert not start_refresh(tmp_path, "default")

    assert spawned == []


def test_a_refresh_that_was_just_started_counts_as_running(tmp_path: Path, spawned: Spawned) -> None:
    assert start_refresh(tmp_path, "default")

    assert refresh_status(tmp_path).running
    assert not start_refresh(tmp_path, "default")
    assert len(spawned) == 1


def test_the_started_process_takes_over_from_the_request(tmp_path: Path, spawned: Spawned) -> None:
    start_refresh(tmp_path, "default")

    with refreshing(tmp_path):
        assert not (tmp_path / REQUEST_FILE).exists()
    assert not refresh_status(tmp_path).running


def test_a_request_whose_process_never_started_expires(tmp_path: Path) -> None:
    request = tmp_path / REQUEST_FILE
    request.touch()
    long_ago = datetime.now(UTC).timestamp() - STARTUP_SECONDS - 1
    os.utime(request, (long_ago, long_ago))

    assert not refresh_status(tmp_path).running


def test_a_failed_spawn_withdraws_the_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def spawn(*_: object, **__: object) -> None:
        raise SpawnError("Couldn't start python")

    monkeypatch.setattr(refresh, "spawn_detached", spawn)

    with pytest.raises(SpawnError):
        start_refresh(tmp_path, "default")

    assert not refresh_status(tmp_path).running


@pytest.mark.parametrize("written", ["", '{"started_at": "2026-10-31T'])
def test_a_status_file_cut_short_reads_as_no_status(tmp_path: Path, written: str) -> None:
    (tmp_path / "refresh.json").write_text(written, encoding="utf-8")

    assert refresh_status(tmp_path) == RefreshStatus()


def test_the_cause_of_a_failure_is_recorded(tmp_path: Path) -> None:
    with pytest.raises(SystemExit), refreshing(tmp_path):
        raise SystemExit(1) from RuntimeError("Nothing cached")

    assert refresh_status(tmp_path).error == "Nothing cached"


def test_progress_is_saved_when_the_phase_changes(tmp_path: Path) -> None:
    with refreshing(tmp_path, interval=3600) as report:
        report(Progress(phase=Phase.INGEST, done=0, total=2))
        first = refresh_status(tmp_path).progress
        report(Progress(phase=Phase.INGEST, done=1, total=2))
        same_phase = refresh_status(tmp_path).progress
        report(Progress(phase=Phase.EMBED, done=0, total=9))
        next_phase = refresh_status(tmp_path).progress

    assert first == Progress(phase=Phase.INGEST, done=0, total=2)
    assert same_phase == first
    assert next_phase == Progress(phase=Phase.EMBED, done=0, total=9)
    assert refresh_status(tmp_path).progress is None


def test_progress_in_one_phase_is_saved_once_the_interval_has_passed(tmp_path: Path) -> None:
    with refreshing(tmp_path, interval=0) as report:
        report(Progress(phase=Phase.EMBED, done=0, total=9))
        report(Progress(phase=Phase.EMBED, done=4, total=9))
        during = refresh_status(tmp_path)

    assert during.progress == Progress(phase=Phase.EMBED, done=4, total=9)
    assert during.running
    assert during.started_at == refresh_status(tmp_path).started_at


def test_a_failed_refresh_keeps_no_progress(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), refreshing(tmp_path) as report:
        report(Progress(phase=Phase.EMBED, done=0, total=9))
        raise RuntimeError("out of memory")

    status = refresh_status(tmp_path)
    assert (status.ok, status.error, status.progress) == (False, "out of memory", None)


class HeldOpen:
    """Path.replace refusing, as on Windows while a reader has the target open, the given number of times."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.refusals = 0
        self.attempts = 0
        self._replace = Path.replace
        monkeypatch.setattr(refresh, "REPLACE_WAIT_SECONDS", 0)
        monkeypatch.setattr(Path, "replace", lambda partial, target: self(partial, target))

    def __call__(self, partial: Path, target: Path) -> Path:
        self.attempts += 1
        if self.attempts <= self.refusals:
            raise PermissionError("[WinError 5] Access is denied")
        return self._replace(partial, target)


@pytest.fixture
def held_open(monkeypatch: pytest.MonkeyPatch) -> HeldOpen:
    return HeldOpen(monkeypatch)


def test_a_status_another_process_has_open_is_replaced_once_it_is_closed(tmp_path: Path, held_open: HeldOpen) -> None:
    held_open.refusals = REPLACE_ATTEMPTS - 1

    with refreshing(tmp_path):
        pass

    assert refresh_status(tmp_path).ok
    assert held_open.attempts == REPLACE_ATTEMPTS + 1


def test_a_status_still_open_after_every_attempt_fails_the_refresh(tmp_path: Path, held_open: HeldOpen) -> None:
    held_open.refusals = REPLACE_ATTEMPTS

    with pytest.raises(PermissionError), refreshing(tmp_path):
        pass

    assert refresh_status(tmp_path) == RefreshStatus()
    assert held_open.attempts == REPLACE_ATTEMPTS


def test_progress_that_cant_be_saved_is_logged_and_the_refresh_goes_on(
    tmp_path: Path, held_open: HeldOpen, caplog: pytest.LogCaptureFixture
) -> None:
    with refreshing(tmp_path) as report:
        held_open.refusals = held_open.attempts + REPLACE_ATTEMPTS
        report(Progress(phase=Phase.EMBED, done=0, total=9))
        held_open.refusals = 0

    assert "progress not saved: [WinError 5] Access is denied" in caplog.text
    assert refresh_status(tmp_path).ok


def test_the_end_of_a_phase_is_saved_within_the_interval(tmp_path: Path) -> None:
    with refreshing(tmp_path, interval=3600) as report:
        report(Progress(phase=Phase.INGEST, done=0, total=2))
        report(Progress(phase=Phase.INGEST, done=1, total=2))
        report(Progress(phase=Phase.INGEST, done=2, total=2))
        ended = refresh_status(tmp_path).progress

    assert ended == Progress(phase=Phase.INGEST, done=2, total=2)


def test_a_status_being_replaced_reads_as_no_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with refreshing(tmp_path):
        pass

    def being_replaced(*_: object, **__: object) -> str:
        raise PermissionError("[WinError 32] The process cannot access the file")

    monkeypatch.setattr(Path, "read_text", being_replaced)

    assert refresh_status(tmp_path) == RefreshStatus()
