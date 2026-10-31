import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from repolore.refresh import (
    LOCK_FILE,
    LOG_FILE,
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


class FakePopen:
    calls: list[tuple[list[str], dict[str, Any]]]

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        FakePopen.calls.append((command, kwargs))


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict[str, Any]]]:
    FakePopen.calls = []
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return FakePopen.calls


def test_a_refresh_is_started_as_a_quiet_detached_process(
    tmp_path: Path, spawned: list[tuple[list[str], dict[str, Any]]]
) -> None:
    data_dir = tmp_path / "data"

    assert start_refresh(data_dir, "experiment")

    [(command, options)] = spawned
    assert command == [sys.executable, "-m", "repolore", "--data-dir", str(data_dir), "refresh", "--db", "experiment"]
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stdout"].name == str(data_dir / LOG_FILE)
    assert options["stderr"] == subprocess.STDOUT
    assert options["env"]["TQDM_DISABLE"] == "1"
    assert options["env"]["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert (data_dir / LOG_FILE).exists()


def test_no_second_refresh_is_started_while_one_runs(
    tmp_path: Path, spawned: list[tuple[list[str], dict[str, Any]]]
) -> None:
    with refreshing(tmp_path):
        assert not start_refresh(tmp_path, "default")

    assert spawned == []


WINDOWS_FLAGS = {
    "CREATE_NO_WINDOW": 0x08000000,
    "CREATE_NEW_PROCESS_GROUP": 0x200,
    "CREATE_BREAKAWAY_FROM_JOB": 0x01000000,
}


@pytest.fixture
def on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    for name, value in WINDOWS_FLAGS.items():
        monkeypatch.setattr(subprocess, name, value, raising=False)


@pytest.mark.usefixtures("on_windows")
def test_on_windows_the_child_leaves_the_servers_job_and_console(
    tmp_path: Path, spawned: list[tuple[list[str], dict[str, Any]]]
) -> None:
    start_refresh(tmp_path, "default")

    [(_, options)] = spawned
    assert options["creationflags"] == 0x08000000 | 0x200 | 0x01000000
    assert "start_new_session" not in options


@pytest.mark.usefixtures("on_windows")
def test_a_job_that_forbids_breaking_away_keeps_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawned: list[tuple[list[str], dict[str, Any]]]
) -> None:
    def popen(command: list[str], **options: Any) -> None:
        if options["creationflags"] & 0x01000000:
            raise PermissionError(13, "Access is denied")
        spawned.append((command, options))

    monkeypatch.setattr(subprocess, "Popen", popen)

    start_refresh(tmp_path, "default")

    [(_, options)] = spawned
    assert options["creationflags"] == 0x08000000 | 0x200


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_elsewhere_the_child_gets_its_own_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spawned: list[tuple[list[str], dict[str, Any]]], platform: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)

    start_refresh(tmp_path, "default")

    [(_, options)] = spawned
    assert options["start_new_session"] is True
    assert "creationflags" not in options


def test_a_refresh_that_was_just_started_counts_as_running(
    tmp_path: Path, spawned: list[tuple[list[str], dict[str, Any]]]
) -> None:
    assert start_refresh(tmp_path, "default")

    assert refresh_status(tmp_path).running
    assert not start_refresh(tmp_path, "default")
    assert len(spawned) == 1


def test_the_started_process_takes_over_from_the_request(
    tmp_path: Path, spawned: list[tuple[list[str], dict[str, Any]]]
) -> None:
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
    def popen(*_: object, **__: object) -> None:
        raise FileNotFoundError(2, "No such file")

    monkeypatch.setattr(subprocess, "Popen", popen)

    with pytest.raises(FileNotFoundError):
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
