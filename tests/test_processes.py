import os
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from repolore.processes import INTERMEDIATE, SpawnError, spawn_detached

NO_WINDOW = 0x08000000
NEW_GROUP = 0x200
BREAKAWAY = 0x01000000
COMMAND = ["python", "-m", "repolore", "refresh"]
ENV = {"TQDM_DISABLE": "1"}


class FakePopen:
    calls: list[tuple[list[str], dict[str, Any]]]
    exit_code = 0

    def __init__(self, command: list[str], **options: Any) -> None:
        FakePopen.calls.append((command, options))

    def wait(self) -> int:
        return FakePopen.exit_code


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict[str, Any]]]:
    FakePopen.calls = []
    FakePopen.exit_code = 0
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return FakePopen.calls


@pytest.fixture
def on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    flags = {
        "CREATE_NO_WINDOW": NO_WINDOW,
        "CREATE_NEW_PROCESS_GROUP": NEW_GROUP,
        "CREATE_BREAKAWAY_FROM_JOB": BREAKAWAY,
    }
    for name, value in flags.items():
        monkeypatch.setattr(subprocess, name, value, raising=False)


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_elsewhere_the_process_gets_its_own_session(
    monkeypatch: pytest.MonkeyPatch, spawned: list[tuple[list[str], dict[str, Any]]], platform: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    log = StringIO()

    spawn_detached(COMMAND, stdout=log, env=ENV)

    [(command, options)] = spawned
    assert command == COMMAND
    assert options == {
        "start_new_session": True,
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "env": ENV,
    }


@pytest.mark.usefixtures("on_windows")
def test_on_windows_an_intermediate_starts_the_process_out_of_the_job_and_console(
    spawned: list[tuple[list[str], dict[str, Any]]],
) -> None:
    log = StringIO()

    spawn_detached(COMMAND, stdout=log, env=ENV)

    flags = NO_WINDOW | NEW_GROUP | BREAKAWAY
    [(command, options)] = spawned
    assert command == [sys.executable, "-c", INTERMEDIATE, str(flags), *COMMAND]
    assert options == {
        "creationflags": flags,
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "env": ENV,
    }


@pytest.mark.usefixtures("on_windows")
def test_a_job_that_forbids_breaking_away_keeps_the_process(
    monkeypatch: pytest.MonkeyPatch, spawned: list[tuple[list[str], dict[str, Any]]]
) -> None:
    def popen(command: list[str], **options: Any) -> FakePopen:
        if options["creationflags"] & BREAKAWAY:
            raise PermissionError(13, "Access is denied")
        return FakePopen(command, **options)

    monkeypatch.setattr(subprocess, "Popen", popen)

    spawn_detached(COMMAND, stdout=StringIO(), env=ENV)

    [(command, options)] = spawned
    assert options["creationflags"] == NO_WINDOW | NEW_GROUP
    assert command[3] == str(NO_WINDOW | NEW_GROUP)


@pytest.mark.usefixtures("on_windows")
def test_an_intermediate_that_fails_says_what_did_not_start(spawned: list[tuple[list[str], dict[str, Any]]]) -> None:
    FakePopen.exit_code = 1

    with pytest.raises(SpawnError, match=r"Couldn't start python: .* exited with code 1"):
        spawn_detached(COMMAND, stdout=StringIO(), env=ENV)


HEARTBEAT = (
    "import os, time\n"
    "print(os.getpid(), flush=True)\n"
    "for beat in range(300):\n"
    "    print(beat, flush=True)\n"
    "    time.sleep(0.1)"
)
SESSION_SERVER = (
    "import os, sys, time\n"
    "from repolore.processes import spawn_detached\n"
    "with open(sys.argv[1], 'w', encoding='utf-8') as log:\n"
    "    spawn_detached([sys.executable, '-c', sys.argv[2]], stdout=log, env=os.environ)\n"
    "print('spawned', flush=True)\n"
    "time.sleep(60)"
)


@pytest.mark.skipif(sys.platform != "win32", reason="Claude Code's taskkill /T is how Windows ends a session's server")
def test_on_windows_the_process_survives_the_session_killing_its_server_tree(tmp_path: Path) -> None:
    log = tmp_path / "detached.log"
    pid = None
    with subprocess.Popen(
        [sys.executable, "-c", SESSION_SERVER, str(log), HEARTBEAT], stdout=subprocess.PIPE, encoding="utf-8"
    ) as server:
        try:
            assert server.stdout is not None
            assert server.stdout.readline().strip() == "spawned"
            pid = _wait_for_beats(log, at_least=2)[0]

            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], check=True, capture_output=True)
            server.wait()
            beats_after_kill = len(_wait_for_beats(log, at_least=1))

            _wait_for_beats(log, at_least=beats_after_kill + 5)
        finally:
            server.kill()
            if pid is not None:
                os.kill(pid, 9)


def _wait_for_beats(log: Path, *, at_least: int) -> list[int]:
    deadline = time.monotonic() + 30
    while len(beats := [int(beat) for beat in log.read_text(encoding="utf-8").split()]) < at_least:
        if time.monotonic() > deadline:
            raise AssertionError(f"The detached process stopped after {len(beats)} lines.")
        time.sleep(0.1)
    return beats
