import os
import stat
import sys
from pathlib import Path

import pytest

from repolore.locks import hold
from repolore.server_state import (
    Discovery,
    ServerFiles,
    connected_relays,
    is_running,
    read_discovery,
    registered,
    request_stop,
    write_discovery,
)

DISCOVERY = Discovery(pid=42, port=61000, token="secret", version="1.2.3", started_at="2026-11-01T12:00:00+00:00")


def test_the_discovery_file_round_trips_and_names_the_url(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path, "default")

    write_discovery(files, DISCOVERY)

    assert read_discovery(files) == DISCOVERY
    assert DISCOVERY.url == "http://127.0.0.1:61000/mcp"
    assert not files.discovery.with_suffix(".tmp").exists()


def test_a_newer_discovery_replaces_the_old_one(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path, "default")
    write_discovery(files, DISCOVERY)

    write_discovery(files, DISCOVERY.model_copy(update={"port": 62000}))

    found = read_discovery(files)
    assert found is not None
    assert found.port == 62000


@pytest.mark.skipif(sys.platform == "win32", reason="Windows keeps the data dir private through the user's profile")
def test_only_the_user_can_read_the_token(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path, "default")

    write_discovery(files, DISCOVERY)

    assert stat.S_IMODE(os.stat(files.discovery).st_mode) == 0o600


@pytest.mark.parametrize("written", [None, "", '{"pid": 1', '{"pid": 1}'])
def test_a_missing_or_partial_discovery_file_reads_as_none(tmp_path: Path, written: str | None) -> None:
    files = ServerFiles(tmp_path, "default")
    if written is not None:
        files.discovery.write_text(written, encoding="utf-8")

    assert read_discovery(files) is None


def test_each_index_has_its_own_files(tmp_path: Path) -> None:
    default, other = ServerFiles(tmp_path, "default"), ServerFiles(tmp_path, "default-2")

    assert {default.lock, default.discovery, default.stop, default.log}.isdisjoint(
        {other.lock, other.discovery, other.stop, other.log}
    )
    assert default.client_lock(7).parent != other.client_lock(7).parent


def test_the_server_runs_while_it_holds_its_lock(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path, "default")
    assert not is_running(files)

    with hold(files.lock, timeout=0):
        assert is_running(files)


def test_a_stop_request_is_a_file(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path / "new", "default")

    request_stop(files)

    assert files.stop.exists()


def test_no_relay_is_connected_before_any_registers(tmp_path: Path) -> None:
    assert connected_relays(ServerFiles(tmp_path, "default")) == 0


def test_a_relay_counts_while_registered(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path, "default")

    with registered(files):
        assert connected_relays(files) == 1
        assert connected_relays(ServerFiles(tmp_path, "other")) == 0

    assert connected_relays(files) == 0


def test_many_relays_count_and_dead_ones_are_cleared(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path, "default")
    files.clients.mkdir(parents=True)
    dead = [files.client_lock(pid) for pid in (1, 2, 3)]
    for lock_file in dead:
        lock_file.touch()
    held = [hold(files.client_lock(pid), timeout=0) for pid in (11, 12)]
    for lock in held:
        lock.acquire()

    try:
        assert connected_relays(files) == 2
        assert not any(lock_file.exists() for lock_file in dead)
    finally:
        for lock in held:
            lock.release()
