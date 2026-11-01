import os
import threading
import time
from pathlib import Path

import anyio
import httpx2
import pytest
from conftest import FakeEmbedder, SharedServer
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types.version import LATEST_MODERN_VERSION

from repolore import __version__
from repolore import shared_server as shared_server_module
from repolore.locks import hold
from repolore.paths import AppPaths
from repolore.server_state import Discovery, is_running, read_discovery, registered
from repolore.shared_server import StaticTokenVerifier

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
}
ACCEPT = {"Accept": "application/json, text/event-stream"}


async def _list_repos(found: Discovery) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {found.token}"}
    async with (
        httpx2.AsyncClient(headers=headers) as http,
        Client(streamable_http_client(found.url, http_client=http), mode=LATEST_MODERN_VERSION) as client,
    ):
        result = await client.call_tool("list_repos", {})
    assert result.structured_content is not None
    return result.structured_content


def test_the_server_says_where_it_listens_and_serves_the_index_with_its_token(shared_server: SharedServer) -> None:
    found = shared_server.start()

    assert found.pid == os.getpid()
    assert found.version == __version__
    assert is_running(shared_server.files)
    repos = anyio.run(_list_repos, found)
    assert [repo["repo"] for repo in repos["repos"]] == ["octo/charts", "octo/repo"]  # type: ignore[index]


def test_each_start_gets_a_new_token(
    data_paths: AppPaths, fake_embedder: FakeEmbedder, shared_server: SharedServer
) -> None:
    first = shared_server.start()
    shared_server.stop()
    second_server = SharedServer(data_paths, fake_embedder)

    second = second_server.start()

    second_server.stop()
    assert second.token != first.token


@pytest.mark.parametrize("token", [None, "wrong"])
def test_a_request_without_the_token_is_refused(shared_server: SharedServer, token: str | None) -> None:
    found = shared_server.start()
    headers = ACCEPT | ({"Authorization": f"Bearer {token}"} if token else {})

    response = httpx2.post(found.url, json=INITIALIZE, headers=headers)

    assert response.status_code == 401


def test_a_page_on_another_site_cannot_reach_it(shared_server: SharedServer) -> None:
    found = shared_server.start()
    headers = ACCEPT | {"Authorization": f"Bearer {found.token}", "Origin": "https://evil.example"}

    response = httpx2.post(found.url, json=INITIALIZE, headers=headers)

    assert response.status_code == 403


def test_a_request_with_the_token_and_no_origin_is_served(shared_server: SharedServer) -> None:
    found = shared_server.start()

    response = httpx2.post(found.url, json=INITIALIZE, headers=ACCEPT | {"Authorization": f"Bearer {found.token}"})

    assert response.status_code == 200


def test_a_stop_request_ends_it_and_clears_its_address(shared_server: SharedServer) -> None:
    shared_server.start()

    shared_server.stop()

    assert shared_server.outcome == [True]
    assert read_discovery(shared_server.files) is None
    assert shared_server.files.stop.exists()
    assert not is_running(shared_server.files)


def test_a_stop_request_left_from_before_it_started_is_ignored(shared_server: SharedServer) -> None:
    shared_server.files.stop.touch()

    shared_server.start()
    time.sleep(0.3)

    assert is_running(shared_server.files)


def test_a_server_waiting_for_one_that_exits_unasked_takes_over(shared_server: SharedServer) -> None:
    held = threading.Event()
    holder = threading.Thread(target=_hold_for, args=(shared_server.files.lock, 0.5, held))
    holder.start()
    assert held.wait(5)

    found = shared_server.start()

    holder.join()
    assert found.pid == os.getpid()
    assert is_running(shared_server.files)


def _hold_for(lock_file: Path, seconds: float, held: threading.Event) -> None:
    with hold(lock_file, timeout=0):
        held.set()
        time.sleep(seconds)


def test_without_any_session_it_exits_after_the_idle_time(
    data_paths: AppPaths, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shared_server_module, "WATCH_SECONDS", 0.05)
    server = SharedServer(data_paths, fake_embedder, idle_minutes=0.5 / 60)

    server.start()
    server.join(seconds=10)

    assert server.outcome == [True]


def test_it_stays_while_a_session_is_connected_and_exits_after_it_leaves(
    data_paths: AppPaths, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shared_server_module, "WATCH_SECONDS", 0.05)
    server = SharedServer(data_paths, fake_embedder, idle_minutes=0.5 / 60)

    with registered(server.files):
        server.start()
        time.sleep(1.5)
        assert is_running(server.files)
    server.join(seconds=10)

    assert server.outcome == [True]


def test_a_second_server_gives_up_while_one_runs(
    data_paths: AppPaths, fake_embedder: FakeEmbedder, shared_server: SharedServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = shared_server.start()
    monkeypatch.setattr(shared_server_module, "LOCK_WAIT_SECONDS", 0.2)
    second = SharedServer(data_paths, fake_embedder)

    second.launch()
    second.join()

    assert second.outcome == [False]
    assert read_discovery(shared_server.files) == found


def test_the_token_check_compares_the_whole_token() -> None:
    verifier = StaticTokenVerifier("secret")

    assert anyio.run(verifier.verify_token, "secret") is not None
    assert anyio.run(verifier.verify_token, "secre") is None
