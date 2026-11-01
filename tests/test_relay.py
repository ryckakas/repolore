import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

import pytest
from conftest import FakeEmbedder, SharedServer
from filelock import FileLock
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError

from repolore import mcp_server, relay
from repolore import shared_server as shared_server_module
from repolore.locks import hold
from repolore.paths import AppPaths
from repolore.refresh import QUIET
from repolore.relay import (
    Relay,
    ServerUnavailable,
    create_relay,
    describe,
    find_or_start,
    start_server,
    stop_server,
)
from repolore.server_state import (
    Discovery,
    ServerFiles,
    connected_relays,
    is_running,
    read_discovery,
    request_stop,
    write_discovery,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _never_start(_: ServerFiles) -> None:
    raise AssertionError("started a server")


def test_a_running_server_of_this_version_is_used(shared_server: SharedServer) -> None:
    running = shared_server.start()

    assert find_or_start(shared_server.files, start=_never_start) == running


def test_without_a_server_one_is_started_and_awaited(shared_server: SharedServer) -> None:
    started: list[Discovery] = []

    found = find_or_start(shared_server.files, start=lambda _: started.append(shared_server.start()))

    assert started == [found]


def test_a_server_of_another_version_is_replaced(
    data_paths: AppPaths, fake_embedder: FakeEmbedder, shared_server: SharedServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = shared_server.start()
    current_version = relay.__version__
    monkeypatch.setattr(relay, "__version__", "9.9.9")
    monkeypatch.setattr(relay, "installed_version", lambda: "9.9.9")
    monkeypatch.setattr(shared_server_module, "__version__", "9.9.9")
    replacement = SharedServer(data_paths, fake_embedder)

    found = find_or_start(shared_server.files, start=lambda _: replacement.start())

    replacement.stop()
    shared_server.join()
    assert (old.version, found.version) == (current_version, "9.9.9")
    assert found.token != old.token


def test_a_server_that_never_comes_up_names_its_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(relay, "STARTUP_SECONDS", 0.3)
    files = ServerFiles(tmp_path, "default")

    with pytest.raises(ServerUnavailable, match=r"didn't start within 0\.3 seconds; its log is .*server-default\.log"):
        find_or_start(files, start=lambda _: None)


def test_an_old_server_that_does_not_stop_names_its_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(relay, "STOP_SECONDS", 0.3)
    files = ServerFiles(tmp_path, "default")
    write_discovery(files, Discovery(pid=1, port=1, token="t", version="0.0.1", started_at="2026-01-01T00:00:00+00:00"))

    with hold(files.lock, timeout=0), pytest.raises(ServerUnavailable, match="shared server didn't stop"):
        find_or_start(files, start=_never_start)

    assert files.stop.exists()


def test_stop_also_stops_a_server_that_was_waiting_for_the_lock(
    data_paths: AppPaths, fake_embedder: FakeEmbedder, shared_server: SharedServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_server.start()
    waiting = threading.Event()

    def hold_and_signal(lock_file: Path, timeout: float) -> FileLock:
        lock = hold(lock_file, timeout=timeout)
        waiting.set()
        return lock

    monkeypatch.setattr(shared_server_module, "hold", hold_and_signal)
    second = SharedServer(data_paths, fake_embedder)
    second.launch()
    assert waiting.wait(10)
    time.sleep(0.2)

    stop_server(shared_server.files)

    shared_server.join()
    second.join()
    assert (shared_server.outcome, second.outcome) == ([True], [True])
    assert not is_running(shared_server.files)
    assert read_discovery(shared_server.files) is None


def test_the_server_is_started_detached_with_its_log_appended(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[tuple[list[str], IO[str], dict[str, str], Path]] = []
    monkeypatch.setattr(
        relay, "spawn_detached", lambda command, stdout, env, cwd: spawned.append((command, stdout, env, cwd))
    )
    files = ServerFiles(tmp_path / "data", "experiment")
    files.data_dir.mkdir()
    files.log.write_text("earlier start\n", encoding="utf-8")

    start_server(files)

    [(command, log, env, cwd)] = spawned
    assert cwd == files.data_dir
    assert command == [
        *[sys.executable, "-P", "-m", "repolore", "--data-dir", str(files.data_dir)],
        *["server", "run", "--db", "experiment"],
    ]
    assert (log.name, log.mode) == (str(files.log), "a")
    assert env.items() >= QUIET.items()


def _starting(server: SharedServer) -> Callable[[ServerFiles], Discovery]:
    def find(files: ServerFiles) -> Discovery:
        return find_or_start(files, start=lambda _: server.start())

    return find


async def test_tools_are_forwarded_to_the_shared_server_past_any_proxy(
    shared_server: SharedServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, f"http://127.0.0.1:{_unused_port()}")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    forwarder = Relay(shared_server.files, _starting(shared_server))

    tools = await forwarder.list_tools(None, None)
    repos = await forwarder.call_tool(None, _params("list_repos", {}))
    failed = await forwarder.call_tool(None, _params("search", {}))

    assert {tool.name for tool in tools.tools} == {"search", "read", "list_repos", "refresh"}
    assert repos.structured_content is not None
    assert [repo["repo"] for repo in repos.structured_content["repos"]] == ["octo/charts", "octo/repo"]
    assert failed.is_error


async def test_a_server_that_went_away_is_found_again_and_the_call_retried(shared_server: SharedServer) -> None:
    running = shared_server.start()
    gone = running.model_copy(update={"port": _unused_port()})
    finds = iter([gone, running])
    forwarder = Relay(shared_server.files, lambda _: next(finds))

    result = await forwarder.call_tool(None, _params("list_repos", {}))

    assert not result.is_error
    assert next(finds, None) is None


async def test_a_server_that_fails_twice_is_reported_with_its_log(shared_server: SharedServer) -> None:
    gone = shared_server.start().model_copy(update={"port": _unused_port()})
    forwarder = Relay(shared_server.files, lambda _: gone)

    result = await forwarder.call_tool(None, _params("list_repos", {}))
    with pytest.raises(MCPError, match="The shared server failed"):
        await forwarder.list_tools(None, None)

    assert result.is_error
    message = result.content[0].text  # type: ignore[union-attr]
    assert message.startswith("The shared server failed: ConnectError")
    assert "ExceptionGroup" not in message
    assert str(shared_server.files.log) in message


async def test_a_server_that_went_away_and_cannot_be_restarted_is_reported(shared_server: SharedServer) -> None:
    gone = shared_server.start().model_copy(update={"port": _unused_port()})
    answers: list[Discovery | ServerUnavailable] = [gone, ServerUnavailable("The shared server didn't start.")]

    def find(_: ServerFiles) -> Discovery:
        answer = answers.pop(0)
        if isinstance(answer, ServerUnavailable):
            raise answer
        return answer

    result = await Relay(shared_server.files, find).call_tool(None, _params("list_repos", {}))

    assert result.is_error
    assert result.content[0].text == "The shared server didn't start."  # type: ignore[union-attr]


async def test_a_server_that_cannot_start_is_reported_without_another_wait(tmp_path: Path) -> None:
    finds: list[int] = []

    def unavailable(files: ServerFiles) -> Discovery:
        finds.append(1)
        raise ServerUnavailable(f"The shared server didn't start; its log is {files.log}.")

    forwarder = Relay(ServerFiles(tmp_path, "default"), unavailable)

    result = await forwarder.call_tool(None, _params("list_repos", {}))
    with pytest.raises(MCPError, match="didn't start"):
        await forwarder.list_tools(None, None)

    assert result.is_error
    assert "didn't start" in result.content[0].text  # type: ignore[union-attr]
    assert len(finds) == 2


def test_an_upgraded_install_asks_to_reconnect_and_leaves_the_server_alone(
    shared_server: SharedServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_server.start()
    monkeypatch.setattr(relay, "installed_version", lambda: "9.9.9")

    with pytest.raises(ServerUnavailable, match=r"upgraded to 9\.9\.9 .* reconnect it in /mcp"):
        find_or_start(shared_server.files, start=_never_start)

    assert is_running(shared_server.files)
    assert not shared_server.files.stop.exists()


def test_a_server_asked_to_stop_is_not_used_and_a_new_one_starts_after_it(shared_server: SharedServer) -> None:
    stopping = Discovery(pid=1, port=1, token="t", version=relay.__version__, started_at="2026-01-01T00:00:00+00:00")
    write_discovery(shared_server.files, stopping)
    request_stop(shared_server.files)
    exiting = threading.Thread(target=_hold_for, args=(shared_server.files.lock, 0.5))
    exiting.start()
    time.sleep(0.1)
    started: list[Discovery] = []

    found = find_or_start(shared_server.files, start=lambda _: started.append(shared_server.start()))

    exiting.join()
    assert started == [found]
    assert found != stopping


def test_a_server_that_exits_before_serving_is_replaced(shared_server: SharedServer) -> None:
    dying = threading.Thread(target=_hold_for, args=(shared_server.files.lock, 0.5))
    dying.start()
    time.sleep(0.1)
    started: list[Discovery] = []

    found = find_or_start(shared_server.files, start=lambda _: started.append(shared_server.start()))

    dying.join()
    assert started == [found]


@pytest.mark.parametrize(
    ("error", "described"),
    [
        (
            ExceptionGroup("tasks", [ExceptionGroup("inner", [MCPError(-32000, "stream ended")])]),
            "MCPError: stream ended",
        ),
        (ExceptionGroup("tasks", [ConnectionResetError()]), "ConnectionResetError"),
        (TimeoutError("no answer"), "TimeoutError: no answer"),
    ],
)
def test_an_error_is_described_by_what_ended_the_call(error: Exception, described: str) -> None:
    assert describe(error) == described


async def test_standalone_speaks_mcp_over_stdio_from_its_own_process(index_file: Path) -> None:
    data_dir = index_file.parent.parent
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-c",
            "from repolore.relay import main; main()",
            "--standalone",
            "--no-preload",
            "--data-dir",
            str(data_dir),
        ],
    )

    async with Client(params) as client:
        result = await client.call_tool("list_repos", {})

    assert result.structured_content is not None
    assert [repo["repo"] for repo in result.structured_content["repos"]] == ["octo/charts", "octo/repo"]
    assert not is_running(ServerFiles(data_dir, "default"))


async def test_the_relay_speaks_mcp_over_stdio_and_starts_the_shared_server(index_file: Path) -> None:
    data_dir = index_file.parent.parent
    files = ServerFiles(data_dir, "default")
    params = StdioServerParameters(
        command=sys.executable, args=["-c", "from repolore.relay import main; main()", "--data-dir", str(data_dir)]
    )

    try:
        async with Client(params) as client:
            result = await client.call_tool("list_repos", {})
            relays = connected_relays(files)
    finally:
        _stop(files)

    assert result.structured_content is not None
    assert [repo["repo"] for repo in result.structured_content["repos"]] == ["octo/charts", "octo/repo"]
    assert relays == 1


def test_standalone_serves_from_the_relay_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[tuple[Path | None, str, bool]] = []
    monkeypatch.setattr(mcp_server, "serve", lambda data_dir, db, preload: served.append((data_dir, db, preload)))
    monkeypatch.setattr(sys, "argv", ["repolore-mcp", "--standalone", "--data-dir", str(tmp_path), "--no-preload"])

    with pytest.raises(SystemExit) as exit_info:
        relay.main()

    assert exit_info.value.code == 0
    assert served == [(tmp_path, "default", False)]


def test_the_relay_loads_no_model_or_index_code() -> None:
    heavy = ["torch", "numpy", "sqlite_vec", "sentence_transformers", "repolore.mcp_server", "repolore.store"]
    script = f"import sys, repolore.relay; print([name for name in {heavy!r} if name in sys.modules])"

    loaded = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout

    assert loaded.strip() == "[]"


async def test_the_relay_answers_the_handshake_itself_and_forwards_tools(shared_server: SharedServer) -> None:
    server = create_relay(shared_server.files, _starting(shared_server))

    async with Client(server) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools.tools} == {"search", "read", "list_repos", "refresh"}


def test_a_server_still_starting_is_awaited_not_started_again(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path, "default")
    starting = Discovery(pid=1, port=1, token="t", version=relay.__version__, started_at="2026-01-01T00:00:00+00:00")
    writer = threading.Timer(0.3, write_discovery, (files, starting))

    with hold(files.lock, timeout=0):
        writer.start()
        found = find_or_start(files, start=_never_start)

    assert found == starting


def test_the_relay_serves_while_registered_as_a_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = ServerFiles(tmp_path.resolve(), "experiment")
    seen: list[tuple[str, int]] = []

    async def serve_stdio(server: relay.Server) -> None:
        seen.append((server.name, connected_relays(files)))

    monkeypatch.setattr(relay, "serve_stdio", serve_stdio)
    monkeypatch.setattr(sys, "argv", ["repolore-mcp", "--data-dir", str(tmp_path), "--db", "experiment"])

    with pytest.raises(SystemExit) as exit_info:
        relay.main()

    assert exit_info.value.code == 0
    assert seen == [("repolore", 1)]
    assert connected_relays(files) == 0


def _params(name: str, arguments: dict[str, object]) -> relay.types.CallToolRequestParams:
    return relay.types.CallToolRequestParams(name=name, arguments=arguments)


def _unused_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _hold_for(lock_file: Path, seconds: float) -> None:
    with hold(lock_file, timeout=0):
        time.sleep(seconds)


def _stop(files: ServerFiles) -> None:
    if not is_running(files):
        return
    request_stop(files)
    deadline = time.monotonic() + 30
    while is_running(files):
        assert time.monotonic() < deadline, "The shared server didn't stop."
        time.sleep(0.1)
