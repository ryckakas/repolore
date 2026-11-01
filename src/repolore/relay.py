import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import anyio
import httpx2
import mcp.types as types
import typer
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.types.version import LATEST_MODERN_VERSION

from repolore import __version__
from repolore.instructions import INSTRUCTIONS
from repolore.paths import resolve_paths
from repolore.processes import SAFE_PATH, spawn_detached
from repolore.refresh import QUIET
from repolore.server_state import Discovery, ServerFiles, is_running, read_discovery, registered, request_stop

log = logging.getLogger(__name__)

# Long enough for a cold start on a slow machine: the server imports the MCP SDK and binds before it writes its address.
STARTUP_SECONDS = 120
STOP_SECONDS = 30
POLL_SECONDS = 0.2
# A search that waits minutes for a cold model load still gets data: the server pings every 15 s while a call runs.
READ_SECONDS = 60


class ServerUnavailable(Exception):
    pass


def start_server(files: ServerFiles) -> None:
    """Start the shared server for this index in a process that outlives the session that started it."""
    files.data_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *SAFE_PATH, "-m", "repolore", "--data-dir", str(files.data_dir), "server", "run"]
    # Appended to, so a second server started in the same moment, which exits at once, can't cut the first one's log.
    with files.log.open("a", encoding="utf-8") as server_log:
        spawn_detached([*command, "--db", files.db], stdout=server_log, env=os.environ | QUIET, cwd=files.data_dir)


def installed_version() -> str:
    return version("repolore")


def find_or_start(files: ServerFiles, start: Callable[[ServerFiles], object] = start_server) -> Discovery:
    """The running shared server of this version, starting one, or replacing one of another version, if needed."""
    # A server started now runs the installed code, which an upgrade during the session made newer than this relay's.
    if (installed := installed_version()) != __version__:
        raise ServerUnavailable(
            f"repolore was upgraded to {installed} while this session ran {__version__}; "
            "reconnect it in /mcp or restart the session."
        )
    found = _serving(files)
    if found is not None and found.version == __version__:
        return found
    if found is not None:
        log.warning("Replacing the shared server of repolore %s with %s.", found.version, __version__)
        stop_server(files)
    started = False

    def started_or_found() -> Discovery | None:
        nonlocal started
        # Also when a server that held the lock exits before it served, as one being stopped or crashing at start does.
        if not started and not is_running(files):
            start(files)
            started = True
        return _current(files)

    return _wait_for(started_or_found, STARTUP_SECONDS, "The shared server didn't start", files)


def stop_server(files: ServerFiles) -> None:
    request_stop(files)
    _wait_for(lambda: not is_running(files), STOP_SECONDS, "The shared server didn't stop", files)


def _serving(files: ServerFiles) -> Discovery | None:
    # A server asked to stop still holds its lock and address while it finishes its calls, but takes no new ones.
    return read_discovery(files) if is_running(files) and not files.stop.exists() else None


def _current(files: ServerFiles) -> Discovery | None:
    found = _serving(files)
    return found if found is not None and found.version == __version__ else None


def _wait_for[T](ready: Callable[[], T | None], seconds: float, failure: str, files: ServerFiles) -> T:
    deadline = time.monotonic() + seconds
    while not (result := ready()):
        if time.monotonic() > deadline:
            raise ServerUnavailable(f"{failure} within {seconds:g} seconds; its log is {files.log}.")
        time.sleep(POLL_SECONDS)
    return result


class Relay:
    """Forwards a session's tool calls to the shared server, starting it, or a new one if it went away."""

    def __init__(self, files: ServerFiles, find: Callable[[ServerFiles], Discovery] = find_or_start) -> None:
        self._files = files
        self._find = find

    async def list_tools(self, _: object, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        cursor = params.cursor if params else None
        try:
            return await self._forward(lambda client: client.list_tools(cursor=cursor))
        except ServerUnavailable as error:
            raise MCPError(code=types.INTERNAL_ERROR, message=str(error)) from error

    async def call_tool(self, _: object, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            return await self._forward(lambda client: client.call_tool(params.name, params.arguments))
        except ServerUnavailable as error:
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(error))], is_error=True)

    async def _forward[T](self, action: Callable[[Client], Awaitable[T]]) -> T:
        try:
            return await self._once(action)
        except ServerUnavailable:
            raise
        except Exception as error:
            log.warning("The shared server went away (%s); finding or starting another.", describe(error))
        try:
            return await self._once(action)
        except ServerUnavailable:
            raise
        except Exception as error:
            raise ServerUnavailable(
                f"The shared server failed: {describe(error)}; its log is {self._files.log}."
            ) from error

    async def _once[T](self, action: Callable[[Client], Awaitable[T]]) -> T:
        server = await anyio.to_thread.run_sync(self._find, self._files)
        headers = {"Authorization": f"Bearer {server.token}"}
        async with (
            # Not through HTTP(S)_PROXY or the system proxy, which would get the token and can't reach loopback anyway.
            httpx2.AsyncClient(
                headers=headers, timeout=httpx2.Timeout(30.0, read=READ_SECONDS), trust_env=False
            ) as http,
            # The relay and the server are the same version, so the newest protocol needs no negotiation.
            Client(
                streamable_http_client(server.url, http_client=http), mode=LATEST_MODERN_VERSION, cache=None
            ) as client,
        ):
            return await action(client)


def describe(error: BaseException) -> str:
    """The error that ended a call, out of the task groups the MCP client wraps it in."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    message = error.message if isinstance(error, MCPError) else str(error)
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def create_relay(files: ServerFiles, find: Callable[[ServerFiles], Discovery] = find_or_start) -> Server:
    """An MCP server that answers the handshake itself and forwards tools to the shared server."""
    forwarder = Relay(files, find)
    return Server(
        "repolore",
        version=__version__,
        instructions=INSTRUCTIONS,
        on_list_tools=forwarder.list_tools,
        on_call_tool=forwarder.call_tool,
    )


async def serve_stdio(server: Server) -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def serve(
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
    db: Annotated[str, typer.Option(help="Name of the index to serve.")] = "default",
    standalone: Annotated[
        bool, typer.Option(help="Serve from this process, with its own model, instead of the shared server.")
    ] = False,
    preload_model: Annotated[
        bool,
        typer.Option("--preload/--no-preload", help="With --standalone, load the model at start, not on first search."),
    ] = True,
) -> None:
    """Serve search over the repolore index to an MCP client on stdin and stdout, through one shared server."""
    if standalone:
        # Imported here so the relay, which runs in every session, never loads the model or index code.
        from repolore import mcp_server

        mcp_server.serve(data_dir, db, preload_model)
        return
    logging.basicConfig(level=logging.WARNING, format="repolore relay: %(message)s")
    files = ServerFiles(resolve_paths(data_dir).data_dir, db)
    with registered(files):
        anyio.run(serve_stdio, create_relay(files))


def main() -> None:
    typer.run(serve)
