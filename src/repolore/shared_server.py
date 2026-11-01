import hmac
import logging
import os
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime

import anyio
import uvicorn
from filelock import Timeout
from mcp.server.auth.provider import AccessToken

from repolore import __version__
from repolore.embedding import Embedder
from repolore.locks import hold
from repolore.mcp_server import create_server, preload
from repolore.paths import AppPaths
from repolore.server_state import HOST, Discovery, ServerFiles, connected_relays, write_discovery

log = logging.getLogger(__name__)

# Of two servers started at the same moment, the second gives up after this long.
LOCK_WAIT_SECONDS = 15
# A stop doesn't wait longer than this for calls still running, such as a search waiting for the model.
GRACEFUL_SECONDS = 10
WATCH_SECONDS = 1.0


class StaticTokenVerifier:
    def __init__(self, token: str) -> None:
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        if hmac.compare_digest(token.encode(), self._token.encode()):
            return AccessToken(token=token, client_id="repolore-relay", scopes=[])
        return None


def run_shared(paths: AppPaths, db: str, idle_minutes: float, embedder_for: Callable[[str], Embedder]) -> bool:
    """Serve the index to every relay until none is left for `idle_minutes`; False if another server runs."""
    files = ServerFiles(paths.data_dir, db)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    lock = hold(files.lock, timeout=LOCK_WAIT_SECONDS)
    try:
        lock.acquire()
    except Timeout:
        log.info("A shared server for %s is already running.", db)
        return False
    try:
        files.discovery.unlink(missing_ok=True)
        files.stop.unlink(missing_ok=True)
        anyio.run(_serve, paths, files, idle_minutes * 60, embedder_for)
    finally:
        files.discovery.unlink(missing_ok=True)
        files.stop.unlink(missing_ok=True)
        lock.release()
    return True


async def _serve(
    paths: AppPaths, files: ServerFiles, idle_seconds: float, embedder_for: Callable[[str], Embedder]
) -> None:
    token = secrets.token_urlsafe(32)
    server = create_server(paths, files.db, embedder_for, StaticTokenVerifier(token))
    app = server.streamable_http_app(host=HOST)
    http = uvicorn.Server(
        uvicorn.Config(app, host=HOST, port=0, log_level="warning", timeout_graceful_shutdown=GRACEFUL_SECONDS)
    )
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_watch, http, paths, files, token, idle_seconds, embedder_for)
        await http.serve()
        tasks.cancel_scope.cancel()


async def _watch(
    http: uvicorn.Server,
    paths: AppPaths,
    files: ServerFiles,
    token: str,
    idle_seconds: float,
    embedder_for: Callable[[str], Embedder],
) -> None:
    while not http.started:
        await anyio.sleep(0.05)
    port = http.servers[0].sockets[0].getsockname()[1]
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    write_discovery(
        files, Discovery(pid=os.getpid(), port=port, token=token, version=__version__, started_at=started_at)
    )
    log.info("Serving %s on %s:%d for repolore %s.", files.db, HOST, port, __version__)
    preload(paths.index_file(files.db), embedder_for)
    last_seen = time.monotonic()
    while True:
        await anyio.sleep(WATCH_SECONDS)
        if files.stop.exists():
            log.info("Stopping: asked to.")
            break
        if connected_relays(files):
            last_seen = time.monotonic()
        elif time.monotonic() - last_seen > idle_seconds:
            log.info("Stopping: no session for %g minutes.", idle_seconds / 60)
            break
    http.should_exit = True
