import hmac
import secrets
import socket
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp

from repolore.paths import AppPaths
from repolore.server_state import HOST
from repolore.ui.client_pages import ClientPages
from repolore.ui.source_pages import SourcePages
from repolore.ui.status_page import StatusPage
from repolore.ui.tools_page import ToolsPage

HERE = Path(__file__).parent
TOKEN_PARAMETER = "token"
UNCHANGING_METHODS = ("GET", "HEAD")
GRACEFUL_SECONDS = 3
HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
        "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    ),
    # Not no-referrer, under which browsers send "Origin: null" on a form POST and the page's own forms are refused.
    "Referrer-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}


class Guard(BaseHTTPMiddleware):
    """Lets in only a browser that has this run's token, and only a POST made by the page itself."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await self._answer(request, call_next)
        response.headers.update(HEADERS)
        return response

    async def _answer(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method not in UNCHANGING_METHODS and not _is_from_the_page(request):
            return PlainTextResponse("This request didn't come from the page.", status_code=403)
        if self._has_token(request):
            return self._with_cookie(request)
        if not self._matches(request.cookies.get(_cookie_name(request))):
            return PlainTextResponse("Open the address that `repolore ui` printed.", status_code=403)
        return await call_next(request)

    def _has_token(self, request: Request) -> bool:
        return (
            request.method == "GET"
            and request.url.path == "/"
            and self._matches(request.query_params.get(TOKEN_PARAMETER))
        )

    def _with_cookie(self, request: Request) -> Response:
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(_cookie_name(request), self._token, httponly=True, samesite="strict")
        return response

    def _matches(self, candidate: str | None) -> bool:
        return candidate is not None and hmac.compare_digest(candidate.encode(), self._token.encode())


def _is_from_the_page(request: Request) -> bool:
    return request.headers.get("origin") == f"http://{request.headers.get('host')}"


def _cookie_name(request: Request) -> str:
    # Cookies aren't kept apart by port, so two runs would otherwise replace each other's.
    return f"repolore_ui_{request.url.port}"


def create_app(paths: AppPaths, db: str, token: str) -> Starlette:
    status = StatusPage(paths, db)
    tools = ToolsPage(paths, db)
    sources = SourcePages(paths, db)
    clients = ClientPages(db)
    return Starlette(
        routes=[
            Route("/", status.page),
            Route("/refresh", status.refresh_state),
            Route("/refresh", status.refresh, methods=["POST"]),
            Route("/checks", status.checks, methods=["POST"]),
            Route("/server/stop", status.stop_server, methods=["POST"]),
            Route("/server/start", status.start_server, methods=["POST"]),
            Route("/tools", tools.page),
            Route("/tools/{name}", tools.run, methods=["POST"]),
            Route("/sources", sources.page),
            Route("/sources/repos", sources.repos),
            Route("/sources/repos", sources.save_repos, methods=["POST"]),
            Route("/sources/matches", sources.matches),
            Route("/sources/local", sources.save_local, methods=["POST"]),
            Route("/sources/folders", sources.folders),
            Route("/sources/folders", sources.add_folder, methods=["POST"]),
            Route("/sources/folders/remove", sources.remove_folder, methods=["POST"]),
            Route("/clients", clients.page),
            Route("/clients/list", clients.listing),
            Route("/clients/register", clients.register, methods=["POST"]),
            Mount("/static", StaticFiles(directory=HERE / "static")),
        ],
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=[HOST], www_redirect=False),
            Middleware(Guard, token=token),
        ],
    )


def serve(paths: AppPaths, db: str, on_ready: Callable[[str], object]) -> None:
    """Serve the page until Ctrl-C; on_ready gets its address, which holds this run's token."""
    token = secrets.token_urlsafe(32)
    config = uvicorn.Config(
        create_app(paths, db, token), log_level="warning", timeout_graceful_shutdown=GRACEFUL_SECONDS
    )
    # Listening before on_ready, so a browser opened from it waits for the page instead of finding nothing there.
    with socket.create_server((HOST, 0)) as listener:
        port = listener.getsockname()[1]
        on_ready(f"http://{HOST}:{port}/?{TOKEN_PARAMETER}={token}")
        # uvicorn raises the signal again once it has shut down.
        with suppress(KeyboardInterrupt):
            uvicorn.Server(config).run(sockets=[listener])
