import hmac
import secrets
import socket
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.types import ASGIApp

from repolore import __version__
from repolore.doctor import Check, diagnose, index_check, refresh_check, server_check
from repolore.paths import AppPaths
from repolore.refresh import refresh_status, start_refresh
from repolore.server_state import HOST, ServerFiles

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

templates = Jinja2Templates(directory=HERE / "templates")


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


class Pages:
    def __init__(self, paths: AppPaths, db: str) -> None:
        self._paths = paths
        self._db = db

    def status(self, request: Request) -> Response:
        return self._status_page(request)

    def refresh_state(self, request: Request) -> Response:
        return self._render(request, "_refresh.html", **self._refresh())

    def refresh(self, request: Request) -> Response:
        try:
            start_refresh(self._paths.data_dir, self._db)
        except OSError as error:
            return self._status_page(request, notice=f"Couldn't start a refresh: {error}", status_code=500)
        return RedirectResponse("/", status_code=303)

    def checks(self, request: Request) -> Response:
        return self._render(request, "_checks.html", checks=list(diagnose(self._paths, self._db)))

    def _status_page(self, request: Request, notice: str | None = None, status_code: int = 200) -> Response:
        files = ServerFiles(self._paths.data_dir, self._db)
        checks: list[Check] = [index_check(self._paths.index_file(self._db)), server_check(files)]
        return self._render(request, "status.html", status_code, checks=checks, notice=notice, **self._refresh())

    def _refresh(self) -> dict[str, Any]:
        data_dir = self._paths.data_dir
        return {"refresh": refresh_check(data_dir), "running": refresh_status(data_dir).running}

    def _render(self, request: Request, template: str, status_code: int = 200, **context: Any) -> Response:
        return templates.TemplateResponse(
            request, template, {"version": __version__, "db": self._db, **context}, status_code=status_code
        )


def create_app(paths: AppPaths, db: str, token: str) -> Starlette:
    pages = Pages(paths, db)
    return Starlette(
        routes=[
            Route("/", pages.status),
            Route("/refresh", pages.refresh_state),
            Route("/refresh", pages.refresh, methods=["POST"]),
            Route("/checks", pages.checks, methods=["POST"]),
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
