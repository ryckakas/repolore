import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.status import HTTP_502_BAD_GATEWAY

from repolore.doctor import Check, diagnose, index_check, refresh_check
from repolore.paths import AppPaths
from repolore.refresh import refresh_status, start_refresh
from repolore.server_state import Discovery, ServerFiles, connected_relays, is_running, read_discovery
from repolore.store import StoreError, connect_read_only, index_settings, repo_counts
from repolore.ui.rendering import render
from repolore.ui.tools_page import TOOLS

LOG_BYTES = 32_000
LOG_LINES = 200


@dataclass(frozen=True)
class SourceRow:
    name: str
    passages: int
    kinds: str


@dataclass(frozen=True)
class ServerView:
    running: bool
    found: Discovery | None = None
    sessions: int = 0
    started: str | None = None

    @property
    def label(self) -> str:
        if not self.running:
            return "Stopped"
        if self.found is None:
            return "Starting"
        return f"Running, {self.sessions} session{'' if self.sessions == 1 else 's'}"


class StatusPage:
    def __init__(self, paths: AppPaths, db: str) -> None:
        self._paths = paths
        self._db = db
        self._files = ServerFiles(paths.data_dir, db)

    def page(self, request: Request) -> Response:
        if not self._paths.config_file.exists():
            return RedirectResponse("/sources", status_code=303)
        return self._status_page(request)

    def refresh_state(self, request: Request) -> Response:
        return render(request, self._db, "_refresh.html", **self._refresh())

    def refresh(self, request: Request) -> Response:
        try:
            start_refresh(self._paths.data_dir, self._db)
        except OSError as error:
            return self._status_page(request, notice=f"Couldn't start a refresh: {error}", status_code=500)
        return RedirectResponse("/", status_code=303)

    def checks(self, request: Request) -> Response:
        return render(request, self._db, "_checks.html", checks=list(diagnose(self._paths, self._db)))

    def stop_server(self, request: Request) -> Response:
        # Imported here for the same reason as the tools: the relay brings the MCP client with it.
        from repolore.relay import stop_server

        return self._change_server(request, stop_server)

    def start_server(self, request: Request) -> Response:
        from repolore.relay import find_or_start

        return self._change_server(request, find_or_start)

    def _change_server(self, request: Request, change: Callable[[ServerFiles], object]) -> Response:
        from repolore.relay import ServerUnavailable

        try:
            change(self._files)
        except (ServerUnavailable, OSError) as error:
            return self._status_page(request, notice=str(error), status_code=HTTP_502_BAD_GATEWAY)
        return RedirectResponse("/", status_code=303)

    def _status_page(self, request: Request, notice: str | None = None, status_code: int = 200) -> Response:
        index, sources = index_summary(self._paths.index_file(self._db))
        return render(
            request,
            self._db,
            "status.html",
            status_code,
            notice=notice,
            index=index,
            sources=sources,
            server=server_view(self._files),
            log=log_tail(self._files.log),
            tools=TOOLS,
            **self._refresh(),
        )

    def _refresh(self) -> dict[str, Any]:
        data_dir = self._paths.data_dir
        return {"refresh": refresh_line(data_dir), "running": refresh_status(data_dir).running}


def index_summary(index_file: Path) -> tuple[Check, list[SourceRow]]:
    if not index_file.exists():
        return index_check(index_file), []
    try:
        with closing(connect_read_only(index_file)) as db:
            counts = repo_counts(db)
            built_at = index_settings(db).get("built_at")
    except (StoreError, sqlite3.Error):
        return index_check(index_file), []
    rows = sorted(
        (SourceRow(name, sum(kinds.values()), ", ".join(kinds)) for name, kinds in counts.items()),
        key=lambda row: (-row.passages, row.name.lower()),
    )
    total = sum(row.passages for row in rows)
    built = when(built_at) if built_at else "at an unknown time"
    sources = f"{len(rows)} source{'' if len(rows) == 1 else 's'}"
    return Check("index", True, f"{total:,} passages from {sources}, built {built}"), rows


def refresh_line(data_dir: Path) -> Check:
    check = refresh_check(data_dir)
    status = refresh_status(data_dir)
    if status.running or not status.ok or status.started_at is None or status.finished_at is None:
        return check
    return replace(
        check, detail=f"finished {when(status.finished_at)} in {took(status.started_at, status.finished_at)}"
    )


def server_view(files: ServerFiles) -> ServerView:
    if not is_running(files):
        return ServerView(running=False)
    found = read_discovery(files)
    if found is None:
        return ServerView(running=True)
    return ServerView(True, found, connected_relays(files), when(found.started_at))


def log_tail(log_file: Path) -> str:
    try:
        with log_file.open("rb") as file:
            start = max(0, log_file.stat().st_size - LOG_BYTES)
            file.seek(start)
            text = file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    whole_lines = text.partition("\n")[2] if start else text
    return "\n".join(whole_lines.splitlines()[-LOG_LINES:])


def when(moment: str, now: datetime | None = None) -> str:
    try:
        local = datetime.fromisoformat(moment).astimezone()
    except ValueError:
        return moment
    today = (now or datetime.now().astimezone()).date()
    return f"today at {local:%H:%M}" if local.date() == today else f"on {local:%Y-%m-%d} at {local:%H:%M}"


def took(started: str, finished: str) -> str:
    seconds = round((datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds())
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60} s"
    return f"{seconds // 3600} h {seconds % 3600 // 60} min"
