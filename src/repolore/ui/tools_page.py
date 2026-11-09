import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

from pydantic import BaseModel, ValidationError
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_502_BAD_GATEWAY

from repolore.documents import DocumentKind
from repolore.paths import AppPaths
from repolore.retrieval import Mode
from repolore.server_state import ServerFiles
from repolore.ui.rendering import form_text, render

GITHUB = "https://github.com/"
SEARCH_FILTERS = ("repo", "kind", "author", "mode")
MAX_K = 25
UNEXPECTED_ANSWER = (
    "The shared server answered in an unexpected form; `repolore server stop` lets the next call start a new one."
)


class BadInputError(Exception):
    pass


class ToolFailedError(Exception):
    pass


class Linked(BaseModel):
    url: str

    @property
    def link(self) -> str | None:
        # Browsers refuse a file:// link from an http page, and nothing else belongs in this page's href.
        return self.url if self.url.startswith(GITHUB) else None

    @property
    def location(self) -> str:
        parts = urlsplit(self.url)
        if parts.scheme != "file":
            return self.url
        return url2pathname(f"//{parts.netloc}{parts.path}" if parts.netloc else parts.path)


class Hit(Linked):
    title: str
    repo: str
    kind: str
    section: str | None
    score: float
    text: str


class Found(BaseModel):
    results: list[Hit]


class Passage(BaseModel):
    header: str
    text: str


class Source(Linked):
    title: str
    repo: str
    kind: str
    passages: list[Passage]
    next_start: int | None


def _texts(form: FormData, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: value for name in names if (value := form_text(form, name))}


def _whole_number(form: FormData, name: str, low: int, high: int | None = None) -> dict[str, Any]:
    text = form_text(form, name)
    if not text:
        return {}
    try:
        number = int(text)
    except ValueError:
        number = low - 1
    if number < low or (high is not None and number > high):
        limits = f"from {low} to {high}" if high is not None else f"of {low} or more"
        raise BadInputError(f"{name} must be a whole number {limits}.")
    return {name: number}


def _search_arguments(form: FormData) -> dict[str, Any]:
    query = form_text(form, "query")
    if not query:
        raise BadInputError("Type what to look for.")
    return {
        "query": query,
        "distinct": "distinct" in form,
        **_texts(form, SEARCH_FILTERS),
        **_whole_number(form, "k", 1, MAX_K),
    }


def _read_arguments(form: FormData) -> dict[str, Any]:
    url = form_text(form, "url")
    if not url:
        raise BadInputError("Give the url of a search result.")
    return {"url": url, **_texts(form, ("section",)), **_whole_number(form, "start", 0)}


def _no_arguments(_: FormData) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class Tool:
    name: str
    summary: str
    description: str
    badge: str
    run_label: str
    arguments: Callable[[FormData], dict[str, Any]]


READ_ONLY = "read only"
TOOLS = (
    Tool(
        "search",
        "Passages that best match a query, with permalinks.",
        "Passages from your PRs, reviews, issues, commits and docs that best match the query, best first, "
        "with permalinks.",
        READ_ONLY,
        "Search",
        _search_arguments,
    ),
    Tool(
        "read",
        "The whole text of a result's source, a page at a time.",
        "The whole text of a search result's source, passage by passage in document order, a page at a time.",
        READ_ONLY,
        "Run read",
        _read_arguments,
    ),
    Tool(
        "list_repos",
        "Indexed repos and passages per kind.",
        "The indexed repos and how many passages of each kind they have.",
        READ_ONLY,
        "Run list_repos",
        _no_arguments,
    ),
    Tool(
        "refresh",
        "Updates the index from GitHub in the background.",
        "Updates the index from GitHub in the background, in minutes, or an hour the first time.",
        "starts a refresh",
        "Start refresh",
        _no_arguments,
    ),
)
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


class ToolsPage:
    def __init__(self, paths: AppPaths, db: str) -> None:
        self._files = ServerFiles(paths.data_dir, db)
        self._db = db

    def page(self, request: Request) -> Response:
        return render(
            request,
            self._db,
            "tools.html",
            tools=TOOLS,
            kinds=[kind.value for kind in DocumentKind],
            modes=[mode.value for mode in Mode],
            max_k=MAX_K,
        )

    async def run(self, request: Request) -> Response:
        # The name only picks a row of the table, so a request can't call anything else on the server.
        tool = TOOLS_BY_NAME.get(request.path_params["name"])
        if tool is None:
            return self._notice(request, HTTP_404_NOT_FOUND, "There's no such tool.")
        form = await request.form()
        try:
            return await self._answer(request, tool, tool.arguments(form))
        except BadInputError as error:
            return self._notice(request, HTTP_400_BAD_REQUEST, str(error))
        except ToolFailedError as error:
            return self._notice(request, HTTP_502_BAD_GATEWAY, str(error))

    async def _answer(self, request: Request, tool: Tool, arguments: dict[str, Any]) -> Response:
        started = time.perf_counter()
        answer = await _call(self._files, tool.name, arguments)
        elapsed = _elapsed(time.perf_counter() - started)
        view = _view(tool.name, answer)
        meta = " · ".join(part for part in (view.pop("summary", None), elapsed) if part)
        pretty = json.dumps(answer, indent=2, ensure_ascii=False)
        return render(request, self._db, "_result.html", meta=meta, json=pretty, **view)

    def _notice(self, request: Request, status_code: int, notice: str) -> Response:
        return render(request, self._db, "_notice.html", status_code, notice=notice)


async def _call(files: ServerFiles, name: str, arguments: dict[str, Any]) -> Any:
    # The MCP client costs about 25 MB and most of a second, which a page that never calls a tool shouldn't pay.
    from mcp.types import CallToolRequestParams

    from repolore.relay import Relay

    result = await Relay(files).call_tool(None, CallToolRequestParams(name=name, arguments=arguments))
    if result.is_error:
        raise ToolFailedError(" ".join(block.text for block in result.content if block.type == "text"))
    return result.structured_content


def _view(name: str, answer: Any) -> dict[str, Any]:
    try:
        if name == "search":
            hits = Found.model_validate(answer).results
            return {"hits": hits, "summary": _passages(len(hits))}
        if name == "read":
            source = Source.model_validate(answer)
            return {"source": source, "summary": _passages(len(source.passages))}
    except ValidationError as error:
        raise ToolFailedError(UNEXPECTED_ANSWER) from error
    return {}


def _passages(count: int) -> str:
    return f"{count} passage{'' if count == 1 else 's'}"


def _elapsed(seconds: float) -> str:
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f} s"
