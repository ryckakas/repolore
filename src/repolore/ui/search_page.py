from urllib.parse import urlsplit
from urllib.request import url2pathname

from pydantic import BaseModel, ValidationError
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_502_BAD_GATEWAY

from repolore.documents import DocumentKind
from repolore.paths import AppPaths
from repolore.server_state import ServerFiles
from repolore.ui.rendering import form_text, render

GITHUB = "https://github.com/"
FILTERS = ("repo", "kind")
UNEXPECTED_ANSWER = (
    "The shared server answered in an unexpected form; `repolore server stop` lets the next search start a new one."
)


class SearchFailedError(Exception):
    pass


class Hit(BaseModel):
    title: str
    url: str
    repo: str
    kind: str
    section: str | None
    score: float
    text: str

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


class Found(BaseModel):
    results: list[Hit]


class SearchPage:
    def __init__(self, paths: AppPaths, db: str) -> None:
        self._files = ServerFiles(paths.data_dir, db)
        self._db = db

    def page(self, request: Request) -> Response:
        return render(request, self._db, "search.html", kinds=[kind.value for kind in DocumentKind])

    async def search(self, request: Request) -> Response:
        form = await request.form()
        query = form_text(form, "query")
        if not query:
            return self._notice(request, HTTP_400_BAD_REQUEST, "Type what to look for.")
        try:
            hits = await _hits(self._files, {"query": query, **_filters(form)})
        except SearchFailedError as error:
            return self._notice(request, HTTP_502_BAD_GATEWAY, str(error))
        return render(request, self._db, "_hits.html", hits=hits)

    def _notice(self, request: Request, status_code: int, notice: str) -> Response:
        return render(request, self._db, "_notice.html", status_code, notice=notice)


def _filters(form: FormData) -> dict[str, str]:
    return {name: value for name in FILTERS if (value := form_text(form, name))}


async def _hits(files: ServerFiles, arguments: dict[str, str]) -> list[Hit]:
    # The MCP client costs about 25 MB and most of a second, which a page that never searches shouldn't pay.
    from mcp.types import CallToolRequestParams

    from repolore.relay import Relay

    result = await Relay(files).call_tool(None, CallToolRequestParams(name="search", arguments=arguments))
    if result.is_error:
        raise SearchFailedError(" ".join(block.text for block in result.content if block.type == "text"))
    try:
        return Found.model_validate(result.structured_content).results
    except ValidationError as error:
        raise SearchFailedError(UNEXPECTED_ANSWER) from error
