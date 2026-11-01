import logging
import sqlite3
import threading
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.auth.provider import TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl, BaseModel, Field

from repolore import __version__
from repolore.documents import DocumentKind
from repolore.embedding import Embedder
from repolore.embedding.local import UnknownModelError, embedder_for
from repolore.hub import ModelUnavailableError
from repolore.instructions import INSTRUCTIONS, SEARCH
from repolore.paths import AppPaths, resolve_paths
from repolore.refresh import RefreshStatus, refresh_status, start_refresh
from repolore.retrieval import Mode
from repolore.retrieval import search as search_index
from repolore.server_state import HOST
from repolore.store import (
    Filters,
    Hit,
    Passage,
    StoreError,
    connect_read_only,
    index_model,
    index_settings,
    repo_counts,
    source_passages,
)

log = logging.getLogger(__name__)

TEXT_CHARS = 1500
READ_CHARS = 12_000
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
REFRESHES = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True)


class Result(BaseModel):
    title: str
    url: str = Field(description="GitHub permalink to the source; cite it.")
    repo: str
    kind: DocumentKind
    section: str | None
    author: str | None
    date: str | None
    score: float = Field(
        description="Relevance to the query, higher is better; the scale depends on mode, "
        "so compare scores only within one search."
    )
    text: str = Field(description=f"Cut at {TEXT_CHARS:,} characters and ended with … when longer; read has it all.")


class SearchResults(BaseModel):
    query: str
    results: list[Result]


class SourcePassage(BaseModel):
    header: str = Field(description="The repo, the source and the headings the passage sits under.")
    text: str


class Source(BaseModel):
    title: str
    url: str
    repo: str
    kind: DocumentKind
    author: str | None
    date: str | None
    passages: list[SourcePassage]
    next_start: int | None = Field(
        description="More passages follow: read again with this start, and the same section, to get them. "
        "Null at the end."
    )


class Repo(BaseModel):
    repo: str
    chunks: dict[str, int] = Field(description="Indexed passages per kind of source.")


class Repos(BaseModel):
    repos: list[Repo]
    built_at: str | None = Field(description="When the index was last built, in UTC.")
    refresh: RefreshStatus = Field(
        description="The last refresh started from here or with `repolore refresh`, with its progress while it runs."
    )


class Refresh(BaseModel):
    started: bool
    message: str


def create_server(
    paths: AppPaths, db_name: str, embedder_for: Callable[[str], Embedder], token_verifier: TokenVerifier | None = None
) -> MCPServer:
    """The tools over one index; with a token verifier, HTTP requests need its bearer token."""
    # A token verifier without an authorization server: no OAuth routes or metadata, just the bearer check.
    auth = AuthSettings(issuer_url=AnyHttpUrl(f"http://{HOST}"), resource_server_url=None) if token_verifier else None
    server = MCPServer(
        "repolore", version=__version__, instructions=INSTRUCTIONS, token_verifier=token_verifier, auth=auth
    )
    index_file = paths.index_file(db_name)

    @server.tool(annotations=READ_ONLY, description=SEARCH)
    def search(
        query: Annotated[str, Field(description="A question or keywords, in plain words.")],
        repo: Annotated[str | None, Field(description="Only this repo, as owner/name.")] = None,
        kind: Annotated[DocumentKind | None, Field(description="Only this kind of source.")] = None,
        author: Annotated[str | None, Field(description="Only this author's GitHub login.")] = None,
        k: Annotated[int, Field(ge=1, le=25, description="How many passages to return.")] = 8,
        mode: Annotated[
            Mode,
            Field(
                description="vector (default) matches meaning; keyword matches exact words, "
                "so use it for identifiers, ticket IDs or error messages; hybrid fuses both by rank."
            ),
        ] = Mode.VECTOR,
        distinct: Annotated[
            bool,
            Field(
                description="true (default) returns at most one passage per source URL, its best section, so k "
                "passages cover k sources; false also returns a source's other matching sections. A review thread "
                "has its own URL, so it counts apart from its PR."
            ),
        ] = True,
    ) -> SearchResults:
        with _open(index_file) as db:
            try:
                embedder = embedder_for(index_model(db)[0])
                filters = Filters(repo=repo, kind=kind, author=author)
                hits = search_index(db, embedder, query, k, filters, mode, distinct=distinct)
            except (StoreError, UnknownModelError, ModelUnavailableError) as error:
                raise ToolError(str(error)) from error
        return SearchResults(query=query, results=[_result(hit) for hit in hits])

    @server.tool(annotations=READ_ONLY)
    def read(
        url: Annotated[str, Field(description="The url of a search result.")],
        section: Annotated[
            str | None, Field(description="Only the passages under this heading, as a search result's section.")
        ] = None,
        start: Annotated[int, Field(ge=0, description="The first passage to return, from next_start.")] = 0,
    ) -> Source:
        """Read the whole text of a search result's source, passage by passage in document order, a page at a time."""
        with _open(index_file) as db:
            try:
                passages = source_passages(db, url)
            except StoreError as error:
                raise ToolError(str(error)) from error
        if not passages:
            raise ToolError(f"Nothing indexed has the url {url}; pass the url of a search result.")
        return _source(url, _in_section(passages, section), start)

    @server.tool(annotations=READ_ONLY)
    def list_repos() -> Repos:
        """List the indexed repos and how many passages of each kind they have."""
        status = refresh_status(paths.data_dir)
        try:
            with _open(index_file) as db:
                counts = repo_counts(db)
                built_at = index_settings(db).get("built_at")
        except (ToolError, StoreError) as error:
            raise ToolError(f"{error}{_refresh_note(status)}") from error
        return Repos(
            repos=[Repo(repo=repo, chunks=chunks) for repo, chunks in counts.items()],
            built_at=built_at,
            refresh=status,
        )

    @server.tool(annotations=REFRESHES)
    def refresh() -> Refresh:
        """Update the index from GitHub in the background, in minutes or an hour the first time; see list_repos."""
        if start_refresh(paths.data_dir, db_name):
            return Refresh(started=True, message="Refresh started in the background.")
        return Refresh(started=False, message="A refresh is already running; list_repos shows how it goes.")

    return server


def preload(index_file: Path, embedder_for: Callable[[str], Embedder]) -> threading.Thread | None:
    """Load the index's model in the background, so the first search doesn't wait for it."""
    if not index_file.exists():
        return None
    thread = threading.Thread(target=_warm_up, args=(index_file, embedder_for), name="preload", daemon=True)
    thread.start()
    return thread


def _warm_up(index_file: Path, embedder_for: Callable[[str], Embedder]) -> None:
    try:
        with closing(connect_read_only(index_file)) as db:
            model_id = index_model(db)[0]
        embedder_for(model_id).warm_up()
    except (StoreError, UnknownModelError, ModelUnavailableError) as error:
        log.warning("Not preloading the model, the first search will report why: %s", error)


@contextmanager
def _open(index_file: Path) -> Generator[sqlite3.Connection]:
    if not index_file.exists():
        raise ToolError(f"No index at {index_file}; use the refresh tool, or run `repolore refresh`, to build it.")
    with closing(connect_read_only(index_file)) as db:
        yield db


def _refresh_note(status: RefreshStatus) -> str:
    if status.running:
        return f" A refresh is running ({status.progress})." if status.progress else " A refresh is running."
    if status.ok is False:
        return f" The last refresh failed: {status.error}"
    return ""


def _result(hit: Hit) -> Result:
    text = hit.text if len(hit.text) <= TEXT_CHARS else hit.text[:TEXT_CHARS] + "…"
    return Result(
        title=hit.title,
        url=hit.url,
        repo=hit.repo,
        kind=DocumentKind(hit.kind),
        section=hit.section,
        author=hit.author,
        date=hit.created_at,
        score=round(hit.score, 4),
        text=text,
    )


def _in_section(passages: list[Passage], section: str | None) -> list[Passage]:
    if section is None:
        return passages
    if matching := [passage for passage in passages if passage.section == section]:
        return matching
    sections = dict.fromkeys(passage.section for passage in passages if passage.section)
    found = f"its sections are: {', '.join(sections)}" if sections else "it has no headed sections"
    raise ToolError(f"No section {section!r} in {passages[0].title}; {found}.")


def _source(url: str, passages: list[Passage], start: int) -> Source:
    first = passages[0]
    page: list[SourcePassage] = []
    size = 0
    next_start = None
    for index, passage in enumerate(passages[start:], start=start):
        size += len(passage.header) + len(passage.text)
        if page and size > READ_CHARS:
            next_start = index
            break
        page.append(SourcePassage(header=passage.header, text=passage.text))
    return Source(
        title=first.title,
        url=url,
        repo=first.repo,
        kind=DocumentKind(first.kind),
        author=first.author,
        date=first.created_at,
        passages=page,
        next_start=next_start,
    )


def serve(data_dir: Path | None, db: str, preload_model: bool) -> None:
    """Serve search over the repolore index to an MCP client on stdin and stdout."""
    paths = resolve_paths(data_dir)
    server = create_server(paths, db, embedder_for)
    if preload_model:
        preload(paths.index_file(db), embedder_for)
    server.run()
