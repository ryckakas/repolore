import logging
import sqlite3
import threading
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

import typer
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from repolore import __version__
from repolore.documents import DocumentKind
from repolore.embedding import Embedder
from repolore.embedding.local import UnknownModelError, embedder_for
from repolore.hub import ModelUnavailableError
from repolore.paths import AppPaths, resolve_paths
from repolore.refresh import RefreshStatus, refresh_status, start_refresh
from repolore.retrieval import Mode
from repolore.retrieval import search as search_index
from repolore.store import Filters, Hit, StoreError, connect_read_only, index_model, index_settings, repo_counts

log = logging.getLogger(__name__)

TEXT_CHARS = 1500
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
REFRESHES = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True)
INSTRUCTIONS = (
    "Searches the user's own engineering history on GitHub: pull request descriptions and discussions, review "
    "threads, issues, commits without a PR, and repository markdown (README, AGENTS.md, docs). Use it when the user "
    "asks why or how something was decided, built, fixed or rejected in their repos, or which repo already did "
    "something. Results are passages with GitHub permalinks; cite the permalinks. A search can wait a minute or "
    "more while the server is still loading the embedding model. list_repos says when the index was built; when it "
    "looks out of date for the question, the refresh tool updates it in the background."
)


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
    text: str


class SearchResults(BaseModel):
    query: str
    results: list[Result]


class Repo(BaseModel):
    repo: str
    chunks: dict[str, int] = Field(description="Indexed passages per kind of source.")


class Repos(BaseModel):
    repos: list[Repo]
    built_at: str | None = Field(description="When the index was last built, in UTC.")
    refresh: RefreshStatus = Field(description="The last refresh started from here or with `repolore refresh`.")


class Refresh(BaseModel):
    started: bool
    message: str


def create_server(paths: AppPaths, db_name: str, embedder_for: Callable[[str], Embedder]) -> MCPServer:
    server = MCPServer("repolore", version=__version__, instructions=INSTRUCTIONS)
    index_file = paths.index_file(db_name)

    @server.tool(annotations=READ_ONLY)
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
        """Find the passages that best match the query, best first, with permalinks."""
        with _open(index_file) as db:
            try:
                embedder = embedder_for(index_model(db)[0])
                filters = Filters(repo=repo, kind=kind, author=author)
                hits = search_index(db, embedder, query, k, filters, mode, distinct=distinct)
            except (StoreError, UnknownModelError, ModelUnavailableError) as error:
                raise ToolError(str(error)) from error
        return SearchResults(query=query, results=[_result(hit) for hit in hits])

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
        return " A refresh is running."
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


def serve(
    data_dir: Annotated[Path | None, typer.Option(help="Override the data directory.")] = None,
    db: Annotated[str, typer.Option(help="Name of the index to serve.")] = "default",
    preload_model: Annotated[
        bool, typer.Option("--preload/--no-preload", help="Load the embedding model at start, not on the first search.")
    ] = True,
) -> None:
    """Serve search over the repolore index to an MCP client on stdin and stdout."""
    paths = resolve_paths(data_dir)
    server = create_server(paths, db, embedder_for)
    if preload_model:
        preload(paths.index_file(db), embedder_for)
    server.run()


def main() -> None:
    typer.run(serve)
