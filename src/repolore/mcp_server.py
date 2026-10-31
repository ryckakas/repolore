import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Annotated

import typer
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from repolore.documents import DocumentKind
from repolore.embedding import Embedder
from repolore.embedding.local import UnknownModelError, embedder_for
from repolore.paths import resolve_paths
from repolore.retrieval import Mode
from repolore.retrieval import search as search_index
from repolore.store import Filters, Hit, StoreError, connect_read_only, index_model, repo_counts

TEXT_CHARS = 1500
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
INSTRUCTIONS = (
    "Searches the user's own engineering history on GitHub: pull request descriptions and discussions, review "
    "threads, issues, commits without a PR, and repository markdown (README, AGENTS.md, docs). Use it when the user "
    "asks why or how something was decided, built, fixed or rejected in their repos, or which repo already did "
    "something. Results are passages with GitHub permalinks; cite the permalinks. The first search in a session "
    "loads the embedding model and can take about a minute."
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


def create_server(index_file: Path, embedder_for: Callable[[str], Embedder]) -> MCPServer:
    server = MCPServer("repolore", instructions=INSTRUCTIONS)

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
    ) -> SearchResults:
        """Find the passages that best match the query, best first, with permalinks."""
        with _open(index_file) as db:
            try:
                embedder = embedder_for(index_model(db)[0])
                hits = search_index(db, embedder, query, k, Filters(repo=repo, kind=kind, author=author), mode)
            except (StoreError, UnknownModelError) as error:
                raise ToolError(str(error)) from error
        return SearchResults(query=query, results=[_result(hit) for hit in hits])

    @server.tool(annotations=READ_ONLY)
    def list_repos() -> Repos:
        """List the indexed repos and how many passages of each kind they have."""
        with _open(index_file) as db:
            return Repos(repos=[Repo(repo=repo, chunks=chunks) for repo, chunks in repo_counts(db).items()])

    return server


@contextmanager
def _open(index_file: Path) -> Iterator[sqlite3.Connection]:
    if not index_file.exists():
        raise ToolError(f"No index at {index_file}; run `repolore ingest` and `repolore build` first.")
    with closing(connect_read_only(index_file)) as db:
        yield db


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
) -> None:
    """Serve search over the repolore index to an MCP client on stdin and stdout."""
    create_server(resolve_paths(data_dir).index_file(db), embedder_for).run()


def main() -> None:
    typer.run(serve)
