import subprocess
from contextlib import closing
from pathlib import Path
from unittest.mock import ANY

import pytest
from conftest import DocumentFactory, FakeEmbedder
from mcp.client import Client
from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

from repolore import __version__, mcp_server
from repolore.chunking import Chunk, chunk_document
from repolore.documents import Document
from repolore.embedding.local import UnknownModelError
from repolore.hub import ModelUnavailableError
from repolore.mcp_server import READ_CHARS, TEXT_CHARS, create_server, preload
from repolore.paths import AppPaths
from repolore.progress import Phase, Progress
from repolore.refresh import refreshing
from repolore.store import connect, write_index

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def server(index_file: Path, embedder: FakeEmbedder) -> MCPServer:
    data_dir = index_file.parent.parent
    paths = AppPaths(config_file=data_dir / "repolore.toml", questions_file=data_dir / "q.yaml", data_dir=data_dir)
    return create_server(paths, index_file.stem, lambda _: embedder)


async def test_tools_are_listed_as_read_only(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"search", "read", "list_repos", "refresh"}
    assert not any("\n" in (tool.description or "") for tool in tools.values())
    assert "engineering history on GitHub" in (tools["search"].description or "")
    assert all(
        (annotations := tools[name].annotations) and annotations.read_only_hint
        for name in ("search", "read", "list_repos")
    )
    assert set(tools["search"].input_schema["properties"]) == {
        "query",
        "repo",
        "kind",
        "author",
        "k",
        "mode",
        "distinct",
    }


async def test_initialize_reports_the_package_version(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        server_info = client.server_info

    assert server_info is not None
    assert server_info.name == "repolore"
    assert server_info.version == __version__


@pytest.mark.parametrize("mode", [None, "vector", "keyword", "hybrid"])
async def test_search_modes(
    index_file: Path, fake_embedder: FakeEmbedder, mode: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_search = mcp_server.search_index
    seen = []

    def record_mode(db, embedder, query, k, filters, mode=mcp_server.Mode.VECTOR, reranker=None, distinct=False):
        seen.append(mode)
        return original_search(db, embedder, query, k, filters, mode, reranker, distinct=distinct)

    monkeypatch.setattr(mcp_server, "search_index", record_mode)
    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "registry", **({"mode": mode} if mode else {})})

    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["results"]
    assert seen == [mcp_server.Mode(mode) if mode else mcp_server.Mode.VECTOR]


async def test_search_rejects_unknown_mode(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "registry", "mode": "invalid"})

    assert result.is_error


async def test_search_returns_ranked_passages_with_permalinks(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "solar battery Modbus", "k": 2})

    assert not result.is_error
    assert result.structured_content is not None
    results = result.structured_content["results"]
    assert len(results) == 2
    assert results[0] | {"score": None} == {
        "title": "Change 3",
        "url": "https://github.com/octo/repo/issues/3",
        "repo": "octo/repo",
        "kind": "issue",
        "section": None,
        "author": "octocat",
        "date": "2026-01-04T00:00:00+00:00",
        "score": None,
        "text": "The solar battery reports power over Modbus TCP.",
    }
    assert results[0]["score"] > results[1]["score"]
    assert isinstance(result.content[0], TextContent)


async def test_search_filters(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "registry", "kind": "doc"})

    assert result.structured_content is not None
    assert [hit["url"] for hit in result.structured_content["results"]] == [
        "https://github.com/octo/repo/blob/main/README.md"
    ]


@pytest.mark.parametrize(("distinct", "urls"), [(None, 3), (True, 3), (False, 2)])
async def test_search_returns_one_passage_per_source_unless_told_not_to(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory, distinct: bool | None, urls: int
) -> None:
    documents = [
        make_document(9, "## Retry\nregistry backoff\n\n## Timeout\nregistry backoff timeout"),
        make_document(1, "The solar battery reports power over Modbus TCP."),
        make_document(2, "Document the snake_case config keys."),
    ]
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    path = tmp_path / "db" / "sections.db"
    with closing(connect(path)) as db:
        vectors = fake_embedder.embed_documents([chunk.embedded_text for chunk in chunks])
        write_index(db, chunks, vectors, fake_embedder.model_id, fake_embedder.dim)

    arguments = {"query": "registry backoff", "k": 3} | ({} if distinct is None else {"distinct": distinct})
    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("search", arguments)

    assert result.structured_content is not None
    hits = result.structured_content["results"]
    assert len(hits) == 3
    assert len({hit["url"] for hit in hits}) == urls


async def test_long_passages_are_shortened(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    chunks = chunk_document(make_document(1, "word " * TEXT_CHARS))
    path = tmp_path / "db" / "long.db"
    with closing(connect(path)) as db:
        write_index(
            db, chunks, fake_embedder.embed_documents([c.embedded_text for c in chunks]), "fake-bag-of-words", 64
        )

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "word"})

    assert result.structured_content is not None
    text = result.structured_content["results"][0]["text"]
    assert len(text) == TEXT_CHARS + 1
    assert text.endswith("…")


async def test_out_of_range_k_is_an_error_the_model_can_read(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "x", "k": 99})

    assert result.is_error


async def test_missing_index_says_how_to_build_it(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(tmp_path / "db" / "missing.db", fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "x"})

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "use the refresh tool, or run `repolore refresh`, to build it" in result.content[0].text


@pytest.mark.parametrize(
    ("tool", "arguments"), [("search", {"query": "x"}), ("read", {"url": "x"}), ("list_repos", {})]
)
async def test_empty_index_says_how_to_build_it(
    tmp_path: Path, fake_embedder: FakeEmbedder, tool: str, arguments: dict[str, str]
) -> None:
    path = tmp_path / "db" / "empty.db"
    with closing(connect(path)):
        pass

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool(tool, arguments)

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "The index is empty; run `repolore build` first." in result.content[0].text


async def test_index_from_another_model_is_refused(
    index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeEmbedder, "model_id", "other-model")

    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "x"})

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "rebuild it" in result.content[0].text


async def test_a_model_it_cant_get_is_an_error_the_model_can_read(
    index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*_: object) -> None:
        raise ModelUnavailableError("voyageai/voyage-4-nano isn't in the Hugging Face cache")

    monkeypatch.setattr(FakeEmbedder, "embed_query", unavailable)

    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "x"})

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "voyageai/voyage-4-nano isn't in the Hugging Face cache" in result.content[0].text


async def test_list_repos_counts_passages_per_kind(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("list_repos", {})

    assert result.structured_content is not None
    assert result.structured_content["repos"] == [
        {"repo": "octo/charts", "chunks": {"pr": 1}},
        {"repo": "octo/repo", "chunks": {"doc": 1, "issue": 1, "pr": 1}},
    ]
    assert result.structured_content["built_at"].endswith("+00:00")
    assert result.structured_content["refresh"]["running"] is False


@pytest.mark.parametrize(("preload_model", "preloads"), [(True, 1), (False, 0)])
def test_serve_serves_the_named_index_from_the_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preload_model: bool, preloads: int
) -> None:
    served: list[Path] = []
    preloaded: list[Path] = []

    class StubServer:
        def __init__(self, paths: AppPaths, db_name: str) -> None:
            self.index_file = paths.index_file(db_name)

        def run(self) -> None:
            served.append(self.index_file)

    monkeypatch.setattr(mcp_server, "create_server", lambda paths, db_name, _: StubServer(paths, db_name))
    monkeypatch.setattr(mcp_server, "preload", lambda index_file, _: preloaded.append(index_file))

    mcp_server.serve(tmp_path, "experiment", preload_model)

    assert served == [tmp_path.resolve() / "db" / "experiment.db"]
    assert preloaded == served * preloads


def test_preload_warms_up_the_index_model(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    models: list[str] = []

    def embedder_for(model_id: str) -> FakeEmbedder:
        models.append(model_id)
        return fake_embedder

    thread = preload(index_file, embedder_for)

    assert thread is not None
    thread.join(timeout=10)
    assert models == [fake_embedder.model_id]
    assert fake_embedder.warm_ups == 1


def test_preload_without_an_index_does_nothing(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    assert preload(tmp_path / "db" / "default.db", lambda _: fake_embedder) is None
    assert fake_embedder.warm_ups == 0


@pytest.mark.parametrize(
    "error", [UnknownModelError("built with another model"), ModelUnavailableError("can't download it")]
)
def test_preload_that_fails_logs_why_and_leaves_it_to_the_first_search(
    index_file: Path, error: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    def embedder_for(_: str) -> FakeEmbedder:
        raise error

    thread = preload(index_file, embedder_for)

    assert thread is not None
    thread.join(timeout=10)
    assert caplog.messages == [f"Not preloading the model, the first search will report why: {error}"]


async def test_refresh_is_annotated_as_writing_and_calling_github(
    index_file: Path, fake_embedder: FakeEmbedder
) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    annotations = tools["refresh"].annotations
    assert annotations is not None
    assert (annotations.read_only_hint, annotations.destructive_hint) == (False, False)
    assert (annotations.idempotent_hint, annotations.open_world_hint) == (True, True)


async def test_refresh_starts_one_in_the_background(
    index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[tuple[Path, str]] = []
    monkeypatch.setattr(mcp_server, "start_refresh", lambda data_dir, db: started.append((data_dir, db)) or True)

    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("refresh", {})

    assert result.structured_content == {"started": True, "message": "Refresh started in the background."}
    assert started == [(index_file.parent.parent, "default")]


async def test_refresh_says_when_one_is_already_running(
    index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: pytest.fail("spawned a refresh"))

    async with Client(server(index_file, fake_embedder)) as client:
        with refreshing(index_file.parent.parent):
            result = await client.call_tool("refresh", {})
            repos = await client.call_tool("list_repos", {})

    assert result.structured_content is not None
    assert result.structured_content["started"] is False
    assert result.structured_content["message"] == "A refresh is already running; list_repos shows how it goes."
    assert repos.structured_content is not None
    assert repos.structured_content["refresh"]["running"] is True


async def test_without_an_index_list_repos_says_how_a_refresh_is_going(
    tmp_path: Path, fake_embedder: FakeEmbedder
) -> None:
    missing = tmp_path / "db" / "default.db"

    async with Client(server(missing, fake_embedder)) as client:
        with refreshing(tmp_path) as report:
            starting = await client.call_tool("list_repos", {})
            report(Progress(phase=Phase.EMBED, done=120, total=5000))
            running = await client.call_tool("list_repos", {})
        with pytest.raises(ValueError, match="no token"), refreshing(tmp_path):
            raise ValueError("no token")
        failed = await client.call_tool("list_repos", {})
        (tmp_path / "refresh.json").unlink()
        never = await client.call_tool("list_repos", {})

    results = (starting, running, failed, never)
    texts = [result.content[0].text for result in results if isinstance(result.content[0], TextContent)]
    assert all(result.is_error for result in results)
    assert texts[0].endswith("to build it. A refresh is running.")
    assert texts[1].endswith("to build it. A refresh is running (embed 120 of 5000).")
    assert texts[2].endswith("to build it. The last refresh failed: no token")
    assert texts[3].endswith("to build it.")


async def test_list_repos_says_how_far_a_running_refresh_has_got(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        with refreshing(index_file.parent.parent) as report:
            report(Progress(phase=Phase.INGEST, done=3, total=50))
            result = await client.call_tool("list_repos", {})

    assert result.structured_content is not None
    refresh = result.structured_content["refresh"]
    assert refresh["running"] is True
    assert refresh["progress"] == {"phase": "ingest", "done": 3, "total": 50}


def index_of(tmp_path: Path, embedder: FakeEmbedder, chunks: list[Chunk]) -> Path:
    path = tmp_path / "db" / "read.db"
    with closing(connect(path)) as db:
        vectors = embedder.embed_documents([chunk.embedded_text for chunk in chunks])
        write_index(db, chunks, vectors, embedder.model_id, embedder.dim)
    return path


def sized(document: Document, ordinal: int, chars: int, section: str | None = None) -> Chunk:
    header = f"part {ordinal}"
    text = "x" * (chars - len(header))
    return Chunk(id=f"{document.url}::{ordinal}", document=document, section=section, header=header, text=text)


def error_text(result: CallToolResult) -> str:
    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    return result.content[0].text


CHANGELOG = (
    "# Changelog\n\n## [0.2.0] - 2026-02-01\n\n### Changed\n\nFaster search.\n\n"
    "## [0.1.0] - 2026-01-01\n\n### Changed\n\nFirst release.\n\n### Fixed\n\nA crash."
)


async def test_read_returns_a_sources_passages_in_document_order(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    document = make_document(1, CHANGELOG)
    path = index_of(tmp_path, fake_embedder, [*chunk_document(make_document(2)), *chunk_document(document)])

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": document.url})

    assert result.structured_content is not None
    source = result.structured_content
    assert {key: source[key] for key in ("title", "url", "repo", "kind", "author", "next_start")} == {
        "title": "Change 1",
        "url": document.url,
        "repo": "octo/repo",
        "kind": "pr",
        "author": "octocat",
        "next_start": None,
    }
    assert source["date"].startswith("2026-01-02")
    assert source["passages"] == [
        {"header": "octo/repo · PR #1 · Change 1 · Changelog > [0.2.0] - 2026-02-01 > Changed", "text": ANY},
        {"header": "octo/repo · PR #1 · Change 1 · Changelog > [0.1.0] - 2026-01-01 > Changed", "text": ANY},
        {"header": "octo/repo · PR #1 · Change 1 · Changelog > [0.1.0] - 2026-01-01 > Fixed", "text": ANY},
    ]
    assert [passage["text"] for passage in source["passages"]] == [
        "### Changed\n\nFaster search.",
        "### Changed\n\nFirst release.",
        "### Fixed\n\nA crash.",
    ]


async def test_read_a_section_returns_every_passage_under_that_heading(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    document = make_document(1, CHANGELOG)
    path = index_of(tmp_path, fake_embedder, chunk_document(document))

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": document.url, "section": "Changed"})

    assert result.structured_content is not None
    assert [passage["text"] for passage in result.structured_content["passages"]] == [
        "### Changed\n\nFaster search.",
        "### Changed\n\nFirst release.",
    ]


async def test_read_a_section_returns_all_the_pieces_of_a_split_section(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    long_section = "\n\n".join(f"Paragraph {number} of the design." for number in range(8))
    document = make_document(1, f"## Design\n\n{long_section}\n\n## Rollout\n\nShip it.")
    chunks = chunk_document(document, max_tokens=30)
    path = index_of(tmp_path, fake_embedder, chunks)

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": document.url, "section": "Design"})

    assert result.structured_content is not None
    passages = result.structured_content["passages"]
    assert len(passages) == sum(1 for chunk in chunks if chunk.section == "Design") > 1
    assert "Paragraph 0" in passages[0]["text"]
    assert "Paragraph 7" in passages[-1]["text"]


async def test_read_a_section_that_isnt_there_names_the_sections_that_are(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    document = make_document(1, CHANGELOG)
    path = index_of(tmp_path, fake_embedder, chunk_document(document))

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": document.url, "section": "Removed"})

    assert error_text(result).endswith("No section 'Removed' in Change 1; its sections are: Changed, Fixed.")


async def test_read_a_url_that_isnt_indexed_says_to_pass_a_search_results(
    index_file: Path, fake_embedder: FakeEmbedder
) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": "https://github.com/octo/repo/pull/404"})

    assert error_text(result).endswith("pass the url of a search result.")


@pytest.mark.parametrize(("extra", "first_page"), [(0, 2), (1, 1)])
async def test_read_returns_at_most_a_page_of_passages(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory, extra: int, first_page: int
) -> None:
    document = make_document(1)
    half = READ_CHARS // 2
    path = index_of(tmp_path, fake_embedder, [sized(document, 0, half), sized(document, 1, half + extra)])

    async with Client(server(path, fake_embedder)) as client:
        first = await client.call_tool("read", {"url": document.url})
        rest = await client.call_tool("read", {"url": document.url, "start": first_page})

    assert first.structured_content is not None
    assert rest.structured_content is not None
    assert len(first.structured_content["passages"]) == first_page
    assert first.structured_content["next_start"] == (None if first_page == 2 else 1)
    assert len(rest.structured_content["passages"]) == 2 - first_page
    assert rest.structured_content["next_start"] is None


async def test_read_returns_a_passage_longer_than_a_page_on_its_own(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    document = make_document(1)
    path = index_of(tmp_path, fake_embedder, [sized(document, 0, READ_CHARS * 2), sized(document, 1, 10)])

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": document.url})

    assert result.structured_content is not None
    assert [len(passage["text"]) for passage in result.structured_content["passages"]] == [READ_CHARS * 2 - 6]
    assert result.structured_content["next_start"] == 1


async def test_read_pages_through_a_section_with_the_same_section(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    document = make_document(1)
    big = READ_CHARS * 2 // 3
    chunks = [sized(document, 0, big, "Design"), sized(document, 1, 10, "Rollout"), sized(document, 2, big, "Design")]
    path = index_of(tmp_path, fake_embedder, chunks)

    async with Client(server(path, fake_embedder)) as client:
        first = await client.call_tool("read", {"url": document.url, "section": "Design"})
        assert first.structured_content is not None
        start = first.structured_content["next_start"]
        rest = await client.call_tool("read", {"url": document.url, "section": "Design", "start": start})

    assert rest.structured_content is not None
    assert [passage["header"] for passage in first.structured_content["passages"]] == ["part 0"]
    assert [passage["header"] for passage in rest.structured_content["passages"]] == ["part 2"]
    assert rest.structured_content["next_start"] is None


async def test_read_a_section_of_a_source_without_headings_says_it_has_none(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    document = make_document(1, "No headings here.")
    path = index_of(tmp_path, fake_embedder, chunk_document(document))

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": document.url, "section": "Design"})

    assert error_text(result).endswith("No section 'Design' in Change 1; it has no headed sections.")


async def test_read_past_the_last_passage_returns_no_passages(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    document = make_document(1, CHANGELOG)
    path = index_of(tmp_path, fake_embedder, chunk_document(document))

    async with Client(server(path, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": document.url, "start": 3})

    assert result.structured_content is not None
    assert result.structured_content["passages"] == []
    assert result.structured_content["next_start"] is None


async def test_read_refuses_a_negative_start(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(server(index_file, fake_embedder)) as client:
        result = await client.call_tool("read", {"url": "https://github.com/octo/repo/pull/1", "start": -1})

    assert result.is_error
