import sys
from contextlib import closing
from pathlib import Path

import pytest
from conftest import DocumentFactory, FakeEmbedder
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import TextContent

from repolore import mcp_server
from repolore.chunking import Chunk, chunk_document
from repolore.mcp_server import TEXT_CHARS, create_server
from repolore.store import connect, write_index

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def index_file(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> Path:
    path = tmp_path / "db" / "default.db"
    with closing(connect(path)) as db:
        vectors = fake_embedder.embed_documents([chunk.embedded_text for chunk in corpus])
        write_index(db, corpus, vectors, fake_embedder.model_id, fake_embedder.dim)
    return path


async def test_tools_are_listed_as_read_only(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(create_server(index_file, lambda _: fake_embedder)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {"search", "list_repos"}
    assert all(tool.annotations and tool.annotations.read_only_hint for tool in tools.values())
    assert set(tools["search"].input_schema["properties"]) == {"query", "repo", "kind", "author", "k", "mode"}


@pytest.mark.parametrize("mode", [None, "vector", "keyword", "hybrid"])
async def test_search_modes(
    index_file: Path, fake_embedder: FakeEmbedder, mode: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_search = mcp_server.search_index
    seen = []

    def record_mode(db, embedder, query, k, filters, mode=mcp_server.Mode.VECTOR, reranker=None):
        seen.append(mode)
        return original_search(db, embedder, query, k, filters, mode, reranker)

    monkeypatch.setattr(mcp_server, "search_index", record_mode)
    async with Client(create_server(index_file, lambda _: fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "registry", **({"mode": mode} if mode else {})})

    assert not result.is_error
    assert result.structured_content is not None
    assert result.structured_content["results"]
    assert seen == [mcp_server.Mode(mode) if mode else mcp_server.Mode.VECTOR]


async def test_search_rejects_unknown_mode(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(create_server(index_file, lambda _: fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "registry", "mode": "invalid"})

    assert result.is_error


async def test_search_returns_ranked_passages_with_permalinks(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(create_server(index_file, lambda _: fake_embedder)) as client:
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
    async with Client(create_server(index_file, lambda _: fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "registry", "kind": "doc"})

    assert result.structured_content is not None
    assert [hit["url"] for hit in result.structured_content["results"]] == [
        "https://github.com/octo/repo/blob/main/README.md"
    ]


async def test_long_passages_are_shortened(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    chunks = chunk_document(make_document(1, "word " * TEXT_CHARS))
    path = tmp_path / "long.db"
    with closing(connect(path)) as db:
        write_index(
            db, chunks, fake_embedder.embed_documents([c.embedded_text for c in chunks]), "fake-bag-of-words", 64
        )

    async with Client(create_server(path, lambda _: fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "word"})

    assert result.structured_content is not None
    text = result.structured_content["results"][0]["text"]
    assert len(text) == TEXT_CHARS + 1
    assert text.endswith("…")


async def test_out_of_range_k_is_an_error_the_model_can_read(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(create_server(index_file, lambda _: fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "x", "k": 99})

    assert result.is_error


async def test_missing_index_says_how_to_build_it(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(create_server(tmp_path / "missing.db", lambda _: fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "x"})

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "run `repolore ingest` and `repolore build` first" in result.content[0].text


async def test_index_from_another_model_is_refused(
    index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeEmbedder, "model_id", "other-model")

    async with Client(create_server(index_file, lambda _: fake_embedder)) as client:
        result = await client.call_tool("search", {"query": "x"})

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "rebuild it" in result.content[0].text


async def test_list_repos_counts_passages_per_kind(index_file: Path, fake_embedder: FakeEmbedder) -> None:
    async with Client(create_server(index_file, lambda _: fake_embedder)) as client:
        result = await client.call_tool("list_repos", {})

    assert result.structured_content == {
        "repos": [
            {"repo": "octo/charts", "chunks": {"pr": 1}},
            {"repo": "octo/repo", "chunks": {"doc": 1, "issue": 1, "pr": 1}},
        ]
    }


async def test_server_speaks_mcp_over_stdio(index_file: Path) -> None:
    data_dir = index_file.parent.parent
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from repolore.mcp_server import main; main()", "--data-dir", str(data_dir)],
    )

    async with Client(params) as client:
        result = await client.call_tool("list_repos", {})

    assert result.structured_content is not None
    assert [repo["repo"] for repo in result.structured_content["repos"]] == ["octo/charts", "octo/repo"]


def test_main_serves_the_named_index_from_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[Path] = []

    class StubServer:
        def __init__(self, index_file: Path) -> None:
            self.index_file = index_file

        def run(self) -> None:
            served.append(self.index_file)

    monkeypatch.setattr(mcp_server, "create_server", lambda index_file, _: StubServer(index_file))
    monkeypatch.setattr(sys, "argv", ["repolore-mcp", "--data-dir", str(tmp_path), "--db", "experiment"])

    with pytest.raises(SystemExit) as exit_info:
        mcp_server.main()

    assert exit_info.value.code == 0
    assert served == [tmp_path.resolve() / "db" / "experiment.db"]
