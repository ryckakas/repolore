import subprocess
import sys
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from conftest import DocumentFactory, FakeEmbedder, SharedServer
from mcp.client import Client
from mcp.server import MCPServer
from mcp.types import CallToolRequestParams, CallToolResult
from starlette.testclient import TestClient

from repolore import mcp_server, relay
from repolore.chunking import chunk_document
from repolore.documents import DocumentKind
from repolore.mcp_server import create_server
from repolore.paths import AppPaths
from repolore.relay import Relay, ServerUnavailable
from repolore.server_state import ServerFiles
from repolore.store import connect, write_index
from repolore.ui.app import create_app
from repolore.ui.tools_page import MAX_K, TOOLS, Hit, _elapsed

TOKEN = "the-token"
BASE = "http://127.0.0.1:1234"
FROM_THE_PAGE = {"Origin": BASE}


class InProcessRelay:
    """Forwards to a real server over the SDK's in-memory transport, where the real relay would go over HTTP."""

    def __init__(self, server: MCPServer, calls: list[dict[str, Any]]) -> None:
        self._server = server
        self._calls = calls

    async def call_tool(self, _: object, params: CallToolRequestParams) -> CallToolResult:
        self._calls.append(params.arguments or {})
        async with Client(self._server) as client:
            return await client.call_tool(params.name, params.arguments)


def browser_of(paths: AppPaths, db: str = "default") -> TestClient:
    client = TestClient(create_app(paths, db, TOKEN), base_url=BASE, follow_redirects=False)
    assert client.get(f"/?token={TOKEN}").status_code == 303
    return client


def search(browser: TestClient, query: str, **filters: str) -> Any:
    return browser.post("/tools/search", data={"query": query, **filters}, headers=FROM_THE_PAGE)


def serve_tools(
    monkeypatch: pytest.MonkeyPatch, paths: AppPaths, embedder: FakeEmbedder, db: str = "default"
) -> list[dict[str, Any]]:
    server = create_server(paths, db, lambda _: embedder)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(relay, "Relay", lambda _: InProcessRelay(server, calls))
    return calls


def no_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(relay, "Relay", lambda _: pytest.fail("the page started a search"))


def index_of(paths: AppPaths, embedder: FakeEmbedder, *documents: Any) -> None:
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    with closing(connect(paths.index_file("default"))) as db:
        write_index(
            db, chunks, embedder.embed_documents([c.embedded_text for c in chunks]), embedder.model_id, embedder.dim
        )


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    return AppPaths(config_file=tmp_path / "repolore.toml", questions_file=tmp_path / "q.yaml", data_dir=tmp_path)


def test_the_page_offers_the_filters_and_doesnt_search(paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    no_relay(monkeypatch)

    html = browser_of(paths).get("/tools").text

    assert 'action="/tools/search"' in html
    assert 'name="repo"' in html
    for kind in DocumentKind:
        assert f'<option value="{kind.value}">' in html
    assert 'href="/tools" aria-current="page"' in html


def test_a_search_reaches_the_real_server_through_the_real_relay(
    shared_server: SharedServer, data_paths: AppPaths
) -> None:
    shared_server.start()

    response = search(browser_of(data_paths), "solar battery Modbus")

    assert response.status_code == 200
    assert "Change 3" in response.text
    assert "octo/repo · issue · score " in response.text


def test_hits_show_their_title_repo_kind_score_and_text(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = serve_tools(monkeypatch, data_paths, fake_embedder)

    response = search(browser_of(data_paths), "solar battery Modbus")

    assert response.status_code == 200
    assert (
        '<a href="https://github.com/octo/repo/issues/3" target="_blank" rel="noopener">Change 3</a>' in response.text
    )
    assert "octo/repo · issue · score " in response.text
    assert "The solar battery reports power over Modbus TCP." in response.text
    assert calls == [{"query": "solar battery Modbus", "distinct": False}]


@pytest.mark.parametrize(
    ("filters", "expected", "left_out"),
    [
        ({"kind": "issue"}, "Change 3", "Change 1<"),
        ({"repo": "octo/charts"}, "Change 2", "Change 1<"),
    ],
)
def test_the_filters_narrow_the_hits(
    data_paths: AppPaths,
    index_file: Path,
    fake_embedder: FakeEmbedder,
    monkeypatch: pytest.MonkeyPatch,
    filters: dict[str, str],
    expected: str,
    left_out: str,
) -> None:
    calls = serve_tools(monkeypatch, data_paths, fake_embedder)

    html = search(browser_of(data_paths), "registry", **filters).text

    assert expected in html
    assert left_out not in html
    assert calls == [{"query": "registry", "distinct": False, **filters}]


def test_empty_filters_arent_sent(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = serve_tools(monkeypatch, data_paths, fake_embedder)

    search(browser_of(data_paths), "  registry  ", kind="", repo="  ")

    assert calls == [{"query": "registry", "distinct": False}]


def test_no_hits_says_so(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_tools(monkeypatch, data_paths, fake_embedder)

    response = search(browser_of(data_paths), "registry", repo="no/such")

    assert response.status_code == 200
    assert "No matches." in response.text


def test_no_index_says_how_to_build_one(
    paths: AppPaths, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_tools(monkeypatch, paths, fake_embedder)

    response = search(browser_of(paths), "registry")

    assert response.status_code == 502
    assert 'class="notice"' in response.text
    assert "No index at" in response.text
    assert "repolore refresh" in response.text


@pytest.mark.parametrize(
    ("failure", "said"),
    [
        (ServerUnavailable("The shared server didn't start within 120 seconds"), "didn&#39;t start"),
        (OSError("no <b>such</b> program"), "no &lt;b&gt;such&lt;/b&gt; program"),
    ],
)
def test_a_server_that_cant_start_is_reported(
    paths: AppPaths, monkeypatch: pytest.MonkeyPatch, failure: Exception, said: str
) -> None:
    def unavailable(_: ServerFiles) -> Any:
        raise failure

    monkeypatch.setattr(relay, "Relay", partial(Relay, find=unavailable))

    response = search(browser_of(paths), "registry")

    assert response.status_code == 502
    assert said in response.text


def test_a_filter_the_tool_refuses_is_reported(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_tools(monkeypatch, data_paths, fake_embedder)

    response = search(browser_of(data_paths), "registry", kind="<b>nonsense</b>")

    assert response.status_code == 502
    assert "<b>nonsense" not in response.text


class AnsweringRelay:
    def __init__(self, answer: Any) -> None:
        self._answer = answer

    async def call_tool(self, _: object, __: CallToolRequestParams) -> CallToolResult:
        return CallToolResult(content=[], structured_content=self._answer)


@pytest.mark.parametrize("answer", [None, {}, {"results": "x"}, {"results": [{"title": "only"}]}])
def test_an_answer_of_another_shape_is_reported(paths: AppPaths, monkeypatch: pytest.MonkeyPatch, answer: Any) -> None:
    monkeypatch.setattr(relay, "Relay", lambda _: AnsweringRelay(answer))

    response = search(browser_of(paths), "registry")

    assert response.status_code == 502
    assert "unexpected form" in response.text


@pytest.mark.parametrize("query", ["", "   "])
def test_an_empty_query_doesnt_search(paths: AppPaths, monkeypatch: pytest.MonkeyPatch, query: str) -> None:
    no_relay(monkeypatch)

    response = search(browser_of(paths), query)

    assert response.status_code == 400
    assert "Type what to look for." in response.text


def test_no_query_field_doesnt_search(paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    no_relay(monkeypatch)

    assert browser_of(paths).post("/tools/search", data={}, headers=FROM_THE_PAGE).status_code == 400


def test_html_in_a_hit_is_shown_as_text(
    paths: AppPaths, fake_embedder: FakeEmbedder, make_document: DocumentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    title = "<img src=x onerror=alert(1)>"
    index_of(paths, fake_embedder, make_document(1, "registry <script>alert(2)</script>", title=title))
    serve_tools(monkeypatch, paths, fake_embedder)

    html = search(browser_of(paths), "registry").text

    assert "<img" not in html
    assert "<script" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in html


def test_a_local_document_shows_its_path_and_no_link(
    paths: AppPaths,
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    make_document: DocumentFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = tmp_path / "my notes" / "plan.md"
    index_of(paths, fake_embedder, make_document(1, "registry", kind=DocumentKind.LOCAL, url=file.as_uri()))
    serve_tools(monkeypatch, paths, fake_embedder)

    html = search(browser_of(paths), "registry").text

    assert '<a href="file:' not in html
    assert f"<code>{file}</code>" in html


def hit(url: str) -> Hit:
    return Hit(title="t", url=url, repo="o/r", kind="pr", section=None, score=1.0, text="x")


def test_a_github_url_is_a_link() -> None:
    assert hit("https://github.com/o/r/pull/1").link == "https://github.com/o/r/pull/1"


@pytest.mark.parametrize(
    "url",
    [
        "file:///home/me/plan.md",
        "javascript:alert(1)",
        "http://github.com/o/r",
        "https://github.com.example.org/o/r",
        "https://example.org/https://github.com/",
        "",
    ],
)
def test_anything_else_is_not_a_link(url: str) -> None:
    assert hit(url).link is None


def test_a_file_on_a_network_share_keeps_its_host() -> None:
    assert "nas" in hit("file://nas/share/plan.md").location


def test_a_url_that_isnt_a_file_is_shown_as_it_is() -> None:
    assert hit("https://example.org/x").location == "https://example.org/x"


def test_the_search_goes_to_the_server_of_this_index(paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[ServerFiles] = []

    def unavailable(files: ServerFiles) -> Any:
        seen.append(files)
        raise ServerUnavailable("not here")

    monkeypatch.setattr(relay, "Relay", partial(Relay, find=unavailable))

    search(browser_of(paths, "other"), "registry")

    assert {(files.data_dir, files.db) for files in seen} == {(paths.data_dir, "other")}


@pytest.mark.parametrize("headers", [{}, {"Origin": "http://evil.example"}])
def test_a_search_the_page_didnt_make_is_refused(
    paths: AppPaths, monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    no_relay(monkeypatch)

    assert browser_of(paths).post("/tools/search", data={"query": "x"}, headers=headers).status_code == 403


def test_the_search_page_needs_the_cookie(paths: AppPaths) -> None:
    client = TestClient(create_app(paths, "default", TOKEN), base_url=BASE, follow_redirects=False)

    assert client.get("/tools").status_code == 403


def test_the_page_loads_no_mcp_client_until_a_search() -> None:
    code = "import sys, repolore.ui.app; sys.exit('mcp' in sys.modules or 'repolore.relay' in sys.modules)"

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def run(browser: TestClient, tool: str, **fields: str) -> Any:
    return browser.post(f"/tools/{tool}", data=fields, headers=FROM_THE_PAGE)


def test_the_page_offers_every_tool_and_the_rest_of_the_search_options(
    paths: AppPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_relay(monkeypatch)

    html = browser_of(paths).get("/tools").text

    for tool in TOOLS:
        assert f'data-tool="{tool.name}"' in html
        assert f'action="/tools/{tool.name}"' in html
    for field in ("author", "k", "mode", "distinct", "url", "section", "start"):
        assert f'name="{field}"' in html
    assert '<option value="hybrid">' in html
    assert f'max="{MAX_K}"' in html


def test_the_tool_table_names_what_the_server_serves(paths: AppPaths, fake_embedder: FakeEmbedder) -> None:
    async def served() -> list[str]:
        async with Client(create_server(paths, "default", lambda _: fake_embedder)) as client:
            return [tool.name for tool in (await client.list_tools()).tools]

    assert sorted(anyio.run(served)) == sorted(tool.name for tool in TOOLS)


def test_every_search_option_reaches_the_tool(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = serve_tools(monkeypatch, data_paths, fake_embedder)

    response = run(
        browser_of(data_paths), "search", query="registry", author="octocat", k="3", mode="keyword", distinct="on"
    )

    assert response.status_code == 200
    assert calls == [{"query": "registry", "distinct": True, "author": "octocat", "mode": "keyword", "k": 3}]


@pytest.mark.parametrize("k", ["0", str(MAX_K + 1), "-1", "two", "1.5", "9" * 5000])
def test_a_k_out_of_range_doesnt_search(paths: AppPaths, monkeypatch: pytest.MonkeyPatch, k: str) -> None:
    no_relay(monkeypatch)

    response = run(browser_of(paths), "search", query="x", k=k)

    assert response.status_code == 400
    assert f"k must be a whole number from 1 to {MAX_K}." in response.text


@pytest.mark.parametrize("k", ["1", str(MAX_K)])
def test_the_ends_of_the_k_range_are_allowed(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch, k: str
) -> None:
    calls = serve_tools(monkeypatch, data_paths, fake_embedder)

    assert run(browser_of(data_paths), "search", query="registry", k=k).status_code == 200
    assert calls[0]["k"] == int(k)


def test_hits_offer_to_read_their_source(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_tools(monkeypatch, data_paths, fake_embedder)

    html = run(browser_of(data_paths), "search", query="solar battery Modbus").text

    assert 'data-read="https://github.com/octo/repo/issues/3"' in html
    assert 'data-view="passages"' in html
    assert 'data-view="json"' in html


def test_a_result_has_a_summary_and_its_json(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_tools(monkeypatch, data_paths, fake_embedder)

    html = run(browser_of(data_paths), "search", query="solar battery Modbus", k="1").text

    assert 'class="meta">1 passage · ' in html
    assert "&#34;query&#34;: &#34;solar battery Modbus&#34;" in html
    assert "data-copy" in html


def test_read_shows_the_source_passage_by_passage(
    paths: AppPaths, fake_embedder: FakeEmbedder, make_document: DocumentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_of(paths, fake_embedder, make_document(1, "# One\n\nfirst part\n\n# Two\n\nsecond part"))
    calls = serve_tools(monkeypatch, paths, fake_embedder)

    response = run(browser_of(paths), "read", url="https://github.com/octo/repo/pull/1")

    assert response.status_code == 200
    assert '<a href="https://github.com/octo/repo/pull/1" target="_blank" rel="noopener">Change 1</a>' in response.text
    assert "first part" in response.text
    assert "second part" in response.text
    assert "More passages follow" not in response.text
    assert calls == [{"url": "https://github.com/octo/repo/pull/1"}]


def test_read_says_when_more_passages_follow(
    paths: AppPaths, fake_embedder: FakeEmbedder, make_document: DocumentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    long = "x" * (mcp_server.READ_CHARS // 2)
    body = "\n\n".join(f"# Part {number}\n\n{long}" for number in range(4))
    index_of(paths, fake_embedder, make_document(1, body))
    serve_tools(monkeypatch, paths, fake_embedder)

    html = run(browser_of(paths), "read", url="https://github.com/octo/repo/pull/1").text

    assert "More passages follow. Read again with start " in html


def test_read_passes_its_section_and_start(
    paths: AppPaths, fake_embedder: FakeEmbedder, make_document: DocumentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_of(paths, fake_embedder, make_document(1, "# One\n\nfirst part\n\n# Two\n\nsecond part"))
    calls = serve_tools(monkeypatch, paths, fake_embedder)

    html = run(browser_of(paths), "read", url="https://github.com/octo/repo/pull/1", section="Two", start="0").text

    assert "second part" in html
    assert "first part" not in html
    assert calls == [{"url": "https://github.com/octo/repo/pull/1", "section": "Two", "start": 0}]


def test_read_of_a_local_document_shows_its_path_and_no_link(
    paths: AppPaths,
    tmp_path: Path,
    fake_embedder: FakeEmbedder,
    make_document: DocumentFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = tmp_path / "plan.md"
    index_of(paths, fake_embedder, make_document(1, "registry", kind=DocumentKind.LOCAL, url=file.as_uri()))
    serve_tools(monkeypatch, paths, fake_embedder)

    html = run(browser_of(paths), "read", url=file.as_uri()).text

    assert '<a href="file:' not in html
    assert f"<code>{file}</code>" in html


@pytest.mark.parametrize("url", ["", "   "])
def test_read_without_a_url_doesnt_call_the_tool(paths: AppPaths, monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    no_relay(monkeypatch)

    response = run(browser_of(paths), "read", url=url)

    assert response.status_code == 400
    assert "Give the url of a search result." in response.text


def test_a_negative_start_doesnt_call_the_tool(paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    no_relay(monkeypatch)

    response = run(browser_of(paths), "read", url="https://github.com/o/r/pull/1", start="-1")

    assert response.status_code == 400
    assert "start must be a whole number of 0 or more." in response.text


def test_reading_something_not_indexed_is_reported(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_tools(monkeypatch, data_paths, fake_embedder)

    response = run(browser_of(data_paths), "read", url="https://github.com/no/such")

    assert response.status_code == 502
    assert "Nothing indexed has the url" in response.text


@pytest.mark.parametrize("answer", [None, {}, {"title": "only"}])
def test_a_read_answer_of_another_shape_is_reported(
    paths: AppPaths, monkeypatch: pytest.MonkeyPatch, answer: Any
) -> None:
    monkeypatch.setattr(relay, "Relay", lambda _: AnsweringRelay(answer))

    response = run(browser_of(paths), "read", url="https://github.com/o/r/pull/1")

    assert response.status_code == 502
    assert "unexpected form" in response.text


def test_list_repos_answers_with_json_alone(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = serve_tools(monkeypatch, data_paths, fake_embedder)

    response = run(browser_of(data_paths), "list_repos")

    assert response.status_code == 200
    assert "octo/repo" in response.text
    assert 'data-view="passages"' not in response.text
    assert '<div data-view-panel="json" class="copyable">' in response.text
    assert calls == [{}]


def test_the_refresh_tool_starts_a_real_refresh(
    data_paths: AppPaths, index_file: Path, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[str] = []
    monkeypatch.setattr(mcp_server, "start_refresh", lambda _, db: started.append(db) or True)
    serve_tools(monkeypatch, data_paths, fake_embedder)

    response = run(browser_of(data_paths), "refresh")

    assert response.status_code == 200
    assert "Refresh started in the background." in response.text
    assert started == ["default"]


@pytest.mark.parametrize("name", ["other", "search2", "__class__"])
def test_only_a_tool_in_the_table_can_be_called(paths: AppPaths, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    no_relay(monkeypatch)

    response = run(browser_of(paths), name)

    assert response.status_code == 404
    assert "There&#39;s no such tool." in response.text


@pytest.mark.parametrize("name", [tool.name for tool in TOOLS])
def test_a_tool_call_the_page_didnt_make_is_refused(
    paths: AppPaths, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    no_relay(monkeypatch)

    assert browser_of(paths).post(f"/tools/{name}", data={"query": "x", "url": "y"}).status_code == 403


@pytest.mark.parametrize(("seconds", "text"), [(0.2124, "212 ms"), (0.9994, "999 ms"), (1.0, "1.0 s"), (3.46, "3.5 s")])
def test_the_time_a_call_took_reads_in_a_sensible_unit(seconds: float, text: str) -> None:
    assert _elapsed(seconds) == text
