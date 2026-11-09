from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import SharedServer
from starlette.testclient import TestClient

from repolore import __version__, relay
from repolore.locks import hold
from repolore.paths import AppPaths
from repolore.refresh import STATUS_FILE, RefreshStatus
from repolore.relay import ServerUnavailable
from repolore.server_state import Discovery, ServerFiles, is_running, registered, write_discovery
from repolore.ui.app import create_app
from repolore.ui.status_page import LOG_BYTES, LOG_LINES, ServerView, log_tail, took, when
from repolore.ui.tools_page import TOOLS

TOKEN = "the-token"
BASE = "http://127.0.0.1:1234"
FROM_THE_PAGE = {"Origin": BASE}
STARTED = "2026-11-08T10:00:00+00:00"


def signed_in(paths: AppPaths) -> TestClient:
    paths.config_file.touch()
    client = TestClient(create_app(paths, "default", TOKEN), base_url=BASE, follow_redirects=False)
    assert client.get(f"/?token={TOKEN}").status_code == 303
    return client


@pytest.fixture
def browser(data_paths: AppPaths) -> TestClient:
    return signed_in(data_paths)


@pytest.fixture
def files(data_paths: AppPaths) -> ServerFiles:
    return ServerFiles(data_paths.data_dir, "default")


def discovery(**changes: Any) -> Discovery:
    fields = {"pid": 4242, "port": 61000, "token": "secret", "version": __version__, "started_at": STARTED}
    return Discovery(**{**fields, **changes})


def page_with_server(browser: TestClient, files: ServerFiles, found: Discovery) -> str:
    write_discovery(files, found)
    with hold(files.lock, timeout=0):
        return browser.get("/").text


def test_the_index_card_counts_passages_and_lists_each_source(browser: TestClient) -> None:
    html = browser.get("/").text

    assert "<strong>index</strong>: " in html
    assert "passages from 2 sources, built today at " in html
    assert "<code>octo/repo</code>" in html
    assert "<code>octo/charts</code>" in html
    assert html.index("<code>octo/repo</code>") < html.index("<code>octo/charts</code>")


def test_no_index_shows_no_table(tmp_path: Path) -> None:
    paths = AppPaths(config_file=tmp_path / "repolore.toml", questions_file=tmp_path / "q.yaml", data_dir=tmp_path)

    html = signed_in(paths).get("/").text

    assert "<table" not in html
    assert "Fix: run `repolore refresh`" in html


def test_an_index_that_cant_be_read_says_how_to_fix_it(browser: TestClient, data_paths: AppPaths) -> None:
    data_paths.index_file("default").write_bytes(b"not a database")

    html = browser.get("/").text

    assert "<table" not in html
    assert "Fix: run `repolore refresh`" in html


def test_a_finished_refresh_says_when_and_how_long_it_took(browser: TestClient, data_paths: AppPaths) -> None:
    start = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
    finished = RefreshStatus(
        started_at=start.isoformat(), finished_at=(start + timedelta(seconds=220)).isoformat(), ok=True
    )
    (data_paths.data_dir / STATUS_FILE).write_text(finished.model_dump_json(), encoding="utf-8")

    assert "in 3 min 40 s" in browser.get("/").text


def test_a_stopped_server_offers_to_start_and_loads_nothing(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(relay, "find_or_start", lambda _: pytest.fail("the page started the server"))

    html = browser.get("/").text

    assert "○ Stopped" in html
    assert 'action="/server/start"' in html
    assert 'action="/server/stop"' not in html


def test_a_running_server_shows_where_it_listens_and_offers_to_stop(browser: TestClient, files: ServerFiles) -> None:
    html = page_with_server(browser, files, discovery())

    assert "● Running, 0 sessions" in html
    assert "http://127.0.0.1:61000/mcp" in html
    assert f"version {__version__}, model not loaded yet" in html
    assert "as pid 4242" in html
    assert 'action="/server/stop"' in html
    assert 'action="/server/start"' not in html
    assert "secret" not in html


def test_a_server_that_has_loaded_the_model_says_so(browser: TestClient, files: ServerFiles) -> None:
    assert "model loaded" in page_with_server(browser, files, discovery(model_ready=True))


def test_sessions_are_counted(browser: TestClient, files: ServerFiles) -> None:
    write_discovery(files, discovery())
    with hold(files.lock, timeout=0), registered(files):
        assert "● Running, 1 session<" in browser.get("/").text


def test_a_server_of_another_version_says_the_next_session_replaces_it(browser: TestClient, files: ServerFiles) -> None:
    html = page_with_server(browser, files, discovery(version="0.0.1"))

    assert f"The next session replaces it with {__version__}." in html


def test_a_server_that_hasnt_written_its_address_is_starting(browser: TestClient, files: ServerFiles) -> None:
    with hold(files.lock, timeout=0):
        html = browser.get("/").text

    assert "○ Starting" in html
    assert 'action="/server/stop"' not in html


@pytest.mark.parametrize(("view", "label"), [(ServerView(False), "Stopped"), (ServerView(True), "Starting")])
def test_the_server_label_without_an_address(view: ServerView, label: str) -> None:
    assert view.label == label


@pytest.mark.parametrize(("action", "name"), [("stop", "stop_server"), ("start", "find_or_start")])
def test_a_server_action_goes_back_to_the_page(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch, files: ServerFiles, action: str, name: str
) -> None:
    called: list[ServerFiles] = []
    monkeypatch.setattr(relay, name, called.append)

    response = browser.post(f"/server/{action}", headers=FROM_THE_PAGE)

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert called == [files]


@pytest.mark.parametrize(
    ("action", "name", "failure"),
    [
        ("stop", "stop_server", ServerUnavailable("didn't <b>stop</b>")),
        ("start", "find_or_start", ServerUnavailable("didn't <b>start</b>")),
        ("start", "find_or_start", OSError("no <b>such</b> program")),
    ],
)
def test_a_server_action_that_fails_is_reported(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch, action: str, name: str, failure: Exception
) -> None:
    def fail(_: ServerFiles) -> None:
        raise failure

    monkeypatch.setattr(relay, name, fail)

    response = browser.post(f"/server/{action}", headers=FROM_THE_PAGE)

    assert response.status_code == 502
    assert "&lt;b&gt;" in response.text
    assert 'class="notice"' in response.text


@pytest.mark.parametrize("action", ["stop", "start"])
@pytest.mark.parametrize("headers", [{}, {"Origin": "http://evil.example"}])
def test_a_server_action_the_page_didnt_make_is_refused(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch, action: str, headers: dict[str, str]
) -> None:
    monkeypatch.setattr(relay, "stop_server", lambda _: pytest.fail("stopped"))
    monkeypatch.setattr(relay, "find_or_start", lambda _: pytest.fail("started"))

    assert browser.post(f"/server/{action}", headers=headers).status_code == 403


def test_stop_server_ends_a_real_server(browser: TestClient, shared_server: SharedServer) -> None:
    shared_server.start()

    response = browser.post("/server/stop", headers=FROM_THE_PAGE)

    assert response.status_code == 303
    assert not is_running(shared_server.files)


def test_the_log_is_shown_as_text(browser: TestClient, files: ServerFiles) -> None:
    files.log.write_text("Serving <b>default</b>\n", encoding="utf-8")

    html = browser.get("/").text

    assert "Show log" in html
    assert "Serving &lt;b&gt;default&lt;/b&gt;" in html


def test_no_log_says_so(browser: TestClient) -> None:
    assert "Nothing logged yet." in browser.get("/").text


def test_the_tools_it_serves_are_listed(browser: TestClient) -> None:
    html = browser.get("/").text

    for tool in TOOLS:
        assert f"<code>{tool.name}</code>" in html
    assert "starts a refresh" in html


@pytest.mark.parametrize(
    ("path", "label"), [("/", "Status"), ("/tools", "Tools"), ("/sources", "Sources"), ("/clients", "Clients")]
)
def test_the_navigation_marks_the_current_page(browser: TestClient, path: str, label: str) -> None:
    html = browser.get(path).text

    assert f'<a href="{path}" aria-current="page">{label}</a>' in html
    assert html.count('aria-current="page"') == 1


def test_a_missing_log_is_empty(tmp_path: Path) -> None:
    assert log_tail(tmp_path / "none.log") == ""


def test_an_empty_log_is_empty(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.touch()

    assert log_tail(log) == ""


def test_one_line_is_kept_whole(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.write_bytes(b"only line\n")

    assert log_tail(log) == "only line"


def test_a_long_log_keeps_its_last_lines(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.write_bytes(b"".join(b"line %d\n" % number for number in range(LOG_LINES * 2)))

    lines = log_tail(log).splitlines()

    assert len(lines) == LOG_LINES
    assert lines[-1] == f"line {LOG_LINES * 2 - 1}"


def test_a_log_over_the_read_limit_drops_the_cut_line(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.write_bytes(b"x" * LOG_BYTES + b"\nwhole\n")

    assert log_tail(log) == "whole"


def test_bytes_that_arent_text_dont_break_the_page(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.write_bytes(b"ok \xff\n")

    assert log_tail(log).startswith("ok ")


NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC).astimezone()


def test_a_moment_today_shows_only_the_time() -> None:
    assert when(NOW.isoformat(), NOW) == f"today at {NOW:%H:%M}"


def test_a_moment_on_another_day_shows_the_date() -> None:
    earlier = NOW - timedelta(days=2)

    assert when(earlier.isoformat(), NOW) == f"on {earlier:%Y-%m-%d} at {earlier:%H:%M}"


def test_a_moment_that_cant_be_read_is_shown_as_it_is() -> None:
    assert when("sometime") == "sometime"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0 s"), (59, "59 s"), (60, "1 min 0 s"), (3599, "59 min 59 s"), (3600, "1 h 0 min"), (3900, "1 h 5 min")],
)
def test_durations_read_in_the_largest_units(seconds: int, text: str) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    assert took(start.isoformat(), (start + timedelta(seconds=seconds)).isoformat()) == text
