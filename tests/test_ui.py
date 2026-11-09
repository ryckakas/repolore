import socket
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from repolore.doctor import Check
from repolore.paths import AppPaths
from repolore.refresh import REQUEST_FILE, STATUS_FILE, RefreshStatus
from repolore.ui import app as ui_app
from repolore.ui import status_page
from repolore.ui.app import HEADERS, create_app, serve

TOKEN = "the-token"
BASE = "http://127.0.0.1:1234"
FROM_THE_PAGE = {"Origin": BASE}
COOKIE = "repolore_ui_1234"
INDEX_LINE = "passages from"


def client_of(paths: AppPaths) -> TestClient:
    paths.config_file.touch()
    return TestClient(create_app(paths, "default", TOKEN), base_url=BASE, follow_redirects=False)


def signed_in(client: TestClient) -> TestClient:
    assert client.get(f"/?token={TOKEN}").status_code == 303
    return client


@pytest.fixture
def anonymous(data_paths: AppPaths) -> TestClient:
    return client_of(data_paths)


@pytest.fixture
def browser(anonymous: TestClient) -> TestClient:
    return signed_in(anonymous)


@pytest.fixture
def refreshes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    started: list[str] = []
    monkeypatch.setattr(status_page, "start_refresh", lambda _, db: started.append(db) or True)
    return started


def test_the_token_is_swapped_for_a_cookie_and_leaves_the_address(anonymous: TestClient) -> None:
    response = anonymous.get(f"/?token={TOKEN}")

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    cookie = response.headers["set-cookie"].lower()
    assert cookie.startswith(f"{COOKIE}={TOKEN}".lower())
    assert "httponly" in cookie
    assert "samesite=strict" in cookie


def test_the_cookie_opens_the_page(browser: TestClient) -> None:
    response = browser.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_a_first_run_with_no_config_opens_on_sources(browser: TestClient, data_paths: AppPaths) -> None:
    data_paths.config_file.unlink()

    response = browser.get("/")

    assert response.status_code == 303
    assert response.headers["location"] == "/sources"


def test_the_token_works_for_the_whole_run(data_paths: AppPaths) -> None:
    first, second = client_of(data_paths), client_of(data_paths)

    assert signed_in(first).get("/").status_code == 200
    assert signed_in(second).get("/").status_code == 200


@pytest.mark.parametrize("path", ["/", "/refresh", "/static/page.js", "/?token=wrong", "/?token="])
def test_without_the_cookie_nothing_is_served(anonymous: TestClient, path: str) -> None:
    response = anonymous.get(path)

    assert response.status_code == 403
    assert "repolore ui" in response.text
    assert COOKIE not in response.headers.get("set-cookie", "")


def test_a_wrong_cookie_is_refused(anonymous: TestClient) -> None:
    anonymous.cookies.set(COOKIE, "not-the-token")

    assert anonymous.get("/").status_code == 403


def test_the_token_only_works_on_the_front_page(anonymous: TestClient) -> None:
    assert anonymous.get(f"/refresh?token={TOKEN}").status_code == 403


@pytest.mark.parametrize("host", ["evil.example", "evil.example:1234", "localhost:1234"])
def test_another_host_is_refused(browser: TestClient, host: str) -> None:
    assert browser.get("/", headers={"Host": host}).status_code == 400


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Origin": "https://127.0.0.1:1234"},
    ],
)
@pytest.mark.parametrize("path", ["/refresh", "/checks"])
def test_a_post_that_the_page_didnt_make_is_refused(
    browser: TestClient, refreshes: list[str], monkeypatch: pytest.MonkeyPatch, path: str, headers: dict[str, str]
) -> None:
    monkeypatch.setattr(status_page, "diagnose", lambda *_: pytest.fail("the checks ran"))

    response = browser.post(path, headers=headers)

    assert response.status_code == 403
    assert refreshes == []


def test_a_post_without_the_cookie_is_refused(anonymous: TestClient, refreshes: list[str]) -> None:
    assert anonymous.post("/refresh", headers=FROM_THE_PAGE).status_code == 403
    assert refreshes == []


@pytest.mark.parametrize("path", ["/", "/refresh", "/static/page.js"])
def test_every_answer_has_the_security_headers(browser: TestClient, path: str) -> None:
    response = browser.get(path)

    assert response.status_code == 200
    assert {name: response.headers[name] for name in HEADERS} == HEADERS


def test_the_referrer_policy_keeps_the_origin_of_the_pages_own_forms(browser: TestClient) -> None:
    assert browser.get("/").headers["referrer-policy"] == "same-origin"


def test_a_refused_answer_has_the_security_headers_too(anonymous: TestClient) -> None:
    assert anonymous.get("/").headers["content-security-policy"] == HEADERS["Content-Security-Policy"]


@pytest.mark.parametrize("file", ["page.js", "page.css", "icon.svg"])
def test_the_page_serves_its_own_files(browser: TestClient, file: str) -> None:
    assert browser.get(f"/static/{file}").status_code == 200


def test_the_page_has_no_inline_script_or_style(browser: TestClient) -> None:
    html = browser.get("/").text

    assert '<script src="/static/page.js"' in html
    assert "<script>" not in html
    assert "<style" not in html
    assert " style=" not in html


def test_the_status_page_sums_up_the_index_and_the_server(browser: TestClient) -> None:
    html = browser.get("/").text

    assert INDEX_LINE in html
    assert "MCP server" in html
    assert "Not running. It starts when" in html
    assert "none yet" in html


def test_without_an_index_the_page_says_how_to_build_one(tmp_path: Path) -> None:
    paths = AppPaths(config_file=tmp_path / "repolore.toml", questions_file=tmp_path / "q.yaml", data_dir=tmp_path)

    html = signed_in(client_of(paths)).get("/").text

    assert "none at" in html
    assert "Fix: run `repolore refresh`" in html
    assert "Refresh now" in html


def test_a_running_refresh_disables_the_button(browser: TestClient, data_paths: AppPaths) -> None:
    (data_paths.data_dir / REQUEST_FILE).touch()

    html = browser.get("/").text

    assert "data-running" in html
    assert "disabled" in html
    assert "running" in html


def test_an_idle_refresh_has_a_working_button(browser: TestClient) -> None:
    html = browser.get("/").text

    assert "data-running" not in html
    assert "disabled" not in html


def test_a_failed_refresh_says_why(browser: TestClient, data_paths: AppPaths) -> None:
    failed = RefreshStatus(started_at="2026-11-08T10:00:00+00:00", finished_at="2026-11-08T10:01:00+00:00", ok=False)
    (data_paths.data_dir / STATUS_FILE).write_text(
        failed.model_copy(update={"error": "GitHub said no"}).model_dump_json(), encoding="utf-8"
    )

    html = browser.get("/").text

    assert "GitHub said no" in html
    assert 'class="fail"' in html


def test_the_refresh_state_is_a_fragment(browser: TestClient, data_paths: AppPaths) -> None:
    (data_paths.data_dir / REQUEST_FILE).touch()

    html = browser.get("/refresh").text

    assert html.startswith('<ul class="checks" id="refresh" data-running>')
    assert "<html" not in html


def test_refresh_now_starts_one_and_goes_back_to_the_page(browser: TestClient, refreshes: list[str]) -> None:
    response = browser.post("/refresh", headers=FROM_THE_PAGE)

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert refreshes == ["default"]


def test_a_refresh_that_cant_start_is_reported(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: Any) -> bool:
        raise OSError("no space left")

    monkeypatch.setattr(status_page, "start_refresh", fail)

    response = browser.post("/refresh", headers=FROM_THE_PAGE)

    assert response.status_code == 500
    assert "Couldn&#39;t start a refresh: no space left" in response.text
    assert INDEX_LINE in response.text


def run_checks(browser: TestClient, monkeypatch: pytest.MonkeyPatch, checks: list[Check]) -> str:
    monkeypatch.setattr(status_page, "diagnose", lambda *_: iter(checks))
    response = browser.post("/checks", headers=FROM_THE_PAGE)
    assert response.status_code == 200
    return response.text


def test_checks_come_back_as_a_fragment(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    html = run_checks(browser, monkeypatch, [Check("GitHub", True, "works")])

    assert '<li class="pass">' in html
    assert "<strong>GitHub</strong>: works" in html
    assert "<html" not in html


def test_a_failed_check_shows_its_fix(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    html = run_checks(browser, monkeypatch, [Check("model", False, "missing", "download it")])

    assert '<li class="fail">' in html
    assert "Fix: download it" in html


def test_a_warning_is_marked_as_one(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert '<li class="warning">' in run_checks(browser, monkeypatch, [Check("x", True, "y", warning=True)])


def test_no_checks_is_an_empty_list(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "<li" not in run_checks(browser, monkeypatch, [])


def test_html_in_a_check_is_shown_as_text(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    html = run_checks(browser, monkeypatch, [Check("<b>x</b>", False, "<script>alert(1)</script>", "<i>fix</i>")])

    assert "<script>" not in html
    assert "<b>" not in html
    assert "<i>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_serve_listens_on_loopback_before_it_gives_the_address(
    data_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    addresses: list[str] = []
    bound: list[tuple[str, int]] = []

    def run(_: object, sockets: list[socket.socket]) -> None:
        bound.append(sockets[0].getsockname())
        with socket.create_connection(sockets[0].getsockname(), timeout=5):
            pass

    monkeypatch.setattr(ui_app.uvicorn.Server, "run", run)

    serve(data_paths, "default", addresses.append)

    host, port = bound[0]
    assert host == "127.0.0.1"
    assert port > 0
    assert addresses[0].startswith(f"http://127.0.0.1:{port}/?token=")
    assert len(addresses[0].split("token=")[1]) >= 32


def test_each_run_has_its_own_token(data_paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    addresses: list[str] = []
    monkeypatch.setattr(ui_app.uvicorn.Server, "run", lambda *_, **__: None)

    serve(data_paths, "default", addresses.append)
    serve(data_paths, "default", addresses.append)

    assert addresses[0].split("token=")[1] != addresses[1].split("token=")[1]


def test_ctrl_c_ends_serve_quietly(data_paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(*_: Any, **__: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(ui_app.uvicorn.Server, "run", interrupted)

    serve(data_paths, "default", lambda _: None)
