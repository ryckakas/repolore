import subprocess
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from repolore.clients import Entry, McpClient, RegistrationError
from repolore.paths import AppPaths
from repolore.ui import client_pages
from repolore.ui.app import create_app

TOKEN = "the-token"
BASE = "http://127.0.0.1:1234"
FROM_THE_PAGE = {"Origin": BASE}
USER_ENTRY = Entry("repolore", ("repolore-mcp",), "user", "`remove user`")
PROJECT_ENTRY = Entry("repolore", ("repolore-mcp",), "project", "`remove project`", for_every_project=False)


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    return AppPaths(config_file=tmp_path / "repolore.toml", questions_file=tmp_path / "q.yaml", data_dir=tmp_path)


@pytest.fixture
def browser(paths: AppPaths) -> TestClient:
    client = TestClient(create_app(paths, "default", TOKEN), base_url=BASE, follow_redirects=False)
    assert client.get(f"/?token={TOKEN}").status_code == 303
    return client


class FakeClient:
    """A client whose registrations a successful add changes, as a real one's do."""

    def __init__(self, entries: list[Entry] | Exception | None = None, refusal: Exception | None = None) -> None:
        self.found = [] if entries is None else entries
        self.refusal = refusal
        self.added = 0

    def entries(self) -> list[Entry]:
        if isinstance(self.found, Exception):
            raise self.found
        return self.found

    def add(self) -> str:
        if self.refusal is not None:
            raise self.refusal
        self.added += 1
        self.found = [USER_ENTRY]
        return "for every project (user scope)"

    def client(self, present: bool = True) -> McpClient:
        def health(_: str) -> str:
            pytest.fail("the page asked the client about the connection, which starts the shared server")

        return McpClient("fake", "Fake", lambda: present, self.entries, self.add, "`fix it by hand`", health)


def use(monkeypatch: pytest.MonkeyPatch, *clients: McpClient) -> None:
    monkeypatch.setattr(client_pages, "CLIENTS", clients)


def register(browser: TestClient, key: str = "fake") -> Any:
    return browser.post("/clients/register", data={"client": key}, headers=FROM_THE_PAGE)


def test_the_clients_page_loads_the_list_afterwards(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch, paths: AppPaths
) -> None:
    fake = FakeClient(RegistrationError("the page waited for it"))
    use(monkeypatch, fake.client())
    paths.config_file.write_bytes(b'include = ["me/alpha"]\n')

    html = browser.get("/clients").text

    assert 'data-load="/clients/list"' in html
    assert "the page waited for it" not in html


def test_a_client_without_repolore_can_be_registered(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeClient().client())

    html = browser.get("/clients/list").text

    assert "repolore isn&#39;t registered" in html
    assert "Register with Fake" in html
    assert 'name="client" value="fake"' in html
    assert "fix it by hand" not in html


def test_a_client_that_isnt_here_isnt_listed(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeClient().client(present=False))

    html = browser.get("/clients/list").text

    assert "No MCP client was found" in html
    assert "Not found: Fake." in html
    assert "Register with Fake" not in html


def test_a_client_already_registered_offers_no_button(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeClient([USER_ENTRY]).client())

    html = browser.get("/clients/list").text

    assert "repolore is registered, connection not checked" in html
    assert "Register with" not in html


def test_a_project_only_registration_offers_no_button(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeClient([PROJECT_ENTRY]).client())

    assert "Register with" not in browser.get("/clients/list").text


def test_a_client_with_two_registrations_says_how_to_keep_one(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, FakeClient([USER_ENTRY, PROJECT_ENTRY]).client())

    html = browser.get("/clients/list").text

    assert "registered 2 times" in html
    assert "keep one: `remove user`; `remove project`" in html
    assert "Register with" not in html


def test_a_client_that_cant_be_listed_offers_no_button(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeClient(RegistrationError("<b>unreadable</b>")).client())

    html = browser.get("/clients/list").text

    assert "can&#39;t list its MCP servers" in html
    assert "&lt;b&gt;unreadable&lt;/b&gt;" in html
    assert "Register with" not in html


def test_registering_adds_it_and_shows_the_new_state(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    use(monkeypatch, fake.client())

    response = register(browser)

    assert response.status_code == 200
    assert fake.added == 1
    assert "Fake: registered repolore for every project (user scope)." in response.text
    assert "repolore is registered, connection not checked" in response.text
    assert "Register with" not in response.text


@pytest.mark.parametrize(
    "refusal",
    [RegistrationError("no <i>way</i>"), OSError("no <i>way</i>"), subprocess.TimeoutExpired("no <i>way</i>", 60)],
)
def test_a_client_that_fails_to_register_says_why_and_can_be_tried_again(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch, refusal: Exception
) -> None:
    use(monkeypatch, FakeClient(refusal=refusal).client())

    response = register(browser)

    assert response.status_code == 400
    assert "Couldn&#39;t register repolore" in response.text
    assert "&lt;i&gt;way&lt;/i&gt;" in response.text
    assert "Register with Fake" in response.text


def test_registering_twice_doesnt_add_it_twice(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    use(monkeypatch, fake.client())

    assert register(browser).status_code == 200
    response = register(browser)

    assert response.status_code == 400
    assert "Fake already has it." in response.text
    assert fake.added == 1


@pytest.mark.parametrize("key", ["other", "", "<b>"])
def test_only_a_client_in_the_table_can_be_registered(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    fake = FakeClient()
    use(monkeypatch, fake.client())

    response = register(browser, key)

    assert response.status_code == 400
    assert "<b>" not in response.text
    assert fake.added == 0


def test_a_client_that_isnt_here_cant_be_registered(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient()
    use(monkeypatch, fake.client(present=False))

    assert register(browser).status_code == 400
    assert fake.added == 0


def test_no_client_field_is_refused(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeClient().client())

    assert browser.post("/clients/register", data={}, headers=FROM_THE_PAGE).status_code == 400


@pytest.mark.parametrize("headers", [{}, {"Origin": "http://evil.example"}])
def test_a_registration_the_page_didnt_make_is_refused(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    fake = FakeClient()
    use(monkeypatch, fake.client())

    response = browser.post("/clients/register", data={"client": "fake"}, headers=headers)

    assert response.status_code == 403
    assert fake.added == 0


def test_the_client_list_needs_the_cookie(paths: AppPaths) -> None:
    client = TestClient(create_app(paths, "default", TOKEN), base_url=BASE, follow_redirects=False)

    assert client.get("/clients/list").status_code == 403


def test_clients_that_arent_here_are_named(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    here = FakeClient()
    other = McpClient("other", "Other", lambda: False, here.entries, here.add, "`x`")
    use(monkeypatch, here.client(), other)

    html = browser.get("/clients/list").text

    assert "Not found: Other." in html
    assert "<strong>Fake</strong>" in html


def test_every_client_here_says_nothing_is_missing(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    use(monkeypatch, FakeClient().client())

    assert "Not found" not in browser.get("/clients/list").text


def test_a_broken_registration_shows_its_fix_and_no_button(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gone = Entry("repolore", (str(tmp_path / "gone" / "repolore-mcp"),), "user", "`remove it`")
    use(monkeypatch, FakeClient([gone]).client())

    html = browser.get("/clients/list").text

    assert "no longer exists" in html
    assert "`remove it`" in html
    assert "Register with" not in html


def test_the_clients_page_shows_the_snippet_and_the_instructions(browser: TestClient) -> None:
    html = browser.get("/clients").text

    assert "repolore-mcp" in html
    assert "data-copy" in html
    assert "Searches the user&#39;s own engineering history on GitHub" in html
