from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from repolore.config import load_config
from repolore.discovery import Discovery, RepoSource, Scope
from repolore.doctor import Check
from repolore.paths import AppPaths
from repolore.refresh import REQUEST_FILE
from repolore.sources.github import RATE_LIMIT_FLOOR, GitHubError, TokenInfo
from repolore.ui import setup_pages
from repolore.ui.app import create_app

TOKEN = "the-token"
BASE = "http://127.0.0.1:1234"
FROM_THE_PAGE = {"Origin": BASE}
SECRET = "ghp_a-very-secret-token"
WORKS = Check("GitHub", True, "me, token from gh, scopes repo")


def source(name: str, scope: Scope = Scope.OWNED) -> RepoSource:
    return RepoSource(name, scope)


class Closable:
    def close(self) -> None:
        pass


@pytest.fixture
def paths(tmp_path: Path) -> AppPaths:
    return AppPaths(config_file=tmp_path / "repolore.toml", questions_file=tmp_path / "q.yaml", data_dir=tmp_path)


@pytest.fixture
def browser(paths: AppPaths) -> TestClient:
    client = TestClient(create_app(paths, "default", TOKEN), base_url=BASE, follow_redirects=False)
    assert client.get(f"/?token={TOKEN}").status_code == 303
    return client


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every repo discovery finds; each discover call appends the config it was given."""
    monkeypatch.setattr(setup_pages, "resolve_token", lambda: SECRET)
    monkeypatch.setattr(setup_pages.GitHubClient, "connect", lambda _: Closable())
    monkeypatch.setattr(
        setup_pages, "github_access", lambda _: (WORKS, TokenInfo("me", frozenset({"repo"}), 5000, 5000))
    )
    configs: list[Any] = []

    def discover(_: object, config: Any) -> Discovery:
        configs.append(config)
        selected = [source("me/alpha"), source("org/gamma", Scope.MEMBER), source("local/notes", Scope.LOCAL)]
        return Discovery(
            "me", [s for s in selected if config.selects(s.name)], [s for s in selected if not config.selects(s.name)]
        )

    monkeypatch.setattr(setup_pages, "discover", discover)
    return configs


def config_with(paths: AppPaths, text: str) -> None:
    paths.config_file.write_bytes(text.encode("utf-8"))


def is_ticked(html: str, name: str) -> bool:
    return f'value="{name}" checked' in html


def save_repos(browser: TestClient, ticked: list[str], *, every: bool = False, shown: list[str] | None = None) -> Any:
    fields = {"repo": ticked, "shown": ["me/alpha", "org/gamma"] if shown is None else shown}
    if every:
        fields["every"] = ["on"]
    return browser.post("/setup/repos", data=fields, headers=FROM_THE_PAGE)


def save_local(browser: TestClient, roots: str, patterns: str) -> Any:
    return browser.post("/setup/local", data={"roots": roots, "patterns": patterns}, headers=FROM_THE_PAGE)


def test_the_setup_page_opens_at_once_and_loads_the_repos_afterwards(
    browser: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_pages, "discover", lambda *_: pytest.fail("the page waited for GitHub"))

    html = browser.get("/setup").text

    assert 'data-load="/setup/repos"' in html
    assert 'action="/setup/local"' in html


def test_the_setup_page_shows_the_current_local_settings(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, 'local_roots = ["/one", "/two"]\nlocal_patterns = ["*.md"]\n')

    html = browser.get("/setup").text

    assert ">/one\n/two</textarea>" in html.replace("\\", "/")
    assert ">*.md</textarea>" in html


def test_the_setup_page_is_empty_with_no_config(browser: TestClient) -> None:
    html = browser.get("/setup").text

    assert 'name="roots" rows="3" spellcheck="false"></textarea>' in html
    assert 'name="patterns" rows="3" spellcheck="false"></textarea>' in html


@pytest.mark.parametrize("content", [b"include = [", b'nonsense = "key"\n', b"\xff\xfe"])
def test_a_config_that_cant_be_read_offers_no_forms(browser: TestClient, paths: AppPaths, content: bytes) -> None:
    paths.config_file.write_bytes(content)

    html = browser.get("/setup").text

    assert "can't be read" in html
    assert "<form" not in html.split("<main>")[1]


def test_html_in_a_local_setting_is_shown_as_text(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, 'local_patterns = ["</textarea><script>alert(1)</script>"]\n')

    html = browser.get("/setup").text

    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(1)" in html


def test_the_repo_list_shows_the_login_and_every_discovered_repo(browser: TestClient, github: list[Any]) -> None:
    html = browser.get("/setup/repos").text

    assert "me, token from gh, scopes repo" in html
    assert 'value="me/alpha"' in html
    assert 'value="org/gamma"' in html
    assert "member" in html
    assert "<html" not in html


def test_the_token_is_never_shown(browser: TestClient, github: list[Any]) -> None:
    assert SECRET not in browser.get("/setup/repos").text


def test_named_folders_arent_in_the_list(browser: TestClient, github: list[Any]) -> None:
    assert "local/notes" not in browser.get("/setup/repos").text


def test_without_a_config_nothing_starts_ticked(browser: TestClient, github: list[Any]) -> None:
    html = browser.get("/setup/repos").text

    assert not is_ticked(html, "me/alpha")
    assert not is_ticked(html, "org/gamma")
    assert 'name="every" checked' not in html


def test_ticks_follow_what_the_config_selects(browser: TestClient, paths: AppPaths, github: list[Any]) -> None:
    config_with(paths, 'include = ["me/*"]\n')

    html = browser.get("/setup/repos").text

    assert is_ticked(html, "me/alpha")
    assert not is_ticked(html, "org/gamma")
    assert 'name="every" checked' not in html


def test_an_empty_config_means_every_repo(browser: TestClient, paths: AppPaths, github: list[Any]) -> None:
    config_with(paths, "")

    html = browser.get("/setup/repos").text

    assert 'name="every" checked' in html
    assert is_ticked(html, "me/alpha")


def test_the_form_warns_that_globs_and_excludes_get_replaced(
    browser: TestClient, paths: AppPaths, github: list[Any]
) -> None:
    config_with(paths, 'include = ["me/*", "org/gamma"]\nexclude = ["org/old"]\n')

    html = browser.get("/setup/repos").text

    assert "Saving replaces these entries" in html
    assert "me/*, org/old" in html


def test_the_form_warns_about_an_exact_name_that_isnt_on_the_list(
    browser: TestClient, paths: AppPaths, github: list[Any]
) -> None:
    config_with(paths, 'include = ["ME/Alpha", "org/not-listed"]\n')

    html = browser.get("/setup/repos").text

    assert "Saving replaces these entries in repolore.toml with the ticked names: org/not-listed" in html


def test_exact_names_get_no_warning(browser: TestClient, paths: AppPaths, github: list[Any]) -> None:
    config_with(paths, 'include = ["me/alpha"]\n')

    assert "Saving replaces" not in browser.get("/setup/repos").text


@pytest.mark.parametrize(
    ("remaining", "listed"), [(RATE_LIMIT_FLOOR - 1, False), (RATE_LIMIT_FLOOR, True), (None, True)]
)
def test_a_nearly_spent_rate_limit_doesnt_start_a_listing_that_would_wait_for_the_reset(
    browser: TestClient, github: list[Any], monkeypatch: pytest.MonkeyPatch, remaining: int | None, listed: bool
) -> None:
    monkeypatch.setattr(setup_pages, "github_access", lambda _: (WORKS, TokenInfo("me", None, remaining, 5000)))

    html = browser.get("/setup/repos").text

    assert bool(github) == listed
    assert ("too few GraphQL points" in html) != listed
    assert ('href="/setup">Check again' in html) != listed


def test_no_repos_at_all_says_so(browser: TestClient, github: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_pages, "discover", lambda *_: Discovery("me", [], []))

    html = browser.get("/setup/repos").text

    assert "lists no repos" in html
    assert "<form" not in html


def test_no_token_shows_the_fix_and_a_way_to_check_again(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = Check("GitHub", False, "No GitHub token: set REPOLORE_GITHUB_TOKEN, or log in with `gh auth login`.")
    monkeypatch.setattr(setup_pages, "github_access", lambda _: (missing, None))
    monkeypatch.setattr(setup_pages, "discover", lambda *_: pytest.fail("repos listed without a token"))

    html = browser.get("/setup/repos").text

    assert "gh auth login" in html
    assert 'href="/setup">Check again' in html
    assert "<form" not in html


def test_a_token_that_fails_shows_the_fix(browser: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    failed = Check("GitHub", False, "the token from gh didn't work: 401", "log in again with `gh auth login`")
    monkeypatch.setattr(setup_pages, "github_access", lambda _: (failed, None))

    html = browser.get("/setup/repos").text

    assert "Fix: log in again" in html
    assert "<form" not in html


def test_a_token_missing_scopes_still_lists_the_repos(
    browser: TestClient, github: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    lacking = Check("GitHub", False, "me; missing read:org", "`gh auth refresh --scopes read:org`")
    monkeypatch.setattr(setup_pages, "github_access", lambda _: (lacking, TokenInfo("me", frozenset(), 5000, 5000)))

    html = browser.get("/setup/repos").text

    assert "missing read:org" in html
    assert 'value="me/alpha"' in html


@pytest.mark.parametrize("error", [GitHubError("Bad gateway"), httpx.ConnectError("no route")])
def test_a_list_that_fails_is_reported(
    browser: TestClient, github: list[Any], monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*_: Any) -> Discovery:
        raise error

    monkeypatch.setattr(setup_pages, "discover", fail)

    html = browser.get("/setup/repos").text

    assert f"Couldn&#39;t list your repos: {error}" in html
    assert 'href="/setup">Check again' in html


def test_a_config_broken_while_the_page_is_open_is_reported(
    browser: TestClient, paths: AppPaths, github: list[Any]
) -> None:
    paths.config_file.write_bytes(b"include = [")

    html = browser.get("/setup/repos").text

    assert 'class="notice"' in html
    assert "<form" not in html


def test_html_in_a_repo_name_is_shown_as_text(
    browser: TestClient, github: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_pages, "discover", lambda *_: Discovery("me", [source('"><script>x</script>')], []))

    html = browser.get("/setup/repos").text

    assert "<script>x" not in html


def test_saving_repos_writes_them_and_reports_the_effect(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, '# mine\ninclude = ["me/alpha"]\n')

    response = save_repos(browser, ["org/gamma"])

    assert response.status_code == 200
    assert "1 repo added, 1 repo leaves the index and its cached data at the next refresh." in response.text
    assert 'action="/refresh"' in response.text
    assert load_config(paths.config_file).include == ["org/gamma"]
    assert "# mine" in paths.config_file.read_text(encoding="utf-8")


def test_several_repos_leaving_read_as_plural(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, 'include = ["me/alpha", "org/gamma"]\n')

    response = save_repos(browser, ["me/third"], shown=["me/alpha", "org/gamma", "me/third"])

    assert "2 repos leave the index and their cached data" in response.text


def test_the_first_save_reports_only_what_was_added(browser: TestClient) -> None:
    response = save_repos(browser, ["me/alpha", "org/gamma"])

    assert "2 repos added." in response.text
    assert "leave" not in response.text


def test_the_switch_saves_every_repo(browser: TestClient, paths: AppPaths) -> None:
    response = save_repos(browser, [], every=True)

    assert response.status_code == 200
    assert paths.config_file.read_bytes() == b""


def test_turning_the_switch_off_with_all_ticked_says_saved(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, "")

    assert "Saved." in save_repos(browser, ["me/alpha", "org/gamma"]).text


def test_unchanged_ticks_say_so_and_offer_no_refresh(browser: TestClient, paths: AppPaths) -> None:
    text = 'include = ["me/*"]\n'
    config_with(paths, text)

    response = save_repos(browser, ["me/alpha"])

    assert "Nothing changed" in response.text
    assert 'action="/refresh"' not in response.text
    assert paths.config_file.read_text(encoding="utf-8") == text


def test_no_ticks_are_refused_on_the_form(browser: TestClient, paths: AppPaths) -> None:
    response = save_repos(browser, [])

    assert response.status_code == 400
    assert "Tick at least one repo" in response.text
    assert not paths.config_file.exists()


def test_more_repos_than_starlettes_default_form_limit_can_be_saved(browser: TestClient, paths: AppPaths) -> None:
    names = [f"me/repo-{number}" for number in range(1001)]

    response = save_repos(browser, names, shown=names)

    assert response.status_code == 200
    assert len(load_config(paths.config_file).include) == 1001


def test_a_save_during_a_refresh_applies_to_the_next_one(browser: TestClient, paths: AppPaths) -> None:
    (paths.data_dir / REQUEST_FILE).touch()

    response = save_repos(browser, ["me/alpha"])

    assert "applies to the next one" in response.text
    assert 'action="/refresh"' not in response.text


@pytest.mark.parametrize("path", ["/setup/repos", "/setup/local", "/setup/folders", "/setup/folders/remove"])
@pytest.mark.parametrize("headers", [{}, {"Origin": "http://evil.example"}])
def test_a_save_the_page_didnt_make_is_refused(
    browser: TestClient, paths: AppPaths, path: str, headers: dict[str, str]
) -> None:
    response = browser.post(path, data={"repo": ["me/alpha"], "shown": ["me/alpha"]}, headers=headers)

    assert response.status_code == 403
    assert not paths.config_file.exists()


@pytest.mark.parametrize("path", ["/setup", "/setup/repos"])
def test_the_setup_pages_need_the_cookie(paths: AppPaths, path: str) -> None:
    client = TestClient(create_app(paths, "default", TOKEN), base_url=BASE, follow_redirects=False)

    assert client.get(path).status_code == 403


@pytest.mark.parametrize("save", ["repos", "local"])
def test_a_config_that_cant_be_read_is_reported_and_left_alone(browser: TestClient, paths: AppPaths, save: str) -> None:
    paths.config_file.write_bytes(b"include = [")

    response = save_repos(browser, ["me/alpha"]) if save == "repos" else save_local(browser, "", "")

    assert response.status_code == 400
    assert 'class="notice"' in response.text
    assert paths.config_file.read_bytes() == b"include = ["


def test_a_read_only_config_is_reported(browser: TestClient, paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*_: Any, **__: Any) -> None:
        raise PermissionError("repolore.toml is read-only")

    monkeypatch.setattr(setup_pages, "save_ticks", denied)

    response = save_repos(browser, ["me/alpha"])

    assert response.status_code == 400
    assert "read-only" in response.text


def test_saving_local_settings_shows_what_they_match(browser: TestClient, paths: AppPaths, tmp_path: Path) -> None:
    config_with(paths, 'include = ["me/alpha"]\n')
    root = tmp_path / "code"
    root.mkdir()

    response = save_local(browser, f"  {root}  \n\n{root}\n", ".plans/**/*.md\n")

    assert response.status_code == 200
    assert "Saved." in response.text
    assert "local checkouts" in response.text
    assert 'action="/refresh"' in response.text
    config = load_config(paths.config_file)
    assert config.local_roots == [root]
    assert config.local_patterns == [".plans/**/*.md"]


def test_clearing_the_local_settings_shows_no_checks(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, 'local_patterns = ["*.md"]\n')

    response = save_local(browser, "", "")

    assert response.status_code == 200
    assert "local checkouts" not in response.text
    assert load_config(paths.config_file).local_patterns == []


@pytest.mark.parametrize(
    ("roots", "patterns", "message"),
    [
        ("relative/path", "*.md", "absolute"),
        ("", "*.md", "together"),
        ("/somewhere", "", "together"),
    ],
)
def test_bad_local_settings_are_shown_on_the_form(
    browser: TestClient, paths: AppPaths, roots: str, patterns: str, message: str
) -> None:
    response = save_local(browser, roots, patterns)

    assert response.status_code == 400
    assert message in response.text
    assert not paths.config_file.exists()


def test_a_missing_folder_is_shown_on_the_form(browser: TestClient, paths: AppPaths, tmp_path: Path) -> None:
    response = save_local(browser, str(tmp_path / "gone"), "*.md")

    assert response.status_code == 400
    assert "not a folder" in response.text
    assert not paths.config_file.exists()


def test_a_bad_pattern_is_shown_on_the_form(browser: TestClient, paths: AppPaths, tmp_path: Path) -> None:
    response = save_local(browser, str(tmp_path), "../outside")

    assert response.status_code == 400
    assert "relative" in response.text
    assert not paths.config_file.exists()


def test_html_in_an_error_is_shown_as_text(browser: TestClient, tmp_path: Path) -> None:
    response = save_local(browser, "<script>alert(1)</script>", "*.md")

    assert "<script>" not in response.text


def test_no_form_fields_count_as_empty(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, 'local_patterns = ["*.md"]\n')

    assert browser.post("/setup/local", data={}, headers=FROM_THE_PAGE).status_code == 200
    assert load_config(paths.config_file).local_patterns == []


def test_a_first_run_saves_the_repos_before_the_local_documents(
    browser: TestClient, paths: AppPaths, tmp_path: Path
) -> None:
    response = save_local(browser, str(tmp_path), ".plans/**/*.md")

    assert response.status_code == 400
    assert "Save your repos first" in response.text
    assert not paths.config_file.exists()


PICKED = 'include = ["me/alpha"]\n'
WITH_A_FOLDER = PICKED + '\n[[local_folders]]\nname = "notes"\npath = "~/notes"\n'


def add_folder(browser: TestClient, name: str, path: str, patterns: str = "") -> Any:
    return browser.post(
        "/setup/folders", data={"name": name, "path": path, "patterns": patterns}, headers=FROM_THE_PAGE
    )


def remove_folder(browser: TestClient, name: str) -> Any:
    return browser.post("/setup/folders/remove", data={"name": name}, headers=FROM_THE_PAGE)


def test_the_setup_page_lists_the_local_folders(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, WITH_A_FOLDER)

    html = browser.get("/setup").text

    assert "local/notes" in html
    assert 'action="/setup/folders/remove"' in html
    assert 'name="name" value="notes"' in html
    assert "**/*.md" in html


def test_no_local_folders_says_so_and_offers_the_form(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, PICKED)

    html = browser.get("/setup").text

    assert "None yet." in html
    assert 'action="/setup/folders"' in html


def test_a_folder_is_added_and_shown_with_what_it_matches(browser: TestClient, paths: AppPaths, tmp_path: Path) -> None:
    config_with(paths, PICKED)
    (tmp_path / "plan.md").write_text("x", encoding="utf-8")

    response = add_folder(browser, " notes ", f"  {tmp_path}  ", "*.md\n\n*.md\n")

    assert response.status_code == 200
    assert "Added local/notes." in response.text
    assert 'name="name" value="notes"' in response.text
    assert 'action="/refresh"' in response.text
    assert f"1 file in {tmp_path}" in response.text
    folder = load_config(paths.config_file).local_folders[0]
    assert (folder.name, folder.path, folder.patterns) == ("notes", tmp_path, ["*.md"])


def test_a_refused_folder_keeps_what_was_typed(browser: TestClient, paths: AppPaths, tmp_path: Path) -> None:
    config_with(paths, PICKED)

    response = add_folder(browser, "bad name", str(tmp_path), "docs/*.md")

    assert response.status_code == 400
    assert "can&#39;t name a folder" in response.text
    assert 'value="bad name"' in response.text
    assert f'value="{tmp_path}"' in response.text
    assert ">docs/*.md</textarea>" in response.text
    assert paths.config_file.read_bytes().decode("utf-8") == PICKED


def test_a_duplicate_folder_is_reported_on_the_form(browser: TestClient, paths: AppPaths, tmp_path: Path) -> None:
    config_with(paths, WITH_A_FOLDER)

    response = add_folder(browser, "Notes", str(tmp_path))

    assert response.status_code == 400
    assert "already a local folder" in response.text
    assert "local/notes" in response.text


def test_a_first_run_saves_the_repos_before_a_folder(browser: TestClient, paths: AppPaths, tmp_path: Path) -> None:
    response = add_folder(browser, "notes", str(tmp_path))

    assert response.status_code == 400
    assert "Save your repos first" in response.text
    assert not paths.config_file.exists()


def test_missing_folder_fields_count_as_empty(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, PICKED)

    response = browser.post("/setup/folders", data={}, headers=FROM_THE_PAGE)

    assert response.status_code == 400
    assert "name and a path" in response.text


def test_a_folder_is_removed_and_says_what_that_does(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, WITH_A_FOLDER)

    response = remove_folder(browser, "notes")

    assert response.status_code == 200
    assert "Removed local/notes. It leaves the index and its cached data at the next refresh." in response.text
    assert "None yet." in response.text
    assert 'action="/refresh"' in response.text
    assert load_config(paths.config_file).local_folders == []


def test_removing_a_folder_during_a_refresh_says_it_applies_to_the_next(
    browser: TestClient, paths: AppPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_with(paths, WITH_A_FOLDER)
    monkeypatch.setattr(setup_pages, "refresh_status", lambda _: type("Status", (), {"running": True}))

    response = remove_folder(browser, "notes")

    assert "applies to the next one" in response.text
    assert 'action="/refresh"' not in response.text


def test_removing_a_folder_that_isnt_there_is_reported(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, WITH_A_FOLDER)

    response = remove_folder(browser, "other")

    assert response.status_code == 400
    assert "no local folder named other" in response.text
    assert "local/notes" in response.text


@pytest.mark.parametrize("change", ["add", "remove"])
def test_a_folder_change_leaves_a_config_that_cant_be_read_alone(
    browser: TestClient, paths: AppPaths, tmp_path: Path, change: str
) -> None:
    paths.config_file.write_bytes(b"include = [")

    response = add_folder(browser, "notes", str(tmp_path)) if change == "add" else remove_folder(browser, "notes")

    assert response.status_code == 400
    assert 'class="notice"' in response.text
    assert paths.config_file.read_bytes() == b"include = ["


@pytest.mark.parametrize("change", ["add", "remove"])
def test_a_folder_change_reports_a_config_that_cant_be_opened(
    browser: TestClient, paths: AppPaths, tmp_path: Path, change: str
) -> None:
    paths.config_file.mkdir()

    response = add_folder(browser, "notes", str(tmp_path)) if change == "add" else remove_folder(browser, "notes")

    assert response.status_code == 400
    assert 'class="notice"' in response.text


def test_a_folder_pattern_is_shown_as_text(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, PICKED + '\n[[local_folders]]\nname = "n"\npath = "~/n"\npatterns = ["<b>x</b>.md"]\n')

    html = browser.get("/setup").text

    assert "<b>x</b>" not in html
    assert "&lt;b&gt;x&lt;/b&gt;.md" in html


def test_a_refused_folder_shows_what_was_typed_as_text(browser: TestClient, paths: AppPaths) -> None:
    config_with(paths, PICKED)

    response = add_folder(browser, '"><script>alert(1)</script>', "~/x")

    assert "<script>alert" not in response.text
