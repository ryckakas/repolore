import os
import sqlite3
import subprocess
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from conftest import FakeEmbedder

from repolore import __version__, doctor
from repolore.chunking import Chunk
from repolore.clients import PLUGIN_SERVER, Entry, McpClient, RegistrationError
from repolore.config import Config, LocalFolder
from repolore.doctor import (
    Check,
    config_check,
    diagnose,
    engine_check,
    github_access,
    github_check,
    index_check,
    local_documents_checks,
    model_check,
    refresh_check,
    registration_check,
    server_check,
    vectors_check,
)
from repolore.embedding.engines import EngineChoice
from repolore.embedding.local import VOYAGE_NANO, onnx_engine
from repolore.locks import hold
from repolore.paths import AppPaths
from repolore.progress import Phase, Progress
from repolore.refresh import LOCK_FILE, STATUS_FILE, RefreshStatus
from repolore.server_state import Discovery, ServerFiles, write_discovery
from repolore.setup import write_config
from repolore.sources import github
from repolore.sources.github import GitHubClient, GitHubError
from repolore.store import StoreError, connect, write_index

Connect = Callable[[str], GitHubClient]
NOW = 1_793_555_000.0
ALL_SCOPES = {"x-oauth-scopes": "repo, read:org", "x-ratelimit-remaining": "4800", "x-ratelimit-limit": "5000"}


def github_answering(status: int = 200, headers: dict[str, str] | None = None) -> Connect:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"data": {"viewer": {"login": "octocat"}}}, headers=headers or ALL_SCOPES)

    http = httpx.Client(base_url=github.API_URL, transport=httpx.MockTransport(handle))
    return lambda _: GitHubClient(http, clock=lambda: NOW)


@pytest.fixture
def env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(github.TOKEN_ENV, "token")


@pytest.fixture
def gh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(github.TOKEN_ENV, raising=False)
    monkeypatch.setattr(doctor, "resolve_token", lambda: "token")


def test_a_passing_check_is_one_line_and_a_failing_one_says_how_to_fix_it() -> None:
    assert str(Check("index", True, "12 chunks")) == "✓ index: 12 chunks"
    assert str(Check("index", False, "none", "run it")) == "✗ index: none\n  fix: run it"


def test_the_vector_extension_loads() -> None:
    check = vectors_check()

    assert check.ok
    assert check.detail.startswith("v")


def test_a_vector_extension_that_cant_load_fails_with_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> str:
        raise StoreError("This Python's sqlite3 can't load extensions")

    monkeypatch.setattr(doctor, "vector_version", fail)

    assert vectors_check() == Check("sqlite-vec", False, "This Python's sqlite3 can't load extensions")


@pytest.mark.usefixtures("env_token")
def test_github_names_the_login_the_token_source_its_scopes_and_the_points_left() -> None:
    check = github_check(github_answering())

    assert check == Check(
        "GitHub",
        True,
        "octocat, token from REPOLORE_GITHUB_TOKEN, scopes read:org, repo, "
        "4,800 of 5,000 GraphQL points left this hour",
    )


@pytest.mark.usefixtures("gh_token")
def test_a_gh_token_missing_a_scope_is_fixed_with_gh_auth_refresh() -> None:
    check = github_check(github_answering(headers={"x-oauth-scopes": "repo"}))

    assert not check.ok
    assert check.detail == "octocat, token from gh auth token, scopes repo; missing read:org"
    assert check.fix == "`gh auth refresh --scopes read:org`"


@pytest.mark.usefixtures("env_token")
def test_a_token_variable_without_scopes_needs_a_new_classic_token() -> None:
    check = github_check(github_answering(headers={"x-oauth-scopes": ""}))

    assert not check.ok
    assert check.detail.endswith("no scopes; missing repo, read:org")
    assert check.fix == "set REPOLORE_GITHUB_TOKEN to a classic token with the repo and read:org scopes"


@pytest.mark.usefixtures("env_token")
def test_a_token_missing_a_scope_still_says_whose_it_is() -> None:
    check, token = github_access(github_answering(headers={"x-oauth-scopes": "repo"}))

    assert not check.ok
    assert token is not None
    assert token.login == "octocat"
    assert token.missing_scopes == ["read:org"]


@pytest.mark.usefixtures("env_token")
def test_a_token_that_doesnt_work_has_no_access() -> None:
    check, token = github_access(github_answering(status=401))

    assert not check.ok
    assert token is None


@pytest.mark.usefixtures("env_token")
def test_a_fine_grained_token_passes_unchecked() -> None:
    check = github_check(github_answering(headers={"x-github-request-id": "1"}))

    assert check == Check(
        "GitHub", True, "octocat, token from REPOLORE_GITHUB_TOKEN, a fine-grained token, so its access isn't checked"
    )


def test_no_token_fails_with_how_to_get_one(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_token() -> str:
        raise GitHubError("No GitHub token: set REPOLORE_GITHUB_TOKEN, or log in with `gh auth login`.")

    monkeypatch.delenv(github.TOKEN_ENV, raising=False)
    monkeypatch.setattr(doctor, "resolve_token", no_token)

    check = github_check(pytest.fail)

    assert not check.ok
    assert check.detail.startswith("No GitHub token")


@pytest.mark.usefixtures("env_token")
def test_a_spent_rate_limit_says_to_wait_instead_of_logging_in_again() -> None:
    check = github_check(
        github_answering(headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(NOW + 500))})
    )

    assert check == Check(
        "GitHub",
        False,
        "the GraphQL rate limit is spent until 2026-11-01T17:51+00:00",
        "wait for the reset; a running refresh may be using the points",
    )


@pytest.mark.parametrize(
    ("token", "fix"),
    [("env_token", "set a new REPOLORE_GITHUB_TOKEN"), ("gh_token", "log in again with `gh auth login`")],
)
def test_a_rejected_token_is_fixed_where_it_came_from(token: str, fix: str, request: pytest.FixtureRequest) -> None:
    request.getfixturevalue(token)

    check = github_check(github_answering(status=401))

    assert not check.ok
    assert "didn't work" in check.detail
    assert "HTTP 401" in check.detail
    assert check.fix == fix


@pytest.mark.usefixtures("env_token")
def test_github_out_of_reach_fails_with_the_network_error() -> None:
    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    def connect(_: str) -> GitHubClient:
        return GitHubClient(httpx.Client(base_url=github.API_URL, transport=httpx.MockTransport(refuse)))

    check = github_check(connect)

    assert check == Check(
        "GitHub", False, "can't reach GitHub: connection refused", "check the network, or HTTPS_PROXY"
    )


def test_without_a_config_file_every_repo_is_in(tmp_path: Path) -> None:
    assert config_check(tmp_path / "repolore.toml") == Check(
        "repolore.toml", True, "none, so every discovered repo is in"
    )


def test_a_config_file_is_summed_up(tmp_path: Path) -> None:
    config_file = tmp_path / "repolore.toml"
    root = tmp_path.as_posix()
    config_file.write_text(f'include = ["octo/*"]\nlocal_roots = ["{root}"]\n', encoding="utf-8")

    check = config_check(config_file)

    assert check == Check(
        "repolore.toml",
        True,
        "1 include glob, 0 exclude globs, 1 local root; `repolore repos` shows what they select",
    )


def test_a_local_root_that_doesnt_exist_fails(tmp_path: Path) -> None:
    config_file = tmp_path / "repolore.toml"
    missing = (tmp_path / "gone").as_posix()
    config_file.write_text(f'exclude = ["a/*", "b/*"]\nlocal_roots = ["{missing}"]\n', encoding="utf-8")

    check = config_check(config_file)

    assert not check.ok
    assert check.detail == f"0 include globs, 2 exclude globs, 1 local root; not found: {Path(missing)}"
    assert check.fix == "fix local_roots"


@pytest.mark.parametrize(
    ("content", "encoding"),
    [
        ("include = [", "utf-8"),
        ('include = "octo/*"', "utf-8"),
        ('includes = ["octo/*"]', "utf-8"),
        ("x = 1", "utf-16"),
    ],
)
def test_an_invalid_config_file_fails_with_its_error_on_one_line(tmp_path: Path, content: str, encoding: str) -> None:
    config_file = tmp_path / "repolore.toml"
    config_file.write_text(content, encoding=encoding)

    check = config_check(config_file)

    assert not check.ok
    assert check.detail.startswith("invalid: ")
    assert "\n" not in check.detail


def test_an_index_is_summed_up(index_file: Path, corpus: list[Chunk]) -> None:
    check = index_check(index_file)

    assert check.ok
    assert check.name == "index default"
    assert check.detail.startswith(f"{len(corpus)} chunks from 2 repos, built 20")
    assert check.detail.endswith("with fake-bag-of-words, whole sections, bot reviews excluded")


def test_an_index_names_its_chunk_limit(tmp_path: Path, corpus: list[Chunk]) -> None:
    check = index_check(index_built_with(tmp_path, VOYAGE_NANO.model_id, corpus))

    assert f"with {VOYAGE_NANO.model_id}, at most 8176 tokens a chunk, bot reviews excluded" in check.detail


def test_a_missing_index_fails_with_how_to_build_it(tmp_path: Path) -> None:
    index_file = tmp_path / "db" / "default.db"

    assert index_check(index_file) == Check("index default", False, f"none at {index_file}", "run `repolore refresh`")


def test_an_empty_index_fails(tmp_path: Path) -> None:
    index_file = tmp_path / "default.db"
    sqlite3.connect(index_file).close()

    check = index_check(index_file)

    assert not check.ok
    assert "The index is empty" in check.detail


def index_built_with(tmp_path: Path, model_id: str, corpus: list[Chunk]) -> Path:
    index_file = tmp_path / "db" / "default.db"
    embedder = FakeEmbedder()
    with closing(connect(index_file)) as db:
        vectors = embedder.embed_documents([chunk.embedded_text for chunk in corpus])
        write_index(db, corpus, vectors, model_id, embedder.dim, {"max_tokens": "8176"})
    return index_file


def test_the_index_model_is_found_in_the_cache(
    tmp_path: Path, corpus: list[Chunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    checked = []
    monkeypatch.setattr(doctor, "is_downloaded", lambda spec, folder: checked.append((spec, folder)) or True)
    models_dir = tmp_path / "models"

    check = model_check(index_built_with(tmp_path, VOYAGE_NANO.model_id, corpus), models_dir)

    assert check == Check("model", True, f"{VOYAGE_NANO.model_id} is in {models_dir / 'voyage-4-nano-1'}")
    assert checked == [(VOYAGE_NANO, models_dir)]


@pytest.mark.parametrize("index", ["none", "empty"])
def test_without_a_readable_index_the_default_model_is_checked(tmp_path: Path, index: str) -> None:
    index_file = tmp_path / "default.db"
    if index == "empty":
        sqlite3.connect(index_file).close()
    folder = tmp_path / "models" / "voyage-4-nano-1"

    check = model_check(index_file, tmp_path / "models")

    assert check == Check(
        "model",
        False,
        f"{VOYAGE_NANO.model_id} isn't in {folder} yet",
        "`repolore refresh` downloads it, 1.4 GB; behind a proxy that blocks github.com, "
        f"put the files from {VOYAGE_NANO.release.url} there by hand",
    )


def test_an_index_built_with_a_model_repolore_cant_load_fails(tmp_path: Path, index_file: Path) -> None:
    check = model_check(index_file, tmp_path)

    assert check == Check(
        "model", False, "the index was built with fake-bag-of-words, which repolore can't load", "`repolore build`"
    )


def write_status(data_dir: Path, status: RefreshStatus) -> None:
    (data_dir / STATUS_FILE).write_text(status.model_dump_json(), encoding="utf-8")


def test_no_refresh_yet_passes(tmp_path: Path) -> None:
    assert refresh_check(tmp_path) == Check("last refresh", True, "none yet")


def test_a_refresh_that_worked_passes(tmp_path: Path) -> None:
    write_status(tmp_path, RefreshStatus(started_at="2026-11-01T10:00", finished_at="2026-11-01T10:05", ok=True))

    assert refresh_check(tmp_path) == Check("last refresh", True, "succeeded at 2026-11-01T10:05")


def test_a_failed_refresh_fails_with_its_error_and_log(tmp_path: Path) -> None:
    status = RefreshStatus(started_at="2026-11-01T10:00", finished_at="2026-11-01T10:01", ok=False, error="HTTP 502")
    write_status(tmp_path, status)

    check = refresh_check(tmp_path)

    assert check == Check(
        "last refresh",
        False,
        "failed at 2026-11-01T10:01: HTTP 502",
        f"run `repolore refresh`; a background run logs to {tmp_path / 'refresh.log'}",
    )


def test_a_refresh_that_stopped_without_finishing_fails(tmp_path: Path) -> None:
    write_status(tmp_path, RefreshStatus(started_at="2026-11-01T10:00"))

    check = refresh_check(tmp_path)

    assert check == Check(
        "last refresh", False, "started at 2026-11-01T10:00 and stopped before it finished", "run `repolore refresh`"
    )


def test_a_running_refresh_shows_its_progress(tmp_path: Path) -> None:
    write_status(
        tmp_path, RefreshStatus(started_at="2026-11-01T10:00", progress=Progress(phase=Phase.EMBED, done=1, total=4))
    )

    with hold(tmp_path / LOCK_FILE, timeout=0):
        check = refresh_check(tmp_path)

    assert check == Check("last refresh", True, "running since 2026-11-01T10:00, at embed 1 of 4")


def test_a_refresh_just_requested_is_running(tmp_path: Path) -> None:
    (tmp_path / "refresh.requested").touch()

    assert refresh_check(tmp_path) == Check("last refresh", True, "running")


def test_no_shared_server_passes(tmp_path: Path) -> None:
    check = server_check(ServerFiles(tmp_path, "default"))

    assert check == Check("shared server", True, "not running; the first search starts it")


def test_a_server_that_holds_its_lock_but_has_no_address_yet_is_starting(tmp_path: Path) -> None:
    files = ServerFiles(tmp_path, "default")

    with hold(files.lock, timeout=0):
        check = server_check(files)

    assert check == Check("shared server", True, "starting")


@pytest.mark.parametrize(
    ("version", "note"), [(__version__, ""), ("0.0.1", f", which the next session replaces with {__version__}")]
)
def test_a_running_server_shows_where_its_version_and_its_sessions(tmp_path: Path, version: str, note: str) -> None:
    files = ServerFiles(tmp_path, "default")
    write_discovery(files, Discovery(pid=4242, port=61000, token="t", version=version, started_at="2026-11-01T10:00"))

    with hold(files.lock, timeout=0), hold(files.client_lock(os.getpid()), timeout=0):
        check = server_check(files)

    assert check == Check(
        "shared server", True, f"pid 4242 at http://127.0.0.1:61000/mcp, repolore {version}{note}; 1 session"
    )


USER_ENTRY = Entry("repolore", ("repolore-mcp",), "user", "`remove user`")
PLUGIN_ENTRY = Entry(PLUGIN_SERVER, ("repolore-mcp",), "plugin", "`remove plugin`")


def never() -> list[Entry]:
    raise AssertionError("shouldn't be called")


def client_with(*entries: Entry, health: Callable[[str], str | None] | None = None) -> McpClient:
    return McpClient("codex", "Codex", lambda: True, lambda: list(entries), lambda: "added", "add it", health)


def connected(_: str) -> str:
    return "✓ Connected"


def test_a_healthy_registration_passes() -> None:
    assert registration_check(client_with(USER_ENTRY, health=connected)) == Check(
        "Codex", True, "repolore is registered: ✓ Connected"
    )


def test_a_registration_by_the_plugin_is_named() -> None:
    assert registration_check(client_with(PLUGIN_ENTRY, health=connected)) == Check(
        "Codex", True, "repolore is registered as plugin:repolore:repolore: ✓ Connected"
    )


def test_a_client_it_cant_check_says_so() -> None:
    assert registration_check(client_with(USER_ENTRY)) == Check(
        "Codex", True, "repolore is registered, connection not checked"
    )


def test_a_status_with_nothing_to_say_passes() -> None:
    assert registration_check(client_with(USER_ENTRY, health=lambda _: None)) == Check(
        "Codex", True, "repolore is registered"
    )


def test_the_health_check_asks_about_the_entry_it_found() -> None:
    asked: list[str] = []

    def health(name: str) -> str:
        asked.append(name)
        return "✓ Connected"

    registration_check(client_with(PLUGIN_ENTRY, health=health))

    assert asked == [PLUGIN_SERVER]


def test_a_client_that_isnt_installed_is_not_checked() -> None:
    absent = McpClient("codex", "Codex", lambda: False, never, lambda: "added", "add it")

    assert registration_check(absent) is None


def test_a_missing_registration_warns_with_how_to_add_it() -> None:
    assert registration_check(client_with()) == Check(
        "Codex", True, "repolore isn't registered", "add it", warning=True
    )


def test_other_servers_dont_count_as_repolore() -> None:
    check = registration_check(client_with(Entry("other", ("other-cmd",), "user", "x")))

    assert check is not None
    assert check.warning


def test_a_warning_prints_its_own_mark_and_its_fix() -> None:
    assert str(Check("Codex", True, "repolore isn't registered", "add it", warning=True)) == (
        "! Codex: repolore isn't registered\n  fix: add it"
    )


def test_a_registration_that_cant_connect_fails() -> None:
    failing = "! Connected · tools fetch failed (Request timed out)"

    check = registration_check(client_with(USER_ENTRY, health=lambda _: failing))

    assert check == Check("Codex", False, f"repolore is registered: {failing}", doctor.CONNECTION_FIX)


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired(["claude"], 120), OSError("denied")])
def test_a_health_check_that_breaks_fails(error: Exception) -> None:
    def broken(_: str) -> str:
        raise error

    check = registration_check(client_with(USER_ENTRY, health=broken))

    assert check is not None
    assert not check.ok
    assert check.detail == f"repolore is registered, but checking it failed: {error}"


@pytest.mark.parametrize(
    "error", [RegistrationError("can't read it"), subprocess.TimeoutExpired(["claude"], 60), OSError("denied")]
)
def test_a_listing_that_cant_be_read_fails(error: Exception) -> None:
    def broken() -> list[Entry]:
        raise error

    check = registration_check(McpClient("codex", "Codex", lambda: True, broken, lambda: "added", "add it"))

    assert check == Check("Codex", False, f"can't list its MCP servers: {error}")


def test_two_registrations_fail_naming_each_and_its_removal() -> None:
    other = Entry("mine", ("uv", "run", "repolore-mcp"), "user", "`remove mine`")

    check = registration_check(client_with(USER_ENTRY, other, health=connected))

    assert check == Check(
        "Codex",
        False,
        "repolore is registered 2 times: repolore (user), mine (user)",
        "keep one: `remove user`; `remove mine`",
    )


def test_a_harmless_duplicate_with_the_same_command_still_fails() -> None:
    check = registration_check(client_with(USER_ENTRY, PLUGIN_ENTRY))

    assert check is not None
    assert not check.ok


def test_a_command_that_no_longer_exists_fails_without_starting_it(tmp_path: Path) -> None:
    gone = str(tmp_path / "repolore-mcp")
    entry = Entry("repolore", (gone,), "user", "`remove user`")

    def not_called(_: str) -> str:
        raise AssertionError("shouldn't be called")

    check = registration_check(client_with(entry, health=not_called))

    assert check == Check(
        "Codex",
        False,
        f"repolore is registered, but {gone} no longer exists",
        "`remove user`, then run `repolore setup`",
    )


@pytest.mark.usefixtures("env_token")
def test_diagnose_checks_every_part_in_order(data_paths: AppPaths) -> None:
    checks = list(diagnose(data_paths, "default", github_answering(), clients=[client_with(USER_ENTRY)]))

    assert [check.name for check in checks] == [
        "sqlite-vec",
        "GitHub",
        "repolore.toml",
        "index default",
        "model",
        "build engine",
        "last refresh",
        "shared server",
        "Codex",
    ]


@pytest.mark.parametrize("engine", ["MLX on the GPU", "ONNX Runtime on the CPU"])
def test_the_build_engine_is_reported_and_never_fails(monkeypatch: pytest.MonkeyPatch, engine: str) -> None:
    monkeypatch.setattr(doctor, "build_engine", lambda: EngineChoice(engine, onnx_engine))

    assert engine_check() == Check("build engine", True, engine)


@pytest.mark.usefixtures("env_token")
def test_diagnose_skips_clients_that_arent_installed(data_paths: AppPaths) -> None:
    absent = McpClient("codex", "Codex", lambda: False, never, lambda: "added", "add it")

    checks = list(diagnose(data_paths, "default", github_answering(), clients=[absent]))

    assert checks[-1].name == "shared server"


def checkout_with_plan(root: Path, name: str = "repo") -> Path:
    directory = root / name
    (directory / ".plans").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(directory)], check=True)
    url = f"https://github.com/octo/{name}.git"
    subprocess.run(["git", "-C", str(directory), "remote", "add", "origin", url], check=True)
    (directory / ".plans" / "plan.md").write_text("# Plan", encoding="utf-8")
    return directory


def local_checks(tmp_path: Path, config: Config) -> list[Check]:
    config_file = tmp_path / "repolore.toml"
    write_config(config_file, config)
    return list(local_documents_checks(config_file))


PLANS = ".plans/**/*.md"


def test_no_local_settings_means_no_local_checks(tmp_path: Path) -> None:
    assert local_checks(tmp_path, Config(include=["me/*"])) == []


def test_no_config_file_means_no_local_checks(tmp_path: Path) -> None:
    assert list(local_documents_checks(tmp_path / "repolore.toml")) == []


def test_an_unreadable_config_is_left_to_the_config_check(tmp_path: Path) -> None:
    config_file = tmp_path / "repolore.toml"
    config_file.write_text("include = [", encoding="utf-8")

    assert list(local_documents_checks(config_file)) == []


def test_checkouts_report_their_files(tmp_path: Path) -> None:
    root = tmp_path / "code"
    checkout_with_plan(root, "one")
    checkout_with_plan(root, "two")

    checks = local_checks(tmp_path, Config(local_roots=[root], local_patterns=[PLANS]))

    assert checks == [Check("local checkouts", True, "2 files in 2 checkouts")]


def test_one_checkout_with_one_file_is_singular(tmp_path: Path) -> None:
    checkout_with_plan(tmp_path / "code")

    checks = local_checks(tmp_path, Config(local_roots=[tmp_path / "code"], local_patterns=[PLANS]))

    assert checks[0].detail == "1 file in 1 checkout"


def test_roots_without_patterns_warn_with_the_line_to_add(tmp_path: Path) -> None:
    check = local_checks(tmp_path, Config(local_roots=[tmp_path]))[0]

    assert (check.ok, check.warning) == (True, True)
    assert "no local_patterns" in check.detail
    assert check.fix == 'add local_patterns = [".plans/**/*.md"] to repolore.toml'


def test_patterns_without_roots_warn(tmp_path: Path) -> None:
    check = local_checks(tmp_path, Config(local_patterns=[PLANS]))[0]

    assert (check.ok, check.warning) == (True, True)
    assert "no local_roots" in check.detail


def test_a_pattern_no_checkout_matches_warns_and_names_it(tmp_path: Path) -> None:
    checkout_with_plan(tmp_path / "code")

    config = Config(local_roots=[tmp_path / "code"], local_patterns=[PLANS, "docs/adr/*.md"])
    check = local_checks(tmp_path, config)[0]

    assert (check.ok, check.warning) == (True, True)
    assert check.detail == "1 file in 1 checkout; no selected GitHub checkout matches docs/adr/*.md"


def test_a_pattern_only_some_checkouts_match_doesnt_warn(tmp_path: Path) -> None:
    checkout_with_plan(tmp_path / "code", "one")
    with_adr = checkout_with_plan(tmp_path / "code", "two")
    (with_adr / "docs").mkdir()
    (with_adr / "docs" / "adr.md").write_text("# ADR", encoding="utf-8")

    checks = local_checks(tmp_path, Config(local_roots=[tmp_path / "code"], local_patterns=[PLANS, "docs/*.md"]))

    assert checks == [Check("local checkouts", True, "3 files in 2 checkouts")]


def test_no_checkouts_at_all_warns_about_every_pattern(tmp_path: Path) -> None:
    (tmp_path / "code").mkdir()

    check = local_checks(tmp_path, Config(local_roots=[tmp_path / "code"], local_patterns=[PLANS]))[0]

    assert check.warning
    assert check.detail == f"0 files in 0 checkouts; no selected GitHub checkout matches {PLANS}"


def test_a_checkout_over_the_limit_fails_naming_it(tmp_path: Path) -> None:
    repo = checkout_with_plan(tmp_path / "code")
    (repo / ".plans" / "more.md").write_text("# More", encoding="utf-8")

    config = Config(local_roots=[tmp_path / "code"], local_patterns=[PLANS], max_local_files=1)
    check = local_checks(tmp_path, config)[0]

    assert not check.ok
    assert not check.warning
    assert check.detail == f"more than 1 files in {repo}"
    assert "max_local_files" in (check.fix or "")


def test_a_checkout_left_out_by_the_globs_isnt_counted_or_limited(tmp_path: Path) -> None:
    checkout_with_plan(tmp_path / "code", "kept")
    left_out = checkout_with_plan(tmp_path / "code", "other")
    (left_out / ".plans" / "more.md").write_text("# More", encoding="utf-8")

    config = Config(include=["octo/kept"], local_roots=[tmp_path / "code"], local_patterns=[PLANS], max_local_files=1)

    assert local_checks(tmp_path, config) == [Check("local checkouts", True, "1 file in 1 checkout")]


def test_exactly_the_limit_passes(tmp_path: Path) -> None:
    checkout_with_plan(tmp_path / "code")

    config = Config(local_roots=[tmp_path / "code"], local_patterns=[PLANS], max_local_files=1)

    assert local_checks(tmp_path, config)[0].ok


def test_a_missing_root_is_not_logged_again(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = Config(local_roots=[tmp_path / "gone"], local_patterns=[PLANS])

    local_checks(tmp_path, config)

    assert "doesn't exist" not in caplog.text


def test_a_folder_reports_its_files(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "notes" / "b.md").write_text("# B", encoding="utf-8")

    checks = local_checks(tmp_path, Config(local_folders=[LocalFolder(name="notes", path=tmp_path / "notes")]))

    assert checks == [Check("local/notes", True, f"2 files in {(tmp_path / 'notes').resolve()}")]


def test_a_folder_matching_nothing_warns(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()

    check = local_checks(tmp_path, Config(local_folders=[LocalFolder(name="notes", path=tmp_path / "notes")]))[0]

    assert (check.ok, check.warning) == (True, True)
    assert "no markdown matches" in check.detail


def test_a_missing_folder_fails_without_logging_it_again(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    gone = tmp_path / "gone"

    check = local_checks(tmp_path, Config(local_folders=[LocalFolder(name="notes", path=gone)]))[0]

    assert not check.ok
    assert check.detail == f"not found: {gone}"
    assert check.fix == "fix its path in repolore.toml"
    assert "doesn't exist" not in caplog.text


def test_a_folder_over_the_limit_fails(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "notes" / "b.md").write_text("# B", encoding="utf-8")

    folder = LocalFolder(name="notes", path=tmp_path / "notes")
    check = local_checks(tmp_path, Config(local_folders=[folder], max_local_files=1))[0]

    assert not check.ok
    assert "more than 1 files" in check.detail


def test_every_folder_gets_its_own_check_after_the_checkouts(tmp_path: Path) -> None:
    folders = [LocalFolder(name="a", path=tmp_path), LocalFolder(name="b", path=tmp_path)]

    checks = local_checks(tmp_path, Config(local_roots=[tmp_path], local_folders=folders))

    assert [check.name for check in checks] == ["local checkouts", "local/a", "local/b"]


def test_diagnose_reports_local_documents_right_after_the_config(data_paths: AppPaths, tmp_path: Path) -> None:
    write_config(data_paths.config_file, Config(local_folders=[LocalFolder(name="notes", path=tmp_path)]))

    names = [check.name for check in diagnose(data_paths, "default", github_answering(), clients=[])]

    assert names[names.index("repolore.toml") + 1] == "local/notes"
