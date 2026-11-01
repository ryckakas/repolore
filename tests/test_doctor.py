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
from repolore.clients import CODEX, Registration
from repolore.doctor import (
    Check,
    config_check,
    diagnose,
    github_check,
    index_check,
    model_check,
    refresh_check,
    registration_check,
    server_check,
    vectors_check,
)
from repolore.embedding.local import BGE_SMALL, VOYAGE_NANO
from repolore.locks import hold
from repolore.paths import AppPaths
from repolore.progress import Phase, Progress
from repolore.refresh import LOCK_FILE, STATUS_FILE, RefreshStatus
from repolore.server_state import Discovery, ServerFiles, write_discovery
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
    check = index_check(index_built_with(tmp_path, BGE_SMALL.model_id, corpus))

    assert f"with {BGE_SMALL.model_id}, at most 8176 tokens a chunk, bot reviews excluded" in check.detail


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
    monkeypatch.setattr(doctor, "is_cached", lambda spec: checked.append(spec) or True)
    monkeypatch.setattr(doctor, "cache_dir", lambda: "/hf")

    check = model_check(index_built_with(tmp_path, BGE_SMALL.model_id, corpus))

    assert check == Check("model", True, f"{BGE_SMALL.model_id} is in /hf")
    assert checked == [BGE_SMALL]


@pytest.mark.parametrize("index", ["none", "empty"])
def test_without_a_readable_index_the_default_model_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index: str
) -> None:
    index_file = tmp_path / "default.db"
    if index == "empty":
        sqlite3.connect(index_file).close()
    monkeypatch.setattr(doctor, "is_cached", lambda _: False)
    monkeypatch.setattr(doctor, "cache_dir", lambda: "/hf")

    check = model_check(index_file)

    assert check == Check(
        "model",
        False,
        f"{VOYAGE_NANO.model_id} isn't in /hf yet",
        "the next build or search downloads it; it needs access to huggingface.co",
    )


def test_an_index_built_with_a_model_repolore_cant_load_fails(index_file: Path) -> None:
    check = model_check(index_file)

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


def registered_as(monkeypatch: pytest.MonkeyPatch, found: Registration | None) -> None:
    monkeypatch.setattr(doctor, "registration", lambda _: found)


def test_a_healthy_registration_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_as(monkeypatch, Registration(found=True, status="✓ Connected"))

    assert registration_check(CODEX) == Check("Codex", True, "repolore is registered: ✓ Connected")


def test_a_client_that_isnt_installed_is_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_as(monkeypatch, None)

    assert registration_check(CODEX) is None


def test_a_missing_registration_fails_with_how_to_add_it(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_as(monkeypatch, Registration(found=False))

    assert registration_check(CODEX) == Check("Codex", False, "repolore isn't registered", CODEX.register)


def test_a_missing_registration_shows_what_the_client_said(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_as(monkeypatch, Registration(found=False, status="unknown command mcp"))

    check = registration_check(CODEX)

    assert check is not None
    assert check.detail == "repolore isn't registered (unknown command mcp)"


def test_a_registration_that_cant_connect_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_as(monkeypatch, Registration(found=True, status="! Connected · tools fetch failed (Request timed out)"))

    check = registration_check(CODEX)

    assert check is not None
    assert not check.ok
    assert check.detail == "repolore is registered: ! Connected · tools fetch failed (Request timed out)"


def test_a_client_cli_that_hangs_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(_: object) -> Registration:
        raise subprocess.TimeoutExpired(["codex"], 120)

    monkeypatch.setattr(doctor, "registration", hang)

    check = registration_check(CODEX)

    assert check is not None
    assert not check.ok
    assert check.detail.startswith("`codex mcp get repolore` failed: ")


@pytest.mark.usefixtures("env_token")
def test_diagnose_checks_every_part_in_order(data_paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    registered_as(monkeypatch, Registration(found=True))

    checks = list(diagnose(data_paths, "default", github_answering(), clients=[CODEX]))

    assert [check.name for check in checks] == [
        "sqlite-vec",
        "GitHub",
        "repolore.toml",
        "index default",
        "model",
        "last refresh",
        "shared server",
        "Codex",
    ]


@pytest.mark.usefixtures("env_token")
def test_diagnose_skips_clients_that_arent_installed(data_paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> None:
    registered_as(monkeypatch, None)

    checks = list(diagnose(data_paths, "default", github_answering(), clients=[CODEX]))

    assert checks[-1].name == "shared server"
