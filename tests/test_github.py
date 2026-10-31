import base64
import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from repolore.sources import github
from repolore.sources.github import (
    GitHubClient,
    GitHubError,
    RawRepo,
    ViewerRepo,
    fetch_contributions,
    fetch_repo,
    involved_repositories,
    resolve_token,
    viewer_repositories,
)

Json = dict[str, Any]
INFO = {"nameWithOwner": "octo/repo", "url": "https://github.com/octo/repo", "isPrivate": False}
NOW = 1_000_000.0


def page(nodes: list[Json], after: str | None = None) -> Json:
    return {"pageInfo": {"hasNextPage": after is not None, "endCursor": after}, "nodes": nodes}


def pull(number: int, files: int = 0, updated: str | None = None) -> Json:
    connection = {"totalCount": 0, "nodes": []}
    return {
        "number": number,
        "files": {"totalCount": files, "nodes": []},
        "commits": connection,
        "comments": connection,
        "reviews": connection,
        "reviewThreads": connection,
    } | ({"updatedAt": updated} if updated else {})


def commit(oid: str, committed: str) -> Json:
    return {"oid": oid, "committedDate": committed}


def tree_entry(path: str, sha: str, mode: str = "100644", kind: str = "blob") -> Json:
    return {"path": path, "sha": sha, "mode": mode, "type": kind}


DEFAULT_TREE = [
    tree_entry("README.md", "readme"),
    tree_entry("CLAUDE.md", "link", mode="120000"),
    tree_entry("docs", "dir", mode="040000", kind="tree"),
    tree_entry("docs/Design.MD", "design"),
    tree_entry("node_modules/pkg/README.md", "dependency"),
    tree_entry("src/main.go", "code"),
]


class FakeGitHub:
    """Serves each GraphQL operation from a list of pages, and the tree and blobs over REST."""

    def __init__(self, branch: str | None = "main", truncated: bool = False, **pages: list[Json]) -> None:
        self.info = {**INFO, "defaultBranchRef": {"name": branch} if branch else None}
        self.pages = {
            "RepoPullRequests": [page([pull(1)])],
            "RepoIssues": [page([{"number": 7}])],
            "DefaultBranchHistory": [page([{"oid": "abc"}])],
        } | pages
        self.tree = DEFAULT_TREE
        self.truncated = truncated
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/graphql":
            body = json.loads(request.content)
            return httpx.Response(200, json={"data": self._graphql(body["operationName"], body["variables"])})
        if "/git/trees/" in request.url.path:
            return httpx.Response(200, json={"truncated": self.truncated, "tree": self.tree})
        sha = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={"content": base64.b64encode(f"# {sha} ✓".encode()).decode()})

    def variables(self, operation: str) -> list[Json]:
        bodies = [json.loads(request.content) for request in self.requests if request.url.path == "/graphql"]
        return [body["variables"] for body in bodies if body["operationName"] == operation]

    def blobs(self) -> list[str]:
        return [request.url.path.rsplit("/", 1)[-1] for request in self.requests if "/git/blobs/" in request.url.path]

    def _graphql(self, operation: str, variables: Json) -> Json:
        if operation == "RepoInfo":
            return {"repository": self.info}
        index = int(variables["after"] or 0)
        connection = self.pages[operation][index]
        if connection["pageInfo"]["hasNextPage"]:
            connection = {**connection, "pageInfo": {"hasNextPage": True, "endCursor": str(index + 1)}}
        return _wrap(operation, connection)


def _wrap(operation: str, connection: Json) -> Json:
    match operation:
        case "RepoPullRequests":
            return {"repository": {"pullRequests": connection}}
        case "RepoIssues":
            return {"repository": {"issues": connection}}
        case "DefaultBranchHistory":
            return {"repository": {"defaultBranchRef": {"target": {"history": connection}}}}
        case "ViewerRepositories":
            return {"viewer": {"login": "octocat", "repositories": connection}}
        case _:
            return {"search": connection}


def client_for(handler: Callable[[httpx.Request], httpx.Response], sleeps: list[float] | None = None) -> GitHubClient:
    http = httpx.Client(base_url=github.API_URL, transport=httpx.MockTransport(handler))
    return GitHubClient(http, sleep=(sleeps if sleeps is not None else []).append, clock=lambda: NOW)


def test_fetch_repo_collects_every_source() -> None:
    fake = FakeGitHub()

    raw = fetch_repo(client_for(fake.handle), "octo/repo")

    assert raw.info["nameWithOwner"] == "octo/repo"
    assert [p["number"] for p in raw.pulls] == [1]
    assert raw.issues == [{"number": 7}]
    assert raw.commits == [{"oid": "abc"}]
    assert raw.docs == [
        {"path": "README.md", "sha": "readme", "text": "# readme ✓"},
        {"path": "docs/Design.MD", "sha": "design", "text": "# design ✓"},
    ]
    assert fake.variables("DefaultBranchHistory") == [{"owner": "octo", "name": "repo", "after": None, "since": None}]


def test_pull_requests_are_paginated() -> None:
    fake = FakeGitHub(RepoPullRequests=[page([pull(2)], after="next"), page([pull(1)])])

    raw = fetch_repo(client_for(fake.handle), "octo/repo")

    assert [p["number"] for p in raw.pulls] == [2, 1]
    assert [v["after"] for v in fake.variables("RepoPullRequests")] == [None, "1"]


def test_cached_repo_fetches_only_what_changed_since() -> None:
    cached = RawRepo(
        info=INFO,
        pulls=[pull(1, updated="2026-01-01T00:00:00Z"), pull(2, updated="2026-02-01T00:00:00Z")],
        issues=[{"number": 7, "updatedAt": "2026-01-15T00:00:00Z", "title": "old"}],
        commits=[commit("aaa", "2026-01-01T00:00:00Z"), commit("bbb", "2026-02-01T00:00:00Z")],
        docs=[],
    )
    newest_first = [
        pull(3, updated="2026-03-01T00:00:00Z"),
        pull(2, updated="2026-02-01T00:00:00Z"),
        pull(1, updated="2026-01-01T00:00:00Z"),
    ]
    fake = FakeGitHub(
        RepoPullRequests=[page(newest_first, after="next"), page([pull(0, updated="2025-01-01T00:00:00Z")])],
        RepoIssues=[page([{"number": 7, "updatedAt": "2026-03-01T00:00:00Z", "title": "edited"}])],
        DefaultBranchHistory=[page([commit("ccc", "2026-03-01T00:00:00Z"), commit("bbb", "2026-02-01T00:00:00Z")])],
    )

    raw = fetch_repo(client_for(fake.handle), "octo/repo", cached)

    assert [p["number"] for p in raw.pulls] == [1, 2, 3]
    assert len(fake.variables("RepoPullRequests")) == 1
    assert raw.issues == [{"number": 7, "updatedAt": "2026-03-01T00:00:00Z", "title": "edited"}]
    assert [c["oid"] for c in raw.commits] == ["aaa", "bbb", "ccc"]
    assert fake.variables("DefaultBranchHistory")[0]["since"] == "2026-02-01T00:00:00Z"


def test_cache_from_before_incremental_fetching_is_fetched_again() -> None:
    cached = RawRepo(info=INFO, pulls=[pull(9)], issues=[], commits=[{"oid": "old"}], docs=[])
    fake = FakeGitHub()

    raw = fetch_repo(client_for(fake.handle), "octo/repo", cached)

    assert [p["number"] for p in raw.pulls] == [1]
    assert raw.commits == [{"oid": "abc"}]
    assert fake.variables("DefaultBranchHistory")[0]["since"] is None


def test_docs_are_synced_by_blob_sha(caplog: pytest.LogCaptureFixture) -> None:
    cached_docs = [
        {"path": "README.md", "sha": "readme", "text": "cached readme"},
        {"path": "docs/Design.MD", "sha": "old-design", "text": "old design"},
        {"path": "docs/removed.md", "sha": "gone", "text": "deleted since"},
    ]
    fake = FakeGitHub()

    with caplog.at_level(logging.INFO):
        raw = fetch_repo(client_for(fake.handle), "octo/repo", RawRepo(INFO, [], [], [], cached_docs))

    assert raw.docs == [
        {"path": "README.md", "sha": "readme", "text": "cached readme"},
        {"path": "docs/Design.MD", "sha": "design", "text": "# design ✓"},
    ]
    assert fake.blobs() == ["design"]
    assert "octo/repo: 1 markdown files removed: docs/removed.md" in caplog.text


def test_repo_without_a_default_branch_has_no_commits_or_docs() -> None:
    fake = FakeGitHub(branch=None)

    raw = fetch_repo(client_for(fake.handle), "octo/repo")

    assert raw.commits == []
    assert raw.docs == []
    assert all(request.url.path == "/graphql" for request in fake.requests)


def test_truncated_inner_connection_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    first = pull(4, files=120)
    first["comments"] = {"totalCount": 2, "nodes": []}
    first["reviewThreads"] = {"totalCount": 3, "nodes": []}
    fake = FakeGitHub(RepoPullRequests=[page([first, pull(5, files=110)])])

    with caplog.at_level(logging.DEBUG):
        fetch_repo(client_for(fake.handle), "octo/repo")

    assert "octo/repo#4: only 0 of 120 files fetched" in caplog.text
    assert ("octo/repo: files truncated to the first 0 (PRs affected: 2)", logging.INFO) in [
        (record.message, record.levelno) for record in caplog.records
    ]
    warnings = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == [
        "octo/repo#4: only 0 of 2 comments fetched",
        "octo/repo#4: only 0 of 3 reviewThreads fetched",
    ]


def test_truncated_tree_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeGitHub(truncated=True)

    with caplog.at_level(logging.WARNING):
        fetch_repo(client_for(fake.handle), "octo/repo")

    assert "file tree is truncated" in caplog.text


def test_missing_repository_is_an_error() -> None:
    client = client_for(lambda _: httpx.Response(200, json={"data": {"repository": None}}))

    with pytest.raises(GitHubError, match="octo/repo: not found"):
        fetch_repo(client, "octo/repo")


def test_viewer_repositories_are_paginated() -> None:
    first = page([{"nameWithOwner": "octocat/repo", "isFork": False, "owner": {"login": "octocat"}}], after="next")
    second = page([{"nameWithOwner": "octo-org/fork", "isFork": True, "owner": {"login": "octo-org"}}])
    fake = FakeGitHub(ViewerRepositories=[first, second])

    login, repos = viewer_repositories(client_for(fake.handle))

    assert login == "octocat"
    assert repos == [ViewerRepo("octocat/repo", "octocat", False), ViewerRepo("octo-org/fork", "octo-org", True)]


def test_involved_repositories_are_listed_once(caplog: pytest.LogCaptureFixture) -> None:
    nodes = [{"repository": {"nameWithOwner": name}} for name in ["other/a", "other/b", "other/a"]]
    fake = FakeGitHub(InvolvedRepositories=[page(nodes) | {"issueCount": 3}])

    names = involved_repositories(client_for(fake.handle), "octocat")

    assert names == ["other/a", "other/b"]
    assert fake.variables("InvolvedRepositories")[0]["query"] == "involves:octocat -user:octocat"
    assert "only the first" not in caplog.text


def test_involved_repositories_beyond_the_search_limit_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    nodes = [{"repository": {"nameWithOwner": "other/a"}}, {}]
    fake = FakeGitHub(InvolvedRepositories=[page(nodes) | {"issueCount": github.SEARCH_LIMIT + 1}])

    with caplog.at_level(logging.WARNING):
        names = involved_repositories(client_for(fake.handle), "octocat")

    assert names == ["other/a"]
    assert f"search found {github.SEARCH_LIMIT + 1} PRs and issues, but only the first 1000" in caplog.text


def test_contributions_are_the_viewers_pull_requests_and_issues() -> None:
    nodes = [pull(5) | {"__typename": "PullRequest"}, {"__typename": "Issue", "number": 6}]
    fake = FakeGitHub(Contributions=[page(nodes)])

    raw = fetch_contributions(client_for(fake.handle), "other/project", "octocat")

    assert [p["number"] for p in raw.pulls] == [5]
    assert raw.issues == [{"__typename": "Issue", "number": 6}]
    assert (raw.commits, raw.docs) == ([], [])
    assert fake.variables("Contributions")[0]["query"] == "repo:other/project involves:octocat"


def test_graphql_errors_are_raised() -> None:
    errors = [{"message": "Bad credentials"}, {"message": "Try again"}]
    client = client_for(lambda _: httpx.Response(200, json={"errors": errors}))

    with pytest.raises(GitHubError, match="RepoInfo: Bad credentials; Try again"):
        client.query("RepoInfo", owner="octo", name="repo")


def test_http_errors_are_raised() -> None:
    client = client_for(lambda _: httpx.Response(401, json={"message": "Bad credentials"}))

    with pytest.raises(GitHubError, match="HTTP 401"):
        client.get("/repos/octo/repo/git/trees/main")


def rate_limited(remaining: int, status: int = 200, **headers: str) -> httpx.Response:
    limits = {"x-ratelimit-remaining": str(remaining), "x-ratelimit-reset": str(int(NOW) + 60)}
    return httpx.Response(status, headers=limits | headers, json={"ok": True})


@pytest.mark.parametrize("remaining", [github.RATE_LIMIT_FLOOR, 4999])
def test_enough_rate_limit_left_means_no_waiting(remaining: int) -> None:
    sleeps: list[float] = []

    result = client_for(lambda _: rate_limited(remaining), sleeps).get("/rate")

    assert result == {"ok": True}
    assert sleeps == []


def test_low_rate_limit_waits_for_the_reset(caplog: pytest.LogCaptureFixture) -> None:
    sleeps: list[float] = []

    with caplog.at_level(logging.WARNING):
        result = client_for(lambda _: rate_limited(github.RATE_LIMIT_FLOOR - 1), sleeps).get("/rate")

    assert result == {"ok": True}
    assert sleeps == [61]
    assert "waiting 61 s" in caplog.text


def test_rate_limited_request_is_retried_after_waiting() -> None:
    responses = iter([rate_limited(0, status=429, **{"retry-after": "5"}), rate_limited(4000)])
    sleeps: list[float] = []

    result = client_for(lambda _: next(responses), sleeps).get("/rate")

    assert result == {"ok": True}
    assert sleeps == [5]


def test_spent_graphql_limit_reported_as_ok_is_retried() -> None:
    spent = httpx.Response(
        200,
        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(NOW) + 60)},
        json={"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]},
    )
    responses = iter([spent, httpx.Response(200, json={"data": {"repository": INFO}})])
    sleeps: list[float] = []

    result = client_for(lambda _: next(responses), sleeps).query("RepoInfo", owner="octo", name="repo")

    assert result == {"repository": INFO}
    assert sleeps == [61]


def test_reset_in_the_past_waits_a_second() -> None:
    sleeps: list[float] = []
    stale = httpx.Response(200, headers={"x-ratelimit-remaining": "5", "x-ratelimit-reset": "1"}, json={})

    client_for(lambda _: stale, sleeps).get("/rate")

    assert sleeps == [1]


def test_request_still_rate_limited_after_retries_fails() -> None:
    sleeps: list[float] = []

    with pytest.raises(GitHubError, match="still rate limited after 3 attempts"):
        client_for(lambda _: rate_limited(0, status=403), sleeps).get("/rate")

    assert len(sleeps) == github.ATTEMPTS - 1


def test_forbidden_without_rate_limit_headers_is_an_error() -> None:
    sleeps: list[float] = []

    with pytest.raises(GitHubError, match="HTTP 403"):
        client_for(lambda _: httpx.Response(403, json={}), sleeps).get("/repos/octo/secret")

    assert sleeps == []


def test_connect_authenticates_every_request() -> None:
    client = GitHubClient.connect("secret")

    assert client._http.headers["Authorization"] == "Bearer secret"
    client.close()


def test_raw_repo_round_trips_through_the_cache(tmp_path: Path) -> None:
    raw = RawRepo(info=INFO, pulls=[{"title": "Grüße"}], issues=[], commits=[], docs=[])
    directory = tmp_path / "octo" / "repo"

    raw.save(directory)

    assert RawRepo.is_cached(directory)
    assert RawRepo.load(directory) == raw
    assert "Grüße" in (directory / "pulls.json").read_text(encoding="utf-8")


def test_raw_repo_removal_keeps_other_files(tmp_path: Path) -> None:
    RawRepo(info=INFO, pulls=[], issues=[], commits=[], docs=[]).save(tmp_path)
    (tmp_path / "plans.json").write_text("[]", encoding="utf-8")

    RawRepo.remove(tmp_path)
    RawRepo.remove(tmp_path)

    assert not RawRepo.is_cached(tmp_path)
    assert [path.name for path in tmp_path.iterdir()] == ["plans.json"]


def test_token_comes_from_the_environment_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(github.TOKEN_ENV, "from-env")
    monkeypatch.setattr(subprocess, "run", pytest.fail)

    assert resolve_token() == "from-env"


def test_token_falls_back_to_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(github.TOKEN_ENV, raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *_, **__: subprocess.CompletedProcess(["gh"], 0, stdout="from-gh\n"))

    assert resolve_token() == "from-gh"


@pytest.mark.parametrize("failure", [FileNotFoundError("gh"), subprocess.CalledProcessError(1, ["gh"])])
def test_missing_token_is_an_error(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def fail(*_: Any, **__: Any) -> None:
        raise failure

    monkeypatch.delenv(github.TOKEN_ENV, raising=False)
    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(GitHubError, match="No GitHub token"):
        resolve_token()
