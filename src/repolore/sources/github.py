import base64
import json
import logging
import os
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any, Self

import httpx

API_URL = "https://api.github.com"
TOKEN_ENV = "REPOLORE_GITHUB_TOKEN"
QUERIES = files("repolore.sources").joinpath("github.graphql").read_text(encoding="utf-8")
INNER_CONNECTIONS = ("files", "commits", "comments", "reviews", "reviewThreads")
DISCUSSION_CONNECTIONS = ("comments", "reviewThreads")
SKIPPED_DIRS = frozenset({"node_modules", "vendor"})
SYMLINK_MODE = "120000"
RATE_LIMIT_FLOOR = 100
RATE_LIMITED = frozenset({403, 429})
ATTEMPTS = 3
SEARCH_LIMIT = 1000

Json = dict[str, Any]
log = logging.getLogger(__name__)


class GitHubError(Exception):
    pass


def resolve_token() -> str:
    if token := os.environ.get(TOKEN_ENV):
        return token
    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitHubError(f"No GitHub token: set {TOKEN_ENV}, or log in with `gh auth login`.") from error
    return result.stdout.strip()


class GitHubClient:
    def __init__(
        self,
        http: httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._http = http
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def connect(cls, token: str) -> Self:
        headers = {"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
        return cls(httpx.Client(base_url=API_URL, headers=headers, timeout=60))

    def close(self) -> None:
        self._http.close()

    def query(self, operation: str, **variables: Any) -> Json:
        body = {"query": QUERIES, "operationName": operation, "variables": variables}
        payload = self._request("POST", "/graphql", json=body)
        if errors := payload.get("errors"):
            raise GitHubError(f"{operation}: {'; '.join(error['message'] for error in errors)}")
        return payload["data"]

    def get(self, path: str, **params: Any) -> Json:
        return self._request("GET", path, params=params)

    def _request(self, method: str, path: str, **kwargs: Any) -> Json:
        attempt = 1
        while True:
            response = self._http.request(method, path, **kwargs)
            wait = _rate_limit_wait(response, self._clock())
            if wait is None:
                return _parse(response)
            rejected = _rejected(response)
            if rejected and attempt == ATTEMPTS:
                raise GitHubError(f"{method} {path}: still rate limited after {ATTEMPTS} attempts")
            log.warning("GitHub rate limit is low, waiting %.0f s for it to reset", wait)
            self._sleep(wait)
            if not rejected:
                return _parse(response)
            attempt += 1


def _rate_limit_wait(response: httpx.Response, now: float) -> float | None:
    headers = response.headers
    if retry_after := headers.get("retry-after"):
        return float(retry_after)
    remaining, reset = headers.get("x-ratelimit-remaining"), headers.get("x-ratelimit-reset")
    if remaining is None or reset is None or int(remaining) >= RATE_LIMIT_FLOOR:
        return None
    return max(float(reset) - now, 0) + 1


def _rejected(response: httpx.Response) -> bool:
    # GraphQL reports a spent limit as HTTP 200 with an error, so the headers say more than the status.
    headers = response.headers
    return (
        response.status_code in RATE_LIMITED or "retry-after" in headers or headers.get("x-ratelimit-remaining") == "0"
    )


def _parse(response: httpx.Response) -> Json:
    if response.is_error:
        raise GitHubError(f"{response.request.method} {response.request.url.path}: HTTP {response.status_code}")
    return response.json()


@dataclass(frozen=True)
class RawRepo:
    info: Json
    pulls: list[Json]
    issues: list[Json]
    commits: list[Json]
    docs: list[Json]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for field in fields(self):
            content = json.dumps(getattr(self, field.name), ensure_ascii=False, indent=1)
            (directory / f"{field.name}.json").write_text(content, encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Self:
        def read(name: str) -> Any:
            return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))

        return cls(**{field.name: read(field.name) for field in fields(cls)})

    @classmethod
    def is_cached(cls, directory: Path) -> bool:
        return (directory / "info.json").exists()

    @classmethod
    def remove(cls, directory: Path) -> None:
        for field in fields(cls):
            (directory / f"{field.name}.json").unlink(missing_ok=True)


@dataclass(frozen=True)
class ViewerRepo:
    name: str
    owner: str
    is_fork: bool


def viewer_repositories(client: GitHubClient) -> tuple[str, list[ViewerRepo]]:
    """The viewer's login and every repo they own or can read as a collaborator or organization member."""
    login = ""
    repos: list[ViewerRepo] = []
    for data in _pages(client, "ViewerRepositories", lambda data: data["viewer"]["repositories"]):
        login = data["viewer"]["login"]
        nodes = data["viewer"]["repositories"]["nodes"]
        repos += [ViewerRepo(node["nameWithOwner"], node["owner"]["login"], node["isFork"]) for node in nodes]
    return login, repos


def involved_repositories(client: GitHubClient, login: str) -> list[str]:
    """Other people's repos with a PR or issue the viewer opened, commented on, reviewed or was assigned."""
    names: dict[str, None] = {}
    for data in _pages(client, "InvolvedRepositories", _search, query=f"involves:{login} -user:{login}"):
        if data["search"]["issueCount"] > SEARCH_LIMIT and not names:
            log.warning(
                "search found %d PRs and issues, but only the first %d are listed",
                data["search"]["issueCount"],
                SEARCH_LIMIT,
            )
        names.update((node["repository"]["nameWithOwner"], None) for node in data["search"]["nodes"] if node)
    return list(names)


def fetch_repo(client: GitHubClient, repo: str, cached: RawRepo | None = None) -> RawRepo:
    """Everything in a repo; with a cached copy, only what changed since it was fetched."""
    owner, name = repo.split("/")
    info = _repo_info(client, repo)
    branch = (info["defaultBranchRef"] or {}).get("name")
    pulls = _fetch_updated(
        client, "RepoPullRequests", "pullRequests", cached.pulls if cached else [], owner=owner, name=name
    )
    _log_truncation(repo, pulls)
    return RawRepo(
        info=info,
        pulls=pulls,
        issues=_fetch_updated(client, "RepoIssues", "issues", cached.issues if cached else [], owner=owner, name=name),
        commits=_fetch_commits(client, owner, name, cached.commits if cached else []) if branch else [],
        docs=_fetch_docs(client, repo, branch, cached.docs if cached else []) if branch else [],
    )


def fetch_contributions(client: GitHubClient, repo: str, login: str) -> RawRepo:
    """Only the PRs and issues in someone else's repo that the viewer took part in."""
    info = _repo_info(client, repo)
    nodes = [
        node
        for data in _pages(client, "Contributions", _search, query=f"repo:{repo} involves:{login}")
        for node in data["search"]["nodes"]
    ]
    pulls = [node for node in nodes if node["__typename"] == "PullRequest"]
    _log_truncation(repo, pulls)
    issues = [node for node in nodes if node["__typename"] == "Issue"]
    return RawRepo(info=info, pulls=pulls, issues=issues, commits=[], docs=[])


def _repo_info(client: GitHubClient, repo: str) -> Json:
    owner, name = repo.split("/")
    info = client.query("RepoInfo", owner=owner, name=name)["repository"]
    if info is None:
        raise GitHubError(f"{repo}: not found, or the token can't read it")
    return info


def _pages(
    client: GitHubClient, operation: str, connection: Callable[[Json], Json], **variables: Any
) -> Iterator[Json]:
    after = None
    while True:
        data = client.query(operation, after=after, **variables)
        yield data
        page_info = connection(data)["pageInfo"]
        if not page_info["hasNextPage"]:
            return
        after = page_info["endCursor"]


def _search(data: Json) -> Json:
    return data["search"]


def _history(data: Json) -> Json:
    return data["repository"]["defaultBranchRef"]["target"]["history"]


def _fetch_updated(
    client: GitHubClient, operation: str, connection: str, cached: list[Json], **variables: Any
) -> list[Json]:
    # Pages come newest-updated first, so everything after the first older node is already cached.
    watermark = _watermark(cached, "updatedAt")
    fresh: list[Json] = []
    for data in _pages(client, operation, lambda data: data["repository"][connection], **variables):
        nodes = data["repository"][connection]["nodes"]
        newer = [node for node in nodes if watermark is None or node["updatedAt"] >= watermark]
        fresh += newer
        if len(newer) < len(nodes):
            break
    return _merge(cached if watermark else [], fresh, "number")


def _fetch_commits(client: GitHubClient, owner: str, name: str, cached: list[Json]) -> list[Json]:
    since = _watermark(cached, "committedDate")
    fresh = [
        node
        for data in _pages(client, "DefaultBranchHistory", _history, owner=owner, name=name, since=since)
        for node in _history(data)["nodes"]
    ]
    return _merge(cached if since else [], fresh, "oid")


def _watermark(cached: list[Json], field: str) -> str | None:
    # A cache written before incremental fetching lacks the field; then everything is fetched again.
    stamps = [node.get(field, "") for node in cached]
    return max(stamps) if stamps and all(stamps) else None


def _merge(cached: list[Json], fresh: list[Json], key: str) -> list[Json]:
    merged = {node[key]: node for node in cached}
    merged.update((node[key], node) for node in fresh)
    return list(merged.values())


def _log_truncation(repo: str, pulls: list[Json]) -> None:
    summaries: Counter[tuple[str, int]] = Counter()
    for pull in pulls:
        for key in INNER_CONNECTIONS:
            fetched, total = len(pull[key]["nodes"]), pull[key]["totalCount"]
            if total <= fetched:
                continue
            if key in DISCUSSION_CONNECTIONS:
                log.warning("%s#%s: only %d of %d %s fetched", repo, pull["number"], fetched, total, key)
            else:
                log.debug("%s#%s: only %d of %d %s fetched", repo, pull["number"], fetched, total, key)
                summaries[key, fetched] += 1
    for (key, fetched), count in summaries.items():
        log.info("%s: %s truncated to the first %d (PRs affected: %d)", repo, key, fetched, count)


def _fetch_docs(client: GitHubClient, repo: str, branch: str, cached: list[Json]) -> list[Json]:
    tree = client.get(f"/repos/{repo}/git/trees/{branch}", recursive=1)
    if tree["truncated"]:
        log.warning("%s: the file tree is truncated, so some markdown files are missing", repo)
    known = {doc["path"]: doc for doc in cached if "sha" in doc}
    docs = [_doc(client, repo, entry, known.get(entry["path"])) for entry in tree["tree"] if _is_markdown_file(entry)]
    if removed := sorted({doc["path"] for doc in cached} - {doc["path"] for doc in docs}):
        log.info("%s: %d markdown files removed: %s", repo, len(removed), ", ".join(removed))
    return docs


def _doc(client: GitHubClient, repo: str, entry: Json, cached: Json | None) -> Json:
    if cached and cached["sha"] == entry["sha"]:
        return cached
    return {"path": entry["path"], "sha": entry["sha"], "text": _read_blob(client, repo, entry["sha"])}


def _is_markdown_file(entry: Json) -> bool:
    path = PurePosixPath(entry["path"])
    return (
        entry["type"] == "blob"
        and entry["mode"] != SYMLINK_MODE
        and path.suffix.lower() == ".md"
        and not SKIPPED_DIRS.intersection(path.parts)
    )


def _read_blob(client: GitHubClient, repo: str, sha: str) -> str:
    blob = client.get(f"/repos/{repo}/git/blobs/{sha}")
    return base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
