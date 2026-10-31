import logging
from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from repolore.config import Config
from repolore.sources.github import (
    GitHubClient,
    GitHubError,
    ViewerOrg,
    ViewerRepo,
    involved_repositories,
    involved_repositories_among,
    viewer_organizations,
    viewer_repositories,
)
from repolore.sources.plans import find_plan_checkouts

log = logging.getLogger(__name__)


class Scope(StrEnum):
    OWNED = "owned"
    MEMBER = "member"
    CONTRIBUTED = "contributed"
    LOCAL = "local"


@dataclass(frozen=True)
class RepoSource:
    """A repo to ingest: its whole history (owned, member), only the viewer's part (contributed) or only local plans."""

    name: str
    scope: Scope
    checkout: Path | None = None


@dataclass(frozen=True)
class Discovery:
    login: str
    included: list[RepoSource]
    excluded: list[RepoSource]


def discover(client: GitHubClient, config: Config) -> Discovery:
    login, viewer_repos = viewer_repositories(client)
    found: dict[str, RepoSource] = {}
    for repo in viewer_repos:
        if not repo.is_fork:
            scope = Scope.OWNED if repo.owner.lower() == login.lower() else Scope.MEMBER
            found[repo.name.lower()] = RepoSource(repo.name, scope)
    for name in _contributed_repositories(client, login, viewer_repos):
        found.setdefault(name.lower(), RepoSource(name, Scope.CONTRIBUTED))
    for name, checkout in find_plan_checkouts(config.local_roots).items():
        known = found.get(name.lower())
        found[name.lower()] = replace(known, checkout=checkout) if known else RepoSource(name, Scope.LOCAL, checkout)
    sources = sorted(found.values(), key=lambda source: source.name.lower())
    return Discovery(
        login=login,
        included=[source for source in sources if config.selects(source.name)],
        excluded=[source for source in sources if not config.selects(source.name)],
    )


def _contributed_repositories(client: GitHubClient, login: str, viewer_repos: list[ViewerRepo]) -> list[str]:
    listed_orgs = _orgs_listed_in_full(_member_organizations(client), viewer_repos)
    forks = [repo.name for repo in viewer_repos if repo.is_fork and repo.owner.lower() in listed_orgs]
    return involved_repositories(client, login, sorted(listed_orgs)) + involved_repositories_among(client, login, forks)


def _member_organizations(client: GitHubClient) -> list[ViewerOrg]:
    try:
        return viewer_organizations(client)
    except GitHubError as error:
        log.warning("can't read your organizations, so the contributions search keeps them in: %s", error)
        return []


def _orgs_listed_in_full(orgs: list[ViewerOrg], viewer_repos: list[ViewerRepo]) -> set[str]:
    # Every repo in these orgs is fetched whole or is a fork searched on its own, so the involves: search can leave
    # them out instead of spending its 1,000 results on them. An org with repos ViewerRepositories misses stays in.
    listed = Counter(repo.owner.lower() for repo in viewer_repos)
    return {org.login.lower() for org in orgs if org.repo_count == listed[org.login.lower()]}
