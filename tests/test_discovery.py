import logging
from pathlib import Path

import pytest

from repolore import discovery as discovery_module
from repolore.config import Config
from repolore.discovery import RepoSource, Scope, discover
from repolore.sources.github import GitHubClient, GitHubError, ViewerOrg, ViewerRepo

CLIENT = GitHubClient.connect("unused")


@pytest.fixture
def found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    viewer_repos = [
        ViewerRepo("octocat/repo", "octocat", is_fork=False),
        ViewerRepo("octocat/fork", "octocat", is_fork=True),
        ViewerRepo("octo-org/service", "octo-org", is_fork=False),
    ]
    monkeypatch.setattr(discovery_module, "viewer_repositories", lambda _: ("OctoCat", viewer_repos))
    monkeypatch.setattr(discovery_module, "viewer_organizations", lambda _: [])
    monkeypatch.setattr(discovery_module, "involved_repositories", lambda *_: ["octo-org/service", "other/project"])
    checkouts = {"OctoCat/Repo": tmp_path / "repo", "someone/clone": tmp_path / "clone"}
    monkeypatch.setattr(discovery_module, "find_plan_checkouts", lambda _: checkouts)
    return tmp_path


def test_repos_are_scoped_by_how_the_viewer_relates_to_them(found: Path) -> None:
    result = discover(CLIENT, Config())

    assert result.login == "OctoCat"
    assert result.included == [
        RepoSource("octo-org/service", Scope.MEMBER),
        RepoSource("octocat/repo", Scope.OWNED, found / "repo"),
        RepoSource("other/project", Scope.CONTRIBUTED),
        RepoSource("someone/clone", Scope.LOCAL, found / "clone"),
    ]
    assert result.excluded == []


def test_config_globs_split_included_from_excluded(found: Path) -> None:
    result = discover(CLIENT, Config(include=["octo*/*"], exclude=["octo-org/*"]))

    assert [source.name for source in result.included] == ["octocat/repo"]
    assert [source.name for source in result.excluded] == ["octo-org/service", "other/project", "someone/clone"]


def test_nothing_found_is_an_empty_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery_module, "viewer_repositories", lambda _: ("octocat", []))
    monkeypatch.setattr(discovery_module, "viewer_organizations", lambda _: [])
    monkeypatch.setattr(discovery_module, "involved_repositories", lambda *_: [])

    result = discover(CLIENT, Config())

    assert (result.included, result.excluded) == ([], [])


ORG_REPOS = [
    ViewerRepo("Octo-Org/service", "Octo-Org", is_fork=False),
    ViewerRepo("Octo-Org/upstream-fork", "Octo-Org", is_fork=True),
]


def searches(monkeypatch: pytest.MonkeyPatch, orgs: list[ViewerOrg]) -> dict[str, list[str]]:
    calls: dict[str, list[str]] = {}

    def search_outside(_: GitHubClient, login: str, excluded_orgs: list[str]) -> list[str]:
        calls["excluded_orgs"] = list(excluded_orgs)
        return ["other/project"]

    def search_among(_: GitHubClient, login: str, repos: list[str]) -> list[str]:
        calls["among"] = list(repos)
        return repos

    monkeypatch.setattr(discovery_module, "viewer_repositories", lambda _: ("octocat", ORG_REPOS))
    monkeypatch.setattr(discovery_module, "viewer_organizations", lambda _: orgs)
    monkeypatch.setattr(discovery_module, "involved_repositories", search_outside)
    monkeypatch.setattr(discovery_module, "involved_repositories_among", search_among)
    monkeypatch.setattr(discovery_module, "find_plan_checkouts", lambda _: {})
    return calls


def test_orgs_listed_in_full_are_left_out_of_the_search_and_their_forks_searched_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = searches(monkeypatch, [ViewerOrg("octo-org", repo_count=2)])

    result = discover(CLIENT, Config())

    assert calls == {"excluded_orgs": ["octo-org"], "among": ["Octo-Org/upstream-fork"]}
    assert result.included == [
        RepoSource("Octo-Org/service", Scope.MEMBER),
        RepoSource("Octo-Org/upstream-fork", Scope.CONTRIBUTED),
        RepoSource("other/project", Scope.CONTRIBUTED),
    ]


def test_an_org_with_repos_the_viewer_is_not_given_stays_in_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = searches(monkeypatch, [ViewerOrg("Octo-Org", repo_count=3)])

    discover(CLIENT, Config())

    assert calls == {"excluded_orgs": [], "among": []}


def test_orgs_without_forks_leave_no_forks_to_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = searches(monkeypatch, [ViewerOrg("Octo-Org", repo_count=2), ViewerOrg("empty-org", repo_count=0)])
    no_forks = [ORG_REPOS[0], ViewerRepo("Octo-Org/api", "Octo-Org", is_fork=False)]
    monkeypatch.setattr(discovery_module, "viewer_repositories", lambda _: ("octocat", no_forks))

    discover(CLIENT, Config())

    assert calls == {"excluded_orgs": ["empty-org", "octo-org"], "among": []}


def test_unreadable_organizations_keep_every_org_in_the_search(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = searches(monkeypatch, [])

    def saml_protected(_: GitHubClient) -> list[ViewerOrg]:
        raise GitHubError("ViewerOrganizations: Resource protected by organization SAML enforcement.")

    monkeypatch.setattr(discovery_module, "viewer_organizations", saml_protected)

    with caplog.at_level(logging.WARNING):
        result = discover(CLIENT, Config())

    assert calls == {"excluded_orgs": [], "among": []}
    assert [source.name for source in result.included] == ["Octo-Org/service", "other/project"]
    assert "can't read your organizations" in caplog.text
    assert "SAML enforcement" in caplog.text
