import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from repolore.documents import GHOST, Document, DocumentKind, Visibility, documents_from, plan_documents
from repolore.sources.github import RawRepo
from repolore.sources.plans import PlanFile

FIXTURES = Path(__file__).parent / "fixtures"
CRASHCAUSE = RawRepo.load(FIXTURES / "crashcause")
OPENWB_854 = json.loads((FIXTURES / "openwb-ui-settings-854.json").read_text(encoding="utf-8"))
OPENWB_INFO = {
    "nameWithOwner": "openWB/openwb-ui-settings",
    "url": "https://github.com/openWB/openwb-ui-settings",
    "isPrivate": False,
    "defaultBranchRef": {"name": "main"},
}
PR_LESS_COMMIT = "https://github.com/ryckakas/crashcause/commit/69531a3aa08bf8d47f1428261efc3038537127b6"


def crashcause(**changes: Any) -> RawRepo:
    return RawRepo(**{**CRASHCAUSE.__dict__, **copy.deepcopy(changes)})


def openwb(pull: dict[str, Any] | None = None) -> RawRepo:
    return RawRepo(info=OPENWB_INFO, pulls=[pull or OPENWB_854], issues=[], commits=[], docs=[])


def by_url(documents: list[Document], url: str) -> Document:
    return next(document for document in documents if document.url == url)


def edited(items: list[dict[str, Any]], index: int, **changes: Any) -> list[dict[str, Any]]:
    items = copy.deepcopy(items)
    items[index] |= changes
    return items


def test_crashcause_sample_yields_one_document_per_kept_source() -> None:
    documents = documents_from(CRASHCAUSE)

    assert Counter(document.kind for document in documents) == {
        DocumentKind.PR: 2,
        DocumentKind.ISSUE: 2,
        DocumentKind.COMMIT: 1,
        DocumentKind.DOC: 2,
    }
    assert {document.repo for document in documents} == {"ryckakas/crashcause"}
    assert {document.visibility for document in documents} == {Visibility.PUBLIC}


def test_templated_pull_request_keeps_its_body() -> None:
    pull = by_url(documents_from(CRASHCAUSE), "https://github.com/ryckakas/crashcause/pull/15")

    assert pull.title == "Prepare the v0.1.0 release"
    assert pull.body.startswith("Cut the CHANGELOG's Unreleased section as 0.1.0")
    assert pull.author == "ryckakas"
    assert pull.state == "MERGED"


def test_bot_pull_request_is_dropped() -> None:
    urls = {document.url for document in documents_from(CRASHCAUSE)}

    assert "https://github.com/ryckakas/crashcause/pull/3" not in urls


def test_empty_pull_request_body_is_built_from_commits_and_files() -> None:
    pull = by_url(documents_from(CRASHCAUSE), "https://github.com/ryckakas/crashcause/pull/1")

    assert pull.body.startswith("## Commits\n\n### ")
    assert "## Files\n\n- `" in pull.body
    assert pull.body.count("### ") == 3


def test_empty_pull_request_without_files_lists_only_commits() -> None:
    pulls = edited(CRASHCAUSE.pulls, 0, files={"totalCount": 0, "nodes": []})

    pull = by_url(documents_from(crashcause(pulls=pulls)), "https://github.com/ryckakas/crashcause/pull/1")

    assert "## Files" not in pull.body


def test_html_comments_and_claude_credit_are_stripped_from_pull_request_bodies() -> None:
    body = "<!-- template hint -->\nReal text.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    pulls = edited(CRASHCAUSE.pulls, 2, body=body)

    pull = by_url(documents_from(crashcause(pulls=pulls)), "https://github.com/ryckakas/crashcause/pull/15")

    assert pull.body == "Real text."
    assert pull.ai_assisted


def test_deleted_issue_author_becomes_ghost() -> None:
    issue = by_url(documents_from(CRASHCAUSE), "https://github.com/ryckakas/crashcause/issues/17")

    assert issue.author == GHOST
    assert issue.kind is DocumentKind.ISSUE


def test_human_comments_are_appended_and_bot_comments_dropped() -> None:
    comments = [
        {
            "author": {"login": "netlify", "__typename": "Bot"},
            "body": "Deploy preview ready",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            "author": {"login": "octocat", "__typename": "User"},
            "body": "Second <!-- hidden -->",
            "createdAt": "2026-01-03T00:00:00Z",
        },
        {"author": None, "body": "First", "createdAt": "2026-01-02T00:00:00Z"},
        {
            "author": {"login": "octocat", "__typename": "User"},
            "body": "<!-- only a comment -->",
            "createdAt": "2026-01-04T00:00:00Z",
        },
    ]
    issues = edited(CRASHCAUSE.issues, 1, comments={"totalCount": 4, "nodes": comments})

    issue = by_url(documents_from(crashcause(issues=issues)), "https://github.com/ryckakas/crashcause/issues/21")

    assert issue.body.endswith("## Discussion\n\n**ghost:** First\n\n**octocat:** Second")
    assert "Deploy preview" not in issue.body


def test_bot_issue_is_dropped() -> None:
    issues = edited(CRASHCAUSE.issues, 0, author={"login": "github-actions", "__typename": "Bot"})

    urls = {document.url for document in documents_from(crashcause(issues=issues))}

    assert "https://github.com/ryckakas/crashcause/issues/17" not in urls


def test_only_commits_without_a_pull_request_are_kept() -> None:
    commits = [document for document in documents_from(CRASHCAUSE) if document.kind is DocumentKind.COMMIT]

    assert [commit.url for commit in commits] == [PR_LESS_COMMIT]


def fetched_commit(oid: str) -> dict[str, Any]:
    pr_less = {key: value for key, value in CRASHCAUSE.commits[2].items() if key != "associatedPullRequests"}
    return pr_less | {"oid": oid}


def commit_kinds(raw: RawRepo) -> list[DocumentKind]:
    return [document.kind for document in documents_from(raw) if document.kind is DocumentKind.COMMIT]


@pytest.mark.parametrize(
    ("oid", "kept"),
    [
        pytest.param("69531a3aa08bf8d47f1428261efc3038537127b6", True, id="in no pull request"),
        pytest.param("1639e4be1a8b129f403d643aad437f01d25c0c8e", False, id="a merged pull request's commit"),
        pytest.param("f662f5cd0424f99d6af7b0281f14a43402e6c2cb", False, id="a pull request's merge commit"),
        pytest.param("0344bf74c36f4be57cf6eb03e411ac2e874c0f7d", False, id="a closed pull request's commit"),
    ],
)
def test_a_fetched_commit_is_matched_to_the_pull_requests_by_oid(oid: str, kept: bool) -> None:
    assert commit_kinds(crashcause(commits=[fetched_commit(oid)])) == ([DocumentKind.COMMIT] if kept else [])


def test_a_repo_without_pull_requests_keeps_its_commits() -> None:
    raw = crashcause(pulls=[], commits=[fetched_commit("1639e4be1a8b129f403d643aad437f01d25c0c8e")])

    assert commit_kinds(raw) == [DocumentKind.COMMIT]


def test_a_commit_cached_with_githubs_answer_keeps_it() -> None:
    cached = fetched_commit("1639e4be1a8b129f403d643aad437f01d25c0c8e") | {"associatedPullRequests": {"totalCount": 0}}

    assert commit_kinds(crashcause(commits=[cached])) == [DocumentKind.COMMIT]


def test_pull_requests_cached_without_oids_still_load() -> None:
    pulls = copy.deepcopy(CRASHCAUSE.pulls)
    del pulls[0]["mergeCommit"]
    for node in pulls[0]["commits"]["nodes"]:
        del node["commit"]["oid"]

    documents = documents_from(crashcause(pulls=pulls))

    assert by_url(documents, "https://github.com/ryckakas/crashcause/pull/1").kind is DocumentKind.PR


def test_commit_headline_is_rejoined_and_claude_trailer_stripped() -> None:
    commit = by_url(documents_from(CRASHCAUSE), PR_LESS_COMMIT)

    assert commit.title == "Engine rules, collectors, AI layer, sinks with full test suites (Milestones 2+4 core)"
    assert commit.body.startswith("- engine: 17 rules")
    assert "Co-Authored-By" not in commit.body
    assert commit.ai_assisted
    assert commit.author == "ryckakas"


def test_commit_without_github_user_is_attributed_to_its_git_name() -> None:
    commits = edited(CRASHCAUSE.commits, 2, author={"name": "Someone Offline", "user": None})

    commit = by_url(documents_from(crashcause(commits=commits)), PR_LESS_COMMIT)

    assert commit.author == "Someone Offline"


def test_bot_commit_is_dropped() -> None:
    commits = edited(CRASHCAUSE.commits, 2, author={"name": "dependabot[bot]", "user": None})

    kinds = {document.kind for document in documents_from(crashcause(commits=commits))}

    assert DocumentKind.COMMIT not in kinds


def test_doc_url_points_at_the_default_branch() -> None:
    doc = by_url(documents_from(CRASHCAUSE), "https://github.com/ryckakas/crashcause/blob/main/README.md")

    assert doc.title == "README.md"
    assert doc.author is None
    assert doc.created_at is None


def test_private_repo_marks_every_document_private() -> None:
    documents = documents_from(crashcause(info={**CRASHCAUSE.info, "isPrivate": True}))

    assert {document.visibility for document in documents} == {Visibility.PRIVATE}


def test_review_thread_becomes_one_document_with_its_diff_and_comments() -> None:
    documents = documents_from(openwb())

    reviews = [document for document in documents if document.kind is DocumentKind.REVIEW]
    assert len(reviews) == 1
    review = reviews[0]
    assert review.title == "PR #854 · src/components/OpenwbPageMessages.vue:185"
    assert review.body.startswith("```diff\n@@")
    assert review.body.count("\n\n**") == 3
    assert review.url.startswith("https://github.com/openWB/openwb-ui-settings/pull/854#discussion_r")
    assert review.state == "OPEN"


def test_review_thread_keeps_only_the_last_lines_of_a_long_hunk() -> None:
    pull = copy.deepcopy(OPENWB_854)
    root = pull["reviewThreads"]["nodes"][0]["comments"]["nodes"][0]
    root["diffHunk"] = "\n".join(f"+line {n}" for n in range(30))

    review = next(document for document in documents_from(openwb(pull)) if document.kind is DocumentKind.REVIEW)

    assert "+line 14\n" not in review.body
    assert "```diff\n+line 15\n" in review.body


@pytest.mark.parametrize(
    ("resolved", "outdated", "state"),
    [(True, True, "RESOLVED"), (False, True, "OUTDATED"), (False, False, "OPEN")],
)
def test_review_thread_state(resolved: bool, outdated: bool, state: str) -> None:
    pull = copy.deepcopy(OPENWB_854)
    pull["reviewThreads"]["nodes"][0] |= {"isResolved": resolved, "isOutdated": outdated}

    review = next(document for document in documents_from(openwb(pull)) if document.kind is DocumentKind.REVIEW)

    assert review.state == state


def test_bot_review_thread_is_dropped() -> None:
    pull = copy.deepcopy(OPENWB_854)
    pull["reviewThreads"]["nodes"][0]["comments"]["nodes"][0]["author"] = {"login": "coderabbitai", "__typename": "Bot"}

    kinds = {document.kind for document in documents_from(openwb(pull))}

    assert DocumentKind.REVIEW not in kinds


def test_bot_review_thread_is_kept_when_asked_for() -> None:
    pull = copy.deepcopy(OPENWB_854)
    pull["reviewThreads"]["nodes"][0]["comments"]["nodes"][0]["author"] = {"login": "coderabbitai", "__typename": "Bot"}

    reviews = [d for d in documents_from(openwb(pull), include_bot_reviews=True) if d.kind is DocumentKind.REVIEW]

    assert [review.author for review in reviews] == ["coderabbitai"]


def test_external_pull_request_carries_its_discussion() -> None:
    pull = by_url(documents_from(openwb()), "https://github.com/openWB/openwb-ui-settings/pull/854")

    assert pull.author == "LKuemmel"
    assert "## Discussion\n\n**benderl:**" in pull.body


def test_plans_are_private_documents_of_their_repo() -> None:
    plan = PlanFile(path=".plans/a/plan.md", url="file:///D:/projects/repo/.plans/a/plan.md", text="# Plan")

    documents = plan_documents("ryckakas/crashcause", [plan])

    assert documents == [
        Document(
            url="file:///D:/projects/repo/.plans/a/plan.md",
            repo="ryckakas/crashcause",
            kind=DocumentKind.PLAN,
            visibility=Visibility.PRIVATE,
            title=".plans/a/plan.md",
            body="# Plan",
        )
    ]


def test_no_plans_means_no_plan_documents() -> None:
    assert plan_documents("ryckakas/crashcause", []) == []
