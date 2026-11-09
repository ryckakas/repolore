import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from repolore.chunking import chunk_document
from repolore.documents import GHOST, Document, DocumentKind, Visibility, documents_from, local_documents
from repolore.sources.github import RawRepo
from repolore.sources.local_docs import LocalFile

FIXTURES = Path(__file__).parent / "fixtures"
CRASHCAUSE = RawRepo.load(FIXTURES / "crashcause")
OPENWB_854 = json.loads((FIXTURES / "openwb-ui-settings-854.json").read_text(encoding="utf-8"))
OPENWB_INFO = {
    "nameWithOwner": "openWB/openwb-ui-settings",
    "url": "https://github.com/openWB/openwb-ui-settings",
    "isPrivate": False,
    "defaultBranchRef": {"name": "main"},
}
BUILT_PULL = "https://github.com/ryckakas/crashcause/pull/1"
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


def built_pull(*messages: tuple[str, str]) -> Document:
    pulls = copy.deepcopy(CRASHCAUSE.pulls)
    template = pulls[0]["commits"]["nodes"][0]
    pulls[0]["commits"]["nodes"] = [
        template | {"commit": template["commit"] | {"messageHeadline": headline, "messageBody": body}}
        for headline, body in messages
    ]
    return by_url(documents_from(crashcause(pulls=pulls)), BUILT_PULL)


def test_empty_pull_request_body_is_built_from_commits_and_files() -> None:
    pull = by_url(documents_from(CRASHCAUSE), BUILT_PULL)

    assert pull.body.startswith(
        "## Commits\n\n- **Fix CI: golangci-lint v2 config schema, x/text + x/net security bumps**\n\n  "
    )
    assert pull.body.count("\n- **") == 3
    assert "## Files\n\n- `" in pull.body


def test_every_commit_headline_of_a_pull_request_without_a_body_reaches_a_chunk() -> None:
    pull = built_pull(("a: with a body", "Text."), ("b: headline only", ""), ("c: headline only", ""))

    texts = [chunk.text for chunk in chunk_document(pull)]

    for headline in ("a: with a body", "b: headline only", "c: headline only"):
        assert any(headline in text for text in texts), headline


def test_a_commit_body_stays_under_its_headline_even_with_headings_of_its_own() -> None:
    pull = built_pull(
        ("a: first", "## Why\n\nBecause.\n\n```\ncode\n```"), ("b: second", ""), ("c: third", "One.\nTwo.")
    )

    assert [chunk.section for chunk in chunk_document(pull)] == ["Commits", "Files"]
    assert pull.body.startswith(
        "## Commits\n\n"
        "- **a: first**\n\n  ## Why\n\n  Because.\n\n  ```\n  code\n  ```\n"
        "- **b: second**\n"
        "- **c: third**\n\n  One.\n  Two.\n\n"
        "## Files"
    )


def test_a_long_commit_list_is_split_without_losing_or_repeating_a_headline() -> None:
    headlines = [f"change number {index}" for index in range(40)]
    pull = built_pull(*((headline, "") for headline in headlines))

    commits = [chunk.text for chunk in chunk_document(pull, max_tokens=100) if chunk.section == "Commits"]

    assert len(commits) > 1
    assert sorted(headline for text in commits for headline in headlines if f"**{headline}**" in text) == sorted(
        headlines
    )


def test_empty_pull_request_without_files_lists_only_commits() -> None:
    pulls = edited(CRASHCAUSE.pulls, 0, files={"totalCount": 0, "nodes": []})

    pull = by_url(documents_from(crashcause(pulls=pulls)), BUILT_PULL)

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

    assert issue.body.endswith("\n\n# Discussion\n\n**ghost:** First\n\n**octocat:** Second")
    assert "Deploy preview" not in issue.body


def test_the_discussion_is_not_filed_under_the_last_heading_of_the_body() -> None:
    comments = [{"author": None, "body": "Looks good.", "createdAt": "2026-01-02T00:00:00Z"}]
    issues = edited(
        CRASHCAUSE.issues, 1, body="# Summary\n\nIt crashes.", comments={"totalCount": 1, "nodes": comments}
    )

    issue = by_url(documents_from(crashcause(issues=issues)), "https://github.com/ryckakas/crashcause/issues/21")

    assert [chunk.header.rsplit(" · ", 1)[-1] for chunk in chunk_document(issue)] == ["Summary", "Discussion"]


def test_bot_issue_is_dropped() -> None:
    issues = edited(CRASHCAUSE.issues, 0, author={"login": "github-actions", "__typename": "Bot"})

    urls = {document.url for document in documents_from(crashcause(issues=issues))}

    assert "https://github.com/ryckakas/crashcause/issues/17" not in urls


def test_only_commits_without_a_pull_request_are_kept() -> None:
    commits = [document for document in documents_from(CRASHCAUSE) if document.kind is DocumentKind.COMMIT]

    assert [commit.url for commit in commits] == [PR_LESS_COMMIT]


def fetched_commit(oid: str) -> dict[str, Any]:
    return CRASHCAUSE.commits[1] | {"oid": oid, "url": f"https://github.com/ryckakas/crashcause/commit/{oid}"}


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


@pytest.mark.parametrize(
    ("parents", "kept"),
    [
        pytest.param(0, True, id="a root commit"),
        pytest.param(1, True, id="one parent"),
        pytest.param(2, False, id="a merge"),
    ],
)
def test_only_a_merge_is_left_out_for_its_parents(parents: int, kept: bool) -> None:
    node = fetched_commit("a" * 40) | {"parents": {"totalCount": parents, "nodes": []}}

    assert commit_kinds(crashcause(commits=[node])) == ([DocumentKind.COMMIT] if kept else [])


def test_a_repo_without_pull_requests_keeps_its_commits() -> None:
    raw = crashcause(pulls=[], commits=[fetched_commit("1639e4be1a8b129f403d643aad437f01d25c0c8e")])

    assert commit_kinds(raw) == [DocumentKind.COMMIT]


def on_a_pull_from(
    head: str,
    patterns: list[str],
    oid: str = "1639e4be1a8b129f403d643aad437f01d25c0c8e",
    cross_repository: bool = False,
) -> RawRepo:
    pulls = copy.deepcopy(CRASHCAUSE.pulls)
    for pull in pulls:
        pull |= {"headRefName": head, "isCrossRepository": cross_repository}
    return crashcause(info=CRASHCAUSE.info | {"branchPatterns": patterns}, pulls=pulls, commits=[fetched_commit(oid)])


@pytest.mark.parametrize(
    ("head", "patterns", "cross_repository", "kept"),
    [
        pytest.param(
            "staging/acme", ["staging/*"], False, True, id="a promotion, whether or not the branch still exists"
        ),
        pytest.param("main", ["staging/*"], False, True, id="from the default branch"),
        pytest.param("fix/ci", ["staging/*"], False, False, id="a feature pull request"),
        pytest.param("Staging/acme", ["staging/*"], False, False, id="branch names are case-sensitive"),
        pytest.param("main", ["staging/*"], True, False, id="a fork's branch"),
        pytest.param("staging/acme", ["staging/*"], True, False, id="a fork's branch that matches"),
        pytest.param("main", [], False, False, id="no branch rules"),
    ],
)
def test_only_a_pull_request_from_an_indexed_branch_leaves_its_commits_indexed(
    head: str, patterns: list[str], cross_repository: bool, kept: bool
) -> None:
    raw = on_a_pull_from(head, patterns, cross_repository=cross_repository)

    assert commit_kinds(raw) == ([DocumentKind.COMMIT] if kept else [])


def test_a_promotions_merge_commit_is_still_left_out() -> None:
    raw = on_a_pull_from("staging/acme", ["staging/*"], oid="f662f5cd0424f99d6af7b0281f14a43402e6c2cb")

    assert commit_kinds(raw) == []


def commit_urls(raw: RawRepo) -> list[str]:
    return [document.url for document in documents_from(raw) if document.kind is DocumentKind.COMMIT]


def test_a_cherry_pick_on_another_branch_is_one_document_and_the_first_stays() -> None:
    original = fetched_commit("a" * 40)
    picked = fetched_commit("b" * 40) | {
        "messageBody": f"{original['messageBody']}\n\n(cherry picked from commit {'a' * 40})",
        "committedDate": "2026-09-01T00:00:00Z",
    }

    assert commit_urls(crashcause(commits=[original, picked])) == [original["url"]]


def test_a_copy_of_a_pull_requests_commit_is_kept_when_the_pull_request_claims_the_first() -> None:
    claimed = fetched_commit("1639e4be1a8b129f403d643aad437f01d25c0c8e")
    picked = fetched_commit("b" * 40)

    assert commit_urls(crashcause(commits=[claimed, picked])) == [picked["url"]]


@pytest.mark.parametrize(
    "difference",
    [
        pytest.param({"authoredDate": "2026-09-01T00:00:00Z"}, id="written at another time"),
        pytest.param({"author": {"name": "Someone Else", "user": None}}, id="another author"),
        pytest.param({"messageHeadline": "Another headline"}, id="another headline"),
    ],
)
def test_commits_that_differ_in_identity_are_both_kept(difference: dict[str, Any]) -> None:
    other = fetched_commit("b" * 40) | difference

    assert len(commit_urls(crashcause(commits=[fetched_commit("a" * 40), other]))) == 2


@pytest.mark.parametrize(
    ("items", "field"),
    [
        pytest.param("pulls", "headRefName", id="a pull request without its source branch"),
        pytest.param("commits", "authoredDate", id="a commit without its authored date"),
    ],
)
def test_a_cached_node_without_a_required_field_is_refused(items: str, field: str) -> None:
    changed = copy.deepcopy(getattr(CRASHCAUSE, items))
    del changed[0][field]

    with pytest.raises(ValidationError):
        documents_from(crashcause(**{items: changed}))


def test_commit_headline_is_rejoined_and_claude_trailer_stripped() -> None:
    commit = by_url(documents_from(CRASHCAUSE), PR_LESS_COMMIT)

    assert commit.title == "Engine rules, collectors, AI layer, sinks with full test suites (Milestones 2+4 core)"
    assert commit.body.startswith("- engine: 17 rules")
    assert "Co-Authored-By" not in commit.body
    assert commit.ai_assisted
    assert commit.author == "ryckakas"


def test_commit_without_github_user_is_attributed_to_its_git_name() -> None:
    commits = edited(CRASHCAUSE.commits, 1, author={"name": "Someone Offline", "user": None})

    commit = by_url(documents_from(crashcause(commits=commits)), PR_LESS_COMMIT)

    assert commit.author == "Someone Offline"


def test_bot_commit_is_dropped() -> None:
    commits = edited(CRASHCAUSE.commits, 1, author={"name": "dependabot[bot]", "user": None})

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
    assert "\n\n# Discussion\n\n**benderl:**" in pull.body


def test_local_files_are_private_documents_of_their_repo() -> None:
    file = LocalFile(path=".plans/a/plan.md", url="file:///D:/projects/repo/.plans/a/plan.md", text="# Plan")

    documents = local_documents("ryckakas/crashcause", [file])

    assert documents == [
        Document(
            url="file:///D:/projects/repo/.plans/a/plan.md",
            repo="ryckakas/crashcause",
            kind=DocumentKind.LOCAL,
            visibility=Visibility.PRIVATE,
            title=".plans/a/plan.md",
            body="# Plan",
        )
    ]


def test_no_local_files_means_no_documents() -> None:
    assert local_documents("ryckakas/crashcause", []) == []
