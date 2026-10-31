import re
from collections.abc import Iterable, Iterator
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from repolore.sources.github import RawRepo
from repolore.sources.models import (
    Actor,
    Comment,
    Commit,
    DocFile,
    Issue,
    PullRequest,
    PullRequestCommit,
    RepoInfo,
    ReviewThread,
)
from repolore.sources.plans import PlanFile

GHOST = "ghost"
DIFF_HUNK_LINES = 15
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
CLAUDE_TRAILER = re.compile(
    r"^[ \t]*(?:co-authored-by:.*\b(?:claude|anthropic)\b.*"
    r"|claude-session:.*"
    r"|https://claude\.ai/code/\S*"
    r"|🤖 generated with \[claude code\].*)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


class DocumentKind(StrEnum):
    PR = "pr"
    REVIEW = "review"
    ISSUE = "issue"
    COMMIT = "commit"
    DOC = "doc"
    PLAN = "plan"


class Visibility(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    repo: str
    kind: DocumentKind
    visibility: Visibility
    title: str
    body: str
    author: str | None = None
    created_at: datetime | None = None
    state: str | None = None
    ai_assisted: bool = False


def documents_from(raw: RawRepo, include_bot_reviews: bool = False) -> list[Document]:
    repo = RepoInfo.model_validate(raw.info)
    pulls = [PullRequest.model_validate(pull) for pull in raw.pulls]
    return [
        *_pull_documents(repo, pulls),
        *_review_documents(repo, pulls, include_bot_reviews),
        *_issue_documents(repo, (Issue.model_validate(issue) for issue in raw.issues)),
        *_commit_documents(repo, (Commit.model_validate(commit) for commit in raw.commits), _pull_commit_oids(pulls)),
        *_doc_documents(repo, (DocFile.model_validate(doc) for doc in raw.docs)),
    ]


def _pull_documents(repo: RepoInfo, pulls: Iterable[PullRequest]) -> Iterator[Document]:
    for pull in pulls:
        if _is_bot(pull.author):
            continue
        body, ai_assisted = _strip_claude_trailers(_strip_html_comments(pull.body))
        commits = [_commit_message(commit) for commit in pull.commits.nodes]
        yield Document(
            url=pull.url,
            repo=repo.name_with_owner,
            kind=DocumentKind.PR,
            visibility=_visibility(repo),
            title=pull.title,
            body=_with_discussion(
                body or _synthetic_pull_body(pull, commits), [*pull.comments.nodes, *pull.reviews.nodes]
            ),
            author=_author(pull.author),
            created_at=pull.created_at,
            state=pull.state,
            ai_assisted=ai_assisted or any(commit_ai for _, commit_ai in commits),
        )


def _synthetic_pull_body(pull: PullRequest, commits: list[tuple[str, bool]]) -> str:
    sections = ["## Commits", *(message for message, _ in commits)]
    if pull.files.nodes:
        sections += ["## Files", "\n".join(f"- `{file.path}`" for file in pull.files.nodes)]
    return "\n\n".join(sections)


def _commit_message(pull_commit: PullRequestCommit) -> tuple[str, bool]:
    commit = pull_commit.commit
    headline, body = _rejoin_headline(commit.message_headline, commit.message_body)
    body, ai_assisted = _strip_claude_trailers(body)
    return "\n\n".join(part for part in (f"### {headline}", body) if part), ai_assisted


def _review_documents(repo: RepoInfo, pulls: Iterable[PullRequest], include_bots: bool) -> Iterator[Document]:
    for pull in pulls:
        for thread in pull.review_threads.nodes:
            if thread.comments.nodes and (include_bots or not _is_bot(thread.comments.nodes[0].author)):
                yield _review_document(repo, pull, thread)


def _review_document(repo: RepoInfo, pull: PullRequest, thread: ReviewThread) -> Document:
    root = thread.comments.nodes[0]
    hunk = "\n".join(root.diff_hunk.splitlines()[-DIFF_HUNK_LINES:])
    conversation = "\n\n".join(
        f"**{_author(comment.author)}:** {comment.body.strip()}" for comment in thread.comments.nodes
    )
    return Document(
        url=root.url,
        repo=repo.name_with_owner,
        kind=DocumentKind.REVIEW,
        visibility=_visibility(repo),
        title=f"PR #{pull.number} · {thread.path}:{thread.line or thread.original_line}",
        body=f"```diff\n{hunk}\n```\n\n{conversation}",
        author=_author(root.author),
        created_at=root.created_at,
        state=_thread_state(thread),
    )


def _thread_state(thread: ReviewThread) -> str:
    if thread.is_resolved:
        return "RESOLVED"
    return "OUTDATED" if thread.is_outdated else "OPEN"


def _issue_documents(repo: RepoInfo, issues: Iterable[Issue]) -> Iterator[Document]:
    for issue in issues:
        if _is_bot(issue.author):
            continue
        body, ai_assisted = _strip_claude_trailers(_strip_html_comments(issue.body))
        yield Document(
            url=issue.url,
            repo=repo.name_with_owner,
            kind=DocumentKind.ISSUE,
            visibility=_visibility(repo),
            title=issue.title,
            body=_with_discussion(body, issue.comments.nodes),
            author=_author(issue.author),
            created_at=issue.created_at,
            state=issue.state,
            ai_assisted=ai_assisted,
        )


def _pull_commit_oids(pulls: Iterable[PullRequest]) -> set[str]:
    """The oids of every PR's merge commit and commits, whatever the PR's state, as GitHub links commits to PRs."""
    oids: set[str] = set()
    for pull in pulls:
        oids.update(node.commit.oid for node in pull.commits.nodes if node.commit.oid)
        if pull.merge_commit:
            oids.add(pull.merge_commit.oid)
    return oids


def _commit_documents(repo: RepoInfo, commits: Iterable[Commit], pull_commit_oids: set[str]) -> Iterator[Document]:
    for commit in commits:
        if commit.parents.total_count > 1 or _came_from_a_pull(commit, pull_commit_oids) or _is_bot_commit(commit):
            continue
        headline, body = _rejoin_headline(commit.message_headline, commit.message_body)
        body, ai_assisted = _strip_claude_trailers(body)
        yield Document(
            url=commit.url,
            repo=repo.name_with_owner,
            kind=DocumentKind.COMMIT,
            visibility=_visibility(repo),
            title=headline,
            body=body,
            author=commit.author.user.login if commit.author.user else commit.author.name,
            created_at=commit.committed_date,
            ai_assisted=ai_assisted,
        )


def _came_from_a_pull(commit: Commit, pull_commit_oids: set[str]) -> bool:
    # Commits cached while history still asked GitHub keep its answer; their PRs were cached without commit oids.
    if commit.associated_pull_requests is not None:
        return commit.associated_pull_requests.total_count > 0
    return commit.oid in pull_commit_oids


def _is_bot_commit(commit: Commit) -> bool:
    login = commit.author.user.login if commit.author.user else commit.author.name
    return login.endswith("[bot]")


def _doc_documents(repo: RepoInfo, docs: Iterable[DocFile]) -> Iterator[Document]:
    branch = repo.default_branch_ref.name if repo.default_branch_ref else "HEAD"
    for doc in docs:
        yield Document(
            url=f"{repo.url}/blob/{branch}/{doc.path}",
            repo=repo.name_with_owner,
            kind=DocumentKind.DOC,
            visibility=_visibility(repo),
            title=doc.path,
            body=doc.text,
        )


def plan_documents(repo: str, plans: Iterable[PlanFile]) -> list[Document]:
    # Plans live in gitignored local folders, so they are private whatever the repo's visibility.
    return [
        Document(
            url=plan.url,
            repo=repo,
            kind=DocumentKind.PLAN,
            visibility=Visibility.PRIVATE,
            title=plan.path,
            body=plan.text,
        )
        for plan in plans
    ]


def _with_discussion(body: str, comments: Iterable[Comment]) -> str:
    replies = [
        f"**{_author(comment.author)}:** {text}"
        for comment in sorted(comments, key=lambda comment: comment.created_at)
        if not _is_bot(comment.author) and (text := _strip_html_comments(comment.body))
    ]
    return "\n\n".join([body, "## Discussion", *replies]) if replies else body


def _rejoin_headline(headline: str, body: str) -> tuple[str, str]:
    # GitHub cuts headlines at ~72 characters and starts the body with the rest of the line.
    if headline.endswith("…") and body.startswith("…"):
        rest, _, body = body[1:].partition("\n")
        return headline[:-1] + rest, body.strip()
    return headline, body.strip()


def _strip_html_comments(text: str) -> str:
    return HTML_COMMENT.sub("", text).strip()


def _strip_claude_trailers(text: str) -> tuple[str, bool]:
    stripped, count = CLAUDE_TRAILER.subn("", text)
    return stripped.strip(), count > 0


def _is_bot(actor: Actor | None) -> bool:
    return actor is not None and actor.is_bot


def _author(actor: Actor | None) -> str:
    return actor.login if actor else GHOST


def _visibility(repo: RepoInfo) -> Visibility:
    return Visibility.PRIVATE if repo.is_private else Visibility.PUBLIC
