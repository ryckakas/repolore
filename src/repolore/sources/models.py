from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class GitHubModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, frozen=True)


class Nodes[T](GitHubModel):
    nodes: list[T]


class Count(GitHubModel):
    total_count: int


class Actor(GitHubModel):
    login: str
    typename: str = Field(alias="__typename")

    @property
    def is_bot(self) -> bool:
        return self.typename == "Bot" or self.login.endswith("[bot]")


class Comment(GitHubModel):
    author: Actor | None
    body: str
    created_at: datetime


class ThreadComment(Comment):
    url: str
    diff_hunk: str


class ReviewThread(GitHubModel):
    is_resolved: bool
    is_outdated: bool
    path: str
    line: int | None
    original_line: int | None
    comments: Nodes[ThreadComment]


class CommitMessage(GitHubModel):
    message_headline: str
    message_body: str


class PullCommitMessage(CommitMessage):
    oid: str | None = None


class PullRequestCommit(GitHubModel):
    commit: PullCommitMessage


class CommitId(GitHubModel):
    oid: str


class ChangedFile(GitHubModel):
    path: str


class PullRequest(GitHubModel):
    number: int
    title: str
    body: str
    url: str
    state: str
    created_at: datetime
    merge_commit: CommitId | None = None
    author: Actor | None
    files: Nodes[ChangedFile]
    commits: Nodes[PullRequestCommit]
    comments: Nodes[Comment]
    reviews: Nodes[Comment]
    review_threads: Nodes[ReviewThread]


class Issue(GitHubModel):
    number: int
    title: str
    body: str
    url: str
    state: str
    created_at: datetime
    author: Actor | None
    comments: Nodes[Comment]


class GitUser(GitHubModel):
    login: str


class GitActor(GitHubModel):
    name: str
    user: GitUser | None


class Commit(CommitMessage):
    oid: str
    url: str
    committed_date: datetime
    parents: Count
    author: GitActor
    associated_pull_requests: Count | None = None


class Branch(GitHubModel):
    name: str


class RepoInfo(GitHubModel):
    name_with_owner: str
    url: str
    is_private: bool
    default_branch_ref: Branch | None


class DocFile(GitHubModel):
    path: str
    text: str
