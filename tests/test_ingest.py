import logging
import shutil
from pathlib import Path

import pytest

from repolore import ingest as ingest_module
from repolore.discovery import Discovery, RepoSource, Scope
from repolore.documents import DocumentKind
from repolore.ingest import SCOPE_FILE, cached_documents, ingest
from repolore.progress import Phase, Progress
from repolore.sources.github import GitHubClient, RawRepo
from repolore.sources.local_docs import LOCAL_FILE, LocalFile, LocalFiles, TooManyFilesError, save_local_files

FIXTURE = Path(__file__).parent / "fixtures" / "crashcause"
REPO = "ryckakas/crashcause"
CLIENT = GitHubClient.connect("unused")


class FakeFetch:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.repos: list[tuple[str, RawRepo | None]] = []
        self.contributions: list[tuple[str, str]] = []
        monkeypatch.setattr(ingest_module, "fetch_repo", self.fetch_repo)
        monkeypatch.setattr(ingest_module, "fetch_contributions", self.fetch_contributions)

    def fetch_repo(self, _: GitHubClient, repo: str, cached: RawRepo | None) -> RawRepo:
        self.repos.append((repo, cached))
        return RawRepo.load(FIXTURE)

    def fetch_contributions(self, _: GitHubClient, repo: str, login: str) -> RawRepo:
        self.contributions.append((repo, login))
        return RawRepo.load(FIXTURE)


def discovery(*sources: RepoSource) -> Discovery:
    return Discovery(login="octocat", included=list(sources), excluded=[])


def plans_checkout(tmp_path: Path, limit: int = 100) -> LocalFiles:
    checkout = tmp_path / "checkout"
    (checkout / ".plans" / "feature").mkdir(parents=True)
    (checkout / ".plans" / "feature" / "plan.md").write_text("# Plan\n\n## Steps\n\nShip it.", encoding="utf-8")
    return LocalFiles(checkout, (".plans/**/*.md",), limit)


def test_cached_documents_come_from_every_cached_repo(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path / REPO)

    documents = cached_documents(tmp_path)

    assert {document.kind for document in documents} == {
        DocumentKind.PR,
        DocumentKind.ISSUE,
        DocumentKind.COMMIT,
        DocumentKind.DOC,
    }


def test_empty_cache_has_no_documents(tmp_path: Path) -> None:
    assert cached_documents(tmp_path / "raw") == []


def test_owned_repo_is_fetched_whole_then_incrementally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeFetch(monkeypatch)
    found = discovery(RepoSource(REPO, Scope.OWNED))

    first = ingest(found, tmp_path, CLIENT)
    ingest(found, tmp_path, CLIENT)
    ingest(found, tmp_path, CLIENT, full=True)

    assert len(first) == 7
    assert [cached for _, cached in fake.repos] == [None, RawRepo.load(FIXTURE), None]
    assert RawRepo.load(tmp_path / REPO) == RawRepo.load(FIXTURE)


def test_repo_that_was_only_contributed_to_is_fetched_whole_once_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetch(monkeypatch)
    ingest(discovery(RepoSource(REPO, Scope.CONTRIBUTED)), tmp_path, CLIENT)

    ingest(discovery(RepoSource(REPO, Scope.MEMBER)), tmp_path, CLIENT)
    ingest(discovery(RepoSource(REPO, Scope.MEMBER)), tmp_path, CLIENT)

    assert [cached for _, cached in fake.repos] == [None, RawRepo.load(FIXTURE)]


def test_contributed_repo_fetches_only_the_viewers_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeFetch(monkeypatch)

    ingest(discovery(RepoSource("other/project", Scope.CONTRIBUTED)), tmp_path, CLIENT)

    assert fake.contributions == [("other/project", "octocat")]
    assert fake.repos == []


def test_local_files_are_cached_and_become_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFetch(monkeypatch)
    local = plans_checkout(tmp_path)

    documents = ingest(discovery(RepoSource(REPO, Scope.OWNED, local)), tmp_path / "raw", CLIENT)

    plans = [document for document in documents if document.kind == DocumentKind.LOCAL]
    assert [(plan.repo, plan.title) for plan in plans] == [(REPO, ".plans/feature/plan.md")]
    assert plans[0].url.startswith("file:///")


def test_dropped_checkout_removes_its_local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFetch(monkeypatch)
    raw_dir = tmp_path / "raw"
    ingest(discovery(RepoSource(REPO, Scope.OWNED, plans_checkout(tmp_path))), raw_dir, CLIENT)

    ingest(discovery(RepoSource(REPO, Scope.OWNED)), raw_dir, CLIENT)

    assert not (raw_dir / REPO / LOCAL_FILE).exists()


def test_repo_left_with_only_local_files_loses_its_github_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeFetch(monkeypatch)
    raw_dir = tmp_path / "raw"
    checkout = plans_checkout(tmp_path)
    ingest(discovery(RepoSource(REPO, Scope.OWNED, checkout)), raw_dir, CLIENT)

    documents = ingest(discovery(RepoSource(REPO, Scope.LOCAL, checkout)), raw_dir, CLIENT)

    assert not RawRepo.is_cached(raw_dir / REPO)
    assert not (raw_dir / REPO / SCOPE_FILE).exists()
    assert {document.kind for document in documents} == {DocumentKind.LOCAL}


def local_file(path: str, url: str = "file:///a") -> LocalFile:
    return LocalFile(path=path, url=url, text="# Local")


def test_a_local_file_github_also_has_is_left_to_the_github_copy(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path / REPO)
    github_path = RawRepo.load(FIXTURE).docs[0]["path"]
    save_local_files(
        tmp_path / REPO, [local_file(github_path, "file:///tracked"), local_file("notes/draft.md", "file:///draft")]
    )

    local = [document.title for document in cached_documents(tmp_path) if document.kind == DocumentKind.LOCAL]

    assert local == ["notes/draft.md"]


def test_a_repo_with_no_github_cache_keeps_every_local_file(tmp_path: Path) -> None:
    save_local_files(tmp_path / REPO, [local_file("README.md")])

    assert [document.title for document in cached_documents(tmp_path)] == ["README.md"]


def test_a_file_two_sources_reach_is_one_document_under_the_first(tmp_path: Path) -> None:
    save_local_files(tmp_path / "a" / "first", [local_file("x.md", "file:///same")])
    save_local_files(tmp_path / "b" / "second", [local_file("x.md", "file:///same")])

    assert [document.repo for document in cached_documents(tmp_path)] == ["a/first"]


def test_too_many_local_files_stops_ingest_and_purges_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFetch(monkeypatch)
    raw_dir = tmp_path / "raw"
    shutil.copytree(FIXTURE, raw_dir / "other" / "kept")
    local = plans_checkout(tmp_path, limit=0)

    with pytest.raises(TooManyFilesError):
        ingest(discovery(RepoSource(REPO, Scope.OWNED, local)), raw_dir, CLIENT)

    assert RawRepo.is_cached(raw_dir / "other" / "kept")


def test_repos_no_longer_discovered_are_purged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    FakeFetch(monkeypatch)
    shutil.copytree(FIXTURE, tmp_path / "former-employer" / "service")
    shutil.copytree(FIXTURE, tmp_path / "ryckakas" / "dropped")

    with caplog.at_level(logging.WARNING):
        documents = ingest(discovery(RepoSource(REPO, Scope.OWNED)), tmp_path, CLIENT)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["ryckakas"]
    assert [path.name for path in (tmp_path / "ryckakas").iterdir()] == ["crashcause"]
    assert "removed former-employer/service from the raw cache" in caplog.text
    assert "removed ryckakas/dropped from the raw cache" in caplog.text
    assert len(documents) == 7


def test_nothing_discovered_purges_everything(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path / REPO)

    assert ingest(discovery(), tmp_path, CLIENT) == []
    assert list(tmp_path.iterdir()) == []


def test_progress_counts_the_repos_fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFetch(monkeypatch)
    reports: list[Progress] = []

    ingest(
        discovery(RepoSource(REPO, Scope.OWNED), RepoSource("other/project", Scope.CONTRIBUTED)),
        tmp_path,
        CLIENT,
        report=reports.append,
    )

    assert reports == [Progress(phase=Phase.INGEST, done=done, total=2) for done in range(3)]


def test_progress_with_nothing_discovered_has_nothing_to_fetch(tmp_path: Path) -> None:
    reports: list[Progress] = []

    ingest(discovery(), tmp_path, CLIENT, report=reports.append)

    assert reports == [Progress(phase=Phase.INGEST, done=0, total=0)]


def test_a_removed_local_folder_is_purged_from_the_cache(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "a.md").write_text("# A", encoding="utf-8")
    source = RepoSource("local/notes", Scope.LOCAL, LocalFiles(folder, ("*.md",), 10))

    kept = ingest(discovery(source), raw_dir, CLIENT)
    removed = ingest(discovery(), raw_dir, CLIENT)

    assert [(document.repo, document.kind) for document in kept] == [("local/notes", DocumentKind.LOCAL)]
    assert removed == []
    assert not (raw_dir / "local").exists()


def test_a_file_in_a_checkout_and_a_folder_stays_under_its_repo(tmp_path: Path) -> None:
    save_local_files(tmp_path / "local" / "notes", [local_file("x.md", "file:///same")])
    save_local_files(tmp_path / "octo" / "repo", [local_file("x.md", "file:///same")])

    assert [document.repo for document in cached_documents(tmp_path)] == ["octo/repo"]
