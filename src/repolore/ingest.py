import logging
import shutil
from pathlib import Path

from repolore.discovery import Discovery, RepoSource, Scope
from repolore.documents import Document, documents_from, plan_documents
from repolore.progress import Phase, Progress, Report, ignore
from repolore.sources.github import GitHubClient, RawRepo, fetch_contributions, fetch_repo
from repolore.sources.plans import PLANS_FILE, load_plans, read_plans, save_plans

SCOPE_FILE = "scope.txt"
WHOLE_HISTORY = frozenset({Scope.OWNED, Scope.MEMBER})

log = logging.getLogger(__name__)


def ingest(
    discovery: Discovery, raw_dir: Path, client: GitHubClient, full: bool = False, report: Report = ignore
) -> list[Document]:
    """Bring the raw cache in line with the discovered repos, then read every document from it."""
    total = len(discovery.included)
    for done, source in enumerate(discovery.included):
        report(Progress(phase=Phase.INGEST, done=done, total=total))
        directory = raw_dir / source.name
        _sync_history(client, source, discovery.login, directory, full)
        _sync_plans(source, directory)
    report(Progress(phase=Phase.INGEST, done=total, total=total))
    _purge(raw_dir, {source.name.lower() for source in discovery.included})
    return cached_documents(raw_dir)


def cached_documents(raw_dir: Path, include_bot_reviews: bool = False) -> list[Document]:
    documents: list[Document] = []
    for directory in _repo_dirs(raw_dir):
        if RawRepo.is_cached(directory):
            documents += documents_from(RawRepo.load(directory), include_bot_reviews)
        documents += plan_documents(_repo_name(directory), load_plans(directory))
    return documents


def _sync_history(client: GitHubClient, source: RepoSource, login: str, directory: Path, full: bool) -> None:
    match source.scope:
        case Scope.OWNED | Scope.MEMBER:
            cached = None if full else _whole_history(directory)
            log.info("fetching %s%s", source.name, " (changes only)" if cached else "")
            fetch_repo(client, source.name, cached).save(directory)
        case Scope.CONTRIBUTED:
            log.info("fetching %s (your PRs and issues only)", source.name)
            fetch_contributions(client, source.name, login).save(directory)
        case _:
            RawRepo.remove(directory)
    _record_scope(directory, source.scope)


def _whole_history(directory: Path) -> RawRepo | None:
    # A cache holding only the viewer's contributions looks complete to the watermark, so it is never reused whole.
    scope_file = directory / SCOPE_FILE
    if not scope_file.exists() or scope_file.read_text(encoding="utf-8") not in WHOLE_HISTORY:
        return None
    return RawRepo.load(directory)


def _record_scope(directory: Path, scope: Scope) -> None:
    if scope is Scope.LOCAL:
        (directory / SCOPE_FILE).unlink(missing_ok=True)
    else:
        (directory / SCOPE_FILE).write_text(scope, encoding="utf-8")


def _sync_plans(source: RepoSource, directory: Path) -> None:
    if source.checkout:
        save_plans(directory, read_plans(source.checkout))
    else:
        (directory / PLANS_FILE).unlink(missing_ok=True)


def _purge(raw_dir: Path, kept: set[str]) -> None:
    for directory in _repo_dirs(raw_dir):
        if _repo_name(directory).lower() not in kept:
            shutil.rmtree(directory)
            log.warning("removed %s from the raw cache: it is no longer discovered or selected", _repo_name(directory))
    for owner in raw_dir.glob("*"):
        if owner.is_dir() and not any(owner.iterdir()):
            owner.rmdir()


def _repo_dirs(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.glob("*/*") if path.is_dir())


def _repo_name(directory: Path) -> str:
    return f"{directory.parent.name}/{directory.name}"
