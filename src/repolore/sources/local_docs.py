import json
import logging
import os
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from repolore.config import Config
from repolore.sources.github import SKIPPED_DIRS

LOCAL_FILE = "local.json"
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
NEVER_MATCHED_DIRS = SKIPPED_DIRS | {".git"}
DOT_SEGMENTS = frozenset({".", ".."})
GITHUB_REMOTE = re.compile(r"github\.com[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")

log = logging.getLogger(__name__)


class TooManyFilesError(Exception):
    pass


class LocalFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    url: str
    text: str


@dataclass(frozen=True)
class LocalFiles:
    """The markdown files some glob patterns match under a folder."""

    root: Path
    patterns: tuple[str, ...]
    limit: int

    def matches(self) -> list[Path]:
        """The matching files, sorted; more than the limit means the search stopped one past it."""
        root = self.root.resolve()
        found: set[Path] = set()
        for path in chain.from_iterable(self.root.glob(pattern) for pattern in self.patterns):
            if self._is_document(path, root):
                found.add(path)
                if len(found) > self.limit:
                    break
        return sorted(found)

    def _is_document(self, path: Path, resolved_root: Path) -> bool:
        # ** follows Windows junctions on Python 3.12, so only the resolved path can tell a file left the root.
        return (
            path.suffix.lower() in MARKDOWN_SUFFIXES
            and not NEVER_MATCHED_DIRS.intersection(path.relative_to(self.root).parts)
            and os.path.isfile(path)
            and path.resolve().is_relative_to(resolved_root)
        )


def find_checkouts(config: Config) -> dict[str, LocalFiles]:
    """The files local_patterns match in each GitHub checkout under the local roots, keyed by its repo."""
    if not config.local_patterns:
        return {}
    found: dict[str, LocalFiles] = {}
    for root in (root.expanduser() for root in config.local_roots):
        if not os.path.isdir(root):
            log.warning("local root %s doesn't exist", root)
            continue
        for checkout in _checkouts_under(root, config.checkout_depth):
            files = LocalFiles(checkout, tuple(config.local_patterns), config.max_local_files)
            if files.matches() and (repo := github_repo(checkout)):
                kept = found.setdefault(repo, files)
                if kept != files:
                    log.warning("%s is checked out in %s and %s; using the first", repo, kept.root, checkout)
    return found


def _checkouts_under(folder: Path, depth: int) -> Iterator[Path]:
    # A .git file marks a worktree or submodule: only its main checkout counts, or a stale worktree could win over it.
    # os.path.isdir because Path.is_dir raises on a folder the user can't read.
    if os.path.isdir(folder / ".git"):
        yield folder
    elif depth > 0:
        for child in _subfolders(folder):
            yield from _checkouts_under(child, depth - 1)


def _subfolders(folder: Path) -> list[Path]:
    try:
        children = sorted(folder.iterdir())
    except OSError as error:
        log.info("can't look inside %s: %s", folder, error)
        return []
    return [child for child in children if _may_hold_checkouts(child)]


def _may_hold_checkouts(path: Path) -> bool:
    return (
        os.path.isdir(path)
        and not path.is_symlink()
        and not path.is_junction()
        and not path.name.startswith(".")
        and path.name not in SKIPPED_DIRS
    )


def github_repo(checkout: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "remote", "get-url", "origin"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        log.info("%s has no origin remote, so its local documents are skipped", checkout)
        return None
    match = GITHUB_REMOTE.search(result.stdout.strip())
    if match and not {match[1], match[2]} & DOT_SEGMENTS:
        return f"{match[1]}/{match[2]}"
    log.info("%s isn't a GitHub checkout, so its local documents are skipped", checkout)
    return None


def read_local_files(files: LocalFiles) -> list[LocalFile]:
    paths = files.matches()
    if len(paths) > files.limit:
        raise TooManyFilesError(
            f"{files.root} matches more than {files.limit} local files; narrow the patterns or raise max_local_files"
        )
    return [
        LocalFile(
            path=path.relative_to(files.root).as_posix(),
            url=path.resolve().as_uri(),
            text=path.read_text(encoding="utf-8", errors="replace"),
        )
        for path in paths
    ]


def save_local_files(directory: Path, files: list[LocalFile]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    content = json.dumps([file.model_dump() for file in files], ensure_ascii=False, indent=1)
    (directory / LOCAL_FILE).write_text(content, encoding="utf-8")


def load_local_files(directory: Path) -> list[LocalFile]:
    path = directory / LOCAL_FILE
    if not path.exists():
        return []
    return [LocalFile.model_validate(file) for file in json.loads(path.read_text(encoding="utf-8"))]
