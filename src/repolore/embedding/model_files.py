import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from repolore.embedding import ModelUnavailableError
from repolore.locks import hold
from repolore.progress import Phase, Progress, Report, ignore

MB = 1 << 20
LOG_EVERY_BYTES = 100 * MB
HASH_READ_BYTES = 1 << 24
TIMEOUT = httpx.Timeout(60, connect=30)
WAIT_FOR_THE_OTHER_DOWNLOAD = -1
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelRelease:
    """Files of an immutable GitHub release, so a new export is a new release and never replaces a file in use."""

    name: str
    url: str
    graph: ModelFile
    weights: ModelFile
    tokenizer: ModelFile

    @property
    def files(self) -> tuple[ModelFile, ...]:
        return (self.graph, self.weights, self.tokenizer)

    def folder(self, models_dir: Path) -> Path:
        return models_dir / self.name


def missing(folder: Path, files: Sequence[ModelFile]) -> list[ModelFile]:
    """The files not there yet. A file only gets its final name once verified, so its size is enough to check."""
    return [file for file in files if not _is_complete(folder / file.name, file)]


def ensure(
    release: ModelRelease,
    files: Sequence[ModelFile],
    models_dir: Path,
    report: Report = ignore,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Download the files that are missing, one process at a time. A failed download resumes on the next run."""
    folder = release.folder(models_dir)
    if not missing(folder, files):
        return
    folder.mkdir(parents=True, exist_ok=True)
    with hold(folder.with_name(f"{folder.name}.lock"), timeout=WAIT_FOR_THE_OTHER_DOWNLOAD):
        if needed := missing(folder, files):
            _download(release.url, folder, needed, report, transport)


def _is_complete(path: Path, file: ModelFile) -> bool:
    return path.is_file() and path.stat().st_size == file.size


def _download(
    url: str, folder: Path, files: list[ModelFile], report: Report, transport: httpx.BaseTransport | None
) -> None:
    total = sum(file.size for file in files)
    done = 0

    def advance(count: int) -> None:
        nonlocal done
        if (done + count) // LOG_EVERY_BYTES > done // LOG_EVERY_BYTES:
            log.info("downloaded %d of %d MB", (done + count) // MB, total // MB)
        done += count
        report(Progress(phase=Phase.DOWNLOAD, done=done // MB, total=total // MB))

    # A client of its own: the release is public, and the GitHub token must not travel to the host it redirects to.
    with httpx.Client(transport=transport, follow_redirects=True, timeout=TIMEOUT) as client:
        for file in files:
            log.info("downloading %s, %d MB", file.name, file.size // MB)
            try:
                _fetch(client, f"{url}/{file.name}", folder / file.name, file, advance)
            except httpx.HTTPError as error:
                raise ModelUnavailableError(_failure(file, f"{url}/{file.name}", folder, error)) from error
            except OSError as error:
                raise ModelUnavailableError(
                    f"{file.name} couldn't be written to {folder}: {_reason(error)}."
                ) from error


def _fetch(client: httpx.Client, url: str, target: Path, file: ModelFile, advance: Callable[[int], None]) -> None:
    partial = target.with_name(f"{target.name}.partial")
    digest, offset = _resumed(partial, file)
    if offset < file.size:
        with client.stream("GET", url, headers={"Range": f"bytes={offset}-"} if offset else None) as response:
            response.raise_for_status()
            if response.status_code != httpx.codes.PARTIAL_CONTENT:
                digest, offset = hashlib.sha256(), 0
            advance(offset)
            with partial.open("ab" if offset else "wb") as output:
                for chunk in response.iter_bytes(MB):
                    output.write(chunk)
                    digest.update(chunk)
                    advance(len(chunk))
    if digest.hexdigest() != file.sha256:
        partial.unlink()
        raise ModelUnavailableError(
            f"{file.name} from {url} didn't match its checksum, so it was removed. Run `repolore refresh` to try again."
        )
    partial.replace(target)


def _resumed(partial: Path, file: ModelFile) -> "tuple[hashlib._Hash, int]":
    digest = hashlib.sha256()
    if not partial.is_file():
        return digest, 0
    offset = partial.stat().st_size
    if offset > file.size:
        partial.unlink()
        return digest, 0
    with partial.open("rb") as existing:
        while block := existing.read(HASH_READ_BYTES):
            digest.update(block)
    return digest, offset


def _failure(file: ModelFile, url: str, folder: Path, error: Exception) -> str:
    return (
        f"{file.name} couldn't be downloaded from {url}: {_reason(error)}. "
        f"It needs access to github.com; behind a proxy that blocks it, put the file in {folder} by hand."
    )


def _reason(error: Exception) -> str:
    return next(iter(str(error).strip().splitlines()), type(error).__name__).rstrip(".")
