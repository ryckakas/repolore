import os
import platform
import sqlite3
import subprocess
import tomllib
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import ValidationError

from repolore import __version__
from repolore.clients import CLIENTS, Entry, McpClient, RegistrationError, is_connected, repolore_entries
from repolore.config import Config, LocalFolder, load_config
from repolore.embedding.engines import build_engine
from repolore.embedding.local import VOYAGE_NANO, is_downloaded, spec_for
from repolore.indexing import WHOLE
from repolore.paths import AppPaths
from repolore.refresh import LOG_FILE, refresh_status
from repolore.server_state import ServerFiles, connected_relays, is_running, read_discovery
from repolore.setup import PLANS_PATTERN
from repolore.sources.github import (
    TOKEN_ENV,
    GitHubClient,
    GitHubError,
    RateLimited,
    TokenInfo,
    resolve_token,
    token_source,
)
from repolore.sources.local_docs import LocalFiles, find_checkouts, folder_files
from repolore.store import StoreError, connect_read_only, index_settings, repo_counts, vector_version

REFRESH_FIX = "run `repolore refresh`"
CONNECTION_FIX = "`repolore server stop`, then reconnect repolore in the client; `repolore server status` names its log"
CLIENT_ERRORS = (OSError, subprocess.SubprocessError, RegistrationError)
CONFIG_ERRORS = (tomllib.TOMLDecodeError, ValidationError, UnicodeDecodeError)
LIMIT_FIX = "narrow the patterns or raise max_local_files in repolore.toml"
PATTERNS_FIX = "check them; `repolore repos` shows what each checkout matches"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str | None = None
    warning: bool = False

    def __str__(self) -> str:
        line = f"{self._mark} {self.name}: {self.detail}"
        return f"{line}\n  fix: {self.fix}" if self.fix else line

    @property
    def _mark(self) -> str:
        if self.warning:
            return "!"
        return "✓" if self.ok else "✗"


def diagnose(
    paths: AppPaths,
    db: str,
    connect: Callable[[str], GitHubClient] = GitHubClient.connect,
    clients: Sequence[McpClient] = CLIENTS,
) -> Iterator[Check]:
    """Check each part of the setup in turn; only the GitHub check uses the network, with one request."""
    index_file = paths.index_file(db)
    yield vectors_check()
    yield github_check(connect)
    yield config_check(paths.config_file)
    yield from local_documents_checks(paths.config_file)
    yield index_check(index_file)
    yield model_check(index_file, paths.models_dir)
    yield engine_check()
    yield refresh_check(paths.data_dir)
    yield server_check(ServerFiles(paths.data_dir, db))
    for client in clients:
        if (check := registration_check(client)) is not None:
            yield check


def vectors_check() -> Check:
    try:
        version = vector_version()
    except (StoreError, sqlite3.Error) as error:
        return Check("sqlite-vec", False, str(error))
    return Check("sqlite-vec", True, f"{version} loads in Python {platform.python_version()}")


def github_check(connect: Callable[[str], GitHubClient]) -> Check:
    return github_access(connect)[0]


def github_access(connect: Callable[[str], GitHubClient]) -> tuple[Check, TokenInfo | None]:
    """The GitHub check, with what the token is when it works, even if it lacks scopes."""
    source = token_source()
    try:
        token = resolve_token()
    except GitHubError as error:
        return Check("GitHub", False, str(error)), None
    login_fix = f"set a new {TOKEN_ENV}" if source == TOKEN_ENV else "log in again with `gh auth login`"
    try:
        with closing(connect(token)) as client:
            info = client.token_info()
    except RateLimited as error:
        return Check("GitHub", False, str(error), "wait for the reset; a running refresh may be using the points"), None
    except GitHubError as error:
        return Check("GitHub", False, f"the token from {source} didn't work: {error}", login_fix), None
    except httpx.TransportError as error:
        return Check("GitHub", False, f"can't reach GitHub: {error}", "check the network, or HTTPS_PROXY"), None
    detail = f"{info.login}, token from {source}, {_scopes(info)}{_points(info)}"
    if missing := info.missing_scopes:
        return Check("GitHub", False, f"{detail}; missing {', '.join(missing)}", _scope_fix(source, missing)), info
    return Check("GitHub", True, detail), info


def _scopes(info: TokenInfo) -> str:
    if info.scopes is None:
        return "a fine-grained token, so its access isn't checked"
    return f"scopes {', '.join(sorted(info.scopes))}" if info.scopes else "no scopes"


def _points(info: TokenInfo) -> str:
    if info.remaining is None or info.limit is None:
        return ""
    return f", {info.remaining:,} of {info.limit:,} GraphQL points left this hour"


def _scope_fix(source: str, missing: list[str]) -> str:
    if source == TOKEN_ENV:
        return f"set {TOKEN_ENV} to a classic token with the {' and '.join(missing)} scopes"
    return f"`gh auth refresh --scopes {','.join(missing)}`"


def config_check(config_file: Path) -> Check:
    name = "repolore.toml"
    if not config_file.exists():
        return Check(name, True, "none, so every discovered repo is in")
    try:
        config = load_config(config_file)
    except CONFIG_ERRORS as error:
        reason = " ".join(line.strip() for line in str(error).splitlines())
        return Check(name, False, f"invalid: {reason}", "fix it; the README's quickstart shows the format")
    counts = [_count(len(config.include), "include glob"), _count(len(config.exclude), "exclude glob")]
    detail = ", ".join([*counts, _count(len(config.local_roots), "local root")])
    if missing := [str(root) for root in config.local_roots if not root.expanduser().is_dir()]:
        return Check(name, False, f"{detail}; not found: {', '.join(missing)}", "fix local_roots")
    return Check(name, True, f"{detail}; `repolore repos` shows what they select")


def local_documents_checks(config_file: Path) -> Iterator[Check]:
    """What the local settings match, with no GitHub access; config_check reports a config that can't be read."""
    try:
        config = load_config(config_file)
    except CONFIG_ERRORS:
        return
    yield from checkout_checks(config)
    for folder in config.local_folders:
        yield folder_check(folder, config.max_local_files)


def checkout_checks(config: Config) -> Iterator[Check]:
    if config.local_roots or config.local_patterns:
        yield _checkouts_check(config)


def _checkouts_check(config: Config) -> Check:
    name = "local checkouts"
    if not config.local_patterns:
        fix = f'add local_patterns = ["{PLANS_PATTERN}"] to repolore.toml'
        return Check(name, True, "local_roots is set but no local_patterns, so nothing in them is indexed", fix, True)
    if not config.local_roots:
        return Check(name, True, "local_patterns is set but no local_roots to search", "add local_roots", warning=True)
    # A missing root is config_check's to report, and looking in it would only log that again.
    existing = [root for root in config.local_roots if os.path.isdir(root.expanduser())]
    found = find_checkouts(config.model_copy(update={"local_roots": existing}))
    selected = {repo: files for repo, files in found.items() if config.selects(repo)}
    checkouts = list(selected.values())
    counts = [len(files.matches()) for files in checkouts]
    if over := [files.root for files, count in zip(checkouts, counts, strict=True) if count > files.limit]:
        return Check(name, False, f"more than {config.max_local_files} files in {over[0]}", LIMIT_FIX)
    repos = f": {', '.join(sorted(selected, key=str.lower))}" if selected else ""
    detail = f"{_count(sum(counts), 'file')} in {_count(len(checkouts), 'checkout')}{repos}"
    if unmatched := _unmatched_patterns(config.local_patterns, checkouts):
        return Check(
            name,
            True,
            f"{detail}; no selected GitHub checkout matches {', '.join(unmatched)}",
            PATTERNS_FIX,
            warning=True,
        )
    return Check(name, True, detail)


def _unmatched_patterns(patterns: list[str], checkouts: list[LocalFiles]) -> list[str]:
    return [
        pattern
        for pattern in patterns
        if not any(LocalFiles(files.root, (pattern,), limit=1).matches() for files in checkouts)
    ]


def folder_check(folder: LocalFolder, limit: int) -> Check:
    files = folder_files(folder, limit)
    if files is None:
        return Check(folder.repo, False, f"not found: {folder.path}", "fix its path in repolore.toml")
    try:
        count = len(files.matches())
    except OSError as error:
        return Check(folder.repo, False, f"can't search {files.root}: {error}", "check its path and patterns")
    if count > limit:
        return Check(folder.repo, False, f"more than {limit} files in {files.root}", LIMIT_FIX)
    if count == 0:
        return Check(folder.repo, True, f"no markdown matches in {files.root}", "check its path and patterns", True)
    return Check(folder.repo, True, f"{_count(count, 'file')} in {files.root}")


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def index_check(index_file: Path) -> Check:
    name = f"index {index_file.stem}"
    if not index_file.exists():
        return Check(name, False, f"none at {index_file}", REFRESH_FIX)
    try:
        with closing(connect_read_only(index_file)) as db:
            meta = index_settings(db)
            counts = repo_counts(db)
    except (StoreError, sqlite3.Error) as error:
        return Check(name, False, f"{index_file}: {error}", REFRESH_FIX)
    chunks = sum(sum(kinds.values()) for kinds in counts.values())
    limit = meta.get("max_tokens")
    size = "whole sections" if limit in (None, WHOLE) else f"at most {limit} tokens a chunk"
    return Check(
        name,
        True,
        f"{chunks:,} chunks from {_count(len(counts), 'repo')}, built {meta.get('built_at', 'at an unknown time')} "
        f"with {meta.get('model_id')}, {size}, "
        f"bot reviews {meta.get('bot_reviews', 'excluded')}",
    )


def model_check(index_file: Path, models_dir: Path) -> Check:
    model_id = _index_model(index_file)
    spec = VOYAGE_NANO if model_id is None else spec_for(model_id)
    if spec is None:
        return Check(
            "model", False, f"the index was built with {model_id}, which repolore can't load", "`repolore build`"
        )
    release = spec.release
    folder = release.folder(models_dir)
    if is_downloaded(spec, models_dir):
        return Check("model", True, f"{spec.model_id} is in {folder}")
    size = sum(file.size for file in release.files) / 1e9
    return Check(
        "model",
        False,
        f"{spec.model_id} isn't in {folder} yet",
        f"`repolore refresh` downloads it, {size:.1f} GB; behind a proxy that blocks github.com, "
        f"put the files from {release.url} there by hand",
    )


def engine_check() -> Check:
    """Which engine a build would embed with; the CPU still works, so it never fails."""
    return Check("build engine", True, build_engine().name)


def _index_model(index_file: Path) -> str | None:
    if not index_file.exists():
        return None
    try:
        with closing(connect_read_only(index_file)) as db:
            return index_settings(db).get("model_id")
    except (StoreError, sqlite3.Error):
        return None


def refresh_check(data_dir: Path) -> Check:
    name = "last refresh"
    status = refresh_status(data_dir)
    if status.running:
        since = f" since {status.started_at}" if status.started_at else ""
        progress = f", at {status.progress}" if status.progress else ""
        return Check(name, True, f"running{since}{progress}")
    if status.started_at is None:
        return Check(name, True, "none yet")
    if status.ok:
        return Check(name, True, f"succeeded at {status.finished_at}")
    if status.ok is False:
        log = data_dir / LOG_FILE
        return Check(
            name,
            False,
            f"failed at {status.finished_at}: {status.error}",
            f"{REFRESH_FIX}; a background run logs to {log}",
        )
    return Check(name, False, f"started at {status.started_at} and stopped before it finished", REFRESH_FIX)


def server_check(files: ServerFiles) -> Check:
    name = "shared server"
    if not is_running(files):
        return Check(name, True, "not running; the first search starts it")
    found = read_discovery(files)
    if found is None:
        return Check(name, True, "starting")
    replaced = "" if found.version == __version__ else f", which the next session replaces with {__version__}"
    sessions = _count(connected_relays(files), "session")
    return Check(name, True, f"pid {found.pid} at {found.url}, repolore {found.version}{replaced}; {sessions}")


def registration_check(client: McpClient, *, connection: bool = True) -> Check | None:
    """What a client that's here has registered; connection=False skips asking it, which can start the shared server."""
    if not client.present():
        return None
    try:
        found = repolore_entries(client)
    except CLIENT_ERRORS as error:
        return Check(client.name, False, f"can't list its MCP servers: {error}")
    if not found:
        return Check(client.name, True, "repolore isn't registered", client.register, warning=True)
    if len(found) > 1:
        return _duplicates_check(client, found)
    return _entry_check(client, found[0], connection)


def _duplicates_check(client: McpClient, found: list[Entry]) -> Check:
    listed = ", ".join(f"{entry.name} ({entry.scope})" for entry in found)
    removals = "; ".join(entry.remove for entry in found)
    return Check(client.name, False, f"repolore is registered {len(found)} times: {listed}", f"keep one: {removals}")


def _entry_check(client: McpClient, entry: Entry, connection: bool) -> Check:
    registered = f"repolore is registered{entry.alias}"
    if entry.command_is_gone:
        detail = f"{registered}, but {entry.command[0]} no longer exists"
        return Check(client.name, False, detail, f"{entry.remove}, then run `repolore setup`")
    if client.health is None or not connection:
        return Check(client.name, True, f"{registered}, connection not checked")
    try:
        status = client.health(entry.name)
    except CLIENT_ERRORS as error:
        return Check(client.name, False, f"{registered}, but checking it failed: {error}")
    detail = registered + (f": {status}" if status else "")
    return (
        Check(client.name, True, detail) if is_connected(status) else Check(client.name, False, detail, CONNECTION_FIX)
    )
