from contextlib import closing
from dataclasses import dataclass
from typing import Any

import httpx
import tomlkit.exceptions
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import Response

from repolore.config import FOLDER_OWNER, Config, load_config
from repolore.discovery import discover
from repolore.doctor import CONFIG_ERRORS, Check, github_access, local_documents_checks
from repolore.paths import AppPaths
from repolore.refresh import refresh_status
from repolore.setup import (
    PLANS_PATTERN,
    RepoChange,
    RepoTicks,
    SelectionError,
    add_local_folder,
    remove_local_folder,
    save_local_settings,
    save_ticks,
)
from repolore.sources.github import RATE_LIMIT_FLOOR, GitHubClient, GitHubError, resolve_token
from repolore.ui.rendering import render

# The repo form sends two fields a repo, and Starlette refuses a form with more than 1,000 by default.
MAX_FORM_FIELDS = 100_000
SAVE_ERRORS = (SelectionError, OSError, tomlkit.exceptions.TOMLKitError, *CONFIG_ERRORS)
GLOB_CHARACTERS = frozenset("*?[")


@dataclass(frozen=True)
class RepoRow:
    name: str
    scope: str
    ticked: bool


@dataclass(frozen=True)
class FolderDraft:
    """What the add-a-folder form holds, so a refused save doesn't lose what was typed."""

    name: str = ""
    path: str = ""
    patterns: str = ""


class SetupPages:
    def __init__(self, paths: AppPaths, db: str) -> None:
        self._paths = paths
        self._db = db

    def page(self, request: Request) -> Response:
        try:
            config = load_config(self._paths.config_file)
        except CONFIG_ERRORS as error:
            return self._render(request, "setup.html", config_error=str(error), config_file=self._paths.config_file)
        return self._render(
            request,
            "setup.html",
            config_file=self._paths.config_file,
            roots="\n".join(str(root) for root in config.local_roots),
            patterns="\n".join(config.local_patterns),
            example_pattern=PLANS_PATTERN,
            folders=config.local_folders,
            draft=FolderDraft(),
        )

    def repos(self, request: Request) -> Response:
        github, access = github_access(GitHubClient.connect)
        if access is None:
            return self._render(request, "_repos.html", github=github, rows=None)
        if access.remaining is not None and access.remaining < RATE_LIMIT_FLOOR:
            notice = (
                "GitHub has too few GraphQL points left this hour to list your repos, so try again after the reset."
            )
            return self._render(request, "_repos.html", github=github, rows=None, notice=notice)
        try:
            config = load_config(self._paths.config_file)
            return self._repo_form(request, github, config)
        except CONFIG_ERRORS as error:
            return self._render(request, "_repos.html", github=github, rows=None, notice=str(error))
        except (GitHubError, httpx.TransportError) as error:
            notice = f"Couldn't list your repos: {error}"
            return self._render(request, "_repos.html", github=github, rows=None, notice=notice)

    async def save_repos(self, request: Request) -> Response:
        form = await request.form(max_fields=MAX_FORM_FIELDS)
        ticks = RepoTicks(_texts(form, "shown"), _texts(form, "repo"), every="every" in form)
        return await run_in_threadpool(self._save_repos, request, ticks)

    async def save_local(self, request: Request) -> Response:
        form = await request.form()
        return await run_in_threadpool(
            self._save_local, request, _lines(form.get("roots")), _lines(form.get("patterns"))
        )

    async def add_folder(self, request: Request) -> Response:
        form = await request.form()
        draft = FolderDraft(_text(form, "name"), _text(form, "path"), _text(form, "patterns"))
        return await run_in_threadpool(self._add_folder, request, draft)

    async def remove_folder(self, request: Request) -> Response:
        form = await request.form()
        return await run_in_threadpool(self._remove_folder, request, _text(form, "name"))

    def _repo_form(self, request: Request, github: Check, config: Config) -> Response:
        with closing(GitHubClient.connect(resolve_token())) as client:
            discovery = discover(client, config)
        existed = self._paths.config_file.exists()
        rows = [
            RepoRow(source.name, source.scope.value, existed and config.selects(source.name))
            for source in sorted([*discovery.included, *discovery.excluded], key=lambda source: source.name.lower())
            if not source.name.startswith(f"{FOLDER_OWNER}/")
        ]
        every = existed and not config.include and not config.exclude
        replaced = _replaced(config, rows)
        return self._render(request, "_repos.html", github=github, rows=rows, every=every, replaced=replaced)

    def _save_repos(self, request: Request, ticks: RepoTicks) -> Response:
        try:
            change = save_ticks(self._paths.config_file, ticks)
        except SAVE_ERRORS as error:
            return self._render(request, "_notice.html", 400, notice=str(error))
        return self._saved(request, _outcome(change), saved=change.saved)

    def _save_local(self, request: Request, roots: list[str], patterns: list[str]) -> Response:
        try:
            save_local_settings(self._paths.config_file, roots, patterns)
        except SAVE_ERRORS as error:
            return self._render(request, "_notice.html", 400, notice=str(error))
        checks = list(local_documents_checks(self._paths.config_file))
        return self._saved(request, "Saved.", saved=True, checks=checks)

    def _add_folder(self, request: Request, draft: FolderDraft) -> Response:
        try:
            add_local_folder(self._paths.config_file, draft.name, draft.path, _lines(draft.patterns))
        except SAVE_ERRORS as error:
            return self._folders(request, 400, draft, notice=str(error))
        checks = list(local_documents_checks(self._paths.config_file))
        return self._changed_folders(request, f"Added {FOLDER_OWNER}/{draft.name}.", checks)

    def _remove_folder(self, request: Request, name: str) -> Response:
        try:
            remove_local_folder(self._paths.config_file, name)
        except SAVE_ERRORS as error:
            return self._folders(request, 400, notice=str(error))
        message = f"Removed {FOLDER_OWNER}/{name}. It leaves the index and its cached data at the next refresh."
        return self._changed_folders(request, message, [])

    def _changed_folders(self, request: Request, message: str, checks: list[Check]) -> Response:
        running = refresh_status(self._paths.data_dir).running
        return self._folders(request, message=message, saved=True, checks=checks, running=running)

    def _folders(
        self, request: Request, status_code: int = 200, draft: FolderDraft | None = None, **context: Any
    ) -> Response:
        try:
            folders = load_config(self._paths.config_file).local_folders
        except (OSError, *CONFIG_ERRORS):
            folders = []
        return self._render(
            request, "_folders.html", status_code, folders=folders, draft=draft or FolderDraft(), **context
        )

    def _saved(self, request: Request, message: str, *, saved: bool, checks: list[Check] | None = None) -> Response:
        running = refresh_status(self._paths.data_dir).running
        return self._render(request, "_saved.html", message=message, saved=saved, checks=checks, running=running)

    def _render(self, request: Request, template: str, status_code: int = 200, **context: Any) -> Response:
        return render(request, self._db, template, status_code, **context)


def _texts(form: FormData, name: str) -> list[str]:
    return [value for value in form.getlist(name) if isinstance(value, str)]


def _text(form: FormData, name: str) -> str:
    value = form.get(name)
    return value.strip() if isinstance(value, str) else ""


def _lines(value: object) -> list[str]:
    text = value if isinstance(value, str) else ""
    return list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


def _replaced(config: Config, rows: list[RepoRow]) -> list[str]:
    """The include and exclude entries that names ticked from these rows wouldn't keep."""
    listed = {row.name.lower() for row in rows}
    kept_only_by_the_file = [
        entry for entry in config.include if GLOB_CHARACTERS & set(entry) or entry.lower() not in listed
    ]
    return [*kept_only_by_the_file, *config.exclude]


def _outcome(change: RepoChange) -> str:
    if not change.saved:
        return "Nothing changed, so the file was left alone."
    parts = []
    if change.added:
        parts.append(f"{change.added} repo{'' if change.added == 1 else 's'} added")
    if change.leaving == 1:
        parts.append("1 repo leaves the index and its cached data at the next refresh")
    elif change.leaving:
        parts.append(f"{change.leaving} repos leave the index and their cached data at the next refresh")
    return f"{', '.join(parts)}." if parts else "Saved."
