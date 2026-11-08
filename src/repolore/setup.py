import json
import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
from pydantic import ValidationError
from tomlkit.toml_document import TOMLDocument

from repolore.config import FOLDER_NAME, Config, absolute_path, load_config, relative_pattern
from repolore.config_files import replace_file

ALL = "all"
PLANS_PATTERN = ".plans/**/*.md"
NUMBERS = re.compile(r"(\d+)(?:-(\d+))?")
FOLDERS_KEY = "local_folders"


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class RepoTicks:
    """What the setup page's repo form was showing and what was ticked when it was saved."""

    shown: list[str]
    ticked: list[str]
    every: bool


@dataclass(frozen=True)
class RepoChange:
    saved: bool
    added: int = 0
    leaving: int = 0


def parse_selection(answer: str, names: list[str]) -> list[str]:
    """The include globs for an answer that names repos by number, range of numbers or owner/name glob.

    "all" selects every repo, which needs no include glob.
    """
    tokens = [token for token in re.split(r"[\s,]+", answer.strip()) if token]
    if not tokens:
        raise SelectionError(f"name at least one repo, or {ALL}")
    if [token.lower() for token in tokens] == [ALL]:
        return []
    picked = [glob for token in tokens for glob in _picked(token, names)]
    return list(dict.fromkeys(picked))


def _picked(token: str, names: list[str]) -> list[str]:
    if match := NUMBERS.fullmatch(token):
        first, last = int(match[1]), int(match[2] or match[1])
        if not 1 <= first <= last <= len(names):
            raise SelectionError(f"{token} isn't a number or range between 1 and {len(names)}")
        return names[first - 1 : last]
    if not any(Config(include=[token]).selects(name) for name in names):
        raise SelectionError(f"{token} matches none of the repos")
    return [token]


def parse_roots(answer: str) -> list[Path]:
    return checked_roots([Path(part.strip()) for part in answer.split(",") if part.strip()])


def checked_roots(roots: list[Path]) -> list[Path]:
    """The roots, which must be folders, made absolute (a typed ~ kept), since a refresh runs in the data dir."""
    if missing := [str(root) for root in roots if not root.expanduser().is_dir()]:
        raise SelectionError(f"not a folder: {', '.join(missing)}")
    return [root if root.expanduser().is_absolute() else Path(os.path.abspath(root)) for root in roots]


def parse_patterns(answer: str) -> list[str]:
    patterns = [part.strip() for part in answer.split(",") if part.strip()]
    if not patterns:
        raise SelectionError("name at least one pattern")
    return checked_patterns(patterns)


def checked_patterns(patterns: list[str]) -> list[str]:
    try:
        return [relative_pattern(pattern) for pattern in patterns]
    except ValueError as error:
        raise SelectionError(str(error)) from error


def require_roots_with_patterns(roots: list[Path], patterns: list[str]) -> None:
    if bool(roots) != bool(patterns):
        raise SelectionError(
            "--local-root and --local-pattern go together, "
            f'such as --local-root ~/projects --local-pattern "{PLANS_PATTERN}"'
        )


def with_unasked_settings(config: Config, replaced_file: Path) -> Config:
    """The config plus what setup can't ask about from the file it replaces, unless that file can't be read."""
    try:
        old = load_config(replaced_file)
    except (tomllib.TOMLDecodeError, ValidationError, UnicodeDecodeError):
        return config
    return config.model_copy(
        update={
            "local_folders": old.local_folders,
            "checkout_depth": old.checkout_depth,
            "max_local_files": old.max_local_files,
        }
    )


def config_text(config: Config) -> str:
    """repolore.toml for the config, with only the keys it sets; JSON's strings and arrays are valid TOML."""
    keys = config.model_dump(mode="json", exclude_defaults=True)
    folders = keys.pop("local_folders", [])
    return _toml_lines(keys) + "".join(f"\n[[local_folders]]\n{_toml_lines(folder)}" for folder in folders)


def _toml_lines(keys: dict[str, Any]) -> str:
    return "".join(f"{key} = {json.dumps(value, ensure_ascii=False)}\n" for key, value in keys.items())


def save_ticks(path: Path, ticks: RepoTicks) -> RepoChange:
    """Write what the ticks select to include and exclude, unless that's what the file already selects."""
    existed = path.exists()
    config = load_config(path)
    before = {name for name in ticks.shown if existed and config.selects(name)}
    after = set(ticks.shown) if ticks.every else set(ticks.shown) & set(ticks.ticked)
    if not after and not ticks.every:
        raise SelectionError("Tick at least one repo, or choose every repo.")
    was_every = existed and not config.include and not config.exclude
    if existed and after == before and ticks.every == was_every:
        return RepoChange(saved=False)
    update_config(path, {"include": [] if ticks.every else sorted(after), "exclude": []})
    return RepoChange(saved=True, added=len(after - before), leaving=len(before - after))


def save_local_settings(path: Path, roots: list[str], patterns: list[str]) -> None:
    """Write the local roots and patterns typed on the setup page, which has no working directory to resolve against."""
    load_config(path)
    if bool(roots) != bool(patterns):
        raise SelectionError(f"Give folders and patterns together, such as {PLANS_PATTERN}, or neither.")
    try:
        checked = checked_roots([absolute_path(Path(root)) for root in roots])
    except ValueError as error:
        raise SelectionError(str(error)) from error
    valid_patterns = checked_patterns(patterns)
    _require_repos_saved(path)
    update_config(path, {"local_roots": [str(root) for root in checked], "local_patterns": valid_patterns})


def add_local_folder(path: Path, name: str, location: str, patterns: list[str]) -> None:
    """Add a [[local_folders]] entry typed on the setup page, keeping the path as typed so a ~ stays one."""
    config = load_config(path)
    _check_new_folder(config, name, location)
    entry: dict[str, Any] = {"name": name, "path": location}
    if patterns:
        entry["patterns"] = checked_patterns(patterns)
    _require_repos_saved(path)
    _edit_config(path, lambda document: _append_folder(document, entry))


def remove_local_folder(path: Path, name: str) -> None:
    if name not in [folder.name for folder in load_config(path).local_folders]:
        raise SelectionError(f"There's no local folder named {name}.")
    _edit_config(path, lambda document: _delete_folder(document, name))


def _check_new_folder(config: Config, name: str, location: str) -> None:
    if not name or not location:
        raise SelectionError("Give the folder a name and a path.")
    if not re.fullmatch(FOLDER_NAME, name):
        raise SelectionError(f"{name} can't name a folder: use letters, digits, ., _ and -, and don't end with a dot.")
    if name.lower() in {folder.name.lower() for folder in config.local_folders}:
        raise SelectionError(f"There's already a local folder named {name}.")
    try:
        absolute = absolute_path(Path(location))
    except ValueError as error:
        raise SelectionError(str(error)) from error
    if not absolute.expanduser().exists():
        raise SelectionError(f"{location} doesn't exist.")


def _require_repos_saved(path: Path) -> None:
    # A file with no include selects every repo, so only saving the repos may create it.
    if not path.exists():
        raise SelectionError("Save your repos first, then the local documents.")


def _append_folder(document: TOMLDocument, entry: dict[str, Any]) -> None:
    if FOLDERS_KEY not in document:
        document[FOLDERS_KEY] = tomlkit.aot()
    folders: Any = document[FOLDERS_KEY]
    folders.append(entry)


def _delete_folder(document: TOMLDocument, name: str) -> None:
    folders: Any = document[FOLDERS_KEY]
    del folders[[folder["name"] for folder in folders].index(name)]
    if not folders:
        del document[FOLDERS_KEY]


def update_config(path: Path, keys: dict[str, list[str]]) -> None:
    """Change only these keys, keeping the file's comments and its other keys; an empty list removes a key."""

    def change(document: TOMLDocument) -> None:
        for key, value in keys.items():
            # Assigning an equal value would still rewrite the array and lose the comments inside it.
            if value == document.get(key, []):
                continue
            if value:
                document[key] = value
            else:
                document.pop(key, None)

    _edit_config(path, change)


def _edit_config(path: Path, change: Callable[[TOMLDocument], None]) -> None:
    target = path.resolve()
    existed = target.exists()
    if existed and not os.access(target, os.W_OK):
        raise PermissionError(f"{target} is read-only")
    text = target.read_text(encoding="utf-8") if existed else ""
    # tomlkit would join a new table onto a last line with no newline, which doesn't parse.
    document = tomlkit.parse(text if text.endswith("\n") or not text else f"{text}\n")
    change(document)
    text = tomlkit.dumps(document)
    Config.model_validate(tomllib.loads(text))
    replace_file(target, text.encode("utf-8"), existed)


def write_config(path: Path, config: Config) -> str:
    text = config_text(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
