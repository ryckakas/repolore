import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from repolore.config import Config, load_config, relative_pattern

ALL = "all"
PLANS_PATTERN = ".plans/**/*.md"
NUMBERS = re.compile(r"(\d+)(?:-(\d+))?")


class SelectionError(ValueError):
    pass


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


def write_config(path: Path, config: Config) -> str:
    text = config_text(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
