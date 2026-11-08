import codecs
import copy
import json
import os
import re
import shutil
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

BACKUP_SUFFIX = ".repolore-backup"
PARTIAL_SUFFIX = ".repolore-partial"
_STRING = r'"(?:\\.|[^"\\])*"'
_COMMENT = r"//[^\r\n]*|/\*.*?\*/"
_COMMENTS = re.compile(f"({_STRING})|{_COMMENT}", re.DOTALL)
_TRAILING_COMMAS = re.compile(rf"({_STRING})|,(?=\s*[}}\]])", re.DOTALL)
_TOKENS = re.compile(rf"{_STRING}|{_COMMENT}|[{{}}\[\]:]", re.DOTALL)


class ConfigFileError(Exception):
    pass


def parse_jsonc(text: str) -> Any:
    """JSON that may have comments and trailing commas, as several clients' settings files do."""
    return json.loads(_TRAILING_COMMAS.sub(_kept_string, _COMMENTS.sub(_kept_string, text)))


def _kept_string(match: re.Match[str]) -> str:
    return match[1] or ""


def add_entry(path: Path, keys: Sequence[str], name: str, entry: Mapping[str, Any]) -> None:
    """Add entry under name in the object at keys, leaving every other byte of the file as it was.

    The first change to a file keeps a copy of it beside it. Nothing is written unless the result checks out.
    """
    target = path.resolve()
    existed = target.exists()
    if existed and not os.access(target, os.W_OK):
        raise ConfigFileError(f"can't add {name} to {path}: it's read-only")
    raw = target.read_bytes() if existed else b""
    mark = codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b""
    try:
        updated = with_entry(raw[len(mark) :].decode("utf-8"), keys, name, entry)
    except (ConfigFileError, UnicodeDecodeError) as error:
        raise ConfigFileError(f"can't add {name} to {path}: {error}") from error
    if existed:
        _keep_first_copy(target)
    _replace(target, mark + updated.encode("utf-8"), existed)


def with_entry(text: str, keys: Sequence[str], name: str, entry: Mapping[str, Any]) -> str:
    """The text with entry added as text rather than re-serialized, which would lose comments and formatting."""
    expected = _with_entry_added(_settings(text), keys, name, entry)
    if not text.strip():
        return json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    updated = _inserted(text, keys, name, entry)
    if not _parses_to(updated, expected):
        raise ConfigFileError("adding the entry wouldn't give the expected settings")
    return updated


def _settings(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        settings = parse_jsonc(text)
    except ValueError as error:
        raise ConfigFileError(f"it isn't valid JSON: {error}") from error
    if not isinstance(settings, dict):
        raise ConfigFileError("it isn't a JSON object")
    return settings


def _with_entry_added(
    settings: dict[str, Any], keys: Sequence[str], name: str, entry: Mapping[str, Any]
) -> dict[str, Any]:
    expected = copy.deepcopy(settings)
    node = expected
    for key in keys:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigFileError(f"its {key} isn't an object")
    if name in node:
        raise ConfigFileError(f"it already has {name}")
    node[name] = dict(entry)
    return expected


def _parses_to(text: str, expected: dict[str, Any]) -> bool:
    try:
        return parse_jsonc(text) == expected
    except ValueError:
        return False


def _inserted(text: str, keys: Sequence[str], name: str, entry: Mapping[str, Any]) -> str:
    position = next(token.end() for token in _TOKENS.finditer(text) if token[0] == "{")
    missing = list(keys)
    while missing and (found := _object_after_key(text, position, missing[0])) is not None:
        position = found
        missing.pop(0)
    key, value = name, dict(entry)
    for outer_key in reversed(missing):
        key, value = outer_key, {key: value}
    return _with_member(text, position, key, value)


def _object_after_key(text: str, start: int, key: str) -> int | None:
    """Where the object under key begins, in the object whose members begin at start."""
    tokens = _symbols(text, start)
    depth = 0
    last_string = None
    for token in tokens:
        symbol = token[0]
        if symbol in "{[":
            depth += 1
        elif symbol in "}]":
            if depth == 0:
                return None
            depth -= 1
        elif symbol != ":":
            last_string = json.loads(symbol)
        elif depth == 0 and last_string == key:
            return next(tokens).end()
    return None


def _symbols(text: str, start: int) -> Iterator[re.Match[str]]:
    return (token for token in _TOKENS.finditer(text, start) if not token[0].startswith("/"))


def _with_member(text: str, position: int, key: str, value: dict[str, Any]) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"
    outer = _indent_of_line(text, position)
    inner = outer + "  "
    rendered = json.dumps(value, indent=2, ensure_ascii=False).replace("\n", newline + inner)
    member = f"{newline}{inner}{json.dumps(key, ensure_ascii=False)}: {rendered}"
    is_empty = next(_symbols(text, position))[0] == "}"
    return text[:position] + member + (newline + outer if is_empty else ",") + text[position:]


def _indent_of_line(text: str, position: int) -> str:
    start = text.rfind("\n", 0, position) + 1
    line = text[start:position]
    return line[: len(line) - len(line.lstrip(" \t"))]


def _keep_first_copy(target: Path) -> None:
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)


def _replace(target: Path, content: bytes, existed: bool) -> None:
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        partial.write_bytes(content)
        if existed:
            shutil.copymode(target, partial)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
