import codecs
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from repolore.config_files import (
    BACKUP_SUFFIX,
    PARTIAL_SUFFIX,
    ConfigFileError,
    add_entry,
    parse_jsonc,
    with_entry,
)

ENTRY = {"command": "repolore-mcp"}
SERVERS = ("mcpServers",)


def added(text: str, keys: tuple[str, ...] = SERVERS) -> str:
    return with_entry(text, keys, "repolore", ENTRY)


def test_plain_json_parses() -> None:
    assert parse_jsonc('{"a": [1, 2]}') == {"a": [1, 2]}


def test_comments_and_trailing_commas_parse() -> None:
    text = """{
      // a line comment
      "a": 1, /* a block
                 comment */
      "b": [1, 2,],
    }"""

    assert parse_jsonc(text) == {"a": 1, "b": [1, 2]}


def test_a_trailing_comma_before_a_comment_parses() -> None:
    assert parse_jsonc('{"a": 1, // last\n}') == {"a": 1}


@pytest.mark.parametrize("value", ["// not a comment", "/* nor this */", "a, }", 'quote " and //', "back\\slash,]"])
def test_strings_that_look_like_comments_or_commas_are_kept(value: str) -> None:
    assert parse_jsonc(json.dumps({"a": value})) == {"a": value}


@pytest.mark.parametrize("text", ["{ not json", '{"a": /* never closed }', '{"a": "never closed}'])
def test_broken_text_doesnt_parse(text: str) -> None:
    with pytest.raises(ValueError):
        parse_jsonc(text)


def test_an_entry_goes_first_in_a_new_object() -> None:
    assert (
        added('{\n  "a": 1\n}\n')
        == """{
  "mcpServers": {
    "repolore": {
      "command": "repolore-mcp"
    }
  },
  "a": 1
}
"""
    )


@pytest.mark.parametrize("text", ["", "  \n"])
def test_blank_text_becomes_an_object_with_the_entry(text: str) -> None:
    assert json.loads(added(text)) == {"mcpServers": {"repolore": ENTRY}}
    assert added(text).endswith("}\n")


def test_an_empty_object_gets_the_entry() -> None:
    assert (
        added("{}\n")
        == """{
  "mcpServers": {
    "repolore": {
      "command": "repolore-mcp"
    }
  }
}
"""
    )


def test_an_empty_servers_object_gets_the_entry() -> None:
    text = added('{\n  "mcpServers": {}\n}\n')

    assert text.startswith('{\n  "mcpServers": {\n    "repolore": {\n      "command": "repolore-mcp"\n    }\n  }\n}')


def test_an_entry_joins_the_other_servers() -> None:
    text = '{\n  "mcpServers": {\n    "other": {"command": "x"}\n  }\n}\n'

    assert (
        added(text)
        == """{
  "mcpServers": {
    "repolore": {
      "command": "repolore-mcp"
    },
    "other": {"command": "x"}
  }
}
"""
    )


@pytest.mark.parametrize(
    "text",
    [
        '{\n  // keep me\n  "a": 1, /* and me */\n  "b": [1, 2,],\n}\n',
        '{\n  "mcpServers": { // servers\n    "other": {},\n  },\n  // end\n}\n',
        '{"a":1,\t"b" :  2}',
        '{ "url": "http://example.com/{x}", "b": "// no" }',
    ],
)
def test_every_other_byte_is_left_alone(text: str) -> None:
    updated = added(text)
    extra = len(updated) - len(text)
    position = next(i for i in range(len(text) + 1) if updated[:i] + updated[i + extra :] == text)

    assert parse_jsonc(updated)["mcpServers"]["repolore"] == ENTRY
    assert updated[:position] + updated[position + extra :] == text


def test_a_nested_key_that_doesnt_exist_is_created() -> None:
    assert parse_jsonc(added('{"a": 1}', ("mcp", "servers"))) == {"a": 1, "mcp": {"servers": {"repolore": ENTRY}}}


def test_a_nested_key_with_its_parent_present_goes_inside_it() -> None:
    text = '{"mcp": {"timeout": 5}}'

    assert parse_jsonc(added(text, ("mcp", "servers"))) == {"mcp": {"timeout": 5, "servers": {"repolore": ENTRY}}}


def test_a_nested_key_already_there_gets_the_entry() -> None:
    text = '{"mcp": {"servers": {"other": {}}}}'

    assert parse_jsonc(added(text, ("mcp", "servers"))) == {"mcp": {"servers": {"other": {}, "repolore": ENTRY}}}


@pytest.mark.parametrize(
    "text",
    [
        '{"a": {"mcpServers": {"deep": 1}}}',
        '{"a": [{"mcpServers": {"deep": 1}}]}',
        '{"name": "mcpServers", "a": {}}',
    ],
)
def test_the_key_is_only_matched_at_its_own_level(text: str) -> None:
    updated = parse_jsonc(added(text))

    assert updated["mcpServers"] == {"repolore": ENTRY}
    assert updated == {**json.loads(text), "mcpServers": {"repolore": ENTRY}}


def test_comments_around_a_keys_colon_are_skipped() -> None:
    text = '{"mcpServers" /* : { */ : // {\n {"other": {}}}'

    assert parse_jsonc(added(text)) == {"mcpServers": {"other": {}, "repolore": ENTRY}}


def test_an_escaped_key_is_the_same_key() -> None:
    assert parse_jsonc(added('{"mcp\\u0053ervers": {"other": {}}}')) == {"mcpServers": {"other": {}, "repolore": ENTRY}}


def test_non_ascii_values_are_written_as_they_are() -> None:
    updated = with_entry('{"a": "é"}', SERVERS, "repolore", {"command": "C:\\Users\\Žydrūnas\\x"})

    assert '"é"' in updated
    assert "Žydrūnas" in updated


def test_a_comment_only_object_is_empty() -> None:
    assert parse_jsonc(added("{ // nothing\n}")) == {"mcpServers": {"repolore": ENTRY}}


def test_crlf_text_gets_crlf_lines() -> None:
    updated = added('{\r\n  "a": 1\r\n}\r\n')

    assert updated.count("\n") == updated.count("\r\n")
    assert parse_jsonc(updated)["a"] == 1


@pytest.mark.parametrize(
    ("text", "complaint"),
    [
        ("{ not json", "isn't valid JSON"),
        ("[1, 2]", "isn't a JSON object"),
        ('"text"', "isn't a JSON object"),
        ("// only a comment", "isn't valid JSON"),
        ('{"mcpServers": []}', "its mcpServers isn't an object"),
        ('{"mcpServers": {"repolore": {}}}', "it already has repolore"),
    ],
)
def test_text_it_cant_add_to_is_refused(text: str, complaint: str) -> None:
    with pytest.raises(ConfigFileError, match=complaint):
        added(text)


@pytest.mark.parametrize(
    "text",
    ['{"mcpServers": {"a": {}}, "mcpServers": {"b": {}}}', '{"mcpServers": 1, "mcpServers": {"b": {}}}'],
)
def test_a_duplicated_key_fails_the_check_on_the_result(text: str) -> None:
    with pytest.raises(ConfigFileError, match="wouldn't give the expected settings"):
        added(text)


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return tmp_path / "settings.json"


def backup_of(path: Path) -> Path:
    return path.with_name(path.name + BACKUP_SUFFIX)


def add(path: Path, name: str = "repolore", entry: dict[str, Any] = ENTRY) -> None:
    add_entry(path, SERVERS, name, entry)


def test_a_new_file_is_created_without_a_backup(config: Path) -> None:
    add(config)

    assert json.loads(config.read_bytes()) == {"mcpServers": {"repolore": ENTRY}}
    assert not backup_of(config).exists()


def test_a_missing_folder_is_created(tmp_path: Path) -> None:
    path = tmp_path / "new" / "folder" / "settings.json"

    add(path)

    assert path.exists()


def test_an_existing_file_is_backed_up_as_it_was(config: Path) -> None:
    original = b'{\n  // mine\n  "a": 1\n}\n'
    config.write_bytes(original)

    add(config)

    assert backup_of(config).read_bytes() == original
    assert b"// mine" in config.read_bytes()
    assert b"repolore-mcp" in config.read_bytes()


def test_only_the_first_change_makes_a_backup(config: Path) -> None:
    original = b'{"a": 1}'
    config.write_bytes(original)

    add(config)
    add(config, "second")

    assert backup_of(config).read_bytes() == original
    assert parse_jsonc(config.read_text(encoding="utf-8"))["mcpServers"].keys() == {"repolore", "second"}


def test_an_empty_file_is_filled_and_backed_up(config: Path) -> None:
    config.write_bytes(b"")

    add(config)

    assert json.loads(config.read_bytes()) == {"mcpServers": {"repolore": ENTRY}}
    assert backup_of(config).read_bytes() == b""


def test_a_byte_order_mark_and_crlf_are_kept(config: Path) -> None:
    config.write_bytes(codecs.BOM_UTF8 + b'{\r\n  "a": 1\r\n}\r\n')

    add(config)

    written = config.read_bytes()
    assert written.startswith(codecs.BOM_UTF8)
    assert written.count(b"\n") == written.count(b"\r\n")
    assert parse_jsonc(written.decode("utf-8-sig"))["a"] == 1


@pytest.mark.parametrize(
    ("original", "complaint"),
    [
        (b"{ not json", "isn't valid JSON"),
        (b"[1]", "isn't a JSON object"),
        (b'{"a": "\xff"}', "codec can't decode"),
        (b'{"mcpServers": {"repolore": {}}}', "already has repolore"),
        (b'{"mcpServers": {"a": {}}, "mcpServers": {"b": {}}}', "wouldn't give the expected settings"),
    ],
)
def test_a_refused_file_is_named_and_nothing_is_written(config: Path, original: bytes, complaint: str) -> None:
    config.write_bytes(original)

    with pytest.raises(ConfigFileError, match=complaint) as caught:
        add(config)

    assert str(config) in str(caught.value)
    assert config.read_bytes() == original
    assert not backup_of(config).exists()
    assert sorted(path.name for path in config.parent.iterdir()) == [config.name]


def test_a_read_only_file_is_refused_and_left_alone(config: Path) -> None:
    original = b'{"a": 1}'
    config.write_bytes(original)
    config.chmod(stat.S_IREAD)

    try:
        with pytest.raises(ConfigFileError, match="it's read-only") as caught:
            add(config)

        assert str(config) in str(caught.value)
        assert config.read_bytes() == original
        assert sorted(path.name for path in config.parent.iterdir()) == [config.name]
    finally:
        config.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_a_failed_replace_leaves_the_file_and_no_partial(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def locked(_self: Path, _target: Path) -> None:
        raise PermissionError("in use")

    original = b'{"a": 1}'
    config.write_bytes(original)
    monkeypatch.setattr(Path, "replace", locked)

    with pytest.raises(PermissionError):
        add(config)

    assert config.read_bytes() == original
    assert not config.with_name(config.name + PARTIAL_SUFFIX).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no POSIX file modes")
def test_the_files_mode_is_kept(config: Path) -> None:
    config.write_bytes(b"{}")
    config.chmod(0o600)

    add(config)

    assert stat.S_IMODE(config.stat().st_mode) == 0o600


def test_a_symlinked_config_stays_a_symlink(tmp_path: Path) -> None:
    real = tmp_path / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_bytes(b'{"a": 1}')
    link = tmp_path / "settings.json"
    try:
        os.symlink(real, link)
    except OSError:
        pytest.skip("this machine can't create symlinks")

    add(link)

    assert link.is_symlink()
    assert b"repolore-mcp" in real.read_bytes()
    assert backup_of(real).exists()
