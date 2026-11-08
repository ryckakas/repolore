from pathlib import Path, PureWindowsPath

import pytest

from repolore.config import Config, LocalFolder, load_config
from repolore.setup import (
    PLANS_PATTERN,
    SelectionError,
    checked_patterns,
    checked_roots,
    config_text,
    parse_patterns,
    parse_roots,
    parse_selection,
    require_roots_with_patterns,
    with_unasked_settings,
    write_config,
)

NAMES = ["me/alpha", "me/beta", "org/gamma", "org/delta"]


def test_numbers_pick_the_repos_listed_under_them() -> None:
    assert parse_selection("1,3", NAMES) == ["me/alpha", "org/gamma"]


def test_all_picks_every_repo_without_an_include_glob() -> None:
    assert parse_selection(" ALL ", NAMES) == []


@pytest.mark.parametrize("answer", ["", "  ", ", ,"])
def test_an_empty_answer_picks_nothing_rather_than_everything(answer: str) -> None:
    with pytest.raises(SelectionError, match="at least one"):
        parse_selection(answer, NAMES)


def test_a_range_picks_every_repo_in_it() -> None:
    assert parse_selection("2-4", NAMES) == ["me/beta", "org/gamma", "org/delta"]


def test_numbers_ranges_and_globs_mix_with_commas_or_spaces_and_repeat_once() -> None:
    assert parse_selection("4 org/*, 1-2 1", NAMES) == ["org/delta", "org/*", "me/alpha", "me/beta"]


@pytest.mark.parametrize("answer", ["1", "4", "1-4", "3-3"])
def test_numbers_from_one_to_the_last_repo_are_in_range(answer: str) -> None:
    assert parse_selection(answer, NAMES)


@pytest.mark.parametrize("answer", ["0", "5", "0-2", "3-5", "3-2"])
def test_numbers_outside_the_list_or_a_backwards_range_are_refused(answer: str) -> None:
    with pytest.raises(SelectionError, match=f"{answer} isn't a number or range between 1 and 4"):
        parse_selection(answer, NAMES)


def test_a_glob_matching_no_repo_is_refused_as_a_likely_typo() -> None:
    with pytest.raises(SelectionError, match="orgg/\\* matches none"):
        parse_selection("1 orgg/*", NAMES)


def test_a_glob_matches_names_case_insensitively() -> None:
    assert parse_selection("ME/Alpha", NAMES) == ["ME/Alpha"]


def test_roots_are_comma_separated_existing_folders(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    assert parse_roots(f" {tmp_path / 'a'} ,{tmp_path / 'b'}") == [tmp_path / "a", tmp_path / "b"]


def test_no_roots_is_an_empty_answer() -> None:
    assert parse_roots("  ") == []


def test_a_root_with_a_tilde_is_kept_as_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert parse_roots("~") == [Path("~")]


def test_a_relative_root_is_made_absolute_for_a_refresh_running_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "code").mkdir()
    monkeypatch.chdir(tmp_path)

    assert parse_roots("code, ., code/../code") == [tmp_path / "code", tmp_path, tmp_path / "code"]
    assert checked_roots([Path("code")]) == [tmp_path / "code"]


def test_a_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match=r"not a folder: .*missing"):
        parse_roots(f"{tmp_path},{tmp_path / 'missing'}")


def test_an_empty_config_writes_no_keys() -> None:
    assert config_text(Config()) == ""


def test_only_the_keys_set_are_written() -> None:
    assert config_text(Config(include=["me/*"])) == 'include = ["me/*"]\n'


def test_default_depth_and_limit_are_left_out_and_others_written() -> None:
    assert "depth" not in config_text(Config(local_patterns=["*.md"]))
    assert config_text(Config(checkout_depth=3, max_local_files=9)) == "checkout_depth = 3\nmax_local_files = 9\n"


def test_patterns_are_comma_separated() -> None:
    assert parse_patterns(f" {PLANS_PATTERN} , docs/*.md,") == [PLANS_PATTERN, "docs/*.md"]


@pytest.mark.parametrize("answer", ["", " , "])
def test_an_empty_pattern_answer_is_refused(answer: str) -> None:
    with pytest.raises(SelectionError, match="at least one pattern"):
        parse_patterns(answer)


def test_a_pattern_that_leaves_the_folder_is_refused() -> None:
    with pytest.raises(SelectionError, match="relative to the folder"):
        checked_patterns(["../*.md"])


@pytest.mark.parametrize(("roots", "patterns"), [([Path("a")], []), ([], ["*.md"])])
def test_roots_and_patterns_need_each_other(roots: list[Path], patterns: list[str]) -> None:
    with pytest.raises(SelectionError, match="--local-root and --local-pattern go together"):
        require_roots_with_patterns(roots, patterns)


@pytest.mark.parametrize(("roots", "patterns"), [([Path("a")], ["*.md"]), ([], [])])
def test_roots_with_patterns_or_neither_are_fine(roots: list[Path], patterns: list[str]) -> None:
    require_roots_with_patterns(roots, patterns)


def test_written_config_reads_back_the_same(tmp_path: Path) -> None:
    config = Config(
        include=["me/*", 'odd"name', "ünïcode/😀"],
        exclude=["me/old-*"],
        local_roots=[Path("~/projects"), Path(str(PureWindowsPath(r"C:\Users\me\code")))],
        local_patterns=[PLANS_PATTERN],
        checkout_depth=1,
    )
    path = tmp_path / "nested" / "repolore.toml"

    text = write_config(path, config)

    assert path.read_text(encoding="utf-8") == text
    assert load_config(path) == config


FOLDERS = [
    LocalFolder(name="notes", path=Path("~/notes")),
    LocalFolder(name="odd", path=Path(str(PureWindowsPath(r"C:\\Users\\me\\od\"d"))), patterns=["a/*.md", "**/*.md"]),
]


def test_folders_are_written_as_tables_that_read_back_the_same(tmp_path: Path) -> None:
    config = Config(include=["me/*"], local_roots=[Path("~/projects")], local_folders=FOLDERS)
    path = tmp_path / "repolore.toml"

    text = write_config(path, config)

    assert text.count("[[local_folders]]") == 2
    assert text.index("local_roots") < text.index("[[local_folders]]")
    assert load_config(path) == config


def test_a_config_of_only_folders_reads_back_the_same(tmp_path: Path) -> None:
    config = Config(local_folders=FOLDERS[:1])
    path = tmp_path / "repolore.toml"

    write_config(path, config)

    assert load_config(path) == config


def test_the_default_folder_pattern_is_left_out() -> None:
    assert "patterns" not in config_text(Config(local_folders=FOLDERS[:1]))


def test_replacing_a_config_keeps_what_setup_doesnt_ask_about(tmp_path: Path) -> None:
    path = tmp_path / "repolore.toml"
    write_config(path, Config(include=["old/*"], local_folders=FOLDERS, checkout_depth=4, max_local_files=9))

    kept = with_unasked_settings(Config(include=["new/*"]), path)

    assert kept == Config(include=["new/*"], local_folders=FOLDERS, checkout_depth=4, max_local_files=9)


def test_nothing_to_replace_changes_nothing(tmp_path: Path) -> None:
    config = Config(include=["new/*"])

    assert with_unasked_settings(config, tmp_path / "missing.toml") == config


@pytest.mark.parametrize("content", [b"include = [", b'nonsense = "key"\n', b"\xff\xfe"])
def test_a_replaced_file_that_cant_be_read_is_just_replaced(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "repolore.toml"
    path.write_bytes(content)
    config = Config(include=["new/*"])

    assert with_unasked_settings(config, path) == config
