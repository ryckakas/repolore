import os
import stat
import tomllib
from pathlib import Path, PureWindowsPath

import pytest
from pydantic import ValidationError

from repolore.config import Config, LocalFolder, load_config
from repolore.setup import (
    PLANS_PATTERN,
    RepoChange,
    RepoTicks,
    SelectionError,
    checked_patterns,
    checked_roots,
    config_text,
    parse_patterns,
    parse_roots,
    parse_selection,
    require_roots_with_patterns,
    save_local_settings,
    save_ticks,
    update_config,
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
    LocalFolder(name="odd", path=Path('~/od"d'), patterns=["a/*.md", "**/*.md"]),
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


SHOWN = ["me/alpha", "me/beta", "org/gamma"]
COMMENTED = """# my repos
include = ["me/alpha"] # the main one

checkout_depth = 3

# notes
[[local_folders]]
name = "notes"
path = "~/notes"
"""


def toml_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "repolore.toml"
    path.write_bytes(text.encode("utf-8"))
    return path


def ticks(ticked: list[str], every: bool = False, shown: list[str] = SHOWN) -> RepoTicks:
    return RepoTicks(shown, ticked, every)


def test_update_keeps_comments_and_other_keys(tmp_path: Path) -> None:
    path = toml_file(tmp_path, COMMENTED)

    update_config(path, {"include": ["me/beta"]})

    text = path.read_bytes().decode("utf-8")
    assert "# my repos" in text
    assert "# the main one" in text
    assert "# notes" in text
    config = load_config(path)
    assert config.include == ["me/beta"]
    assert config.checkout_depth == 3
    assert [folder.name for folder in config.local_folders] == ["notes"]


def test_a_new_key_doesnt_land_inside_a_table(tmp_path: Path) -> None:
    path = toml_file(tmp_path, COMMENTED)

    update_config(path, {"exclude": ["org/*"]})

    assert load_config(path).exclude == ["org/*"]


def test_an_empty_list_removes_the_key(tmp_path: Path) -> None:
    path = toml_file(tmp_path, COMMENTED)

    update_config(path, {"include": []})

    assert "include" not in path.read_bytes().decode("utf-8")


def test_an_unchanged_key_keeps_the_comments_inside_its_array(tmp_path: Path) -> None:
    text = 'include = [\n  "me/alpha", # why\n]\n'
    path = toml_file(tmp_path, text)

    update_config(path, {"include": ["me/alpha"]})

    assert path.read_bytes().decode("utf-8") == text


def test_removing_a_key_that_isnt_there_is_fine(tmp_path: Path) -> None:
    path = toml_file(tmp_path, COMMENTED)

    update_config(path, {"exclude": []})

    assert path.read_bytes().decode("utf-8") == COMMENTED


def test_update_creates_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "new" / "repolore.toml"

    update_config(path, {"include": ["me/alpha"]})

    assert load_config(path).include == ["me/alpha"]


def test_a_windows_path_survives_as_valid_toml(tmp_path: Path) -> None:
    path = toml_file(tmp_path, "")

    update_config(path, {"local_roots": ["C:\\Users\\me\\code"], "local_patterns": ["*.md"]})

    assert tomllib.loads(path.read_text(encoding="utf-8"))["local_roots"] == ["C:\\Users\\me\\code"]


def test_a_result_that_isnt_a_valid_config_is_refused_and_leaves_the_file(tmp_path: Path) -> None:
    path = toml_file(tmp_path, COMMENTED)

    with pytest.raises(ValidationError):
        update_config(path, {"local_patterns": ["../outside"]})

    assert path.read_bytes().decode("utf-8") == COMMENTED


def test_a_read_only_file_is_refused_and_leaves_no_partial_file(tmp_path: Path) -> None:
    path = toml_file(tmp_path, COMMENTED)
    path.chmod(stat.S_IREAD)

    with pytest.raises(PermissionError, match="read-only"):
        update_config(path, {"include": ["me/beta"]})

    path.chmod(stat.S_IWRITE | stat.S_IREAD)
    assert path.read_bytes().decode("utf-8") == COMMENTED
    assert [file.name for file in tmp_path.iterdir()] == ["repolore.toml"]


def test_a_symlinked_config_stays_a_symlink_when_updated(tmp_path: Path) -> None:
    real = toml_file(tmp_path, COMMENTED)
    link = tmp_path / "link.toml"
    try:
        os.symlink(real, link)
    except OSError:
        pytest.skip("this machine can't create symlinks")

    update_config(link, {"include": ["me/beta"]})

    assert link.is_symlink()
    assert load_config(real).include == ["me/beta"]


def test_the_first_pick_writes_sorted_names(tmp_path: Path) -> None:
    path = tmp_path / "repolore.toml"

    change = save_ticks(path, ticks(["org/gamma", "me/alpha"]))

    assert change == RepoChange(saved=True, added=2, leaving=0)
    assert load_config(path).include == ["me/alpha", "org/gamma"]


def test_one_tick_is_enough(tmp_path: Path) -> None:
    path = tmp_path / "repolore.toml"

    assert save_ticks(path, ticks(["me/beta"])) == RepoChange(saved=True, added=1, leaving=0)


def test_ticking_every_repo_writes_the_names(tmp_path: Path) -> None:
    path = tmp_path / "repolore.toml"

    save_ticks(path, ticks(SHOWN))

    assert load_config(path).include == sorted(SHOWN)


def test_the_switch_writes_both_keys_empty(tmp_path: Path) -> None:
    path = toml_file(tmp_path, 'include = ["me/alpha"]\nexclude = ["org/*"]\n')

    change = save_ticks(path, ticks([], every=True))

    assert change == RepoChange(saved=True, added=2, leaving=0)
    assert path.read_bytes() == b""


def test_the_switch_on_a_first_run_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "repolore.toml"

    assert save_ticks(path, ticks([], every=True)).saved
    assert path.exists()


def test_the_switch_ignores_the_ticks(tmp_path: Path) -> None:
    path = tmp_path / "repolore.toml"

    save_ticks(path, ticks(["me/alpha"], every=True))

    assert load_config(path).include == []


def test_turning_the_switch_off_with_everything_ticked_writes_the_names(tmp_path: Path) -> None:
    path = toml_file(tmp_path, "")

    change = save_ticks(path, ticks(SHOWN))

    assert change == RepoChange(saved=True, added=0, leaving=0)
    assert load_config(path).include == sorted(SHOWN)


def test_unticking_counts_what_leaves(tmp_path: Path) -> None:
    path = toml_file(tmp_path, 'include = ["me/alpha", "me/beta"]\n')

    change = save_ticks(path, ticks(["me/beta", "org/gamma"]))

    assert change == RepoChange(saved=True, added=1, leaving=1)


def test_unchanged_ticks_leave_globs_and_the_file_alone(tmp_path: Path) -> None:
    text = '# mine\ninclude = ["me/*"]\n'
    path = toml_file(tmp_path, text)

    assert save_ticks(path, ticks(["me/alpha", "me/beta"])) == RepoChange(saved=False)
    assert path.read_bytes().decode("utf-8") == text


def test_the_switch_left_on_changes_nothing(tmp_path: Path) -> None:
    path = toml_file(tmp_path, "# every repo\n")

    assert save_ticks(path, ticks([], every=True)) == RepoChange(saved=False)


def test_changed_ticks_replace_globs_and_excludes(tmp_path: Path) -> None:
    path = toml_file(tmp_path, 'include = ["me/*"]\nexclude = ["me/beta"]\n')

    save_ticks(path, ticks(["me/beta"]))

    config = load_config(path)
    assert config.include == ["me/beta"]
    assert config.exclude == []


@pytest.mark.parametrize("existing", [None, 'include = ["me/alpha"]\n', ""])
def test_no_ticks_are_refused_since_an_empty_include_means_every_repo(tmp_path: Path, existing: str | None) -> None:
    path = tmp_path / "repolore.toml"
    if existing is not None:
        toml_file(tmp_path, existing)

    with pytest.raises(SelectionError, match="at least one"):
        save_ticks(path, ticks([]))

    assert path.exists() == (existing is not None)
    if existing is not None:
        assert path.read_bytes().decode("utf-8") == existing


def test_a_tick_for_a_repo_that_wasnt_shown_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "repolore.toml"

    save_ticks(path, ticks(["me/alpha", "evil/repo"]))

    assert load_config(path).include == ["me/alpha"]


def test_a_tick_for_only_a_repo_that_wasnt_shown_counts_as_none(tmp_path: Path) -> None:
    with pytest.raises(SelectionError):
        save_ticks(tmp_path / "repolore.toml", ticks(["evil/repo"]))


def test_saving_ticks_keeps_comments_and_folders(tmp_path: Path) -> None:
    path = toml_file(tmp_path, COMMENTED)

    save_ticks(path, ticks(["org/gamma"]))

    assert "# notes" in path.read_bytes().decode("utf-8")
    assert [folder.name for folder in load_config(path).local_folders] == ["notes"]


@pytest.mark.parametrize("save", ["ticks", "local"])
@pytest.mark.parametrize("content", [b"include = [", b'nonsense = "key"\n', b"\xff\xfe"])
def test_a_config_that_cant_be_read_is_never_overwritten(tmp_path: Path, save: str, content: bytes) -> None:
    path = tmp_path / "repolore.toml"
    path.write_bytes(content)

    with pytest.raises((tomllib.TOMLDecodeError, ValidationError, UnicodeDecodeError)):
        if save == "ticks":
            save_ticks(path, ticks(["me/alpha"]))
        else:
            save_local_settings(path, [], [])

    assert path.read_bytes() == content


def test_local_settings_are_written(tmp_path: Path) -> None:
    path = toml_file(tmp_path, 'include = ["me/alpha"]\n')
    root = tmp_path / "code"
    root.mkdir()

    save_local_settings(path, [str(root)], [PLANS_PATTERN, "docs/*.md"])

    config = load_config(path)
    assert config.local_roots == [root]
    assert config.local_patterns == [PLANS_PATTERN, "docs/*.md"]


def test_a_root_starting_with_a_tilde_is_kept_as_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    (home / "code").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    path = toml_file(tmp_path, "")

    save_local_settings(path, ["~/code"], [PLANS_PATTERN])

    assert load_config(path).local_roots == [Path("~/code")]


@pytest.mark.parametrize("patterns", [[], [PLANS_PATTERN]])
def test_local_settings_cant_create_the_file_that_would_select_every_repo(tmp_path: Path, patterns: list[str]) -> None:
    path = tmp_path / "repolore.toml"
    roots = [str(tmp_path)] if patterns else []

    with pytest.raises(SelectionError, match="repos first"):
        save_local_settings(path, roots, patterns)

    assert not path.exists()


def test_local_settings_can_be_cleared(tmp_path: Path) -> None:
    path = toml_file(tmp_path, 'local_roots = ["~/x"]\nlocal_patterns = ["*.md"]\n# stays\n')

    save_local_settings(path, [], [])

    assert path.read_bytes().decode("utf-8") == "# stays\n"


def test_folders_without_patterns_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match="together"):
        save_local_settings(tmp_path / "repolore.toml", [str(tmp_path)], [])


def test_patterns_without_folders_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match="together"):
        save_local_settings(tmp_path / "repolore.toml", [], [PLANS_PATTERN])


@pytest.mark.parametrize("root", ["code", "./code"])
def test_a_relative_root_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "code").mkdir()

    with pytest.raises(SelectionError, match="absolute"):
        save_local_settings(tmp_path / "repolore.toml", [root], [PLANS_PATTERN])

    assert not (tmp_path / "repolore.toml").exists()


def test_a_root_for_an_unknown_user_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match=r"absolute|not a folder"):
        save_local_settings(tmp_path / "repolore.toml", ["~nobody-here-at-all/code"], [PLANS_PATTERN])


def test_a_root_that_isnt_a_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match="not a folder"):
        save_local_settings(tmp_path / "repolore.toml", [str(tmp_path / "gone")], [PLANS_PATTERN])


def test_a_bad_pattern_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SelectionError, match="relative"):
        save_local_settings(tmp_path / "repolore.toml", [str(tmp_path)], ["../outside"])
