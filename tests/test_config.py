import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from repolore.config import Config, LocalFolder, load_config


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "repolore.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_settings_are_loaded(tmp_path: Path) -> None:
    config = load_config(
        write(tmp_path, 'include = ["octo/*"]\nexclude = ["octo/old-*"]\nlocal_roots = ["D:/projects"]')
    )

    assert config == Config(include=["octo/*"], exclude=["octo/old-*"], local_roots=[Path("D:/projects")])


@pytest.mark.parametrize("content", [None, ""])
def test_missing_or_empty_config_selects_everything(tmp_path: Path, content: str | None) -> None:
    path = write(tmp_path, content) if content is not None else tmp_path / "repolore.toml"

    config = load_config(path)

    assert config == Config()
    assert config.selects("anyone/anything")


@pytest.mark.parametrize(
    ("config", "selected"),
    [
        (Config(include=["octo/*"]), ["octo/repo", "Octo/Old-Demo"]),
        (Config(exclude=["*/old-*"]), ["octo/repo", "other/repo"]),
        (Config(include=["octo/*"], exclude=["octo/old-*"]), ["octo/repo"]),
        (Config(include=["octo/repo"]), ["octo/repo"]),
    ],
)
def test_include_and_exclude_globs_ignore_case(config: Config, selected: list[str]) -> None:
    repos = ["octo/repo", "Octo/Old-Demo", "other/repo"]

    assert [repo for repo in repos if config.selects(repo)] == selected


@pytest.mark.parametrize("content", ['repos = ["octo/repo"]', 'include = "octo/*"', "local_roots = [1]"])
def test_invalid_config_is_rejected(tmp_path: Path, content: str) -> None:
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, content))


def test_local_document_settings_are_loaded(tmp_path: Path) -> None:
    config = load_config(
        write(tmp_path, 'checkout_depth = 3\nmax_local_files = 50\nlocal_patterns = [".plans/**/*.md", "docs/*.md"]')
    )

    assert config == Config(checkout_depth=3, max_local_files=50, local_patterns=[".plans/**/*.md", "docs/*.md"])


def test_local_document_settings_default_to_nothing_local() -> None:
    assert (Config().local_patterns, Config().checkout_depth, Config().max_local_files) == ([], 2, 2000)


@pytest.mark.parametrize(
    "pattern", ["", ".", "..", "../x/*.md", "a/../b.md", "/abs/*.md", "C:/x/*.md", r"C:\x\*.md", r"..\x.md"]
)
def test_a_pattern_that_could_leave_its_folder_is_rejected(pattern: str) -> None:
    with pytest.raises(ValidationError, match="relative to the folder"):
        Config(local_patterns=[pattern])


@pytest.mark.parametrize("pattern", ["docs/**.md", "**.md", "a**/*.md", "***/x.md"])
def test_a_double_star_inside_a_path_part_is_rejected_since_glob_would_raise(pattern: str) -> None:
    with pytest.raises(ValidationError, match="whole path part"):
        Config(local_patterns=[pattern])


@pytest.mark.parametrize("pattern", [".plans/**/*.md", "**/*.md", "*.md", "a..b/*.md"])
def test_relative_patterns_are_accepted(pattern: str) -> None:
    assert Config(local_patterns=[pattern]).local_patterns == [pattern]


def test_depth_zero_and_a_limit_of_one_are_the_smallest_allowed() -> None:
    assert Config(checkout_depth=0, max_local_files=1)


@pytest.mark.parametrize("setting", [{"checkout_depth": -1}, {"max_local_files": 0}])
def test_a_depth_or_limit_below_the_smallest_is_rejected(setting: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(setting)


def test_broken_toml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        load_config(write(tmp_path, "include = ["))


def test_local_folders_are_loaded_with_the_default_patterns(tmp_path: Path) -> None:
    config = load_config(
        write(
            tmp_path,
            '[[local_folders]]\nname = "notes"\npath = "~/notes"\n\n'
            '[[local_folders]]\nname = "adr"\npath = "~/adr"\npatterns = ["*.md"]\n',
        )
    )

    assert config.local_folders == [
        LocalFolder(name="notes", path=Path("~/notes"), patterns=["**/*.md"]),
        LocalFolder(name="adr", path=Path("~/adr"), patterns=["*.md"]),
    ]
    assert config.local_folders[0].repo == "local/notes"


def test_no_local_folders_is_the_default() -> None:
    assert Config().local_folders == []


def test_a_home_directory_that_cant_be_found_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_such_user(_: Path) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "expanduser", no_such_user)

    with pytest.raises(ValidationError, match="must be absolute"):
        LocalFolder(name="notes", path=Path("~typo/notes"))


@pytest.mark.parametrize("name", ["", ".", "..", "notes.", "notes..", "a/b", r"a\\b", "a b", "a:b"])
def test_a_folder_name_that_isnt_a_plain_word_is_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValidationError):
        LocalFolder(name=name, path=tmp_path)


@pytest.mark.parametrize("name", ["notes", "my.notes_2-a", "a", "..a"])
def test_plain_folder_names_are_accepted(tmp_path: Path, name: str) -> None:
    assert LocalFolder(name=name, path=tmp_path).name == name


@pytest.mark.parametrize("path", ["notes", "./notes", "../notes", ""])
def test_a_relative_folder_path_is_rejected(path: str) -> None:
    with pytest.raises(ValidationError, match="must be absolute"):
        LocalFolder(name="notes", path=Path(path))


def test_a_folder_needs_at_least_one_pattern(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        LocalFolder(name="notes", path=tmp_path, patterns=[])


def test_a_folder_pattern_that_could_leave_it_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="relative to the folder"):
        LocalFolder(name="notes", path=tmp_path, patterns=["../*.md"])


def test_two_folders_named_alike_are_rejected_whatever_the_case(tmp_path: Path) -> None:
    folders = [LocalFolder(name="Notes", path=tmp_path), LocalFolder(name="notes", path=tmp_path)]

    with pytest.raises(ValidationError, match="more than one local_folders entry is named notes"):
        Config(local_folders=folders)


def test_a_named_folder_skips_include_and_exclude_but_other_repos_dont(tmp_path: Path) -> None:
    config = Config(include=["me/*"], exclude=["local/*"], local_folders=[LocalFolder(name="Notes", path=tmp_path)])

    assert config.selects("local/notes")
    assert not config.selects("local/other")
    assert not config.selects("someone/else")
