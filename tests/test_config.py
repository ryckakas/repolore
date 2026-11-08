import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from repolore.config import Config, load_config


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
