import os
from pathlib import Path

import pytest

from repolore.paths import CONFIG_FILENAME, HOME_ENV, QUESTIONS_FILENAME, PathSource, child_environment, resolve_paths


def test_default_paths_are_per_user_app_dirs() -> None:
    paths = resolve_paths()

    assert paths.config_file.name == CONFIG_FILENAME
    assert paths.config_file.parent.name == "repolore"
    assert paths.data_dir.name == "repolore"
    assert paths.data_dir.is_absolute()


def test_data_dir_override_is_resolved_to_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    paths = resolve_paths(Path("relative-data"))

    assert paths.data_dir == tmp_path.resolve() / "relative-data"


def test_data_dir_override_leaves_config_location_alone(tmp_path: Path) -> None:
    assert resolve_paths(tmp_path).config_file == resolve_paths().config_file


def test_questions_file_lives_in_the_config_dir() -> None:
    paths = resolve_paths()

    assert paths.questions_file == paths.config_file.parent / "evals" / QUESTIONS_FILENAME


def test_the_default_paths_say_they_are_the_defaults() -> None:
    paths = resolve_paths()

    assert (paths.config_from, paths.data_from) == (PathSource.DEFAULT, PathSource.DEFAULT)


def test_a_home_holds_the_config_the_questions_and_the_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOME_ENV, str(tmp_path))

    paths = resolve_paths()

    assert paths.config_file == tmp_path.resolve() / CONFIG_FILENAME
    assert paths.questions_file == tmp_path.resolve() / "evals" / QUESTIONS_FILENAME
    assert paths.data_dir == tmp_path.resolve()
    assert (paths.config_from, paths.data_from) == (PathSource.HOME, PathSource.HOME)


def test_a_data_dir_given_wins_over_the_home_for_the_data_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "home"))

    paths = resolve_paths(tmp_path / "data")

    assert paths.config_file == tmp_path.resolve() / "home" / CONFIG_FILENAME
    assert paths.data_dir == tmp_path.resolve() / "data"
    assert (paths.config_from, paths.data_from) == (PathSource.HOME, PathSource.OPTION)


def test_an_empty_home_variable_is_no_home(monkeypatch: pytest.MonkeyPatch) -> None:
    default = resolve_paths()
    monkeypatch.setenv(HOME_ENV, "")

    assert resolve_paths() == default


def test_a_relative_home_is_made_absolute_for_the_processes_started_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(HOME_ENV, "home")
    monkeypatch.setenv("REPOLORE_TEST_PASSED_ON", "yes")

    environment = child_environment()

    assert environment[HOME_ENV] == str(tmp_path.resolve() / "home")
    assert environment["REPOLORE_TEST_PASSED_ON"] == "yes"
    assert os.environ[HOME_ENV] == "home"


def test_without_a_home_the_environment_is_passed_on_as_it_is() -> None:
    assert child_environment() == dict(os.environ)
