import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "repolore"
CONFIG_FILENAME = "repolore.toml"
QUESTIONS_FILENAME = "questions.yaml"
HOME_ENV = "REPOLORE_HOME"


class PathSource(StrEnum):
    DEFAULT = "default"
    HOME = HOME_ENV
    OPTION = "--data-dir"


@dataclass(frozen=True)
class AppPaths:
    config_file: Path
    questions_file: Path
    data_dir: Path
    config_from: PathSource = PathSource.DEFAULT
    data_from: PathSource = PathSource.DEFAULT

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def embedding_cache(self) -> Path:
        return self.data_dir / "cache" / "embeddings.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def index_file(self, name: str) -> Path:
        return self.data_dir / "db" / f"{name}.db"


def resolve_paths(data_dir: Path | None = None) -> AppPaths:
    """REPOLORE_HOME holds the config and the data in one folder; without it they follow the platform's conventions.
    A data dir given wins over both for the data."""
    home = _home()
    if home is not None:
        config_dir = default_data_dir = home
        source = PathSource.HOME
    else:
        # Roaming config follows the user across machines; the data dir holds rebuildable caches and DBs, so it stays
        # local.
        config_dir = Path(user_config_dir(APP_NAME, appauthor=False, roaming=True))
        default_data_dir = Path(user_data_dir(APP_NAME, appauthor=False))
        source = PathSource.DEFAULT
    return AppPaths(
        config_file=config_dir / CONFIG_FILENAME,
        questions_file=config_dir / "evals" / QUESTIONS_FILENAME,
        data_dir=(data_dir or default_data_dir).resolve(),
        config_from=source,
        data_from=PathSource.OPTION if data_dir else source,
    )


def child_environment() -> dict[str, str]:
    """This process's environment for a process that starts in another folder, with REPOLORE_HOME made absolute."""
    environment = dict(os.environ)
    if (home := _home()) is not None:
        environment[HOME_ENV] = str(home)
    return environment


def _home() -> Path | None:
    value = os.environ.get(HOME_ENV)
    return Path(value).expanduser().resolve() if value else None
