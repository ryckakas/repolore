import tomllib
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path, PureWindowsPath
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

FOLDER_OWNER = "local"


def relative_pattern(pattern: str) -> str:
    # PureWindowsPath understands both separators and drive letters, so a pattern is judged the same on every platform.
    path = PureWindowsPath(pattern)
    if not path.parts or path.anchor or ".." in path.parts:
        raise ValueError(f"{pattern!r} must be a glob relative to the folder, without ..")
    if any("**" in part and part != "**" for part in path.parts):
        raise ValueError(f"{pattern!r} has a ** that isn't a whole path part, such as **/*.md")
    return pattern


LocalPattern = Annotated[str, AfterValidator(relative_pattern)]


def absolute_path(path: Path) -> Path:
    # A refresh runs in the data dir, so a relative path would quietly look there.
    try:
        absolute = path.expanduser().is_absolute()
    except RuntimeError:
        absolute = False
    if not absolute:
        raise ValueError(f"{path} must be absolute or start with ~")
    return path


class LocalFolder(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]*[A-Za-z0-9_-]$")]
    path: Annotated[Path, AfterValidator(absolute_path)]
    patterns: Annotated[list[LocalPattern], Field(min_length=1)] = ["**/*.md"]

    @property
    def repo(self) -> str:
        return f"{FOLDER_OWNER}/{self.name}"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    include: list[str] = []
    exclude: list[str] = []
    local_roots: list[Path] = []
    checkout_depth: int = Field(default=2, ge=0)
    local_patterns: list[LocalPattern] = []
    local_folders: list[LocalFolder] = []
    max_local_files: int = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def _folder_names_are_unique(self) -> Self:
        counts = Counter(folder.name.lower() for folder in self.local_folders)
        if repeated := [name for name, count in counts.items() if count > 1]:
            raise ValueError(f"more than one local_folders entry is named {', '.join(repeated)}")
        return self

    def selects(self, repo: str) -> bool:
        """Whether a discovered repo is ingested: it is a named local folder, or it matches an include glob (or none
        are set) and no exclude glob."""
        if repo.lower() in {folder.repo.lower() for folder in self.local_folders}:
            return True
        included = not self.include or _matches_any(repo, self.include)
        return included and not _matches_any(repo, self.exclude)


def _matches_any(repo: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(repo.lower(), pattern.lower()) for pattern in patterns)


def load_config(path: Path) -> Config:
    """The config file, or the defaults (every discovered repo, no local documents) when there is none."""
    if not path.exists():
        return Config()
    return Config.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
