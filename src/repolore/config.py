import tomllib
from fnmatch import fnmatchcase
from pathlib import Path, PureWindowsPath
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def relative_pattern(pattern: str) -> str:
    # PureWindowsPath understands both separators and drive letters, so a pattern is judged the same on every platform.
    path = PureWindowsPath(pattern)
    if not path.parts or path.anchor or ".." in path.parts:
        raise ValueError(f"{pattern!r} must be a glob relative to the folder, without ..")
    if any("**" in part and part != "**" for part in path.parts):
        raise ValueError(f"{pattern!r} has a ** that isn't a whole path part, such as **/*.md")
    return pattern


LocalPattern = Annotated[str, AfterValidator(relative_pattern)]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    include: list[str] = []
    exclude: list[str] = []
    local_roots: list[Path] = []
    checkout_depth: int = Field(default=2, ge=0)
    local_patterns: list[LocalPattern] = []
    max_local_files: int = Field(default=2000, ge=1)

    def selects(self, repo: str) -> bool:
        """Whether a discovered repo is ingested: matches an include glob (or none are set) and no exclude glob."""
        included = not self.include or _matches_any(repo, self.include)
        return included and not _matches_any(repo, self.exclude)


def _matches_any(repo: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(repo.lower(), pattern.lower()) for pattern in patterns)


def load_config(path: Path) -> Config:
    """The config file, or the defaults (every discovered repo, no local documents) when there is none."""
    if not path.exists():
        return Config()
    return Config.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
