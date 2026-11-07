from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, Field


class Phase(StrEnum):
    DISCOVER = "discover"
    INGEST = "ingest"
    DOWNLOAD = "download"
    CHUNK = "chunk"
    EMBED = "embed"
    WRITE = "write"


class Progress(BaseModel):
    phase: Phase = Field(
        description="discover, ingest (fetch repos), download (the embedding model, once), chunk, "
        "embed (new chunks only) or write."
    )
    done: int | None = Field(default=None, description="Repos fetched, MB downloaded or new chunks embedded so far.")
    total: int | None = Field(default=None, description="Repos to fetch, MB to download or new chunks to embed in all.")

    def __str__(self) -> str:
        if self.total is None:
            return self.phase
        unit = " MB" if self.phase is Phase.DOWNLOAD else ""
        return f"{self.phase} {self.done} of {self.total}{unit}"


Report = Callable[[Progress], None]


def ignore(_: Progress) -> None:
    pass
