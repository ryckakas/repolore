from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, Field


class Phase(StrEnum):
    DISCOVER = "discover"
    INGEST = "ingest"
    CHUNK = "chunk"
    EMBED = "embed"
    WRITE = "write"


class Progress(BaseModel):
    phase: Phase = Field(description="discover, ingest (fetch repos), chunk, embed (new chunks only) or write.")
    done: int | None = Field(default=None, description="Repos fetched, or new chunks embedded, so far.")
    total: int | None = Field(default=None, description="Repos to fetch, or new chunks to embed, in all.")

    def __str__(self) -> str:
        return self.phase if self.total is None else f"{self.phase} {self.done} of {self.total}"


Report = Callable[[Progress], None]


def ignore(_: Progress) -> None:
    pass
