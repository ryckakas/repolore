import os
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from filelock import Timeout
from pydantic import BaseModel, ValidationError

from repolore.locks import hold, is_locked

HOST = "127.0.0.1"


class Discovery(BaseModel):
    pid: int
    port: int
    token: str
    version: str
    started_at: str

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/mcp"


@dataclass(frozen=True)
class ServerFiles:
    """Where the shared server for one index keeps its lock, address, stop request and log, and relays register."""

    data_dir: Path
    db: str

    @property
    def lock(self) -> Path:
        return self.data_dir / f"server-{self.db}.lock"

    @property
    def discovery(self) -> Path:
        return self.data_dir / f"server-{self.db}.json"

    @property
    def stop(self) -> Path:
        return self.data_dir / f"server-{self.db}.stop"

    @property
    def log(self) -> Path:
        return self.data_dir / f"server-{self.db}.log"

    @property
    def clients(self) -> Path:
        return self.data_dir / "clients" / self.db

    def client_lock(self, pid: int) -> Path:
        return self.clients / f"{pid}.lock"


def is_running(files: ServerFiles) -> bool:
    return is_locked(files.lock)


def read_discovery(files: ServerFiles) -> Discovery | None:
    try:
        return Discovery.model_validate_json(files.discovery.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        # Missing before the server has bound its port, locked on Windows for the moment it is replaced, or left
        # garbled by a server that was killed writing it.
        return None


def write_discovery(files: ServerFiles, discovery: Discovery) -> None:
    """Write the address and token where only this user can read them, and atomically, for relays polling it."""
    partial = files.discovery.with_suffix(".tmp")
    partial.unlink(missing_ok=True)
    with os.fdopen(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as file:
        file.write(discovery.model_dump_json())
    partial.replace(files.discovery)


def request_stop(files: ServerFiles) -> None:
    files.data_dir.mkdir(parents=True, exist_ok=True)
    files.stop.touch()


@contextmanager
def registered(files: ServerFiles) -> Generator[None]:
    """Hold this relay's lock while it runs, so the server knows a session is open; a killed relay drops it."""
    files.clients.mkdir(parents=True, exist_ok=True)
    # Not at once: the server may be checking this new file at the moment the relay takes it.
    with hold(files.client_lock(os.getpid()), timeout=5):
        yield


def connected_relays(files: ServerFiles) -> int:
    """Count the relays holding their lock, deleting the files of those that exited."""
    connected = 0
    for lock_file in files.clients.glob("*.lock"):
        try:
            with hold(lock_file, timeout=0):
                pass
        except Timeout:
            connected += 1
            continue
        with suppress(OSError):
            lock_file.unlink()
    return connected
