import subprocess
import sys
from collections.abc import Mapping
from typing import IO, Any

INTERMEDIATE = (
    "import subprocess, sys; "
    "subprocess.Popen(sys.argv[2:], creationflags=int(sys.argv[1]), "
    "stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=subprocess.STDOUT)"
)


class SpawnError(OSError):
    pass


def spawn_detached(command: list[str], *, stdout: IO[str], env: Mapping[str, str]) -> None:
    """Start a process that outlives this one and the client session that started it, writing to `stdout`."""
    options: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": stdout, "stderr": subprocess.STDOUT, "env": env}
    if sys.platform != "win32":
        subprocess.Popen(command, start_new_session=True, **options)
        return
    # Claude Code stops a server on Windows with `taskkill /T`, which follows parent PIDs to every descendant, whatever
    # the creation flags. A process whose parent has already exited is linked to nothing, so the intermediate starts
    # the real one and exits before this returns.
    # No console window for the process or the `gh auth token` a refresh runs, and no Ctrl+C from the server's group.
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    intermediate = [sys.executable, "-c", INTERMEDIATE]
    try:
        # Out of the client's job object, whose closing would otherwise kill the process along with the server.
        flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
        process = subprocess.Popen([*intermediate, str(flags), *command], creationflags=flags, **options)
    except PermissionError:
        # A job that forbids breaking away, as the Python MCP SDK's client makes, keeps the process in it.
        flags &= ~subprocess.CREATE_BREAKAWAY_FROM_JOB
        process = subprocess.Popen([*intermediate, str(flags), *command], creationflags=flags, **options)
    if (code := process.wait()) != 0:
        raise SpawnError(f"Couldn't start {command[0]}: the process that starts it exited with code {code}.")
