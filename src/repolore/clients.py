import shutil
import subprocess
from dataclasses import dataclass

SERVER = "repolore"
GET_SECONDS = 120


@dataclass(frozen=True)
class McpClient:
    """An MCP client whose CLI can show how repolore is registered with it."""

    name: str
    command: str
    register: str


CLAUDE_CODE = McpClient(
    "Claude Code", "claude", "`claude mcp add --scope user --transport stdio repolore -- repolore-mcp`"
)
CODEX = McpClient("Codex", "codex", "add [mcp_servers.repolore] to ~/.codex/config.toml, as the README shows")
CLIENTS = (CLAUDE_CODE, CODEX)


@dataclass(frozen=True)
class Registration:
    found: bool
    status: str | None = None

    @property
    def healthy(self) -> bool:
        """Registered, and connected when the client checked the server; Codex doesn't check."""
        return self.found and (
            self.status is None or ("Connected" in self.status and "fail" not in self.status.lower())
        )


def registration(client: McpClient) -> Registration | None:
    """How the client has repolore registered, or None when its CLI isn't on PATH.

    Claude Code starts the server to check it, which can take a while.
    """
    executable = shutil.which(client.command)
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "mcp", "get", SERVER],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GET_SECONDS,
    )
    if result.returncode != 0:
        said = (result.stderr or result.stdout).strip().splitlines()
        return Registration(found=False, status=said[0] if said else None)
    status, issue = _field(result.stdout, "Status"), _field(result.stdout, "Issue")
    return Registration(found=True, status=f"{status} ({issue})" if status and issue else status)


def _field(output: str, name: str) -> str | None:
    prefix = f"{name}:"
    lines = (line.strip() for line in output.splitlines())
    return next((line.removeprefix(prefix).strip() for line in lines if line.startswith(prefix)), None)
