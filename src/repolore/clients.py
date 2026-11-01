import json
import os
import shutil
import subprocess
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SERVER = "repolore"
RELAY = "repolore-mcp"
GET_SECONDS = 120
ADD_SECONDS = 60
CODEX_ENTRY = f"""[mcp_servers.{SERVER}]
command = "{RELAY}"
startup_timeout_sec = 60
tool_timeout_sec = 300
"""
CODEX_PASSES_ON = (
    "REPOLORE_GITHUB_TOKEN",
    "HF_HOME",
    "HTTPS_PROXY",
    "REPOLORE_DEVICE",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
)


class RegistrationError(Exception):
    pass


def _claude_add(executable: str) -> str:
    command = [executable, "mcp", "add", "--scope", "user", "--transport", "stdio", SERVER, "--", RELAY]
    result = _run(command, ADD_SECONDS)
    if result.returncode != 0:
        raise RegistrationError(_first_line(result) or f"`claude mcp add` exited with {result.returncode}")
    return "for every project (user scope)"


def _codex_add(_executable: str) -> str:
    config = codex_home() / "config.toml"
    text = config.read_text(encoding="utf-8") if config.exists() else ""
    try:
        servers = tomllib.loads(text).get("mcp_servers", {})
    except tomllib.TOMLDecodeError as error:
        raise RegistrationError(f"can't read {config}: {error}") from error
    if SERVER in servers:
        raise RegistrationError(f"{config} already has [mcp_servers.{SERVER}]")
    addition = _separator(text) + codex_entry()
    try:
        tomllib.loads(text + addition)
    except tomllib.TOMLDecodeError as error:
        raise RegistrationError(f"can't add [mcp_servers.{SERVER}] to {config}: {error}") from error
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open("a", encoding="utf-8") as file:
        file.write(addition)
    return f"in {config}"


def _separator(text: str) -> str:
    if not text:
        return ""
    return "\n" if text.endswith("\n") else "\n\n"


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def codex_entry() -> str:
    """Codex's entry for repolore, passing on the variables repolore reads that are set here; Codex drops the rest."""
    passed = [name for name in CODEX_PASSES_ON if os.environ.get(name)]
    return CODEX_ENTRY + (f"env_vars = {json.dumps(passed)}\n" if passed else "")


@dataclass(frozen=True)
class McpClient:
    """An MCP client whose CLI can show how repolore is registered with it, and how to register it."""

    name: str
    command: str
    register: str
    add: Callable[[str], str]


CLAUDE_CODE = McpClient(
    "Claude Code",
    "claude",
    f"`claude mcp add --scope user --transport stdio {SERVER} -- {RELAY}`",
    _claude_add,
)
CODEX = McpClient(
    "Codex", "codex", f"add [mcp_servers.{SERVER}] to ~/.codex/config.toml, as the README shows", _codex_add
)
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
    result = _run([executable, "mcp", "get", SERVER], GET_SECONDS)
    if result.returncode != 0:
        return Registration(found=False, status=_first_line(result))
    status, issue = _field(result.stdout, "Status"), _field(result.stdout, "Issue")
    return Registration(found=True, status=f"{status} ({issue})" if status and issue else status)


def register(client: McpClient) -> str:
    """Register repolore with the client, whose CLI must be on PATH, and say where."""
    executable = shutil.which(client.command)
    if executable is None:
        raise RegistrationError(f"`{client.command}` isn't on PATH")
    return client.add(executable)


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _first_line(result: subprocess.CompletedProcess[str]) -> str | None:
    said = (result.stderr or result.stdout).strip().splitlines()
    return said[0] if said else None


def _field(output: str, name: str) -> str | None:
    prefix = f"{name}:"
    lines = (line.strip() for line in output.splitlines())
    return next((line.removeprefix(prefix).strip() for line in lines if line.startswith(prefix)), None)
