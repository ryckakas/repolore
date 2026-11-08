import json
import os
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

SERVER = "repolore"
RELAY = "repolore-mcp"
# `claude mcp get repolore` doesn't find the server the plugin provides: Claude Code names it this way.
PLUGIN_SERVER = f"plugin:{SERVER}:{SERVER}"
GET_SECONDS = 120
ADD_SECONDS = 60
LIST_SECONDS = 60
CODEX_ENTRY = f"""[mcp_servers.{SERVER}]
command = "{RELAY}"
startup_timeout_sec = 60
tool_timeout_sec = 300
"""
CODEX_PASSES_ON = (
    "REPOLORE_GITHUB_TOKEN",
    "HTTPS_PROXY",
    "REPOLORE_DEVICE",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_CACHE_HOME",
)


class RegistrationError(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    """A server registered with a client: what it runs, where it's registered and how to remove it."""

    name: str
    command: tuple[str, ...]
    scope: str
    remove: str
    for_every_project: bool = True

    @property
    def alias(self) -> str:
        """ " as <name>" when the client knows repolore under another name, such as the plugin's."""
        return "" if self.name == SERVER else f" as {self.name}"

    @property
    def is_repolore(self) -> bool:
        return self.name in (SERVER, PLUGIN_SERVER) or any(PureWindowsPath(part).stem == RELAY for part in self.command)

    @property
    def command_is_gone(self) -> bool:
        """An absolute command that no longer exists; a bare name depends on the client's PATH, which we can't see."""
        return bool(self.command) and Path(self.command[0]).is_absolute() and shutil.which(self.command[0]) is None


@dataclass(frozen=True)
class McpClient:
    """An MCP client: whether it's here, what it has registered, how to register repolore, and how to check it."""

    name: str
    present: Callable[[], bool]
    entries: Callable[[], list[Entry]]
    add: Callable[[], str]
    register: str
    health: Callable[[str], str | None] | None = None


def repolore_entries(client: McpClient) -> list[Entry]:
    """Every registration of repolore with the client, under any name."""
    return [entry for entry in client.entries() if entry.is_repolore]


def is_connected(status: str | None) -> bool:
    """Whether a client's status says it reached the server; no status line says nothing against it."""
    return status is None or ("Connected" in status and "fail" not in status.lower())


def _on_path(command: str) -> Callable[[], bool]:
    return lambda: shutil.which(command) is not None


def _executable(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise RegistrationError(f"`{command}` isn't on PATH")
    return executable


def _claude_add() -> str:
    command = [_executable("claude"), "mcp", "add", "--scope", "user", "--transport", "stdio", SERVER, "--", RELAY]
    result = _run(command, ADD_SECONDS)
    if result.returncode != 0:
        raise RegistrationError(_first_line(result) or f"`claude mcp add` exited with {result.returncode}")
    return "for every project (user scope)"


def _claude_entries() -> list[Entry]:
    config = _read_config(_claude_config(), json.loads)
    project = _read_config(Path.cwd() / ".mcp.json", json.loads)
    return [
        *_claude_scope(config.get("mcpServers"), "user", for_every_project=True),
        *_claude_scope(_this_folder(config).get("mcpServers"), "local", for_every_project=False),
        *_claude_scope(project.get("mcpServers"), "project", for_every_project=False),
        *_plugin_entries(),
    ]


def _claude_config() -> Path:
    folder = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(folder) / ".claude.json" if folder else Path.home() / ".claude.json"


def _claude_scope(servers: object, scope: str, for_every_project: bool) -> Iterator[Entry]:
    for name, command in _commands(servers):
        yield Entry(name, command, scope, f"`claude mcp remove {name} --scope {scope}`", for_every_project)


def _this_folder(config: dict[str, Any]) -> dict[str, Any]:
    projects = config.get("projects")
    if not isinstance(projects, dict):
        return {}
    here = _comparable(Path.cwd())
    return next(
        (
            settings
            for folder, settings in projects.items()
            if _comparable(folder) == here and isinstance(settings, dict)
        ),
        {},
    )


def _comparable(folder: str | Path) -> str:
    return os.path.normcase(os.path.abspath(folder))


def _plugin_entries() -> Iterator[Entry]:
    for plugin in _enabled_plugins():
        plugin_id = str(plugin.get("id", ""))
        plugin_name = plugin_id.partition("@")[0]
        for server, command in _commands(plugin.get("mcpServers")):
            yield Entry(
                f"plugin:{plugin_name}:{server}",
                command,
                "plugin",
                f"`claude plugin uninstall {plugin_id} --scope {plugin.get('scope')}`",
                plugin.get("scope") == "user",
            )


def _enabled_plugins() -> list[dict[str, Any]]:
    result = _run([_executable("claude"), "plugin", "list", "--json"], LIST_SECONDS)
    if result.returncode != 0:
        said = _first_line(result) or f"exited with {result.returncode}"
        raise RegistrationError(f"`claude plugin list --json` failed: {said}")
    try:
        plugins = json.loads(result.stdout)
    except ValueError as error:
        raise RegistrationError(f"`claude plugin list --json` printed something else: {error}") from error
    if not isinstance(plugins, list):
        raise RegistrationError("`claude plugin list --json` didn't print a list")
    return [plugin for plugin in plugins if isinstance(plugin, dict) and _applies_here(plugin)]


def _applies_here(plugin: dict[str, Any]) -> bool:
    """`enabled` comes from merged settings, so an install that belongs to another folder can read as enabled here."""
    folder = plugin.get("projectPath")
    return bool(plugin.get("enabled")) and (not folder or _comparable(folder) == _comparable(Path.cwd()))


def _claude_status(name: str) -> str | None:
    """What Claude Code says about a server it has; it starts the server to find out, which can take a while."""
    result = _run([_executable("claude"), "mcp", "get", name], GET_SECONDS)
    if result.returncode != 0:
        return _first_line(result) or f"`claude mcp get {name}` exited with {result.returncode}"
    status, issue = _field(result.stdout, "Status"), _field(result.stdout, "Issue")
    return f"{status} ({issue})" if status and issue else status


def _codex_add() -> str:
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


def _codex_entries() -> list[Entry]:
    user = codex_home() / "config.toml"
    project = Path.cwd() / ".codex" / "config.toml"
    entries = list(_codex_file(user, "user", for_every_project=True))
    if project.resolve() != user.resolve():
        entries += _codex_file(project, "project", for_every_project=False)
    return entries


def _codex_file(path: Path, scope: str, for_every_project: bool) -> Iterator[Entry]:
    for name, command in _commands(_read_config(path, tomllib.loads).get("mcp_servers")):
        yield Entry(name, command, scope, f"delete [mcp_servers.{name}] from {path}", for_every_project)


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def codex_entry() -> str:
    """Codex's entry for repolore, passing on the variables repolore reads that are set here; Codex drops the rest."""
    passed = [name for name in CODEX_PASSES_ON if os.environ.get(name)]
    return CODEX_ENTRY + (f"env_vars = {json.dumps(passed)}\n" if passed else "")


def _read_config(path: Path, parse: Callable[[str], Any]) -> dict[str, Any]:
    """A client's settings, empty when it has no such file; a file that can't be read raises, because what it says
    decides whether registering is safe."""
    try:
        settings = parse(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except ValueError as error:
        raise RegistrationError(f"can't read {path}: {error}") from error
    if not isinstance(settings, dict):
        raise RegistrationError(f"can't read {path}: it isn't a set of named settings")
    return settings


def _commands(servers: object) -> Iterator[tuple[str, tuple[str, ...]]]:
    if isinstance(servers, dict):
        for name, server in servers.items():
            yield str(name), _command(server)


def _command(server: object) -> tuple[str, ...]:
    if not isinstance(server, dict):
        return ()
    command, args = server.get("command"), server.get("args")
    parts = [command] if isinstance(command, str) else []
    return tuple(parts + ([str(arg) for arg in args] if isinstance(args, list) else []))


CLAUDE_CODE = McpClient(
    "Claude Code",
    _on_path("claude"),
    _claude_entries,
    _claude_add,
    f"`claude mcp add --scope user --transport stdio {SERVER} -- {RELAY}`, or install the plugin as the README shows",
    _claude_status,
)
CODEX = McpClient(
    "Codex",
    _on_path("codex"),
    _codex_entries,
    _codex_add,
    f"add [mcp_servers.{SERVER}] to ~/.codex/config.toml, as the README shows",
)
CLIENTS = (CLAUDE_CODE, CODEX)


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _first_line(result: subprocess.CompletedProcess[str]) -> str | None:
    said = (result.stderr or result.stdout).strip().splitlines()
    return said[0] if said else None


def _field(output: str, name: str) -> str | None:
    prefix = f"{name}:"
    lines = (line.strip() for line in output.splitlines())
    return next((line.removeprefix(prefix).strip() for line in lines if line.startswith(prefix)), None)
