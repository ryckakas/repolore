import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from repolore.config_files import ConfigFileError, add_entry, parse_jsonc

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
NOT_SECRET = ("REPOLORE_DEVICE", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")
PASSED_ON = ("REPOLORE_GITHUB_TOKEN", "HTTPS_PROXY", *NOT_SECRET)
TOOL_SECONDS = 300
MCP_SERVERS = ("mcpServers",)


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

    key: str
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


def named_clients(names: str, available: Sequence[McpClient]) -> list[McpClient]:
    """The clients named in a comma-separated list of their keys."""
    by_key = {client.key: client for client in available}
    keys = [key.strip() for key in names.split(",") if key.strip()]
    if not keys or any(key not in by_key for key in keys):
        raise RegistrationError(f"--clients takes a comma-separated list of {', '.join(by_key)}; got {names!r}")
    return [by_key[key] for key in dict.fromkeys(keys)]


def _on_path(*commands: str) -> Callable[[], bool]:
    return lambda: any(shutil.which(command) is not None for command in commands)


def _folder_exists(folder: Callable[[], Path]) -> Callable[[], bool]:
    return lambda: folder().is_dir()


def _either(first: Callable[[], bool], second: Callable[[], bool]) -> Callable[[], bool]:
    return lambda: first() or second()


def _executable(command: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise RegistrationError(f"`{command}` isn't on PATH")
    return executable


def _add_with_cli(command: list[str]) -> str:
    result = _run(command, ADD_SECONDS)
    if result.returncode != 0:
        said = _first_line(result) or f"`{Path(command[0]).stem} mcp add` exited with {result.returncode}"
        raise RegistrationError(said)
    return "for every project (user scope)"


def _passed_on(names: Sequence[str] = PASSED_ON) -> list[str]:
    return [name for name in names if os.environ.get(name)]


def _env_references() -> dict[str, str]:
    return {name: f"${{env:{name}}}" for name in _passed_on()}


def _claude_add() -> str:
    return _add_with_cli(
        [_executable("claude"), "mcp", "add", "--scope", "user", "--transport", "stdio", SERVER, "--", RELAY]
    )


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
    passed = _passed_on()
    return CODEX_ENTRY + (f"env_vars = {json.dumps(passed)}\n" if passed else "")


def _home_folder(variable: str, *fallback: str) -> Path:
    return Path(os.environ.get(variable) or Path.home().joinpath(*fallback))


def _roaming() -> Path:
    return _home_folder("APPDATA", "AppData", "Roaming")


def _local_app_data() -> Path:
    return _home_folder("LOCALAPPDATA", "AppData", "Local")


def _xdg_config() -> Path:
    return _home_folder("XDG_CONFIG_HOME", ".config")


def _app_support() -> Path:
    return Path.home() / "Library" / "Application Support"


def _app_config(name: str) -> Path:
    if sys.platform == "win32":
        return _roaming() / name
    return _app_support() / name if sys.platform == "darwin" else _xdg_config() / name


def _json_entries(user: Path, projects: Sequence[Path], *server_keys: tuple[str, ...]) -> list[Entry]:
    entries = list(_json_file(user, "user", True, server_keys))
    for project in projects:
        if project.resolve() != user.resolve():
            entries += _json_file(project, "project", False, server_keys)
    return entries


def _json_file(
    path: Path, scope: str, for_every_project: bool, server_keys: Sequence[tuple[str, ...]]
) -> Iterator[Entry]:
    settings = _read_config(path, _jsonc_settings)
    for keys in server_keys:
        for name, command in _commands(_nested(settings, keys)):
            yield Entry(name, command, scope, f"delete {name} from {path}", for_every_project)


def _jsonc_settings(text: str) -> Any:
    return parse_jsonc(text) if text.strip() else {}


def _nested(settings: dict[str, Any], keys: tuple[str, ...]) -> object:
    value: object = settings
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _json_client(
    key: str,
    name: str,
    present: Callable[[], bool],
    file: Callable[[], Path],
    project_files: Sequence[str],
    servers_at: tuple[str, ...],
    entry: Callable[[], dict[str, Any]],
    listed_at: Sequence[tuple[str, ...]] = (),
    then: str = "",
) -> McpClient:
    """A client with no usable command for registering a server, whose settings file gets the entry directly."""

    def entries() -> list[Entry]:
        return _json_entries(file(), [Path.cwd() / path for path in project_files], *(listed_at or (servers_at,)))

    def add() -> str:
        path = file()
        try:
            add_entry(path, servers_at, SERVER, entry())
        except ConfigFileError as error:
            raise RegistrationError(str(error)) from error
        return f"in {path}{then}"

    return McpClient(key, name, present, entries, add, f"`repolore setup --clients {key}`")


def _relay_path() -> str:
    """Where repolore-mcp is: an app started from a dock or the Start menu may not have the tool folder on its PATH."""
    relay = _executable(RELAY)
    if Path(relay).is_relative_to(sys.prefix):
        raise RegistrationError(f"{relay} is inside a checkout or temporary environment; run this from a tool install")
    return relay


def _gemini_home() -> Path:
    return _home_folder("GEMINI_CLI_HOME") / ".gemini"


def _gemini_entry() -> dict[str, Any]:
    return {"command": RELAY, "args": [], "env": {name: f"${name}" for name in _passed_on()}}


def _claude_desktop_folder() -> Path:
    if sys.platform == "win32":
        packages = sorted((_local_app_data() / "Packages").glob("Claude_*"))
        return packages[0] / "LocalCache" / "Roaming" / "Claude" if packages else _roaming() / "Claude"
    return _app_config("Claude")


def _claude_desktop_entry() -> dict[str, Any]:
    return {"command": _relay_path(), "args": [], "env": {name: os.environ[name] for name in _passed_on(NOT_SECRET)}}


def _cursor_entry() -> dict[str, Any]:
    return {"type": "stdio", "command": _relay_path(), "args": [], "env": _env_references()}


def _vscode_folder() -> Path:
    return _app_config("Code") / "User"


def _vscode_entry() -> dict[str, Any]:
    return {"type": "stdio", "command": _relay_path()}


def _zed_config() -> Path:
    if sys.platform == "win32":
        return _roaming() / "Zed"
    return (Path.home() / ".config" if sys.platform == "darwin" else _xdg_config()) / "zed"


def _zed_data() -> Path:
    if sys.platform == "win32":
        return _local_app_data() / "Zed"
    return (
        _app_support() / "Zed" if sys.platform == "darwin" else _home_folder("XDG_DATA_HOME", ".local", "share") / "zed"
    )


def _zed_entry() -> dict[str, Any]:
    return {"command": _relay_path(), "args": [], "env": {}, "timeout": TOOL_SECONDS}


def _opencode_config() -> Path:
    folder = _xdg_config() / "opencode"
    commented = folder / "opencode.jsonc"
    return commented if commented.exists() else folder / "opencode.json"


def _opencode_entry() -> dict[str, Any]:
    return {"type": "local", "command": [_relay_path()], "enabled": True, "timeout": TOOL_SECONDS * 1000}


def _devin_config() -> Path:
    return (_roaming() if sys.platform == "win32" else _xdg_config()) / "devin" / "mcp_config.json"


def _devin_entries() -> list[Entry]:
    projects = [Path.cwd() / ".devin" / "mcp_config.json", Path.cwd() / ".devin" / "mcp_config.local.json"]
    return _json_entries(_devin_config(), projects, MCP_SERVERS)


def _devin_add() -> str:
    flags = [part for name, value in _env_references().items() for part in ("-e", f"{name}={value}")]
    return _add_with_cli([_executable("devin"), "mcp", "add", "-s", "user", *flags, SERVER, "--", _relay_path()])


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
    parts = [command] if isinstance(command, str) else command if isinstance(command, list) else []
    return tuple(str(part) for part in [*parts, *(args if isinstance(args, list) else [])])


CLAUDE_CODE = McpClient(
    "claude",
    "Claude Code",
    _on_path("claude"),
    _claude_entries,
    _claude_add,
    f"`claude mcp add --scope user --transport stdio {SERVER} -- {RELAY}`, or install the plugin as the README shows",
    _claude_status,
)
CODEX = McpClient(
    "codex",
    "Codex",
    _on_path("codex"),
    _codex_entries,
    _codex_add,
    f"add [mcp_servers.{SERVER}] to ~/.codex/config.toml, as the README shows",
)
# Not `gemini mcp add`: it saves the settings it loaded with variables expanded, writing other servers' secrets to disk.
GEMINI = _json_client(
    "gemini",
    "Gemini CLI",
    _on_path("gemini"),
    lambda: _gemini_home() / "settings.json",
    (".gemini/settings.json",),
    MCP_SERVERS,
    _gemini_entry,
)
CLAUDE_DESKTOP = _json_client(
    "claude-desktop",
    "Claude Desktop",
    _folder_exists(_claude_desktop_folder),
    lambda: _claude_desktop_folder() / "claude_desktop_config.json",
    (),
    MCP_SERVERS,
    _claude_desktop_entry,
    then="; quit and restart Claude Desktop to load it",
)
CURSOR = _json_client(
    "cursor",
    "Cursor",
    _either(_folder_exists(lambda: Path.home() / ".cursor"), _on_path("agent")),
    lambda: Path.home() / ".cursor" / "mcp.json",
    (".cursor/mcp.json",),
    MCP_SERVERS,
    _cursor_entry,
)
VSCODE = _json_client(
    "vscode",
    "VS Code",
    _either(_on_path("code"), _folder_exists(_vscode_folder)),
    lambda: _vscode_folder() / "mcp.json",
    (".vscode/mcp.json",),
    ("servers",),
    _vscode_entry,
)
ZED = _json_client(
    "zed",
    "Zed",
    _folder_exists(_zed_data),
    lambda: _zed_config() / "settings.json",
    (".zed/settings.json",),
    ("context_servers",),
    _zed_entry,
)
OPENCODE = _json_client(
    "opencode",
    "OpenCode",
    _on_path("opencode", "opencode2"),
    _opencode_config,
    ("opencode.json", "opencode.jsonc"),
    ("mcp",),
    _opencode_entry,
    listed_at=(("mcp",), ("mcp", "servers")),
)
DEVIN = McpClient(
    "devin", "Devin CLI", _on_path("devin"), _devin_entries, _devin_add, "`repolore setup --clients devin`"
)
CLIENTS = (CLAUDE_CODE, CODEX, GEMINI, CLAUDE_DESKTOP, CURSOR, VSCODE, ZED, OPENCODE, DEVIN)


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _first_line(result: subprocess.CompletedProcess[str]) -> str | None:
    said = (result.stderr or result.stdout).strip().splitlines()
    return said[0] if said else None


def _field(output: str, name: str) -> str | None:
    prefix = f"{name}:"
    lines = (line.strip() for line in output.splitlines())
    return next((line.removeprefix(prefix).strip() for line in lines if line.startswith(prefix)), None)
