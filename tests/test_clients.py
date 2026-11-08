import json
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from repolore.clients import (
    ADD_SECONDS,
    CLAUDE_CODE,
    CLAUDE_DESKTOP,
    CLIENTS,
    CODEX,
    CURSOR,
    DEVIN,
    GEMINI,
    GET_SECONDS,
    LIST_SECONDS,
    OPENCODE,
    PASSED_ON,
    PLUGIN_SERVER,
    RELAY,
    VSCODE,
    ZED,
    Entry,
    McpClient,
    RegistrationError,
    codex_home,
    is_connected,
    named_clients,
    repolore_entries,
)
from repolore.config_files import parse_jsonc

CLAUDE_HEALTHY = """repolore:
  Scope: User config (available in all your projects)
  Status: ✓ Connected
  Type: stdio
  Command: repolore-mcp
"""
CLAUDE_TIMED_OUT = """repolore:
  Scope: User config (available in all your projects)
  Status: ! Connected · tools fetch failed
  Issue: Request timed out
  Type: stdio
"""
RELAY_SERVER = {"command": "repolore-mcp", "args": []}
PLUGIN = {"id": "repolore@repolore", "enabled": True, "scope": "user", "mcpServers": {"repolore": RELAY_SERVER}}


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    monkeypatch.setattr(shutil, "which", lambda command: f"/bin/{command}")
    return []


def answer(
    monkeypatch: pytest.MonkeyPatch,
    runs: list[list[str]],
    timeout: int,
    code: int,
    stdout: str = "",
    stderr: str = "",
) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == timeout
        runs.append(command)
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", run)


@pytest.fixture
def claude_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runs: list[list[str]]) -> Path:
    config = tmp_path / "config"
    project = tmp_path / "project"
    config.mkdir()
    project.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    monkeypatch.chdir(project)
    answer(monkeypatch, runs, LIST_SECONDS, 0, "[]")
    return config


def write_json(path: Path, content: object) -> None:
    path.write_text(json.dumps(content), encoding="utf-8")


def with_plugins(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]], *plugins: dict[str, Any]) -> None:
    answer(monkeypatch, runs, LIST_SECONDS, 0, json.dumps(plugins))


def test_claude_code_with_nothing_registered_has_no_entries(claude_dir: Path, runs: list[list[str]]) -> None:
    assert CLAUDE_CODE.entries() == []
    assert runs == [["/bin/claude", "plugin", "list", "--json"]]
    assert not claude_dir.joinpath(".claude.json").exists()


def test_claude_code_lists_a_user_scope_server(claude_dir: Path) -> None:
    write_json(claude_dir / ".claude.json", {"mcpServers": {"repolore": {"command": "repolore-mcp", "args": ["-x"]}}})

    assert CLAUDE_CODE.entries() == [
        Entry("repolore", ("repolore-mcp", "-x"), "user", "`claude mcp remove repolore --scope user`")
    ]


def test_claude_code_lists_only_this_folders_local_servers(claude_dir: Path) -> None:
    here = Path.cwd().as_posix()
    projects = {here: {"mcpServers": {"mine": RELAY_SERVER}}, "/elsewhere": {"mcpServers": {"theirs": RELAY_SERVER}}}
    write_json(claude_dir / ".claude.json", {"projects": projects})

    assert CLAUDE_CODE.entries() == [
        Entry("mine", ("repolore-mcp",), "local", "`claude mcp remove mine --scope local`", for_every_project=False)
    ]


@pytest.mark.parametrize("projects", ["a list", {"unrelated": "not a mapping"}])
def test_claude_code_ignores_project_settings_of_another_shape(claude_dir: Path, projects: object) -> None:
    write_json(claude_dir / ".claude.json", {"projects": projects})

    assert CLAUDE_CODE.entries() == []


@pytest.mark.usefixtures("claude_dir")
def test_claude_code_lists_this_folders_project_servers() -> None:
    write_json(Path.cwd() / ".mcp.json", {"mcpServers": {"repolore": RELAY_SERVER}})

    assert CLAUDE_CODE.entries() == [
        Entry(
            "repolore",
            ("repolore-mcp",),
            "project",
            "`claude mcp remove repolore --scope project`",
            for_every_project=False,
        )
    ]


@pytest.mark.usefixtures("claude_dir")
def test_claude_code_lists_the_plugins_server_by_its_own_name(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    with_plugins(monkeypatch, runs, PLUGIN)

    assert CLAUDE_CODE.entries() == [
        Entry(PLUGIN_SERVER, ("repolore-mcp",), "plugin", "`claude plugin uninstall repolore@repolore --scope user`")
    ]


@pytest.mark.usefixtures("claude_dir")
def test_a_local_plugin_install_is_removed_from_its_own_scope(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    with_plugins(monkeypatch, runs, {**PLUGIN, "scope": "local", "projectPath": str(Path.cwd())})

    assert [entry.remove for entry in CLAUDE_CODE.entries()] == [
        "`claude plugin uninstall repolore@repolore --scope local`"
    ]


@pytest.mark.usefixtures("claude_dir")
def test_a_plugin_installed_for_another_folder_isnt_listed_here(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]], tmp_path: Path
) -> None:
    elsewhere = {**PLUGIN, "scope": "local", "projectPath": str(tmp_path / "elsewhere")}
    with_plugins(monkeypatch, runs, elsewhere, PLUGIN)

    assert [entry.scope for entry in CLAUDE_CODE.entries()] == ["plugin"]
    assert [entry.for_every_project for entry in CLAUDE_CODE.entries()] == [True]


@pytest.mark.usefixtures("claude_dir")
def test_a_disabled_plugin_has_no_entries(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    with_plugins(monkeypatch, runs, {**PLUGIN, "enabled": False})

    assert CLAUDE_CODE.entries() == []


@pytest.mark.usefixtures("claude_dir")
def test_a_plugin_installed_for_a_project_isnt_for_every_project(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    with_plugins(monkeypatch, runs, {**PLUGIN, "scope": "project"})

    assert [entry.for_every_project for entry in CLAUDE_CODE.entries()] == [False]


def test_every_claude_code_scope_is_listed_together(
    claude_dir: Path, monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    write_json(claude_dir / ".claude.json", {"mcpServers": {"repolore": RELAY_SERVER}})
    write_json(Path.cwd() / ".mcp.json", {"mcpServers": {"repolore": RELAY_SERVER}})
    with_plugins(monkeypatch, runs, PLUGIN)

    assert [entry.scope for entry in repolore_entries(CLAUDE_CODE)] == ["user", "project", "plugin"]


@pytest.mark.parametrize("value", [None, ""])
def test_claude_codes_config_defaults_to_the_home_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runs: list[list[str]], value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    else:
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", value)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    answer(monkeypatch, runs, LIST_SECONDS, 0, "[]")
    write_json(tmp_path / ".claude.json", {"mcpServers": {"repolore": RELAY_SERVER}})

    assert [entry.name for entry in CLAUDE_CODE.entries()] == ["repolore"]


@pytest.mark.parametrize(
    ("content", "complaint"),
    [("{ not json", "can't read"), ("[1, 2]", "isn't a set of named settings")],
)
def test_an_unreadable_claude_config_is_an_error_naming_it(claude_dir: Path, content: str, complaint: str) -> None:
    (claude_dir / ".claude.json").write_text(content, encoding="utf-8")

    with pytest.raises(RegistrationError, match=complaint) as caught:
        CLAUDE_CODE.entries()

    assert str(claude_dir / ".claude.json") in str(caught.value)


@pytest.mark.usefixtures("claude_dir")
def test_a_project_file_with_a_byte_order_mark_is_read() -> None:
    (Path.cwd() / ".mcp.json").write_text(json.dumps({"mcpServers": {"x": RELAY_SERVER}}), encoding="utf-8-sig")

    assert [entry.name for entry in CLAUDE_CODE.entries()] == ["x"]


@pytest.mark.parametrize("servers", ["a string", ["a", "list"], None])
def test_claude_servers_of_another_shape_are_no_entries(claude_dir: Path, servers: object) -> None:
    write_json(claude_dir / ".claude.json", {"mcpServers": servers})

    assert CLAUDE_CODE.entries() == []


def test_a_server_without_a_command_has_an_empty_one(claude_dir: Path) -> None:
    servers = {"remote": {"type": "http", "url": "https://example.com"}, "odd": "text"}
    write_json(claude_dir / ".claude.json", {"mcpServers": servers})

    assert [entry.command for entry in CLAUDE_CODE.entries()] == [(), ()]


@pytest.mark.usefixtures("claude_dir")
@pytest.mark.parametrize(
    ("code", "stdout", "stderr", "complaint"),
    [
        (1, "", "unknown command 'plugin'\nmore", "failed: unknown command 'plugin'$"),
        (3, "", "", "failed: exited with 3$"),
        (0, "Loading...", "", "printed something else"),
        (0, "{}", "", "didn't print a list"),
    ],
)
def test_a_plugin_listing_that_fails_is_an_error(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]], code: int, stdout: str, stderr: str, complaint: str
) -> None:
    answer(monkeypatch, runs, LIST_SECONDS, code, stdout, stderr)

    with pytest.raises(RegistrationError, match=complaint):
        CLAUDE_CODE.entries()


@pytest.mark.usefixtures("claude_dir")
def test_a_plugin_listing_ignores_what_isnt_a_plugin(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    answer(monkeypatch, runs, LIST_SECONDS, 0, json.dumps(["text", PLUGIN]))

    assert [entry.name for entry in CLAUDE_CODE.entries()] == [PLUGIN_SERVER]


@pytest.mark.parametrize(
    ("name", "command", "is_repolore"),
    [
        ("repolore", ("something-else",), True),
        (PLUGIN_SERVER, ("something-else",), True),
        ("mine", ("repolore-mcp",), True),
        ("mine", ("/home/me/.local/bin/repolore-mcp",), True),
        ("mine", ("C:\\Users\\me\\.local\\bin\\repolore-mcp.exe",), True),
        ("mine", ("uv", "run", "--directory", "/code/repolore", "--no-sync", "repolore-mcp"), True),
        ("mine", ("uvx", "--managed-python", "--python", "3.12", "repolore-mcp"), True),
        ("mine", ("repolore",), False),
        ("mine", ("not-repolore-mcp",), False),
        ("mine", (), False),
    ],
)
def test_an_entry_is_repolores_by_name_or_by_what_it_runs(
    name: str, command: tuple[str, ...], is_repolore: bool
) -> None:
    assert Entry(name, command, "user", "x").is_repolore is is_repolore


def test_repolore_entries_leaves_other_servers_out() -> None:
    ours = Entry("repolore", ("repolore-mcp",), "user", "x")
    others = Entry("other", ("other-cmd",), "user", "x")
    client = McpClient("client", "Client", lambda: True, lambda: [others, ours], lambda: "added", "add it")

    assert repolore_entries(client) == [ours]


def test_an_entry_under_repolores_own_name_has_no_alias() -> None:
    assert Entry("repolore", (), "user", "x").alias == ""


def test_an_entry_under_another_name_has_one() -> None:
    assert Entry(PLUGIN_SERVER, (), "plugin", "x").alias == " as plugin:repolore:repolore"


def test_an_absolute_command_that_exists_isnt_gone(tmp_path: Path) -> None:
    relay = tmp_path / "repolore-mcp.exe"
    relay.touch(mode=0o755)

    assert not Entry("repolore", (str(relay),), "user", "x").command_is_gone


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows finds a command without its extension")
def test_an_absolute_command_without_its_extension_isnt_gone(tmp_path: Path) -> None:
    (tmp_path / "repolore-mcp.exe").touch()

    assert not Entry("repolore", (str(tmp_path / "repolore-mcp"),), "user", "x").command_is_gone


def test_an_absolute_command_that_doesnt_exist_is_gone(tmp_path: Path) -> None:
    assert Entry("repolore", (str(tmp_path / "repolore-mcp"),), "user", "x").command_is_gone


@pytest.mark.parametrize("command", [("repolore-mcp",), ()])
def test_a_bare_or_empty_command_is_never_gone(command: tuple[str, ...]) -> None:
    assert not Entry("repolore", command, "user", "x").command_is_gone


def claude_status(name: str) -> str | None:
    assert CLAUDE_CODE.health is not None
    return CLAUDE_CODE.health(name)


def test_claude_codes_status_is_what_it_says(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    answer(monkeypatch, runs, GET_SECONDS, 0, CLAUDE_HEALTHY)

    assert claude_status(PLUGIN_SERVER) == "✓ Connected"
    assert runs == [["/bin/claude", "mcp", "get", PLUGIN_SERVER]]


def test_a_failed_health_check_keeps_its_issue(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    answer(monkeypatch, runs, GET_SECONDS, 0, CLAUDE_TIMED_OUT)

    assert claude_status("repolore") == "! Connected · tools fetch failed (Request timed out)"


def test_output_without_a_status_line_has_no_status(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    answer(monkeypatch, runs, GET_SECONDS, 0, "repolore:\n  Type: stdio\n")

    assert claude_status("repolore") is None


def test_a_refused_status_says_what_claude_said(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    answer(monkeypatch, runs, GET_SECONDS, 1, "", "No MCP server named repolore")

    assert claude_status("repolore") == "No MCP server named repolore"


def test_a_silently_refused_status_names_the_exit_code(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    answer(monkeypatch, runs, GET_SECONDS, 2)

    assert claude_status("repolore") == "`claude mcp get repolore` exited with 2"


@pytest.mark.parametrize(
    ("status", "connected"),
    [
        (None, True),
        ("✓ Connected", True),
        ("! Connected · tools fetch failed (Request timed out)", False),
        ("✗ Failed to connect", False),
        ("Pending", False),
    ],
)
def test_a_status_is_connected_only_when_it_says_so(status: str | None, connected: bool) -> None:
    assert is_connected(status) is connected


@pytest.mark.usefixtures("runs")
@pytest.mark.parametrize(("client", "command"), [(CLAUDE_CODE, "claude"), (CODEX, "codex")])
def test_a_client_is_present_when_its_cli_is_on_path(client: McpClient, command: str) -> None:
    assert client.present()
    assert shutil.which(command) == f"/bin/{command}"


@pytest.mark.parametrize("client", [CLAUDE_CODE, CODEX])
def test_a_client_is_absent_when_its_cli_isnt(monkeypatch: pytest.MonkeyPatch, client: McpClient) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    assert not client.present()


@pytest.fixture
def codex_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for name in PASSED_ON:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_codex_without_a_config_has_no_entries(codex_dir: Path) -> None:
    assert CODEX.entries() == []
    assert not (codex_dir / "config.toml").exists()


def test_codex_lists_its_user_servers(codex_dir: Path) -> None:
    config = codex_dir / "config.toml"
    config.write_text('[mcp_servers.repolore]\ncommand = "repolore-mcp"\nargs = ["-x"]\n', encoding="utf-8")

    assert CODEX.entries() == [
        Entry("repolore", ("repolore-mcp", "-x"), "user", f"delete [mcp_servers.repolore] from {config}")
    ]


def test_codex_lists_this_folders_project_servers(codex_dir: Path) -> None:
    project = codex_dir / ".codex" / "config.toml"
    project.parent.mkdir()
    project.write_text('[mcp_servers.mine]\ncommand = "repolore-mcp"\n', encoding="utf-8")

    assert CODEX.entries() == [
        Entry(
            "mine", ("repolore-mcp",), "project", f"delete [mcp_servers.mine] from {project}", for_every_project=False
        )
    ]


def test_codex_in_its_own_home_doesnt_count_the_file_twice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / ".codex"
    home.mkdir()
    (home / "config.toml").write_text('[mcp_servers.repolore]\ncommand = "repolore-mcp"\n', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    assert [entry.scope for entry in CODEX.entries()] == ["user"]


def test_an_unreadable_codex_listing_is_an_error_naming_it(codex_dir: Path) -> None:
    (codex_dir / "config.toml").write_text("model = ", encoding="utf-8")

    with pytest.raises(RegistrationError, match="can't read") as caught:
        CODEX.entries()

    assert str(codex_dir / "config.toml") in str(caught.value)


def adding(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]], code: int, stderr: str = "") -> None:
    answer(monkeypatch, runs, ADD_SECONDS, code, "", stderr)


def test_claude_code_registers_repolore_for_every_project(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    adding(monkeypatch, runs, 0)

    assert CLAUDE_CODE.add() == "for every project (user scope)"
    assert runs == [
        ["/bin/claude", "mcp", "add", "--scope", "user", "--transport", "stdio", "repolore", "--", "repolore-mcp"]
    ]


def test_a_refused_claude_registration_says_why(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    adding(monkeypatch, runs, 1, "MCP server repolore already exists in user config\nmore")

    with pytest.raises(RegistrationError, match=r"^MCP server repolore already exists in user config$"):
        CLAUDE_CODE.add()


def test_a_silently_refused_claude_registration_names_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    adding(monkeypatch, runs, 3)

    with pytest.raises(RegistrationError, match="exited with 3"):
        CLAUDE_CODE.add()


def test_a_client_not_on_path_cant_be_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(RegistrationError, match="`claude` isn't on PATH"):
        CLAUDE_CODE.add()


def test_codex_gets_a_config_entry_with_longer_timeouts(codex_dir: Path) -> None:
    assert CODEX.add() == f"in {codex_dir / 'config.toml'}"

    text = (codex_dir / "config.toml").read_text(encoding="utf-8")
    assert text.startswith("[mcp_servers.repolore]\n")
    assert tomllib.loads(text) == {
        "mcp_servers": {"repolore": {"command": "repolore-mcp", "startup_timeout_sec": 60, "tool_timeout_sec": 300}}
    }


def test_codex_entry_follows_the_existing_config(codex_dir: Path) -> None:
    (codex_dir / "config.toml").write_text('model = "o3"\n[mcp_servers.other]\ncommand = "other"', encoding="utf-8")

    CODEX.add()

    text = (codex_dir / "config.toml").read_text(encoding="utf-8")
    assert text.startswith('model = "o3"\n[mcp_servers.other]\ncommand = "other"\n\n[mcp_servers.repolore]\n')
    assert tomllib.loads(text)["mcp_servers"].keys() == {"other", "repolore"}


def test_codex_entry_passes_on_the_variables_repolore_reads(codex_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOLORE_GITHUB_TOKEN", "token")
    monkeypatch.setenv("XDG_DATA_HOME", str(codex_dir))
    monkeypatch.setenv("HTTPS_PROXY", "")

    CODEX.add()

    config = tomllib.loads((codex_dir / "config.toml").read_text(encoding="utf-8"))
    assert config["mcp_servers"]["repolore"]["env_vars"] == ["REPOLORE_GITHUB_TOKEN", "XDG_DATA_HOME"]


def test_codex_already_having_repolore_is_left_alone(codex_dir: Path) -> None:
    original = '[mcp_servers.repolore]\ncommand = "elsewhere"\n'
    (codex_dir / "config.toml").write_text(original, encoding="utf-8")

    with pytest.raises(RegistrationError, match=r"already has \[mcp_servers.repolore\]"):
        CODEX.add()

    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == original


def test_codex_servers_in_an_inline_table_are_left_alone(codex_dir: Path) -> None:
    original = 'mcp_servers = { other = { command = "other" } }\n'
    (codex_dir / "config.toml").write_text(original, encoding="utf-8")

    with pytest.raises(RegistrationError, match=r"can't add \[mcp_servers.repolore\]"):
        CODEX.add()

    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == original


def test_an_unreadable_codex_config_is_left_alone(codex_dir: Path) -> None:
    (codex_dir / "config.toml").write_text("model = ", encoding="utf-8")

    with pytest.raises(RegistrationError, match="can't read"):
        CODEX.add()

    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == "model = "


@pytest.mark.parametrize("value", [None, ""])
def test_codex_home_defaults_to_the_home_folder(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("CODEX_HOME", raising=False)
    else:
        monkeypatch.setenv("CODEX_HOME", value)

    assert codex_home() == Path.home() / ".codex"


RELAY_AT = f"/tools/{RELAY}"


@dataclass
class Machine:
    home: Path
    project: Path
    on_path: set[str] = field(default_factory=lambda: {RELAY})

    def path(self, relative: str) -> Path:
        return self.home.joinpath(*relative.split("/"))

    def write(self, relative: str, text: str) -> Path:
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Machine:
    made = Machine(tmp_path / "home", tmp_path / "project")
    made.home.mkdir()
    made.project.mkdir()
    monkeypatch.setattr(Path, "home", lambda: made.home)
    monkeypatch.chdir(made.project)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda command: f"/tools/{command}" if command in made.on_path else None)
    for name in (*PASSED_ON, "APPDATA", "LOCALAPPDATA", "GEMINI_CLI_HOME"):
        monkeypatch.delenv(name, raising=False)
    return made


DESKTOP_FILE = "claude_desktop_config.json"
USER_FILES = [
    (CLAUDE_DESKTOP, "linux", f".config/Claude/{DESKTOP_FILE}", ("mcpServers",)),
    (CLAUDE_DESKTOP, "darwin", f"Library/Application Support/Claude/{DESKTOP_FILE}", ("mcpServers",)),
    (CLAUDE_DESKTOP, "win32", f"AppData/Roaming/Claude/{DESKTOP_FILE}", ("mcpServers",)),
    (GEMINI, "linux", ".gemini/settings.json", ("mcpServers",)),
    (GEMINI, "win32", ".gemini/settings.json", ("mcpServers",)),
    (CURSOR, "linux", ".cursor/mcp.json", ("mcpServers",)),
    (CURSOR, "win32", ".cursor/mcp.json", ("mcpServers",)),
    (VSCODE, "linux", ".config/Code/User/mcp.json", ("servers",)),
    (VSCODE, "darwin", "Library/Application Support/Code/User/mcp.json", ("servers",)),
    (VSCODE, "win32", "AppData/Roaming/Code/User/mcp.json", ("servers",)),
    (ZED, "linux", ".config/zed/settings.json", ("context_servers",)),
    (ZED, "darwin", ".config/zed/settings.json", ("context_servers",)),
    (ZED, "win32", "AppData/Roaming/Zed/settings.json", ("context_servers",)),
    (OPENCODE, "linux", ".config/opencode/opencode.json", ("mcp",)),
    (OPENCODE, "win32", ".config/opencode/opencode.json", ("mcp",)),
]


def added_entry(client: McpClient, *keys: str) -> dict[str, Any]:
    path = Path(client.add().removeprefix("in ").partition(";")[0])
    settings = json.loads(path.read_bytes())
    for key in keys:
        settings = settings[key]
    return settings["repolore"]


@pytest.mark.parametrize(("client", "platform", "relative", "keys"), USER_FILES)
def test_a_file_client_writes_its_own_file_and_finds_what_it_wrote(
    machine: Machine,
    monkeypatch: pytest.MonkeyPatch,
    client: McpClient,
    platform: str,
    relative: str,
    keys: tuple[str, ...],
) -> None:
    monkeypatch.setattr(sys, "platform", platform)

    assert client.add().startswith(f"in {machine.path(relative)}")

    settings = json.loads(machine.path(relative).read_bytes())
    for key in keys:
        settings = settings[key]
    assert "repolore" in settings
    assert [(entry.name, entry.scope, entry.for_every_project) for entry in repolore_entries(client)] == [
        ("repolore", "user", True)
    ]


@pytest.mark.usefixtures("machine")
@pytest.mark.parametrize(
    ("client", "keys", "expected"),
    [
        (GEMINI, ("mcpServers",), {"command": RELAY, "args": [], "env": {}}),
        (CLAUDE_DESKTOP, ("mcpServers",), {"command": RELAY_AT, "args": [], "env": {}}),
        (CURSOR, ("mcpServers",), {"type": "stdio", "command": RELAY_AT, "args": [], "env": {}}),
        (VSCODE, ("servers",), {"type": "stdio", "command": RELAY_AT}),
        (ZED, ("context_servers",), {"command": RELAY_AT, "args": [], "env": {}, "timeout": 300}),
        (OPENCODE, ("mcp",), {"type": "local", "command": [RELAY_AT], "enabled": True, "timeout": 300_000}),
    ],
)
def test_each_file_client_gets_the_entry_it_expects(
    client: McpClient, keys: tuple[str, ...], expected: dict[str, Any]
) -> None:
    assert added_entry(client, *keys) == expected


def test_claude_desktop_never_gets_the_token_but_gets_what_isnt_secret(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPOLORE_GITHUB_TOKEN", "secret-token")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pass@proxy")
    monkeypatch.setenv("XDG_DATA_HOME", "/data")
    monkeypatch.setenv("REPOLORE_DEVICE", "cpu")

    env = added_entry(CLAUDE_DESKTOP, "mcpServers")["env"]

    assert env == {"REPOLORE_DEVICE": "cpu", "XDG_DATA_HOME": "/data"}
    assert b"secret-token" not in machine.path(f".config/Claude/{DESKTOP_FILE}").read_bytes()


def test_cursor_gets_references_to_the_variables_that_are_set(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPOLORE_GITHUB_TOKEN", "secret-token")
    monkeypatch.setenv("HTTPS_PROXY", "")

    assert added_entry(CURSOR, "mcpServers")["env"] == {"REPOLORE_GITHUB_TOKEN": "${env:REPOLORE_GITHUB_TOKEN}"}
    assert b"secret-token" not in machine.path(".cursor/mcp.json").read_bytes()


def test_claude_desktop_says_to_restart_it(machine: Machine) -> None:
    assert CLAUDE_DESKTOP.add() == (
        f"in {machine.path(f'.config/Claude/{DESKTOP_FILE}')}; quit and restart Claude Desktop to load it"
    )


def test_claude_desktops_store_install_wins_over_its_usual_folder(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    machine.path("AppData/Roaming/Claude").mkdir(parents=True)
    packaged = machine.path("AppData/Local/Packages/Claude_abc123/LocalCache/Roaming/Claude")
    packaged.mkdir(parents=True)

    assert CLAUDE_DESKTOP.present()
    assert CLAUDE_DESKTOP.add().startswith(f"in {packaged / DESKTOP_FILE}")
    assert not machine.path(f"AppData/Roaming/Claude/{DESKTOP_FILE}").exists()


def test_windows_folders_follow_their_variables(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(machine.path("elsewhere/roaming")))

    assert CLAUDE_DESKTOP.add().startswith(f"in {machine.path(f'elsewhere/roaming/Claude/{DESKTOP_FILE}')}")


def test_xdg_config_home_moves_the_linux_folders(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(machine.path("xdg")))

    assert ZED.add().startswith(f"in {machine.path('xdg/zed/settings.json')}")


def test_zed_on_a_mac_ignores_xdg_config_home(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(machine.path("xdg")))

    assert ZED.add().startswith(f"in {machine.path('.config/zed/settings.json')}")


def test_a_file_client_keeps_what_the_file_already_has(machine: Machine) -> None:
    original = '{\n  // my theme\n  "theme": "dark",\n}\n'
    path = machine.write(".config/zed/settings.json", original)

    ZED.add()

    text = path.read_text(encoding="utf-8")
    assert "// my theme" in text
    assert '"theme": "dark"' in text
    assert path.with_name("settings.json.repolore-backup").read_text(encoding="utf-8") == original


def test_a_file_client_without_the_relay_on_path_is_refused(machine: Machine) -> None:
    machine.on_path.clear()

    with pytest.raises(RegistrationError, match=f"`{RELAY}` isn't on PATH"):
        CURSOR.add()

    assert not machine.path(".cursor/mcp.json").exists()


def test_a_relay_inside_this_environment_isnt_recorded(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "prefix", "/tools")

    with pytest.raises(RegistrationError, match="checkout or temporary environment"):
        CURSOR.add()

    assert not machine.path(".cursor/mcp.json").exists()


def test_a_settings_file_it_cant_change_is_a_registration_error(machine: Machine) -> None:
    path = machine.write(".cursor/mcp.json", "{ not json")

    with pytest.raises(RegistrationError, match="isn't valid JSON") as caught:
        CURSOR.add()

    assert str(path) in str(caught.value)


def test_an_unreadable_settings_file_fails_the_listing(machine: Machine) -> None:
    path = machine.write(".cursor/mcp.json", "{ not json")

    with pytest.raises(RegistrationError, match="can't read") as caught:
        CURSOR.entries()

    assert str(path) in str(caught.value)


@pytest.mark.usefixtures("machine")
def test_a_client_with_no_settings_has_no_entries() -> None:
    assert CURSOR.entries() == []


def test_a_blank_settings_file_has_no_entries(machine: Machine) -> None:
    machine.write(".cursor/mcp.json", "  \n")

    assert CURSOR.entries() == []


def test_a_settings_file_with_comments_is_listed(machine: Machine) -> None:
    machine.write(
        ".config/zed/settings.json", '{\n  // servers\n  "context_servers": {"mine": {"command": "repolore-mcp",},},\n}'
    )

    assert [entry.name for entry in repolore_entries(ZED)] == ["mine"]


def test_this_folders_project_file_is_listed_as_not_for_every_project(machine: Machine) -> None:
    path = machine.project / ".cursor" / "mcp.json"
    path.parent.mkdir()
    write_json(path, {"mcpServers": {"repolore": RELAY_SERVER}})

    assert CURSOR.entries() == [
        Entry("repolore", ("repolore-mcp",), "project", f"delete repolore from {path}", for_every_project=False)
    ]


def test_a_user_file_isnt_counted_twice_from_its_own_folder(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    write_json(machine.write(".cursor/mcp.json", ""), {"mcpServers": {"repolore": RELAY_SERVER}})
    monkeypatch.chdir(machine.home)

    assert [entry.scope for entry in CURSOR.entries()] == ["user"]


def test_vscode_project_servers_are_under_its_own_key(machine: Machine) -> None:
    path = machine.project / ".vscode" / "mcp.json"
    path.parent.mkdir()
    write_json(path, {"servers": {"repolore": RELAY_SERVER}})

    assert [entry.scope for entry in VSCODE.entries()] == ["project"]


def test_opencode_prefers_an_existing_commented_file(machine: Machine) -> None:
    path = machine.write(".config/opencode/opencode.jsonc", "// mine\n{}\n")

    assert OPENCODE.add() == f"in {path}"
    assert not machine.path(".config/opencode/opencode.json").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"mcp": {"mine": {"type": "local", "command": ["repolore-mcp", "-x"]}}},
        {"mcp": {"servers": {"mine": {"type": "local", "command": ["repolore-mcp", "-x"]}}}},
    ],
)
def test_opencode_servers_are_found_in_both_shapes(machine: Machine, settings: dict[str, Any]) -> None:
    write_json(machine.write(".config/opencode/opencode.json", ""), settings)

    assert [(entry.name, entry.command) for entry in repolore_entries(OPENCODE)] == [("mine", ("repolore-mcp", "-x"))]


@pytest.mark.parametrize("name", ["opencode.json", "opencode.jsonc"])
def test_opencode_project_files_of_either_name_are_listed(machine: Machine, name: str) -> None:
    write_json(machine.project / name, {"mcp": {"repolore": {"command": ["repolore-mcp"]}}})

    assert [entry.scope for entry in repolore_entries(OPENCODE)] == ["project"]


def test_opencode_gets_its_flat_entry_beside_existing_v2_servers(machine: Machine) -> None:
    path = machine.write(".config/opencode/opencode.json", '{\n  "mcp": {\n    "servers": {"other": {}}\n  }\n}\n')

    OPENCODE.add()

    mcp = parse_jsonc(path.read_text(encoding="utf-8"))["mcp"]
    assert mcp["servers"] == {"other": {}}
    assert mcp["repolore"]["command"] == [RELAY_AT]


def test_gemini_gets_references_to_the_variables_that_are_set(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPOLORE_GITHUB_TOKEN", "secret-token")

    assert added_entry(GEMINI, "mcpServers")["env"] == {"REPOLORE_GITHUB_TOKEN": "$REPOLORE_GITHUB_TOKEN"}
    assert b"secret-token" not in machine.path(".gemini/settings.json").read_bytes()


def test_a_refused_devin_registration_names_it(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    machine.on_path.add("devin")
    answer(monkeypatch, [], ADD_SECONDS, 4)

    with pytest.raises(RegistrationError, match="`devin mcp add` exited with 4"):
        DEVIN.add()


def test_gemini_lists_its_user_and_project_files(machine: Machine) -> None:
    write_json(machine.write(".gemini/settings.json", ""), {"mcpServers": {"repolore": RELAY_SERVER}})
    project = machine.project / ".gemini" / "settings.json"
    project.parent.mkdir()
    write_json(project, {"mcpServers": {"mine": RELAY_SERVER}})

    assert [(entry.name, entry.scope) for entry in repolore_entries(GEMINI)] == [
        ("repolore", "user"),
        ("mine", "project"),
    ]


def test_gemini_cli_home_moves_its_settings(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_CLI_HOME", str(machine.path("elsewhere")))
    write_json(machine.write("elsewhere/.gemini/settings.json", ""), {"mcpServers": {"repolore": RELAY_SERVER}})

    assert [entry.scope for entry in repolore_entries(GEMINI)] == ["user"]


def test_devin_registers_for_the_user_with_the_relays_full_path(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    machine.on_path.add("devin")
    monkeypatch.setenv("XDG_DATA_HOME", "/data")
    runs: list[list[str]] = []
    answer(monkeypatch, runs, ADD_SECONDS, 0)

    assert DEVIN.add() == "for every project (user scope)"
    assert runs == [
        [
            "/tools/devin",
            "mcp",
            "add",
            "-s",
            "user",
            "-e",
            "XDG_DATA_HOME=${env:XDG_DATA_HOME}",
            "repolore",
            "--",
            RELAY_AT,
        ]
    ]


@pytest.mark.parametrize(
    ("platform", "relative"),
    [("linux", ".config/devin/mcp_config.json"), ("win32", "AppData/Roaming/devin/mcp_config.json")],
)
def test_devin_lists_its_user_file(
    machine: Machine, monkeypatch: pytest.MonkeyPatch, platform: str, relative: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    write_json(machine.write(relative, ""), {"mcpServers": {"repolore": RELAY_SERVER}})

    assert [entry.scope for entry in repolore_entries(DEVIN)] == ["user"]


@pytest.mark.parametrize("name", ["mcp_config.json", "mcp_config.local.json"])
def test_devin_lists_both_project_files(machine: Machine, name: str) -> None:
    path = machine.project / ".devin" / name
    path.parent.mkdir()
    write_json(path, {"mcpServers": {"repolore": RELAY_SERVER}})

    assert [entry.scope for entry in repolore_entries(DEVIN)] == ["project"]


@pytest.mark.usefixtures("machine")
@pytest.mark.parametrize("client", CLIENTS)
def test_nothing_on_the_machine_means_no_client_is_present(client: McpClient) -> None:
    assert not client.present()


@pytest.mark.parametrize(
    ("client", "command"),
    [
        (GEMINI, "gemini"),
        (CURSOR, "agent"),
        (VSCODE, "code"),
        (OPENCODE, "opencode"),
        (OPENCODE, "opencode2"),
        (DEVIN, "devin"),
    ],
)
def test_a_client_is_found_by_its_command(machine: Machine, client: McpClient, command: str) -> None:
    machine.on_path.add(command)

    assert client.present()


def test_a_gemini_folder_without_the_cli_isnt_gemini(machine: Machine) -> None:
    machine.path(".gemini").mkdir()

    assert not GEMINI.present()


@pytest.mark.parametrize(
    ("client", "platform", "folder"),
    [
        (CLAUDE_DESKTOP, "linux", ".config/Claude"),
        (CLAUDE_DESKTOP, "darwin", "Library/Application Support/Claude"),
        (CLAUDE_DESKTOP, "win32", "AppData/Roaming/Claude"),
        (CURSOR, "linux", ".cursor"),
        (VSCODE, "linux", ".config/Code/User"),
        (VSCODE, "darwin", "Library/Application Support/Code/User"),
        (VSCODE, "win32", "AppData/Roaming/Code/User"),
        (ZED, "linux", ".local/share/zed"),
        (ZED, "darwin", "Library/Application Support/Zed"),
        (ZED, "win32", "AppData/Local/Zed"),
    ],
)
def test_a_client_is_found_by_its_folder(
    machine: Machine, monkeypatch: pytest.MonkeyPatch, client: McpClient, platform: str, folder: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    machine.path(folder).mkdir(parents=True)

    assert client.present()


def test_zeds_config_folder_alone_isnt_zed(machine: Machine) -> None:
    machine.path(".config/zed").mkdir(parents=True)

    assert not ZED.present()


def test_every_client_has_its_own_key() -> None:
    keys = [client.key for client in CLIENTS]

    assert len(keys) == 9
    assert len(set(keys)) == len(keys)


def test_clients_are_named_by_key_in_the_order_given() -> None:
    assert named_clients("codex,claude", CLIENTS) == [CODEX, CLAUDE_CODE]


def test_one_client_can_be_named() -> None:
    assert named_clients("zed", CLIENTS) == [ZED]


def test_spaces_and_repeats_in_client_names_dont_matter() -> None:
    assert named_clients(" zed , cursor,zed,", CLIENTS) == [ZED, CURSOR]


@pytest.mark.parametrize("names", ["", " , ", "zed,nope", "Zed"])
def test_a_list_without_known_clients_is_refused_naming_the_choices(names: str) -> None:
    with pytest.raises(RegistrationError, match="comma-separated list of claude, codex, gemini"):
        named_clients(names, CLIENTS)
