import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from repolore.clients import (
    ADD_SECONDS,
    CLAUDE_CODE,
    CODEX,
    CODEX_PASSES_ON,
    GET_SECONDS,
    LIST_SECONDS,
    PLUGIN_SERVER,
    Entry,
    McpClient,
    RegistrationError,
    codex_home,
    is_connected,
    repolore_entries,
)

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
    client = McpClient("Client", lambda: True, lambda: [others, ours], lambda: "added", "add it")

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
    for name in CODEX_PASSES_ON:
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
