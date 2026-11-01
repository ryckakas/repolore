import shutil
import subprocess
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
    Registration,
    RegistrationError,
    codex_home,
    register,
    registration,
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
CODEX_FOUND = """repolore
  enabled: true
  transport: stdio
  command: repolore-mcp
"""


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    monkeypatch.setattr(shutil, "which", lambda command: f"/bin/{command}")
    return []


def answer(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]], code: int, stdout: str = "") -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == GET_SECONDS
        runs.append(command)
        return subprocess.CompletedProcess(command, code, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", run)


def test_a_healthy_registration_carries_claude_codes_status(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    answer(monkeypatch, runs, 0, CLAUDE_HEALTHY)

    assert registration(CLAUDE_CODE) == Registration(found=True, status="✓ Connected")
    assert runs == [["/bin/claude", "mcp", "get", "repolore"]]


def test_a_client_without_a_status_line_is_healthy_when_registered(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    answer(monkeypatch, runs, 0, CODEX_FOUND)

    assert registration(CODEX) == Registration(found=True)


def test_a_failed_health_check_keeps_its_issue(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    answer(monkeypatch, runs, 0, CLAUDE_TIMED_OUT)

    assert registration(CLAUDE_CODE) == Registration(
        found=True, status="! Connected · tools fetch failed (Request timed out)"
    )


@pytest.mark.parametrize(
    ("found", "healthy"),
    [
        (Registration(found=True), True),
        (Registration(found=True, status="✓ Connected"), True),
        (Registration(found=True, status="! Connected · tools fetch failed (Request timed out)"), False),
        (Registration(found=True, status="✗ Failed to connect"), False),
        (Registration(found=True, status="Pending"), False),
        (Registration(found=False), False),
    ],
)
def test_a_registration_is_healthy_when_found_and_connected_if_checked(found: Registration, healthy: bool) -> None:
    assert found.healthy is healthy


def test_a_client_that_doesnt_know_repolore_has_no_registration(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    answer(monkeypatch, runs, 1, "No MCP server named repolore")

    assert registration(CODEX) == Registration(found=False, status="No MCP server named repolore")


def test_a_client_that_fails_silently_has_no_registration(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    answer(monkeypatch, runs, 2)

    assert registration(CODEX) == Registration(found=False)


def test_a_client_not_on_path_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(subprocess, "run", pytest.fail)

    assert registration(CLAUDE_CODE) is None


@pytest.fixture
def codex_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    for name in CODEX_PASSES_ON:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def adding(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]], code: int, stderr: str = "") -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["timeout"] == ADD_SECONDS
        runs.append(command)
        return subprocess.CompletedProcess(command, code, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", run)


def test_claude_code_registers_repolore_for_every_project(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    adding(monkeypatch, runs, 0)

    assert register(CLAUDE_CODE) == "for every project (user scope)"
    assert runs == [
        ["/bin/claude", "mcp", "add", "--scope", "user", "--transport", "stdio", "repolore", "--", "repolore-mcp"]
    ]


def test_a_refused_claude_registration_says_why(monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]) -> None:
    adding(monkeypatch, runs, 1, "MCP server repolore already exists in user config\nmore")

    with pytest.raises(RegistrationError, match=r"^MCP server repolore already exists in user config$"):
        register(CLAUDE_CODE)


def test_a_silently_refused_claude_registration_names_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, runs: list[list[str]]
) -> None:
    adding(monkeypatch, runs, 3)

    with pytest.raises(RegistrationError, match="exited with 3"):
        register(CLAUDE_CODE)


def test_a_client_not_on_path_cant_be_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(RegistrationError, match="`claude` isn't on PATH"):
        register(CLAUDE_CODE)


@pytest.mark.usefixtures("runs")
def test_codex_gets_a_config_entry_with_longer_timeouts(codex_dir: Path) -> None:
    assert register(CODEX) == f"in {codex_dir / 'config.toml'}"

    text = (codex_dir / "config.toml").read_text(encoding="utf-8")
    assert text.startswith("[mcp_servers.repolore]\n")
    assert tomllib.loads(text) == {
        "mcp_servers": {"repolore": {"command": "repolore-mcp", "startup_timeout_sec": 60, "tool_timeout_sec": 300}}
    }


@pytest.mark.usefixtures("runs")
def test_codex_entry_follows_the_existing_config(codex_dir: Path) -> None:
    (codex_dir / "config.toml").write_text('model = "o3"\n[mcp_servers.other]\ncommand = "other"', encoding="utf-8")

    register(CODEX)

    text = (codex_dir / "config.toml").read_text(encoding="utf-8")
    assert text.startswith('model = "o3"\n[mcp_servers.other]\ncommand = "other"\n\n[mcp_servers.repolore]\n')
    assert tomllib.loads(text)["mcp_servers"].keys() == {"other", "repolore"}


@pytest.mark.usefixtures("runs")
def test_codex_entry_passes_on_the_variables_repolore_reads(codex_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOLORE_GITHUB_TOKEN", "token")
    monkeypatch.setenv("HF_HOME", str(codex_dir))
    monkeypatch.setenv("HTTPS_PROXY", "")

    register(CODEX)

    config = tomllib.loads((codex_dir / "config.toml").read_text(encoding="utf-8"))
    assert config["mcp_servers"]["repolore"]["env_vars"] == ["REPOLORE_GITHUB_TOKEN", "HF_HOME"]


@pytest.mark.usefixtures("runs")
def test_codex_already_having_repolore_is_left_alone(codex_dir: Path) -> None:
    original = '[mcp_servers.repolore]\ncommand = "elsewhere"\n'
    (codex_dir / "config.toml").write_text(original, encoding="utf-8")

    with pytest.raises(RegistrationError, match=r"already has \[mcp_servers.repolore\]"):
        register(CODEX)

    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == original


@pytest.mark.usefixtures("runs")
def test_codex_servers_in_an_inline_table_are_left_alone(codex_dir: Path) -> None:
    original = 'mcp_servers = { other = { command = "other" } }\n'
    (codex_dir / "config.toml").write_text(original, encoding="utf-8")

    with pytest.raises(RegistrationError, match=r"can't add \[mcp_servers.repolore\]"):
        register(CODEX)

    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == original


@pytest.mark.usefixtures("runs")
def test_an_unreadable_codex_config_is_left_alone(codex_dir: Path) -> None:
    (codex_dir / "config.toml").write_text("model = ", encoding="utf-8")

    with pytest.raises(RegistrationError, match="can't read"):
        register(CODEX)

    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == "model = "


@pytest.mark.parametrize("value", [None, ""])
def test_codex_home_defaults_to_the_home_folder(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("CODEX_HOME", raising=False)
    else:
        monkeypatch.setenv("CODEX_HOME", value)

    assert codex_home() == Path.home() / ".codex"
