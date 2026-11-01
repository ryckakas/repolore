import shutil
import subprocess
from typing import Any

import pytest

from repolore.clients import CLAUDE_CODE, CODEX, GET_SECONDS, Registration, registration

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
