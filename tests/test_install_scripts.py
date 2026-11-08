import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
FROM_PYPI = "uv tool install --managed-python --python 3.12 repolore-mcp"
FROM_GIT = "uv tool install --managed-python --python 3.12 --upgrade git+https://github.com/ryckakas/repolore"
UPGRADE = "uv tool upgrade repolore-mcp"
NO_TERMINAL = "there's no terminal"

UV_STUB = """#!/bin/sh
printf 'uv %s\\n' "$*" >> "$STUB_LOG"
case "$1 $2" in
"tool list")
    [ -z "${STUB_LIST_FAILS:-}" ] || exit 1
    [ -z "${STUB_INSTALLED:-}" ] || printf 'repolore-mcp v0.3.0\\n- repolore\\n'
    ;;
"tool dir") if [ -n "${STUB_TOOLS_ELSEWHERE:-}" ]; then printf '/elsewhere\\n'; else dirname "$0"; fi ;;
"tool upgrade") exit "${STUB_UPGRADE_EXIT:-0}" ;;
"tool update-shell") exit "${STUB_UPDATE_SHELL_EXIT:-0}" ;;
esac
"""
REPOLORE_STUB = """#!/bin/sh
printf 'repolore %s\\n' "$*" >> "$STUB_LOG"
exit "${STUB_SETUP_EXIT:-0}"
"""
CURL_STUB = """#!/bin/sh
printf 'curl %s\\n' "$*" >> "$STUB_LOG"
[ -n "${STUB_INSTALLER_FAILS:-}" ] || printf 'mkdir -p "$HOME/.local/bin"; cp "$STUB_UV" "$HOME/.local/bin/uv"\\n'
"""


@dataclass
class ShellRun:
    stubs: Path
    log: Path
    env: dict[str, str]

    def __call__(self, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["sh", "-s", "--", *args],
            input=(REPO / "install.sh").read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**self.env, **extra},
            start_new_session=True,
            timeout=60,
            check=False,
        )

    @property
    def calls(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []


def write_stub(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))
    path.chmod(0o755)


@pytest.fixture
def shell(tmp_path: Path) -> ShellRun:
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("no sh on this machine")
    stubs = tmp_path / "stubs"
    parked = tmp_path / "parked"
    home = tmp_path / "home"
    for folder in (stubs, parked, home):
        folder.mkdir()
    write_stub(stubs / "uv", UV_STUB)
    write_stub(stubs / "repolore", REPOLORE_STUB)
    write_stub(stubs / "curl", CURL_STUB)
    write_stub(parked / "uv", UV_STUB)
    log = tmp_path / "calls.log"
    path = os.pathsep.join([str(stubs), str(Path(sh).parent), "/usr/bin", "/bin"])
    return ShellRun(stubs, log, {"PATH": path, "HOME": str(home), "STUB_LOG": str(log), "STUB_UV": str(parked / "uv")})


SETUP_ARGS = ("--include", "you/*", "--register")


def test_sh_installs_and_runs_setup_with_the_arguments(shell: ShellRun) -> None:
    result = shell(*SETUP_ARGS)

    assert result.returncode == 0, result.stderr
    assert FROM_PYPI in shell.calls
    assert shell.calls[-1] == "repolore setup --include you/* --register"


def test_sh_upgrades_an_existing_install_and_never_reinstalls_it(shell: ShellRun) -> None:
    result = shell(*SETUP_ARGS, STUB_INSTALLED="1")

    assert result.returncode == 0, result.stderr
    assert UPGRADE in shell.calls
    assert not any(call.startswith("uv tool install") for call in shell.calls)


def test_sh_stops_when_the_upgrade_fails(shell: ShellRun) -> None:
    result = shell(*SETUP_ARGS, STUB_INSTALLED="1", STUB_UPGRADE_EXIT="1")

    assert result.returncode == 1
    assert "couldn't upgrade repolore-mcp" in result.stderr
    assert not any(call.startswith("uv tool install") for call in shell.calls)
    assert not any(call.startswith("repolore") for call in shell.calls)


@pytest.mark.parametrize("installed", ["", "1"])
def test_sh_installs_main_from_git_when_asked_and_says_about_extras(shell: ShellRun, installed: str) -> None:
    result = shell(*SETUP_ARGS, REPOLORE_INSTALL_FROM_GIT="1", STUB_INSTALLED=installed)

    assert result.returncode == 0, result.stderr
    assert FROM_GIT in shell.calls
    assert UPGRADE not in shell.calls
    assert "extras such as rerank" in result.stdout


def test_sh_without_a_terminal_or_arguments_leaves_setup_to_the_user(shell: ShellRun) -> None:
    result = shell()

    assert result.returncode == 0, result.stderr
    assert NO_TERMINAL in result.stdout
    assert not any(call.startswith("repolore") for call in shell.calls)


def test_sh_passes_on_setups_failure(shell: ShellRun) -> None:
    result = shell(*SETUP_ARGS, STUB_SETUP_EXIT="1")

    assert result.returncode == 1
    assert shell.calls[-1].startswith("repolore setup")
    assert "every question needs its flag" in result.stderr


def test_sh_stops_rather_than_guess_when_it_cant_list_the_tools(shell: ShellRun) -> None:
    result = shell(*SETUP_ARGS, STUB_INSTALLED="1", STUB_LIST_FAILS="1")

    assert result.returncode == 1
    assert "couldn't list uv's tools" in result.stderr
    assert not any(call.startswith("uv tool install") for call in shell.calls)


def test_sh_with_the_tools_on_path_leaves_the_shell_files_alone(shell: ShellRun) -> None:
    result = shell(*SETUP_ARGS)

    assert result.returncode == 0, result.stderr
    assert not any("update-shell" in call for call in shell.calls)
    assert "open a new terminal" not in result.stdout


def test_sh_with_the_tools_off_path_updates_the_shell_and_says_so(shell: ShellRun) -> None:
    result = shell(*SETUP_ARGS, STUB_TOOLS_ELSEWHERE="1")

    assert result.returncode == 0, result.stderr
    assert "uv tool update-shell" in shell.calls
    assert "open a new terminal" in result.stdout


def test_sh_goes_on_when_update_shell_fails(shell: ShellRun) -> None:
    result = shell(*SETUP_ARGS, STUB_TOOLS_ELSEWHERE="1", STUB_UPDATE_SHELL_EXIT="2")

    assert result.returncode == 0, result.stderr
    assert shell.calls[-1].startswith("repolore setup")


def test_sh_installs_uv_when_its_missing_and_uses_it_at_once(shell: ShellRun) -> None:
    (shell.stubs / "uv").unlink()

    result = shell(*SETUP_ARGS)

    assert result.returncode == 0, result.stderr
    assert shell.calls[0] == "curl -fsSL https://astral.sh/uv/install.sh"
    assert FROM_PYPI in shell.calls


def test_sh_stops_when_uv_cant_be_installed(shell: ShellRun) -> None:
    (shell.stubs / "uv").unlink()

    result = shell(*SETUP_ARGS, STUB_INSTALLER_FAILS="1")

    assert result.returncode == 1
    assert "couldn't install uv" in result.stderr


UV_CMD = """@echo off
echo uv %*>> "%STUB_LOG%"
if "%1 %2"=="tool list" echo a warning on stderr 1>&2
if "%1 %2"=="tool list" if defined STUB_LIST_FAILS exit /b 1
if "%1 %2"=="tool list" if defined STUB_INSTALLED echo repolore-mcp v0.3.0
if "%1 %2"=="tool dir" echo %STUB_TOOLS%
if "%1 %2"=="tool upgrade" exit /b %STUB_UPGRADE_EXIT%
if "%1 %2"=="tool update-shell" exit /b %STUB_UPDATE_SHELL_EXIT%
exit /b 0
"""
REPOLORE_CMD = """@echo off
echo repolore %*>> "%STUB_LOG%"
if "%1"=="setup" exit /b %STUB_SETUP_EXIT%
exit /b 0
"""
POWERSHELL_CMD = """@echo off
echo powershell %*>> "%STUB_LOG%"
if defined STUB_INSTALLER_FAILS exit /b 0
mkdir "%USERPROFILE%\\.local\\bin"
copy /y "%STUB_UV%" "%USERPROFILE%\\.local\\bin\\uv.cmd" > nul
"""


@dataclass
class PowerShellRun:
    pwsh: str
    stubs: Path
    log: Path
    env: dict[str, str]

    def __call__(self, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return self.typed(" ".join(f"'{arg}'" for arg in args), **extra)

    def typed(self, arguments: str, before: str = "", **extra: str) -> subprocess.CompletedProcess[str]:
        script = str(REPO / "install.ps1")
        command = f"{before}& ([scriptblock]::Create([IO.File]::ReadAllText('{script}'))) {arguments}"
        return subprocess.run(
            [self.pwsh, "-NoProfile", "-Command", command],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**self.env, **extra},
            timeout=120,
            check=False,
        )

    @property
    def calls(self) -> list[str]:
        return [line.strip() for line in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []


@pytest.fixture(params=["pwsh", "powershell"])
def powershell(request: pytest.FixtureRequest, tmp_path: Path) -> PowerShellRun:
    pwsh = shutil.which(request.param)
    if sys.platform != "win32" or pwsh is None:
        pytest.skip(f"install.ps1 is for Windows, and needs {request.param} here")
    stubs = tmp_path / "stubs"
    parked = tmp_path / "parked"
    home = tmp_path / "home"
    for folder in (stubs, parked, home):
        folder.mkdir()
    (stubs / "uv.cmd").write_text(UV_CMD, encoding="utf-8")
    (stubs / "repolore.cmd").write_text(REPOLORE_CMD, encoding="utf-8")
    (stubs / "powershell.cmd").write_text(POWERSHELL_CMD, encoding="utf-8")
    (parked / "uv.cmd").write_text(UV_CMD, encoding="utf-8")
    log = tmp_path / "calls.log"
    system = os.environ["SYSTEMROOT"]
    path = os.pathsep.join([str(stubs), str(Path(pwsh).parent), rf"{system}\System32", system])
    env = {
        **os.environ,
        "PATH": path,
        "USERPROFILE": str(home),
        "STUB_LOG": str(log),
        "STUB_UV": str(parked / "uv.cmd"),
        "STUB_TOOLS": str(stubs),
        "STUB_UPGRADE_EXIT": "0",
        "STUB_UPDATE_SHELL_EXIT": "0",
        "STUB_SETUP_EXIT": "0",
    }
    env.pop("REPOLORE_INSTALL_FROM_GIT", None)
    return PowerShellRun(pwsh, stubs, log, env)


def test_ps_fails_when_the_command_isnt_there_after_installing(powershell: PowerShellRun) -> None:
    (powershell.stubs / "repolore.cmd").unlink()

    result = powershell(*SETUP_ARGS)

    assert FROM_PYPI in powershell.calls
    assert result.returncode != 0
    assert "wasn't found after installing" in result.stdout + result.stderr


def test_ps_runs_setup_with_the_arguments_after_stopping_the_server(powershell: PowerShellRun) -> None:
    result = powershell(*SETUP_ARGS)

    assert result.returncode == 0, result.stderr
    assert powershell.calls[0] == "repolore server stop"
    assert powershell.calls[-1] == "repolore setup --include you/* --register"


def test_ps_keeps_an_unquoted_comma_list_as_one_argument(powershell: PowerShellRun) -> None:
    result = powershell.typed("--clients claude,codex --register")

    assert result.returncode == 0, result.stderr
    assert powershell.calls[-1] == "repolore setup --clients claude,codex --register"


def test_ps_stops_rather_than_guess_when_it_cant_list_the_tools(powershell: PowerShellRun) -> None:
    result = powershell(*SETUP_ARGS, STUB_INSTALLED="1", STUB_LIST_FAILS="1")

    assert result.returncode != 0
    assert "couldn't list uv's tools" in result.stdout + result.stderr
    assert not any(call.startswith("uv tool install") for call in powershell.calls)


def test_ps_survives_a_caller_that_stops_on_errors(powershell: PowerShellRun) -> None:
    result = powershell.typed("'--register'", before="$ErrorActionPreference = 'Stop'; ")

    assert result.returncode == 0, result.stderr
    assert powershell.calls[-1] == "repolore setup --register"


def test_ps_upgrades_an_existing_install_and_never_reinstalls_it(powershell: PowerShellRun) -> None:
    result = powershell(*SETUP_ARGS, STUB_INSTALLED="1")

    assert result.returncode == 0, result.stderr
    assert UPGRADE in powershell.calls
    assert not any(call.startswith("uv tool install") for call in powershell.calls)


def test_ps_stops_when_the_upgrade_fails_and_says_what_to_close(powershell: PowerShellRun) -> None:
    result = powershell(*SETUP_ARGS, STUB_INSTALLED="1", STUB_UPGRADE_EXIT="1")

    assert result.returncode != 0
    assert "close the sessions" in result.stdout + result.stderr
    assert not any(call.startswith("repolore setup") for call in powershell.calls)


@pytest.mark.parametrize("installed", ["", "1"])
def test_ps_installs_main_from_git_when_asked(powershell: PowerShellRun, installed: str) -> None:
    extra = {"STUB_INSTALLED": "1"} if installed else {}

    result = powershell(*SETUP_ARGS, REPOLORE_INSTALL_FROM_GIT="1", **extra)

    assert result.returncode == 0, result.stderr
    assert FROM_GIT in powershell.calls
    assert UPGRADE not in powershell.calls


def test_ps_without_a_terminal_or_arguments_leaves_setup_to_the_user(powershell: PowerShellRun) -> None:
    result = powershell()

    assert result.returncode == 0, result.stderr
    assert NO_TERMINAL in result.stdout
    assert not any(call.startswith("repolore setup") for call in powershell.calls)


def test_ps_fails_when_setup_does(powershell: PowerShellRun) -> None:
    result = powershell(*SETUP_ARGS, STUB_SETUP_EXIT="1")

    assert result.returncode != 0
    assert "setup stopped" in result.stdout + result.stderr


def test_ps_with_the_tools_on_path_leaves_the_shell_files_alone(powershell: PowerShellRun) -> None:
    result = powershell(*SETUP_ARGS)

    assert result.returncode == 0, result.stderr
    assert not any("update-shell" in call for call in powershell.calls)


def test_ps_goes_on_when_update_shell_fails(powershell: PowerShellRun, tmp_path: Path) -> None:
    result = powershell(*SETUP_ARGS, STUB_TOOLS=str(tmp_path / "elsewhere"), STUB_UPDATE_SHELL_EXIT="2")

    assert result.returncode == 0, result.stderr
    assert "uv tool update-shell" in powershell.calls
    assert powershell.calls[-1].startswith("repolore setup")


def test_ps_installs_uv_when_its_missing_and_uses_it_at_once(powershell: PowerShellRun) -> None:
    (powershell.stubs / "uv.cmd").unlink()

    result = powershell(*SETUP_ARGS)

    assert result.returncode == 0, result.stderr
    assert any("astral.sh/uv/install.ps1" in call for call in powershell.calls)
    assert FROM_PYPI in powershell.calls


def test_ps_stops_when_uv_cant_be_installed(powershell: PowerShellRun) -> None:
    (powershell.stubs / "uv.cmd").unlink()

    result = powershell(*SETUP_ARGS, STUB_INSTALLER_FAILS="1")

    assert result.returncode != 0
    assert "couldn't install uv" in result.stdout + result.stderr
