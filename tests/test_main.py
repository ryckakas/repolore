import runpy
import subprocess
import sys

import pytest

from repolore import __version__


def test_the_package_runs_as_the_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["repolore", "version"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("repolore", run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.splitlines()[0] == f"repolore {__version__}"


def test_python_dash_m_repolore_starts_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "repolore", "version"], capture_output=True, text=True, encoding="utf-8", check=True
    )

    assert result.stdout.splitlines()[0] == f"repolore {__version__}"
