import json
import tomllib
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from repolore.cli import PLUGIN_REFRESH_HOURS, SESSION_REFRESH_HOURS, app
from repolore.clients import RELAY, SERVER

ROOT = Path(__file__).parent.parent
PLUGIN = ROOT / "plugins" / "repolore"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scripts() -> dict[str, str]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]


def session_start_hooks() -> list[dict[str, Any]]:
    groups = read_json(PLUGIN / "hooks" / "hooks.json")["hooks"]["SessionStart"]
    return [hook | {"matcher": group["matcher"]} for group in groups for hook in group["hooks"]]


def test_the_marketplace_lists_the_plugin_in_its_directory() -> None:
    marketplace = read_json(ROOT / ".claude-plugin" / "marketplace.json")
    (entry,) = marketplace["plugins"]

    assert (ROOT / entry["source"]).resolve() == PLUGIN.resolve()
    assert entry["name"] == read_json(PLUGIN / ".claude-plugin" / "plugin.json")["name"] == SERVER


def test_the_plugin_starts_the_relay_under_repolores_server_name() -> None:
    servers = read_json(PLUGIN / ".mcp.json")["mcpServers"]

    assert servers == {SERVER: {"command": RELAY}}
    assert scripts()[RELAY] == "repolore.relay:main"


def test_the_hook_starts_a_background_refresh_when_a_session_starts() -> None:
    (hook,) = session_start_hooks()

    assert hook["matcher"] == "startup"
    assert hook["async"] is True
    assert hook["command"] in scripts()
    assert CliRunner().invoke(app, [*hook["args"], "--help"]).exit_code == 0


def test_the_hook_takes_no_plugin_option_in_its_arguments() -> None:
    (hook,) = session_start_hooks()

    # Claude Code fails a hook whose ${user_config.x} isn't set, and an option left at its default isn't set.
    assert not any("${user_config." in arg for arg in hook["args"])


def test_the_refresh_option_reaches_session_start_under_its_env_var_with_the_same_default() -> None:
    options = read_json(PLUGIN / ".claude-plugin" / "plugin.json")["userConfig"]

    assert {f"CLAUDE_PLUGIN_OPTION_{key.upper()}" for key in options} == {PLUGIN_REFRESH_HOURS}
    assert options["refresh_hours"]["default"] == SESSION_REFRESH_HOURS
