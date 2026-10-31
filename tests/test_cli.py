import io
import json
import shutil
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import FakeEmbedder
from typer.testing import CliRunner

from repolore import __version__, cli
from repolore.cli import app
from repolore.config import Config
from repolore.discovery import Discovery, RepoSource, Scope
from repolore.documents import Document
from repolore.paths import resolve_paths
from repolore.sources.github import GitHubClient, GitHubError
from repolore.store import connect, index_settings

runner = CliRunner()
EXAMPLE_FILE = Path(__file__).parent.parent / "evals" / "questions.example.yaml"
FIXTURES = Path(__file__).parent / "fixtures"


def test_version_prints_version_and_default_paths() -> None:
    result = runner.invoke(app, ["version"])

    defaults = resolve_paths()
    assert result.exit_code == 0
    assert f"repolore {__version__}" in result.output
    assert f"config: {defaults.config_file}" in result.output
    assert f"data:   {defaults.data_dir}" in result.output


def test_data_dir_option_overrides_data_path(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--data-dir", str(tmp_path), "version"])

    assert result.exit_code == 0
    assert f"data:   {tmp_path.resolve()}" in result.output


def test_no_command_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "version" in result.output


def test_questions_summarizes_a_valid_file() -> None:
    result = runner.invoke(app, ["questions", "--file", str(EXAMPLE_FILE)])

    assert result.exit_code == 0
    assert "total:     3 (1 held out)" in result.output
    assert "factual       1" in result.output
    assert "commit-only   0" in result.output


def test_questions_defaults_to_the_config_dir_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "evals" / "questions.yaml"
    monkeypatch.setattr("repolore.cli.resolve_paths", lambda _: replace(resolve_paths(), questions_file=missing))

    result = runner.invoke(app, ["questions"])

    assert result.exit_code == 1
    assert f"No question file at {missing}" in result.output


@pytest.mark.parametrize("content", ["questions: [", "questions:\n  - id: Not-Kebab\n"])
def test_questions_reports_an_invalid_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "questions.yaml"
    path.write_text(content, encoding="utf-8")

    result = runner.invoke(app, ["questions", "--file", str(path)])

    assert result.exit_code == 1
    assert f"{path} is invalid" in result.output


def ingest_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str | None = None) -> Path:
    config_file = tmp_path / "repolore.toml"
    if config is not None:
        config_file.write_text(config, encoding="utf-8")
    monkeypatch.setattr(
        "repolore.cli.resolve_paths", lambda _: replace(resolve_paths(tmp_path), config_file=config_file)
    )
    return tmp_path


def online(monkeypatch: pytest.MonkeyPatch, discovery: Discovery | None = None) -> list[Config]:
    configs: list[Config] = []

    def fake_discover(_: GitHubClient, config: Config) -> Discovery:
        configs.append(config)
        return discovery or Discovery(login="octocat", included=[], excluded=[])

    monkeypatch.setattr("repolore.cli.resolve_token", lambda: "token")
    monkeypatch.setattr("repolore.cli.discover", fake_discover)
    return configs


def test_offline_ingest_prints_counts_per_repo_and_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = ingest_setup(tmp_path, monkeypatch)
    shutil.copytree(FIXTURES / "crashcause", data_dir / "raw" / "ryckakas" / "crashcause")

    result = runner.invoke(app, ["ingest", "--offline"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0].split() == ["repo", "pr", "review", "issue", "commit", "doc", "plan"]
    assert lines[1].split() == ["ryckakas/crashcause", "2", "0", "2", "1", "2", "0"]
    assert lines[2].split() == ["total", "2", "0", "2", "1", "2", "0"]


@pytest.mark.parametrize("config", ["include = [", 'repos = ["octo/repo"]'])
def test_invalid_config_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str) -> None:
    ingest_setup(tmp_path, monkeypatch, config)

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert "is invalid" in result.output


def test_offline_ingest_without_cache_says_to_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ingest_setup(tmp_path, monkeypatch)

    result = runner.invoke(app, ["ingest", "--offline"])

    assert result.exit_code == 1
    assert "run `repolore ingest` first" in result.output


def test_ingest_without_token_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_token() -> str:
        raise GitHubError("No GitHub token")

    ingest_setup(tmp_path, monkeypatch)
    monkeypatch.setattr("repolore.cli.resolve_token", no_token)

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert "No GitHub token" in result.output


def test_online_ingest_passes_the_discovery_and_closes_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    calls: list[tuple[Discovery, bool]] = []
    discovery = Discovery(login="octocat", included=[RepoSource("octo/repo", Scope.OWNED)], excluded=[])
    ingest_setup(tmp_path, monkeypatch, 'exclude = ["octo/old"]')
    configs = online(monkeypatch, discovery)
    monkeypatch.setattr("repolore.cli.ingest_repos", lambda found, _, __, full: calls.append((found, full)) or [])
    monkeypatch.setattr(GitHubClient, "close", lambda _: closed.append(True))

    result = runner.invoke(app, ["ingest", "--full"])

    assert result.exit_code == 0
    assert configs == [Config(exclude=["octo/old"])]
    assert calls == [(discovery, True)]
    assert closed == [True]


def test_ingest_reports_github_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object) -> list[Document]:
        raise GitHubError("octo/repo: not found, or the token can't read it")

    ingest_setup(tmp_path, monkeypatch)
    online(monkeypatch)
    monkeypatch.setattr("repolore.cli.ingest_repos", fail)

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert "octo/repo: not found" in result.output


def test_repos_lists_each_repo_with_its_scope_and_plans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "repo"
    discovery = Discovery(
        login="octocat",
        included=[
            RepoSource("octocat/repo", Scope.OWNED, checkout),
            RepoSource("other/project", Scope.CONTRIBUTED),
        ],
        excluded=[RepoSource("octocat/demo", Scope.OWNED), RepoSource("octocat/old", Scope.OWNED)],
    )
    ingest_setup(tmp_path, monkeypatch)
    online(monkeypatch, discovery)

    result = runner.invoke(app, ["repos"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0].split() == ["octocat/repo", "owned", "+", "plans", "in", str(checkout)]
    assert lines[1].split() == ["other/project", "contributed"]
    assert lines[2] == "2 repos for octocat"
    assert lines[3].startswith("2 excluded by ")
    assert lines[3].endswith("repolore.toml")
    assert len(lines) == 4


def test_repos_lists_the_excluded_repos_on_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    discovery = Discovery(
        login="octocat",
        included=[RepoSource("octocat/repo", Scope.OWNED)],
        excluded=[RepoSource("octocat/a-much-longer-name", Scope.MEMBER, tmp_path / "checkout")],
    )
    ingest_setup(tmp_path, monkeypatch)
    online(monkeypatch, discovery)

    result = runner.invoke(app, ["repos", "--excluded"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0] == f"{'octocat/repo':<26}  owned"
    assert lines[1] == "octocat/a-much-longer-name  excluded"
    assert lines[2] == "1 repo for octocat"
    assert lines[3].startswith("1 excluded by ")
    assert len(lines) == 4


def test_repos_without_a_config_uses_the_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ingest_setup(tmp_path, monkeypatch)
    configs = online(monkeypatch)

    result = runner.invoke(app, ["repos"])

    assert result.exit_code == 0
    assert configs == [Config()]
    assert result.output == "0 repos for octocat\n"


def test_inspect_shows_counts_histogram_and_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = ingest_setup(tmp_path, monkeypatch)
    shutil.copytree(FIXTURES / "crashcause", data_dir / "raw" / "ryckakas" / "crashcause")

    result = runner.invoke(app, ["inspect", "--samples", "2", "--seed", "1"])

    assert result.exit_code == 0
    lines = [line.split() for line in result.output.splitlines()]
    assert lines[0] == ["7", "documents,", "24", "chunks"]
    assert lines[1:7] == [["pr", "5"], ["review", "0"], ["issue", "15"], ["commit", "1"], ["doc", "3"], ["plan", "0"]]
    assert lines[7] == ["tokens", "(estimated)"]
    assert [line[0] for line in lines[8:14]] == ["1-50", "51-100", "101-200", "201-400", "401-800", ">800"]
    assert sum(int(line[1]) for line in lines[8:14]) == 24
    assert result.output.count("\n--- https://github.com/ryckakas/crashcause/") == 2


def test_inspect_without_cache_says_to_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ingest_setup(tmp_path, monkeypatch)

    result = runner.invoke(app, ["inspect"])

    assert result.exit_code == 1
    assert "run `repolore ingest` first" in result.output


def test_output_streams_that_are_not_files_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    cli._use_utf8_output()

    assert sys.stdout is stdout


@pytest.fixture
def cached_crashcause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = ingest_setup(tmp_path, monkeypatch)
    shutil.copytree(FIXTURES / "crashcause", data_dir / "raw" / "ryckakas" / "crashcause")
    monkeypatch.setattr("repolore.cli.LocalEmbedder", lambda *_: FakeEmbedder())
    monkeypatch.setattr("repolore.cli.embedder_for", lambda _: FakeEmbedder())
    return data_dir


def test_build_embeds_new_chunks_and_reuses_them_next_time(cached_crashcause: Path) -> None:
    first = runner.invoke(app, ["build"])
    second = runner.invoke(app, ["build", "--db", "experiment"])

    assert first.exit_code == 0
    assert "24 chunks indexed in" in first.output
    assert "24 embedded, 0 reused" in first.output
    assert "0 embedded, 24 reused" in second.output
    assert (cached_crashcause / "db" / "experiment.db").exists()


def test_search_prints_ranked_hits_with_links(cached_crashcause: Path) -> None:
    runner.invoke(app, ["build"])

    result = runner.invoke(app, ["search", "security context violation runAsNonRoot", "-k", "2", "--kind", "issue"])

    assert result.exit_code == 0
    assert result.output.startswith("1. ")
    assert "\n   https://github.com/ryckakas/crashcause/issues/17\n" in result.output
    assert "3. " not in result.output


def test_search_distinct_limits_results_to_one_chunk_per_url(cached_crashcause: Path) -> None:
    assert runner.invoke(app, ["build"]).exit_code == 0
    result = runner.invoke(app, ["search", "security context", "-k", "10", "--distinct"])

    assert result.exit_code == 0
    urls = [line.strip() for line in result.output.splitlines() if line.strip().startswith("https://")]
    assert urls
    assert len(urls) == len(set(urls))


def test_search_with_filters_that_match_nothing(cached_crashcause: Path) -> None:
    runner.invoke(app, ["build"])

    result = runner.invoke(app, ["search", "anything", "--repo", "octo/missing", "--author", "nobody"])

    assert result.exit_code == 0
    assert result.output.strip() == "No matches."


def test_search_without_an_index_says_to_build(cached_crashcause: Path) -> None:
    result = runner.invoke(app, ["search", "anything"])

    assert result.exit_code == 1
    assert "run `repolore build` first" in result.output


def test_search_refuses_an_index_from_another_model(cached_crashcause: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner.invoke(app, ["build"])
    monkeypatch.setattr(FakeEmbedder, "model_id", "other-model")

    result = runner.invoke(app, ["search", "anything"])

    assert result.exit_code == 1
    assert "rebuild it" in result.output


EVAL_QUESTIONS = """
questions:
  - id: runasnonroot
    question: "Which cause code did a runAsNonRoot pod with a root image get?"
    kind: factual
    expected:
      - url: https://github.com/ryckakas/crashcause/issues/17
    answer: "image_pull_other"
  - id: missing-source
    question: "What did PR 999 change?"
    kind: factual
    expected:
      - url: https://github.com/ryckakas/crashcause/pull/999
    answer: "Nothing; it doesn't exist."
  - id: kafka
    question: "Which repo uses Kafka?"
    kind: unanswerable
    answer: "None."
  - id: held-out
    question: "What does the engine classify?"
    kind: commit-only
    expected:
      - url: https://github.com/ryckakas/crashcause/commit/69531a3aa08bf8d47f1428261efc3038537127b6
    answer: "Crashes."
    holdout: true
"""


@pytest.fixture
def eval_questions(cached_crashcause: Path) -> Path:
    path = cached_crashcause / "questions.yaml"
    path.write_text(EVAL_QUESTIONS, encoding="utf-8")
    runner.invoke(app, ["build"])
    return path


def test_eval_scores_saves_and_lists_misses(eval_questions: Path, cached_crashcause: Path) -> None:
    result = runner.invoke(app, ["eval", "--file", str(eval_questions), "--show-misses"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "3 tuning questions on index default (fake-bag-of-words, vector)"
    assert lines[1] == "  not in the index: https://github.com/ryckakas/crashcause/pull/999"
    assert lines[3].split() == ["recall@5", "0.500"]
    assert "  factual       0.500  (2)" in lines
    assert any(line.startswith("top score: ") for line in lines)
    assert any(
        line.startswith("search: median ") and line.endswith("over 24 chunks, query embedding included")
        for line in lines
    )
    assert "miss: missing-source: What did PR 999 change?" in result.output
    assert "  expected https://github.com/ryckakas/crashcause/pull/999 (rank >10)" in lines
    assert len(list((cached_crashcause / "evals" / "runs").glob("*.json"))) == 1


def test_second_eval_compares_with_the_previous_run(eval_questions: Path) -> None:
    runner.invoke(app, ["eval", "--file", str(eval_questions)])

    result = runner.invoke(app, ["eval", "--file", str(eval_questions)])

    assert next(line for line in result.output.splitlines() if line.startswith("recall@5")).split() == [
        "recall@5",
        "0.500",
        "0.500",
        "+0.000",
    ]


def test_eval_on_the_holdout_set(eval_questions: Path) -> None:
    result = runner.invoke(app, ["eval", "--file", str(eval_questions), "--holdout"])

    assert result.exit_code == 0
    assert result.output.startswith("1 held-out questions on index default")
    assert "top score" not in result.output


def test_eval_with_no_questions_in_the_set_reports_no_search_time(eval_questions: Path) -> None:
    tuning_only = eval_questions.with_name("tuning-only.yaml")
    tuning_only.write_text(eval_questions.read_text(encoding="utf-8").replace("holdout: true", ""), encoding="utf-8")

    result = runner.invoke(app, ["eval", "--file", str(tuning_only), "--holdout"])

    assert result.exit_code == 0
    assert result.output.startswith("0 held-out questions")
    assert "search: median" not in result.output


def test_eval_without_an_index_says_to_build(cached_crashcause: Path) -> None:
    path = cached_crashcause / "questions.yaml"
    path.write_text(EVAL_QUESTIONS, encoding="utf-8")

    result = runner.invoke(app, ["eval", "--file", str(path)])

    assert result.exit_code == 1
    assert "run `repolore build` first" in result.output


def test_eval_refuses_an_index_from_another_model(eval_questions: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FakeEmbedder, "model_id", "other-model")

    result = runner.invoke(app, ["eval", "--file", str(eval_questions)])

    assert result.exit_code == 1
    assert "rebuild it" in result.output


def test_build_with_smaller_chunks_records_the_setting(cached_crashcause: Path) -> None:
    result = runner.invoke(app, ["build", "--db", "small", "--max-tokens", "50"])

    assert result.exit_code == 0
    assert int(result.output.split(" chunks indexed")[0]) > 26
    with closing(connect(cached_crashcause / "db" / "small.db")) as db:
        assert index_settings(db)["max_tokens"] == "50"
        assert index_settings(db)["bot_reviews"] == "excluded"


def test_build_with_bot_reviews_records_the_setting(cached_crashcause: Path) -> None:
    result = runner.invoke(app, ["build", "--db", "bots", "--bot-reviews"])

    assert result.exit_code == 0
    with closing(connect(cached_crashcause / "db" / "bots.db")) as db:
        assert index_settings(db)["bot_reviews"] == "included"


def test_eval_records_the_index_settings(eval_questions: Path, cached_crashcause: Path) -> None:
    runner.invoke(app, ["eval", "--file", str(eval_questions), "--mode", "hybrid"])

    [run_file] = (cached_crashcause / "evals" / "runs").glob("*.json")
    config = json.loads(run_file.read_text(encoding="utf-8"))["config"]
    assert config["max_tokens"] == "whole"
    assert config["bot_reviews"] == "excluded"
    assert config["retrieval"] == "hybrid"
