import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEmbedder

from repolore import copying
from repolore.chunking import Chunk
from repolore.copying import CopyError, clone_or_copy, copy_home
from repolore.paths import AppPaths
from repolore.refresh import LOCK_FILE, REQUEST_FILE, STARTUP_SECONDS, RefreshRunning, refreshing
from repolore.store import EmbeddingCache, connect, connect_read_only, index_settings, repo_counts, write_index

KEPT = {
    "repolore.toml": "include = ['octo/*']\n",
    "evals/questions.yaml": "questions: []\n",
    "evals/runs/2026-11-01.json": "{}",
    "raw/octo/repo/pulls.json": "[]",
    "models/voyage-4-nano-1/tokenizer.json": "{}",
    "refresh.json": '{"started_at": "2026-11-01T10:00:00+00:00"}',
    "refresh.log": "done\n",
}
LEFT = {
    "refresh.lock": "",
    "refresh.json.tmp": "{",
    "server-default.json": '{"token": "secret"}',
    "server-default.lock": "",
    "server-default.log": "served\n",
    "clients/default/123.lock": "",
    "models/voyage-4-nano-1.lock": "",
    "models/voyage-4-nano-1/voyage-4-nano.onnx.partial": "half",
}


def write_files(folder: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def paths_in(folder: Path) -> AppPaths:
    """A home on macOS's layout, where the config and the data share one folder."""
    return AppPaths(
        config_file=folder / "repolore.toml", questions_file=folder / "evals" / "questions.yaml", data_dir=folder
    )


@pytest.fixture
def home(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> AppPaths:
    """A home with an index, an embedding cache, the settings, the raw cache, the model and a running setup's state."""
    paths = paths_in(tmp_path / "old")
    with closing(connect(paths.index_file("default"))) as index:
        vectors = fake_embedder.embed_documents([chunk.embedded_text for chunk in corpus])
        write_index(index, corpus, vectors, fake_embedder.model_id, fake_embedder.dim)
    with closing(connect(paths.embedding_cache)) as cache:
        EmbeddingCache(cache).put(["key"], np.ones((1, 4), np.float32))
    write_files(paths.data_dir, KEPT | LEFT)
    return paths


def snapshot(folder: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(folder).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != LOCK_FILE
    }


def test_the_copy_holds_the_data_and_the_settings_and_leaves_the_running_state_behind(
    home: AppPaths, tmp_path: Path
) -> None:
    target = tmp_path / "new-home"

    copied = copy_home(home, target)

    assert copied.target == target.resolve()
    for name, text in KEPT.items():
        assert (target / name).read_text(encoding="utf-8") == text
    for name in LEFT:
        assert not (target / name).exists(), name
    assert not (target / "clients").exists()
    assert copied.files == len(KEPT) + 2


def test_repos_named_like_what_is_left_behind_are_copied(home: AppPaths, tmp_path: Path) -> None:
    repos = [
        *["acme/server-api", "acme/clients", "clients/tool", "acme/x.lock", "me/dev-journal"],
        *["me/repolore.toml", "evals/questions.yaml", "me/refresh.requested"],
    ]
    write_files(home.data_dir, {f"raw/{repo}/pulls.json": "[]" for repo in repos})

    copy_home(home, tmp_path / "new-home")

    for repo in repos:
        assert (tmp_path / "new-home" / "raw" / repo / "pulls.json").exists(), repo


def test_the_databases_are_copied_whole_while_a_server_reads_and_a_writer_holds_changes_in_the_wal(
    home: AppPaths, tmp_path: Path
) -> None:
    index_file = home.index_file("default")
    target = tmp_path / "new-home"
    with closing(connect_read_only(index_file)) as reader, closing(connect(index_file)) as writer:
        writer.execute("insert into meta (key, value) values ('only_in_the_wal', 'yes')")
        writer.commit()
        assert Path(f"{index_file}-wal").stat().st_size > 0
        expected = repo_counts(reader)

        copy_home(home, target)

    assert not list(target.rglob("*-wal")) and not list(target.rglob("*-shm"))
    with closing(connect_read_only(target / "db" / "default.db")) as copy:
        assert repo_counts(copy) == expected
        assert index_settings(copy)["only_in_the_wal"] == "yes"
        assert copy.execute("pragma integrity_check").fetchone() == ("ok",)
    with closing(connect(target / "cache" / "embeddings.db")) as cache:
        assert EmbeddingCache(cache).get(["key"])["key"].tolist() == [1.0] * 4


def test_no_file_of_the_home_copied_changes(home: AppPaths, tmp_path: Path) -> None:
    before = snapshot(home.data_dir)

    copy_home(home, tmp_path / "new-home")

    after = snapshot(home.data_dir)
    assert {name: after[name] for name in before} == before
    # What any reader of a database in WAL mode adds, the shared server's included.
    added = {name: content for name, (content, _) in after.items() if name not in before}
    assert set(added) <= {"db/default.db-wal", "db/default.db-shm"}
    assert added.get("db/default.db-wal", b"") == b""


def test_settings_kept_elsewhere_are_copied_to_the_top_of_the_new_home(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    write_files(config_dir, {"repolore.toml": "include = ['me/*']\n", "evals/questions.yaml": "questions: []\n"})
    write_files(tmp_path / "data", {"raw/me/a/pulls.json": "[]"})
    paths = replace(
        paths_in(tmp_path / "data"),
        config_file=config_dir / "repolore.toml",
        questions_file=config_dir / "evals" / "questions.yaml",
    )

    copy_home(paths, tmp_path / "new-home")

    assert (tmp_path / "new-home" / "repolore.toml").read_text(encoding="utf-8") == "include = ['me/*']\n"
    assert (tmp_path / "new-home" / "evals" / "questions.yaml").exists()
    assert (tmp_path / "new-home" / "raw" / "me" / "a" / "pulls.json").exists()


def test_a_config_in_the_data_dir_that_isnt_read_is_left_out(tmp_path: Path) -> None:
    write_files(tmp_path / "data", {"repolore.toml": "include = ['stale/*']\n", "raw/me/a/pulls.json": "[]"})
    paths = replace(paths_in(tmp_path / "data"), config_file=tmp_path / "config" / "repolore.toml")

    copy_home(paths, tmp_path / "new-home")

    assert not (tmp_path / "new-home" / "repolore.toml").exists()


def test_without_a_data_dir_only_the_settings_are_copied_and_nothing_is_created_there(tmp_path: Path) -> None:
    write_files(tmp_path / "config", {"repolore.toml": "include = []\n"})
    paths = replace(paths_in(tmp_path / "data"), config_file=tmp_path / "config" / "repolore.toml")

    copied = copy_home(paths, tmp_path / "new-home")

    assert copied.files == 1
    assert sorted(path.name for path in (tmp_path / "new-home").iterdir()) == ["repolore.toml"]
    assert not paths.data_dir.exists()


def test_an_empty_folder_can_take_the_copy(home: AppPaths, tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    copy_home(home, tmp_path / "empty")

    assert (tmp_path / "empty" / "db" / "default.db").exists()


def test_a_folder_that_isnt_empty_is_refused(home: AppPaths, tmp_path: Path) -> None:
    write_files(tmp_path / "taken", {"notes.md": "mine"})

    with pytest.raises(CopyError, match="isn't empty"):
        copy_home(home, tmp_path / "taken")

    assert [path.name for path in (tmp_path / "taken").iterdir()] == ["notes.md"]


def test_a_file_is_refused(home: AppPaths, tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")

    with pytest.raises(CopyError, match="isn't a folder"):
        copy_home(home, tmp_path / "file")


@pytest.mark.parametrize("inside", ["", "backup"])
def test_a_folder_inside_the_home_is_refused(home: AppPaths, inside: str) -> None:
    with pytest.raises(CopyError, match="inside the home"):
        copy_home(home, home.data_dir / inside)

    assert not (home.data_dir / "backup").exists()


def test_a_running_refresh_is_refused_and_nothing_is_copied(home: AppPaths, tmp_path: Path) -> None:
    with refreshing(home.data_dir), pytest.raises(RefreshRunning):
        copy_home(home, tmp_path / "new-home")

    assert not (tmp_path / "new-home").exists()


def test_a_refresh_starting_is_refused_but_an_old_request_is_not(home: AppPaths, tmp_path: Path) -> None:
    request = home.data_dir / REQUEST_FILE
    request.touch()
    with pytest.raises(RefreshRunning):
        copy_home(home, tmp_path / "new-home")

    long_ago = request.stat().st_mtime - STARTUP_SECONDS - 1
    os.utime(request, (long_ago, long_ago))
    copy_home(home, tmp_path / "new-home")

    assert not (tmp_path / "new-home" / REQUEST_FILE).exists()


@pytest.fixture
def failing_halfway(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(source: Path, target: Path) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(copying, "back_up", fail)


@pytest.mark.usefixtures("failing_halfway")
def test_a_copy_that_fails_halfway_removes_the_folder_it_made(home: AppPaths, tmp_path: Path) -> None:
    before = snapshot(home.data_dir)

    with pytest.raises(sqlite3.OperationalError):
        copy_home(home, tmp_path / "parent" / "new-home")

    assert not (tmp_path / "parent" / "new-home").exists()
    assert snapshot(home.data_dir) == before


def test_a_copy_that_fails_at_the_end_empties_the_folder_it_was_given(
    home: AppPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(*_: object) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(copying, "_copy_setting", full)
    (tmp_path / "empty").mkdir()

    with pytest.raises(OSError, match="No space left"):
        copy_home(home, tmp_path / "empty")

    assert list((tmp_path / "empty").iterdir()) == []


def test_a_refresh_can_run_again_after_a_copy_fails(home: AppPaths, tmp_path: Path) -> None:
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "x").touch()
    with pytest.raises(CopyError):
        copy_home(home, tmp_path / "taken")

    with refreshing(home.data_dir):
        pass


def test_a_file_system_without_clones_gets_a_plain_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_clones(source: Path, target: Path) -> None:
        raise OSError("Operation not supported")

    monkeypatch.setattr(copying, "_clone", no_clones)
    (tmp_path / "model").write_bytes(b"weights")
    os.utime(tmp_path / "model", (1_700_000_000, 1_700_000_000))

    assert not clone_or_copy(tmp_path / "model", tmp_path / "copy")

    assert (tmp_path / "copy").read_bytes() == b"weights"
    assert (tmp_path / "copy").stat().st_mtime == 1_700_000_000


def test_a_home_on_a_file_system_without_clones_is_copied_whole(
    home: AppPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_clones(source: Path, target: Path) -> None:
        raise OSError("Operation not supported")

    monkeypatch.setattr(copying, "_clone", no_clones)

    copied = copy_home(home, tmp_path / "new-home")

    assert (copied.files, copied.cloned) == (len(KEPT) + 2, 0)
    assert (tmp_path / "new-home" / "raw" / "octo" / "repo" / "pulls.json").read_text(encoding="utf-8") == "[]"


def test_a_clone_or_copy_has_the_same_bytes_and_times(tmp_path: Path) -> None:
    (tmp_path / "model").write_bytes(b"weights" * 1000)
    os.utime(tmp_path / "model", (1_700_000_000, 1_700_000_000))

    clone_or_copy(tmp_path / "model", tmp_path / "copy")

    assert (tmp_path / "copy").read_bytes() == b"weights" * 1000
    assert (tmp_path / "copy").stat().st_mtime == 1_700_000_000
