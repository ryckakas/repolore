import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEmbedder

from repolore.chunking import Chunk
from repolore.store import (
    EmbeddingCache,
    Filters,
    StoreError,
    cache_key,
    changed_settings,
    connect,
    index_model,
    index_settings,
    keyword_matches,
    nearest,
    write_index,
)


@pytest.fixture
def index(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> Iterator[sqlite3.Connection]:
    with closing(connect(tmp_path / "db" / "test.db")) as db:
        vectors = fake_embedder.embed_documents([chunk.embedded_text for chunk in corpus])
        write_index(db, corpus, vectors, fake_embedder.model_id, fake_embedder.dim)
        yield db


def search(
    index: sqlite3.Connection, embedder: FakeEmbedder, query: str, k: int = 4, filters: Filters | None = None
) -> list[str]:
    return [hit.url for hit in nearest(index, embedder.embed_query(query), k, filters or Filters())]


def test_nearest_ranks_the_closest_chunk_first(index: sqlite3.Connection, fake_embedder: FakeEmbedder) -> None:
    hits = nearest(index, fake_embedder.embed_query("registry times out, retry with backoff"), 2, Filters())

    assert hits[0].url == "https://github.com/octo/repo/pull/1"
    assert hits[0].header == "octo/repo · PR #1 · Change 1"
    assert hits[0].score > hits[1].score
    assert len(hits) == 2


def test_nearest_returns_every_chunk_when_k_exceeds_the_index(
    index: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    assert len(search(index, fake_embedder, "anything", k=50)) == 4


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"repo": "octo/charts"}, ["https://github.com/octo/repo/pull/2"]),
        ({"kind": "issue"}, ["https://github.com/octo/repo/issues/3"]),
        ({"author": "hubot"}, ["https://github.com/octo/repo/pull/2"]),
        ({"repo": "octo/repo", "kind": "pr"}, ["https://github.com/octo/repo/pull/1"]),
        ({"repo": "octo/missing"}, []),
    ],
)
def test_filters_apply_inside_the_knn_query(
    index: sqlite3.Connection, fake_embedder: FakeEmbedder, filters: dict[str, str], expected: list[str]
) -> None:
    assert search(index, fake_embedder, "registry", filters=Filters(**filters)) == expected


def test_index_records_its_model(index: sqlite3.Connection) -> None:
    assert index_model(index) == ("fake-bag-of-words", 64)


def test_empty_database_has_no_model(tmp_path: Path) -> None:
    with closing(connect(tmp_path / "empty.db")) as db, pytest.raises(StoreError, match="run `repolore build` first"):
        index_model(db)


def test_rewriting_replaces_the_previous_index(
    index: sqlite3.Connection, corpus: list[Chunk], fake_embedder: FakeEmbedder
) -> None:
    write_index(index, corpus[:1], fake_embedder.embed_documents([corpus[0].embedded_text]), "fake-bag-of-words", 64)

    assert search(index, fake_embedder, "registry") == ["https://github.com/octo/repo/pull/1"]


def test_keyword_index_keeps_snake_case_words_whole(index: sqlite3.Connection) -> None:
    rows = index.execute("select rowid from chunks_fts where chunks_fts match '\"snake_case\"'").fetchall()

    assert len(rows) == 1
    assert index.execute("select count(*) from chunks_fts where chunks_fts match 'snake'").fetchone() == (0,)


def test_embedding_cache_round_trips_vectors(tmp_path: Path) -> None:
    db = connect(tmp_path / "cache.db")
    cache = EmbeddingCache(db)
    vectors = np.array([[1.0, 0.5], [0.0, -2.0]], dtype=np.float32)

    cache.put(["a", "b"], vectors)

    found = cache.get(["a", "b", "missing"])
    assert set(found) == {"a", "b"}
    np.testing.assert_array_equal(found["b"], vectors[1])
    db.close()


def test_cache_key_depends_on_model_and_text() -> None:
    keys = {cache_key("m1", "text"), cache_key("m2", "text"), cache_key("m1", "other")}

    assert len(keys) == 3
    assert cache_key("m1", "text") == cache_key("m1", "text")


def keyword(index: sqlite3.Connection, query: str, **filters: str) -> list[str]:
    return [hit.url for hit in keyword_matches(index, query, 10, Filters(**filters))]


def test_keyword_search_ranks_rarer_matching_words_first(index: sqlite3.Connection) -> None:
    hits = keyword_matches(index, "Modbus registry", 10, Filters())

    assert {hit.url for hit in hits} == {
        "https://github.com/octo/repo/issues/3",
        "https://github.com/octo/repo/pull/1",
        "https://github.com/octo/repo/pull/2",
    }
    assert hits[0].url == "https://github.com/octo/repo/issues/3"
    assert all(hit.score > 0 for hit in hits)


def test_keyword_search_treats_query_syntax_as_plain_words(index: sqlite3.Connection) -> None:
    assert keyword(index, 'snake_case" OR NOT (config*') == ["https://github.com/octo/repo/blob/main/README.md"]


@pytest.mark.parametrize("query", ["", "?!", "unmatched"])
def test_keyword_search_without_matching_words_is_empty(index: sqlite3.Connection, query: str) -> None:
    assert keyword(index, query) == []


def test_keyword_search_applies_filters(index: sqlite3.Connection) -> None:
    assert keyword(index, "registry", repo="octo/charts", author="hubot") == ["https://github.com/octo/repo/pull/2"]
    assert keyword(index, "registry", kind="issue") == []


def test_build_settings_are_kept_beside_the_model(
    tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder
) -> None:
    with closing(connect(tmp_path / "settings.db")) as db:
        vectors = fake_embedder.embed_documents([chunk.embedded_text for chunk in corpus])
        write_index(db, corpus, vectors, fake_embedder.model_id, fake_embedder.dim, {"max_tokens": "200"})

        assert index_settings(db) == {"max_tokens": "200", "model_id": "fake-bag-of-words", "dim": "64"}


def test_python_without_sqlite_extensions_gets_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class NoExtensions:
        closed = False

        def close(self) -> None:
            self.closed = True

    connection = NoExtensions()
    monkeypatch.setattr(sqlite3, "connect", lambda *_, **__: connection)

    with pytest.raises(StoreError, match="can't load extensions"):
        connect(tmp_path / "index.db")

    assert connection.closed


def test_changed_settings_reports_only_existing_differences(index: sqlite3.Connection) -> None:
    assert changed_settings(index, {"model_id": "other", "new": "value"}) == {
        "model_id": ("fake-bag-of-words", "other")
    }
    assert changed_settings(index, {"model_id": "fake-bag-of-words"}) == {}


def test_changed_settings_accepts_an_empty_index(tmp_path: Path) -> None:
    with closing(connect(tmp_path / "empty.db")) as db:
        assert changed_settings(db, {"model_id": "other"}) == {}
