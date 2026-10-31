from contextlib import closing
from pathlib import Path

import pytest
from conftest import DocumentFactory, FakeEmbedder

from repolore.chunking import Chunk, chunk_document
from repolore.embedding.local import BGE_SMALL, ModelName
from repolore.indexing import BATCH_SIZE, BuildSettings, build_index
from repolore.store import EmbeddingCache, connect, index_model


def build(tmp_path: Path, chunks: list[Chunk], embedder: FakeEmbedder, name: str = "test") -> int:
    with closing(connect(tmp_path / f"{name}.db")) as index, closing(connect(tmp_path / "cache.db")) as cache:
        stats = build_index(chunks, embedder, index, EmbeddingCache(cache))
    assert stats.chunks == len(chunks)
    return stats.embedded


def test_first_build_embeds_every_chunk(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> None:
    assert build(tmp_path, corpus, fake_embedder) == len(corpus)


def test_second_build_embeds_nothing(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> None:
    build(tmp_path, corpus, fake_embedder)

    assert build(tmp_path, corpus, fake_embedder) == 0


def test_only_changed_chunks_are_embedded_again(
    tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    build(tmp_path, corpus, fake_embedder)
    changed = chunk_document(make_document(1, "Retry with linear backoff instead."))

    assert build(tmp_path, [*changed, *corpus[1:]], fake_embedder) == 1


def test_the_cache_is_shared_between_indexes(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> None:
    build(tmp_path, corpus, fake_embedder, name="baseline")

    assert build(tmp_path, corpus, fake_embedder, name="experiment") == 0


def test_many_chunks_are_embedded_in_batches(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    chunks = [chunk for n in range(1, 2 * BATCH_SIZE + 2) for chunk in chunk_document(make_document(n, f"Text {n}."))]

    assert build(tmp_path, chunks, fake_embedder) == len(chunks)
    assert len(fake_embedder.embedded) == len(chunks)


def test_empty_corpus_builds_an_empty_index(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    build(tmp_path, [], fake_embedder)

    with closing(connect(tmp_path / "test.db")) as index:
        assert index_model(index) == ("fake-bag-of-words", 64)
        assert index.execute("select count(*) from chunks").fetchone() == (0,)


def test_progress_is_logged(
    tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        build(tmp_path, corpus, fake_embedder)

    assert f"embedded {len(corpus)} of {len(corpus)} chunks" in caplog.text


def test_an_index_without_recorded_settings_builds_with_the_defaults() -> None:
    assert BuildSettings.recorded({}) == BuildSettings(max_tokens=None, bot_reviews=False, model=ModelName.VOYAGE_NANO)


def test_recorded_settings_round_trip() -> None:
    settings = BuildSettings(max_tokens=8192, bot_reviews=True, model=ModelName.BGE_SMALL)

    assert BuildSettings.recorded(settings.meta() | {"model_id": BGE_SMALL.model_id}) == settings


def test_an_unknown_recorded_model_falls_back_to_the_default() -> None:
    assert BuildSettings.recorded({"model_id": "fake-bag-of-words"}).model == ModelName.VOYAGE_NANO


def test_only_given_options_override_the_recorded_settings() -> None:
    recorded = BuildSettings(max_tokens=50, bot_reviews=True, model=ModelName.BGE_SMALL)

    assert recorded.overridden() == recorded
    assert recorded.overridden(max_tokens=0, bot_reviews=False) == BuildSettings(model=ModelName.BGE_SMALL)
    assert recorded.overridden(max_tokens=1, model=ModelName.VOYAGE_NANO) == BuildSettings(
        1, True, ModelName.VOYAGE_NANO
    )


def test_settings_describe_themselves() -> None:
    assert str(BuildSettings()) == "whole sections, bot reviews excluded, voyage-4-nano"
    assert str(BuildSettings(8192, True)) == "sections split at 8192 tokens, bot reviews included, voyage-4-nano"
