from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import pytest
from conftest import DocumentFactory, FakeEmbedder

from repolore import indexing
from repolore.chunking import Chunk, chunk_document
from repolore.embedding.local import BGE_SMALL, VOYAGE_NANO, ModelName
from repolore.indexing import MAX_BATCH, BuildSettings, build_index
from repolore.store import EmbeddingCache, cache_key, connect, index_model


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


def same_length_chunks(make_document: DocumentFactory, count: int) -> list[Chunk]:
    return [chunk for n in range(1, count + 1) for chunk in chunk_document(make_document(n, f"Text {n % 10}."))]


def test_short_chunks_share_batches_up_to_the_cap(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> None:
    chunks = same_length_chunks(make_document, 2 * MAX_BATCH + 1)

    assert build(tmp_path, chunks, fake_embedder) == len(chunks)
    assert [len(batch) for batch in fake_embedder.batches] == [MAX_BATCH, MAX_BATCH, 1]


def test_a_batch_ends_before_its_padded_size_passes_the_budget(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = same_length_chunks(make_document, 7)
    [tokens] = fake_embedder.count_tokens([chunks[0].embedded_text])
    monkeypatch.setattr(indexing, "TOKEN_BUDGET", 3 * tokens)

    build(tmp_path, chunks, fake_embedder)

    assert [len(batch) for batch in fake_embedder.batches] == [3, 3, 1]


def test_a_chunk_over_the_budget_is_embedded_alone_after_shorter_ones(
    tmp_path: Path, fake_embedder: FakeEmbedder, make_document: DocumentFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    [long] = chunk_document(make_document(9, " ".join(["word"] * 100)))
    short = same_length_chunks(make_document, 2)
    monkeypatch.setattr(indexing, "TOKEN_BUDGET", 50)

    build(tmp_path, [long, *short], fake_embedder)

    first, second = fake_embedder.batches
    assert sorted(first) == sorted(chunk.embedded_text for chunk in short)
    assert second == [long.embedded_text]


def test_each_chunk_keeps_its_own_vector_when_batches_reorder_them(
    tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder
) -> None:
    build(tmp_path, list(reversed(corpus)), fake_embedder)

    with closing(connect(tmp_path / "cache.db")) as db:
        cached = EmbeddingCache(db)
        for chunk in corpus:
            key = cache_key(fake_embedder.model_id, chunk.embedded_text)
            assert cached.get([key])[key].tolist() == fake_embedder.embed_query(chunk.embedded_text).tolist()


class NoCountingEmbedder(FakeEmbedder):
    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        raise AssertionError("nothing to embed, so nothing to count")


def test_nothing_to_embed_counts_no_tokens(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> None:
    build(tmp_path, corpus, fake_embedder)

    assert build(tmp_path, corpus, NoCountingEmbedder()) == 0


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
    settings = BuildSettings(max_tokens=50, bot_reviews=True, model=ModelName.BGE_SMALL)

    assert settings.meta(50) == {"max_tokens": "50", "bot_reviews": "included"}
    assert BuildSettings.recorded(settings.meta(50) | {"model_id": BGE_SMALL.model_id}) == settings


@pytest.mark.parametrize(
    ("model_id", "limit", "max_tokens"),
    [
        (VOYAGE_NANO.model_id, "8192", None),
        (VOYAGE_NANO.model_id, "8191", 8191),
        (VOYAGE_NANO.model_id, "whole", None),
        (BGE_SMALL.model_id, "496", None),
        (BGE_SMALL.model_id, "8192", 8192),
    ],
)
def test_a_recorded_limit_reads_back_as_the_default_when_it_is_the_models_default(
    model_id: str, limit: str, max_tokens: int | None
) -> None:
    assert BuildSettings.recorded({"model_id": model_id, "max_tokens": limit}).max_tokens == max_tokens


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
    assert BuildSettings().described(8192) == (
        "sections split at 8192 tokens (the default), bot reviews excluded, voyage-4-nano"
    )
    assert BuildSettings(50, True).described(50) == "sections split at 50 tokens, bot reviews included, voyage-4-nano"
