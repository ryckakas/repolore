import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path

import pytest
from conftest import DocumentFactory, FakeEmbedder

from repolore.chunking import Chunk, chunk_document
from repolore.retrieval import RRF_K, Mode, fuse, search
from repolore.store import Filters, Hit, StoreError, connect, write_index


@pytest.fixture
def index(tmp_path: Path, corpus: list[Chunk], fake_embedder: FakeEmbedder) -> Iterator[sqlite3.Connection]:
    with closing(connect(tmp_path / "test.db")) as db:
        vectors = fake_embedder.embed_documents([chunk.embedded_text for chunk in corpus])
        write_index(db, corpus, vectors, fake_embedder.model_id, fake_embedder.dim)
        yield db


def test_search_embeds_the_query_and_returns_the_nearest_chunks(
    index: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    hits = search(index, fake_embedder, "solar battery over Modbus", 1, Filters())

    assert [hit.url for hit in hits] == ["https://github.com/octo/repo/issues/3"]


def test_search_refuses_an_index_built_with_another_model(
    index: sqlite3.Connection, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeEmbedder, "model_id", "other-model")

    with pytest.raises(StoreError, match="built with fake-bag-of-words, not other-model"):
        search(index, fake_embedder, "anything", 5, Filters())


def urls(hits: list[Hit]) -> list[str]:
    return [hit.url for hit in hits]


def test_keyword_mode_uses_bm25_only(index: sqlite3.Connection, fake_embedder: FakeEmbedder) -> None:
    hits = search(index, fake_embedder, "snake_case", 5, Filters(), Mode.KEYWORD)

    assert urls(hits) == ["https://github.com/octo/repo/blob/main/README.md"]


def test_hybrid_mode_fuses_both_rankings(index: sqlite3.Connection, fake_embedder: FakeEmbedder) -> None:
    hits = search(index, fake_embedder, "registry backoff", 4, Filters(), Mode.HYBRID)

    assert urls(hits)[0] == "https://github.com/octo/repo/pull/1"
    assert len(hits) == 4
    assert hits[0].score == pytest.approx(2 / (RRF_K + 1))


def test_fusion_adds_reciprocal_ranks_across_lists(corpus: list[Chunk]) -> None:
    a, b, c = (hit_for(chunk) for chunk in corpus[:3])

    fused = fuse([a, b], [c, b])

    assert [hit.chunk_id for hit in fused] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert fused[0].score == pytest.approx(1 / (RRF_K + 2) * 2)
    assert fused[1].score == pytest.approx(1 / (RRF_K + 1))


def test_reranker_reorders_a_deeper_candidate_list(index: sqlite3.Connection, fake_embedder: FakeEmbedder) -> None:
    reranker = ReverseReranker()

    hits = search(index, fake_embedder, "registry", 1, Filters(), Mode.VECTOR, reranker)

    assert reranker.seen == 4
    assert len(hits) == 1
    assert hits[0].url != search(index, fake_embedder, "registry", 1, Filters())[0].url


@pytest.fixture
def section_index(
    index: sqlite3.Connection, corpus: list[Chunk], fake_embedder: FakeEmbedder, make_document: DocumentFactory
) -> sqlite3.Connection:
    doc = make_document(9, "## Retry\nregistry backoff\n\n## Timeout\nregistry backoff timeout")
    chunks = corpus[2:3] + list(chunk_document(doc))
    write_index(
        index,
        chunks,
        fake_embedder.embed_documents([c.embedded_text for c in chunks]),
        fake_embedder.model_id,
        fake_embedder.dim,
    )
    return index


def test_distinct_returns_highest_ranked_chunk_per_url(
    section_index: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    hits = search(section_index, fake_embedder, "registry backoff", 2, Filters(), distinct=True)
    assert len(hits) == 2
    assert len({hit.url for hit in hits}) == 2


def test_default_search_keeps_multiple_sections(section_index: sqlite3.Connection, fake_embedder: FakeEmbedder) -> None:
    hits = search(section_index, fake_embedder, "registry backoff", 6, Filters())
    assert sum(hit.url.endswith("/9") for hit in hits) == 2


def test_distinct_keeps_the_rerankers_best_section(
    section_index: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    original = search(section_index, fake_embedder, "registry backoff", 6, Filters())
    sections = [hit for hit in original if hit.url.endswith("/9")]
    assert len(sections) == 2
    hits = search(
        section_index, fake_embedder, "registry backoff", 6, Filters(), reranker=ReverseReranker(), distinct=True
    )
    assert [hit.chunk_id for hit in hits if hit.url.endswith("/9")] == [sections[-1].chunk_id]


class ReverseReranker:
    def __init__(self) -> None:
        self.seen = 0

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        self.seen = len(hits)
        return list(reversed(hits))


def hit_for(chunk: Chunk) -> Hit:
    document = chunk.document
    return Hit(
        chunk.id,
        document.url,
        document.repo,
        document.kind,
        document.title,
        chunk.section,
        document.author,
        None,
        chunk.header,
        chunk.text,
        0.0,
    )
