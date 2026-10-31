import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from enum import StrEnum

from repolore.embedding import Embedder
from repolore.reranking import Reranker
from repolore.store import Filters, Hit, StoreError, index_model, keyword_matches, nearest

CANDIDATES = 50
RRF_K = 60


class Mode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


def search(
    db: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    k: int,
    filters: Filters,
    mode: Mode = Mode.VECTOR,
    reranker: Reranker | None = None,
    distinct: bool = False,
) -> list[Hit]:
    model_id, _ = index_model(db)
    if model_id != embedder.model_id:
        raise StoreError(f"The index was built with {model_id}, not {embedder.model_id}; rebuild it.")
    depth = max(k, CANDIDATES) if reranker or distinct else k
    hits = _candidates(db, embedder, query, depth, filters, mode)
    ranked = reranker.rerank(query, hits) if reranker else hits
    if distinct:
        ranked = _distinct_sources(ranked)
    return ranked[:k]


def _distinct_sources(hits: list[Hit]) -> list[Hit]:
    seen: set[str] = set()
    distinct = []
    for hit in hits:
        if hit.url not in seen:
            distinct.append(hit)
            seen.add(hit.url)
    return distinct


def _candidates(
    db: sqlite3.Connection, embedder: Embedder, query: str, depth: int, filters: Filters, mode: Mode
) -> list[Hit]:
    if mode is Mode.VECTOR:
        return nearest(db, embedder.embed_query(query), depth, filters)
    if mode is Mode.KEYWORD:
        return keyword_matches(db, query, depth, filters)
    depth = max(depth, CANDIDATES)
    vector = nearest(db, embedder.embed_query(query), depth, filters)
    return fuse(vector, keyword_matches(db, query, depth, filters))[:depth]


def fuse(*rankings: Sequence[Hit]) -> list[Hit]:
    """Reciprocal rank fusion: ranks, not scores, are combined, because cosine and BM25 aren't on one scale."""
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (RRF_K + rank)
            hits.setdefault(hit.chunk_id, hit)
    ranked = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [replace(hits[chunk_id], score=scores[chunk_id]) for chunk_id in ranked]
