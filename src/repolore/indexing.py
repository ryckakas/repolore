import logging
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import numpy as np

from repolore.chunking import Chunk, chunk_limit
from repolore.embedding import Embedder, Vectors
from repolore.embedding.local import MODELS, ModelName, model_named_in
from repolore.store import EmbeddingCache, cache_key, write_index

TOKEN_BUDGET = 16_384
MAX_BATCH = 64
WHOLE = "whole"
INCLUDED = "included"
EXCLUDED = "excluded"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildSettings:
    max_tokens: int | None = None
    bot_reviews: bool = False
    model: ModelName = ModelName.VOYAGE_NANO

    @classmethod
    def recorded(cls, meta: Mapping[str, str]) -> Self:
        """The settings an index was built with; anything it doesn't record keeps the default."""
        model = model_named_in(meta.get("model_id", "")) or cls.model
        limit = meta.get("max_tokens", WHOLE)
        # The default limit is recorded as a number but read back as the default, so it follows a new model.
        default = limit == WHOLE or int(limit) == chunk_limit(None, MODELS[model].max_tokens)
        return cls(
            max_tokens=None if default else int(limit),
            bot_reviews=meta.get("bot_reviews") == INCLUDED,
            model=model,
        )

    def overridden(
        self, max_tokens: int | None = None, bot_reviews: bool | None = None, model: ModelName | None = None
    ) -> Self:
        """These settings with every option that was given; a max_tokens of 0 means the default limit."""
        return replace(
            self,
            max_tokens=self.max_tokens if max_tokens is None else (max_tokens or None),
            bot_reviews=self.bot_reviews if bot_reviews is None else bot_reviews,
            model=model or self.model,
        )

    def meta(self, limit: int) -> dict[str, str]:
        """What to record, with the limit the chunker applies rather than the option, which can name the same one."""
        return {"max_tokens": str(limit), "bot_reviews": self._bot_reviews}

    def described(self, limit: int) -> str:
        default = " (the default)" if self.max_tokens is None else ""
        return f"sections split at {limit} tokens{default}, bot reviews {self._bot_reviews}, {self.model}"

    @property
    def _bot_reviews(self) -> str:
        return INCLUDED if self.bot_reviews else EXCLUDED


@dataclass(frozen=True)
class BuildStats:
    chunks: int
    embedded: int


def build_index(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    db: sqlite3.Connection,
    cache: EmbeddingCache,
    settings: Mapping[str, str] | None = None,
) -> BuildStats:
    keys = [cache_key(embedder.model_id, chunk.embedded_text) for chunk in chunks]
    vectors = cache.get(set(keys))
    missing = {key: chunk.embedded_text for key, chunk in zip(keys, chunks, strict=True) if key not in vectors}
    done = 0
    tokens = embedder.count_tokens(list(missing.values())) if missing else []
    for batch in _token_batches(missing, tokens):
        batch_keys = [key for key, _ in batch]
        embedded = embedder.embed_documents([text for _, text in batch])
        cache.put(batch_keys, embedded)
        vectors.update(zip(batch_keys, embedded, strict=True))
        done += len(batch)
        log.info("embedded %d of %d chunks", done, len(missing))
    write_index(db, chunks, _stack(vectors, keys, embedder.dim), embedder.model_id, embedder.dim, settings)
    return BuildStats(chunks=len(chunks), embedded=len(missing))


def _token_batches(texts: Mapping[str, str], token_counts: list[int]) -> Iterator[list[tuple[str, str]]]:
    # A batch is padded to its longest text, so it costs its size times that length. Shortest first, a batch grows
    # until that cost passes the budget; one long chunk then pads nothing but itself.
    batch: list[tuple[str, str]] = []
    for count, key, text in sorted(zip(token_counts, texts, texts.values(), strict=True)):
        if batch and ((len(batch) + 1) * count > TOKEN_BUDGET or len(batch) == MAX_BATCH):
            yield batch
            batch = []
        batch.append((key, text))
    if batch:
        yield batch


def _stack(vectors: dict[str, Vectors], keys: list[str], dim: int) -> Vectors:
    return np.stack([vectors[key] for key in keys]) if keys else np.empty((0, dim), dtype=np.float32)
