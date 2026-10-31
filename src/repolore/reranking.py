import threading
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Protocol

from repolore.hub import downloading
from repolore.store import Hit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
# Query plus passage, in model tokens: longer passages are cut, which keeps 50 candidates to a couple of seconds.
MAX_LENGTH = 256


class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


class CrossEncoderReranker:
    """Scores each candidate together with the query, which is slower than comparing vectors but reads both at once."""

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None
        self._loading = threading.Lock()

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        scores = self._loaded().predict([(query, f"{hit.header}\n\n{hit.text}") for hit in hits])
        scored = [replace(hit, score=float(score)) for hit, score in zip(hits, scores, strict=True)]
        return sorted(scored, key=lambda hit: hit.score, reverse=True)

    def _loaded(self) -> "CrossEncoder":
        with self._loading:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                load = partial(CrossEncoder, MODEL, revision=REVISION, max_length=MAX_LENGTH, device="cpu")
                # A pinned revision never changes, so a cached copy is current and asking the Hub only costs requests.
                try:
                    self._model = load(local_files_only=True)
                except OSError:
                    with downloading(MODEL, REVISION):
                        self._model = load()
            return self._model
