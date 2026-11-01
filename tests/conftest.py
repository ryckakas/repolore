import hashlib
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from repolore.chunking import Chunk, chunk_document
from repolore.documents import Document, DocumentKind, Visibility
from repolore.embedding import Vectors

FAKE_DIM = 64


class FakeEmbedder:
    """Bag-of-words vectors: texts sharing words land close together, deterministically and offline."""

    model_id = "fake-bag-of-words"
    dim = FAKE_DIM
    max_tokens = 10_000

    def __init__(self) -> None:
        self.embedded: list[str] = []
        self.batches: list[list[str]] = []
        self.warm_ups = 0

    def warm_up(self) -> None:
        self.warm_ups += 1

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.embedded += texts
        self.batches.append(list(texts))
        return np.stack([self._vector(text) for text in texts]) if texts else np.empty((0, FAKE_DIM), np.float32)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [len(text.split()) for text in texts]

    def embed_query(self, text: str) -> Vectors:
        return self._vector(text)

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(FAKE_DIM, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % FAKE_DIM] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


DocumentFactory = Callable[..., Document]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_document() -> DocumentFactory:
    def make(number: int = 1, body: str = "Body.", **changes: object) -> Document:
        fields: dict[str, object] = {
            "url": f"https://github.com/octo/repo/pull/{number}",
            "repo": "octo/repo",
            "kind": DocumentKind.PR,
            "visibility": Visibility.PUBLIC,
            "title": f"Change {number}",
            "body": body,
            "author": "octocat",
            "created_at": datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=number),
        }
        return Document.model_validate(fields | changes)

    return make


@pytest.fixture
def corpus(make_document: DocumentFactory) -> list[Chunk]:
    documents = [
        make_document(1, "Retry the upload with exponential backoff when the registry times out."),
        make_document(2, "Cache the Helm chart in the OCI registry.", repo="octo/charts", author="hubot"),
        make_document(
            3,
            "The solar battery reports power over Modbus TCP.",
            kind=DocumentKind.ISSUE,
            url="https://github.com/octo/repo/issues/3",
        ),
        make_document(
            4,
            "Document the snake_case config keys.",
            kind=DocumentKind.DOC,
            author=None,
            url="https://github.com/octo/repo/blob/main/README.md",
            title="README.md",
        ),
    ]
    return [chunk for document in documents for chunk in chunk_document(document)]
