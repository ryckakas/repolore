from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def max_tokens(self) -> int:
        """The most text tokens the model reads in one document; the rest would be cut."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Vectors for the texts in one forward pass: the caller decides what fits in a batch."""
        ...

    def count_tokens(self, texts: Sequence[str]) -> list[int]: ...

    def embed_query(self, text: str) -> Vectors: ...

    def warm_up(self) -> None:
        """Load the model now instead of on the first embed."""
        ...
