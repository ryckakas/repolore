import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache, partial
from itertools import batched
from typing import TYPE_CHECKING

import numpy as np

from repolore.embedding import Vectors
from repolore.hub import downloading

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer

COUNT_BATCH = 1024
PROMPT_MARGIN = 16


@dataclass(frozen=True)
class LocalModel:
    name: str
    revision: str
    dim: int
    max_seq_length: int
    trust_remote_code: bool = False
    prompts: dict[str, str] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return f"{self.name}@{self.revision[:12]}"


VOYAGE_NANO = LocalModel(
    name="voyageai/voyage-4-nano",
    # The model runs its own code (trust_remote_code), so it is pinned to a reviewed commit.
    revision="5fb34e870f49f19e79c94d5e118f5ee730133279",
    dim=1024,
    max_seq_length=32_768,
    trust_remote_code=True,
)
BGE_SMALL = LocalModel(
    name="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dim=384,
    max_seq_length=512,
    # The model card's query instruction; the checkpoint doesn't ship it as a saved prompt.
    prompts={"query": "Represent this sentence for searching relevant passages: "},
)


class ModelName(StrEnum):
    VOYAGE_NANO = "voyage-4-nano"
    BGE_SMALL = "bge-small"


MODELS = {ModelName.VOYAGE_NANO: VOYAGE_NANO, ModelName.BGE_SMALL: BGE_SMALL}


def model_named_in(model_id: str) -> ModelName | None:
    """The model a recorded id names, whichever revision it was pinned to then."""
    return next((name for name, spec in MODELS.items() if model_id.startswith(f"{spec.name}@")), None)


class UnknownModelError(Exception):
    pass


class LocalEmbedder:
    """A sentence-transformers model on the CPU. The default, voyage-4-nano, shares Voyage's hosted embedding space."""

    def __init__(self, model: LocalModel = VOYAGE_NANO) -> None:
        self._spec = model
        self._model: SentenceTransformer | None = None
        self._tokenizer: Tokenizer | None = None
        self._loading = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._spec.model_id

    @property
    def dim(self) -> int:
        return self._spec.dim

    @property
    def max_tokens(self) -> int:
        # Room for the document prompt (7 tokens for voyage-4-nano) and special tokens (2 for bge-small).
        return self._spec.max_seq_length - PROMPT_MARGIN

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        batch = list(texts)
        vectors = self._loaded().encode_document(batch, batch_size=max(len(batch), 1), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Tokens of each text before truncation, without the document prompt or special tokens."""
        tokenizer = self._tokenizer_loaded()
        # An encoding keeps ids, offsets and masks (about 100 bytes a token), so a whole corpus at once costs GBs.
        return [
            len(encoding.ids)
            for part in batched(texts, COUNT_BATCH)
            for encoding in tokenizer.encode_batch(list(part), add_special_tokens=False)
        ]

    def embed_query(self, text: str) -> Vectors:
        return np.asarray(self._loaded().encode_query(text, normalize_embeddings=True), dtype=np.float32)

    def _loaded(self) -> "SentenceTransformer":
        # The MCP server runs tool calls on worker threads; parallel first searches must not load the model twice.
        with self._loading:
            return self._load_once()

    def _tokenizer_loaded(self) -> "Tokenizer":
        with self._loading:
            if self._tokenizer is None:
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(_tokenizer_file(self._spec))
                # Some tokenizer.json files carry the model's padding or truncation, which would distort counts.
                tokenizer.no_padding()
                tokenizer.no_truncation()
                self._tokenizer = tokenizer
            return self._tokenizer

    def _load_once(self) -> "SentenceTransformer":
        if self._model is None:
            # Imported on first use: loading torch takes seconds, and most commands never embed anything.
            import torch
            from sentence_transformers import SentenceTransformer

            spec = self._spec
            # BF16 checkpoints run about 9x slower than float32 on CPUs without native BF16 math.
            load = partial(
                SentenceTransformer,
                spec.name,
                revision=spec.revision,
                trust_remote_code=spec.trust_remote_code,
                truncate_dim=spec.dim,
                prompts=spec.prompts or None,
                device="cpu",
                model_kwargs={"dtype": torch.float32},
            )
            self._model = _from_cache_or_hub(spec, load)
        return self._model


def _from_cache_or_hub(spec: LocalModel, load: Callable[..., "SentenceTransformer"]) -> "SentenceTransformer":
    from huggingface_hub import try_to_load_from_cache

    # A pinned revision never changes, so a cached copy is current and asking the Hub only costs requests. A cache
    # without modules.json would quietly load as a mean-pooling model instead of failing, so it goes to the Hub.
    if isinstance(try_to_load_from_cache(spec.name, "modules.json", revision=spec.revision), str):
        with suppress(OSError):
            return load(local_files_only=True)
    with downloading(spec.name, spec.revision):
        return load()


def _tokenizer_file(spec: LocalModel) -> str:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(spec.name, "tokenizer.json", revision=spec.revision)
    if isinstance(cached, str):
        return cached
    with downloading(spec.name, spec.revision):
        return hf_hub_download(spec.name, "tokenizer.json", revision=spec.revision)


@cache
def embedder_for(model_id: str) -> LocalEmbedder:
    """The embedder an index was built with, so queries land in its vector space; one per model, loaded once."""
    for spec in MODELS.values():
        if spec.model_id == model_id:
            return LocalEmbedder(spec)
    raise UnknownModelError(f"The index was built with {model_id}, which repolore can't load; rebuild it.")
