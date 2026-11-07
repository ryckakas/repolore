import os
import platform
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from repolore.embedding import Vectors
from repolore.embedding.model_files import ModelFile, ModelRelease, ensure, missing
from repolore.hub import ModelUnavailableError
from repolore.progress import Report, ignore

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    from tokenizers import Tokenizer

COUNT_BATCH = 1024
PROMPT_MARGIN = 16
SMALLEST_NORM = 1e-12


@dataclass(frozen=True)
class LocalModel:
    name: str
    revision: str
    dim: int
    max_seq_length: int
    document_prompt: str
    query_prompt: str
    release: ModelRelease

    @property
    def model_id(self) -> str:
        return f"{self.name}@{self.revision[:12]}"

    @property
    def max_tokens(self) -> int:
        # Room for the document prompt, 7 tokens.
        return self.max_seq_length - PROMPT_MARGIN


VOYAGE_NANO = LocalModel(
    name="voyageai/voyage-4-nano",
    # The revision the release was exported from. Its vectors match that revision's to float rounding, so an embedding
    # cache and an index built from it stay valid under the same id.
    revision="5fb34e870f49f19e79c94d5e118f5ee730133279",
    dim=1024,
    max_seq_length=32_768,
    document_prompt="Represent the document for retrieval: ",
    query_prompt="Represent the query for retrieving supporting documents: ",
    release=ModelRelease(
        name="voyage-4-nano-1",
        url="https://github.com/ryckakas/repolore-models/releases/download/voyage-4-nano-1",
        graph=ModelFile(
            "voyage-4-nano.onnx", 1_135_877, "e2df8004432a30bec197d9e498c8d3d136029af8e4e67ea634a47cfc108ed2e5"
        ),
        weights=ModelFile(
            "voyage-4-nano.onnx.data", 1_385_807_872, "a76ace9892d55dbfb96609e022a60fe1ebf15b542e7283942de6d2403e1b5191"
        ),
        tokenizer=ModelFile(
            "tokenizer.json", 7_031_645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
        ),
    ),
)


class UnknownModelError(Exception):
    pass


class Engine(Protocol):
    def embed(self, ids: list[list[int]]) -> Vectors:
        """Normalized vectors for texts already tokenized, prompt included."""
        ...


class OnnxEngine:
    """The exported graph on ONNX Runtime's CPU kernels."""

    def __init__(self, graph_file: Path, dim: int) -> None:
        # Imported on first use: most commands never embed anything.
        import onnxruntime

        options = onnxruntime.SessionOptions()
        if sys.platform == "darwin" and platform.machine() == "arm64":
            # ONNX Runtime's default thread count is about 20% slower on Apple Silicon.
            options.intra_op_num_threads = os.cpu_count() or 0
        self._session: InferenceSession = onnxruntime.InferenceSession(
            str(graph_file), options, providers=["CPUExecutionProvider"]
        )
        self._dim = dim

    def embed(self, ids: list[list[int]]) -> Vectors:
        # One text per run: flash attention needs a graph with no padding mask, so padding would change vectors.
        return np.stack([self._embed_one(text) for text in ids])

    def _embed_one(self, ids: list[int]) -> Vectors:
        tokens = np.array([ids], dtype=np.int64)
        outputs = self._session.run(None, {"input_ids": tokens, "attention_mask": np.ones_like(tokens)})
        return pool(np.asarray(outputs[0], dtype=np.float32)[0], self._dim)


def pool(token_vectors: Vectors, dim: int) -> Vectors:
    """Mean over the tokens, normalized, cut to the model's dimensions and normalized again."""
    return _unit(_unit(token_vectors.mean(axis=0))[:dim])


def _unit(vector: Vectors) -> Vectors:
    return vector / max(float(np.linalg.norm(vector)), SMALLEST_NORM)


class _Tokenizers(NamedTuple):
    counting: "Tokenizer"
    truncating: "Tokenizer"


class LocalEmbedder:
    """A model run on this machine: tokenizers, prompts and truncation here, the forward pass in its engine."""

    def __init__(self, model: LocalModel, models_dir: Path) -> None:
        self._spec = model
        self._models_dir = models_dir
        self._folder = model.release.folder(models_dir)
        self._engine: Engine | None = None
        self._tokenizers: _Tokenizers | None = None
        self._loading = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._spec.model_id

    @property
    def dim(self) -> int:
        return self._spec.dim

    @property
    def max_tokens(self) -> int:
        return self._spec.max_tokens

    def download(self, report: Report = ignore) -> None:
        release = self._spec.release
        ensure(release, release.files, self._models_dir, report)

    def download_tokenizer(self) -> None:
        release = self._spec.release
        ensure(release, [release.tokenizer], self._models_dir)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._embedded([self._spec.document_prompt + text for text in texts])

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Tokens of each text before truncation, without the document prompt."""
        tokenizer = self._tokenizers_loaded().counting
        # An encoding keeps ids, offsets and masks (about 100 bytes a token), so a whole corpus at once costs GBs.
        return [
            len(encoding.ids)
            for part in batched(texts, COUNT_BATCH)
            for encoding in tokenizer.encode_batch(list(part), add_special_tokens=False)
        ]

    def embed_query(self, text: str) -> Vectors:
        return self._embedded([self._spec.query_prompt + text])[0]

    def warm_up(self) -> None:
        self._engine_loaded()
        self._tokenizers_loaded()

    def _embedded(self, texts: list[str]) -> Vectors:
        encodings = self._tokenizers_loaded().truncating.encode_batch(texts, add_special_tokens=False)
        return self._engine_loaded().embed([encoding.ids for encoding in encodings])

    def _engine_loaded(self) -> Engine:
        # The MCP server runs tool calls on worker threads; parallel first searches must not load the model twice.
        with self._loading:
            if self._engine is None:
                release = self._spec.release
                self._require(release.files)
                self._engine = OnnxEngine(self._folder / release.graph.name, self._spec.dim)
            return self._engine

    def _tokenizers_loaded(self) -> _Tokenizers:
        with self._loading:
            if self._tokenizers is None:
                tokenizer = self._spec.release.tokenizer
                self._require([tokenizer])
                self._tokenizers = _load_tokenizers(self._folder / tokenizer.name, self._spec.max_seq_length)
            return self._tokenizers

    def _require(self, files: Sequence[ModelFile]) -> None:
        if missing(self._folder, files):
            raise ModelUnavailableError(f"{self._spec.name} isn't downloaded: run `repolore refresh`.")


def _load_tokenizers(tokenizer_file: Path, max_length: int) -> _Tokenizers:
    from tokenizers import Tokenizer

    # Two instances, because worker threads share them and a tokenizer's truncation can't change per call.
    counting, truncating = (Tokenizer.from_file(str(tokenizer_file)) for _ in range(2))
    for tokenizer in (counting, truncating):
        # Some tokenizer.json files carry the model's padding or truncation, which would distort counts.
        tokenizer.no_padding()
        tokenizer.no_truncation()
    truncating.enable_truncation(max_length=max_length)
    return _Tokenizers(counting, truncating)


def is_downloaded(spec: LocalModel, models_dir: Path) -> bool:
    return not missing(spec.release.folder(models_dir), spec.release.files)


def spec_for(model_id: str) -> LocalModel | None:
    """The model an index records, if repolore can load that revision."""
    return VOYAGE_NANO if model_id == VOYAGE_NANO.model_id else None


@cache
def embedder_for(model_id: str, models_dir: Path) -> LocalEmbedder:
    """The embedder an index was built with, so queries land in its vector space; one per model, loaded once."""
    if spec := spec_for(model_id):
        return LocalEmbedder(spec, models_dir)
    raise UnknownModelError(f"The index was built with {model_id}, which repolore can't load; rebuild it.")
