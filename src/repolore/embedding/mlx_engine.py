import json
from collections.abc import Generator
from contextlib import contextmanager
from importlib.resources import files
from math import ceil, prod
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from repolore.embedding import ModelUnavailableError, Vectors
from repolore.embedding.local import SMALLEST_NORM, Architecture, LocalModel

MANIFEST = "voyage-4-nano.tensors.json"
# MLX 0.32's kernels run about 2x slower on an M5 when the sequence length isn't a multiple of 32, and padding to it
# costs at most 31 masked tokens.
ALIGN = 32

Weights = dict[str, mx.array]


class MlxEngine:
    """The transformer written out in MLX for Apple Silicon's GPU, run on padded batches with a key mask."""

    def __init__(self, weights: Weights, architecture: Architecture, dim: int) -> None:
        self._weights = weights
        self._arch = architecture
        self._dim = dim

    def embed(self, ids: list[list[int]]) -> Vectors:
        padded = ceil(max(len(text) for text in ids) / ALIGN) * ALIGN
        tokens = mx.array([text + [0] * (padded - len(text)) for text in ids])
        keep = mx.array([[True] * len(text) + [False] * (padded - len(text)) for text in ids])
        mask = None if all(len(text) == padded for text in ids) else keep[:, None, None, :]
        with _on_the_gpu():
            # Converting to numpy evaluates the lazy graph here, once per batch.
            return np.array(pool(self._token_vectors(tokens, mask), keep, self._dim), dtype=np.float32)

    def _token_vectors(self, ids: mx.array, mask: mx.array | None) -> mx.array:
        hidden = self._weights["model.embed_tokens.weight"][ids]
        for layer in range(self._arch.layers):
            prefix = f"model.layers.{layer}"
            normed = self._norm(f"{prefix}.input_layernorm", hidden)
            hidden = hidden + self._attention(f"{prefix}.self_attn", normed, mask)
            hidden = hidden + self._mlp(f"{prefix}.mlp", self._norm(f"{prefix}.post_attention_layernorm", hidden))
        return self._linear("linear", self._norm("model.norm", hidden))

    def _attention(self, prefix: str, hidden: mx.array, mask: mx.array | None) -> mx.array:
        arch = self._arch
        batch, length, _ = hidden.shape
        queries = self._rotated(prefix, "q", hidden, arch.heads)
        keys = self._rotated(prefix, "k", hidden, arch.kv_heads)
        values = self._split(self._linear(f"{prefix}.v_proj", hidden), arch.kv_heads)
        # The kernel groups query heads over the key and value heads itself, so those stay unrepeated.
        attended = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=arch.head_dim**-0.5, mask=mask)
        merged = attended.transpose(0, 2, 1, 3).reshape(batch, length, arch.heads * arch.head_dim)
        return self._linear(f"{prefix}.o_proj", merged)

    def _rotated(self, prefix: str, name: str, hidden: mx.array, heads: int) -> mx.array:
        arch = self._arch
        batch, length, _ = hidden.shape
        projected = self._linear(f"{prefix}.{name}_proj", hidden).reshape(batch, length, heads, arch.head_dim)
        normed = mx.fast.rms_norm(projected, self._weights[f"{prefix}.{name}_norm.weight"], arch.eps)
        return mx.fast.rope(
            normed.transpose(0, 2, 1, 3), arch.head_dim, traditional=False, base=arch.rope_base, scale=1.0, offset=0
        )

    def _split(self, projected: mx.array, heads: int) -> mx.array:
        batch, length, _ = projected.shape
        return projected.reshape(batch, length, heads, self._arch.head_dim).transpose(0, 2, 1, 3)

    def _mlp(self, prefix: str, hidden: mx.array) -> mx.array:
        gate = self._linear(f"{prefix}.gate_proj", hidden)
        return self._linear(f"{prefix}.down_proj", gate * mx.sigmoid(gate) * self._linear(f"{prefix}.up_proj", hidden))

    def _norm(self, name: str, hidden: mx.array) -> mx.array:
        return mx.fast.rms_norm(hidden, self._weights[f"{name}.weight"], self._arch.eps)

    def _linear(self, name: str, hidden: mx.array) -> mx.array:
        return hidden @ self._weights[f"{name}.weight"].T


def pool(token_vectors: mx.array, keep: mx.array, dim: int) -> mx.array:
    """Mean over each text's own tokens, normalized, cut to the model's dimensions and normalized again."""
    counted = keep.astype(mx.float32)[:, :, None]
    mean = (token_vectors * counted).sum(axis=1) / counted.sum(axis=1)
    return _unit(_unit(mean)[:, :dim])


def _unit(vectors: mx.array) -> mx.array:
    return vectors / mx.maximum(mx.linalg.norm(vectors, axis=-1, keepdims=True), SMALLEST_NORM)


def mlx_engine(spec: LocalModel, folder: Path) -> MlxEngine:
    with _on_the_gpu():
        weights = load_weights(folder / spec.release.weights.name, manifest())
    return MlxEngine(weights, spec.architecture, spec.dim)


@contextmanager
def _on_the_gpu() -> Generator[None]:
    """Turns an MLX failure, such as a Metal timeout or an allocation that doesn't fit, into one line and a way out."""
    try:
        yield
    except RuntimeError as error:
        reason = next(iter(str(error).strip().splitlines()), type(error).__name__).rstrip(".")
        raise ModelUnavailableError(
            f"MLX failed on the GPU: {reason}. Set REPOLORE_DEVICE=cpu to build on the CPU instead."
        ) from error


def manifest() -> dict[str, Any]:
    """Where each tensor sits in the release's weights file; repolore-models' tensors.py writes it."""
    return json.loads(files("repolore.embedding").joinpath(MANIFEST).read_text(encoding="utf-8"))


def load_weights(weights_file: Path, tensors: dict[str, Any]) -> Weights:
    data = np.memmap(weights_file, dtype=np.uint8, mode="r")
    weights = {name: mx.array(_stored(data, entry)) for name, entry in tensors["tensors"].items()}
    mx.eval(weights)
    return weights


def _stored(data: np.memmap, entry: dict[str, Any]) -> np.ndarray:
    shape = entry["shape"]
    stored = np.frombuffer(data, dtype=np.float32, count=prod(shape), offset=entry["offset"]).reshape(shape)
    # The export keeps matmul weights transposed; turning them back keeps the matmul layout the speed was measured on.
    return np.ascontiguousarray(stored.T) if entry["transposed"] else stored
