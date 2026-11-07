import itertools
from collections.abc import Iterator
from dataclasses import replace
from math import prod
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from repolore.embedding import ModelUnavailableError, mlx_engine
from repolore.embedding.local import VOYAGE_NANO, Architecture, LocalEmbedder, LocalModel
from repolore.embedding.local import pool as numpy_pool
from repolore.embedding.mlx_engine import ALIGN, MlxEngine, load_weights, manifest
from repolore.embedding.model_files import ModelFile
from repolore.paths import resolve_paths

TINY = Architecture(layers=2, heads=4, kv_heads=2, head_dim=8, eps=1e-6, rope_base=10_000.0)
HIDDEN, MLP, VOCABULARY, PROJECTION, DIM = 16, 24, 50, 24, 8
NumpyWeights = dict[str, np.ndarray]


@pytest.fixture(autouse=True)
def on_the_cpu() -> Iterator[None]:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny_weights(seed: int = 0) -> NumpyWeights:
    random = np.random.default_rng(seed)

    def matrix(rows: int, columns: int) -> np.ndarray:
        return random.normal(0, 0.3, (rows, columns)).astype(np.float32)

    def scale(size: int) -> np.ndarray:
        return (1 + random.normal(0, 0.1, size)).astype(np.float32)

    weights = {
        "model.embed_tokens.weight": matrix(VOCABULARY, HIDDEN),
        "model.norm.weight": scale(HIDDEN),
        "linear.weight": matrix(PROJECTION, HIDDEN),
    }
    for layer in range(TINY.layers):
        prefix = f"model.layers.{layer}"
        weights |= {
            f"{prefix}.input_layernorm.weight": scale(HIDDEN),
            f"{prefix}.post_attention_layernorm.weight": scale(HIDDEN),
            f"{prefix}.self_attn.q_proj.weight": matrix(TINY.heads * TINY.head_dim, HIDDEN),
            f"{prefix}.self_attn.k_proj.weight": matrix(TINY.kv_heads * TINY.head_dim, HIDDEN),
            f"{prefix}.self_attn.v_proj.weight": matrix(TINY.kv_heads * TINY.head_dim, HIDDEN),
            f"{prefix}.self_attn.o_proj.weight": matrix(HIDDEN, TINY.heads * TINY.head_dim),
            f"{prefix}.self_attn.q_norm.weight": scale(TINY.head_dim),
            f"{prefix}.self_attn.k_norm.weight": scale(TINY.head_dim),
            f"{prefix}.mlp.gate_proj.weight": matrix(MLP, HIDDEN),
            f"{prefix}.mlp.up_proj.weight": matrix(MLP, HIDDEN),
            f"{prefix}.mlp.down_proj.weight": matrix(HIDDEN, MLP),
        }
    return weights


def rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + TINY.eps) * weight


def rotated(x: np.ndarray) -> np.ndarray:
    """Rotary positions with the halves layout, x being (heads, tokens, head_dim)."""
    half = TINY.head_dim // 2
    angles = np.arange(x.shape[1])[:, None] * TINY.rope_base ** (-np.arange(half) / half)[None, :]
    first, second = x[..., :half], x[..., half:]
    return np.concatenate(
        [first * np.cos(angles) - second * np.sin(angles), second * np.cos(angles) + first * np.sin(angles)], -1
    )


def reference_vector(weights: NumpyWeights, ids: list[int]) -> np.ndarray:
    """The forward pass and pooling written out in numpy for one unpadded text, independent of the engine."""
    w = {name: value.astype(np.float64) for name, value in weights.items()}
    hidden = w["model.embed_tokens.weight"][ids]
    tokens, group = len(ids), TINY.heads // TINY.kv_heads
    for layer in range(TINY.layers):
        prefix = f"model.layers.{layer}"
        normed = rms_norm(hidden, w[f"{prefix}.input_layernorm.weight"])

        def heads(name: str, count: int, x: np.ndarray = normed, p: str = prefix) -> np.ndarray:
            return (
                (x @ w[f"{p}.self_attn.{name}_proj.weight"].T).reshape(tokens, count, TINY.head_dim).transpose(1, 0, 2)
            )

        queries = rotated(rms_norm(heads("q", TINY.heads), w[f"{prefix}.self_attn.q_norm.weight"]))
        keys = rotated(rms_norm(heads("k", TINY.kv_heads), w[f"{prefix}.self_attn.k_norm.weight"]))
        values = heads("v", TINY.kv_heads)
        keys, values = np.repeat(keys, group, axis=0), np.repeat(values, group, axis=0)
        scores = queries @ keys.transpose(0, 2, 1) / np.sqrt(TINY.head_dim)
        attention = np.exp(scores - scores.max(-1, keepdims=True))
        attended = (attention / attention.sum(-1, keepdims=True)) @ values
        merged = attended.transpose(1, 0, 2).reshape(tokens, -1)
        hidden = hidden + merged @ w[f"{prefix}.self_attn.o_proj.weight"].T
        normed = rms_norm(hidden, w[f"{prefix}.post_attention_layernorm.weight"])
        gate = normed @ w[f"{prefix}.mlp.gate_proj.weight"].T
        up = normed @ w[f"{prefix}.mlp.up_proj.weight"].T
        hidden = hidden + (gate / (1 + np.exp(-gate)) * up) @ w[f"{prefix}.mlp.down_proj.weight"].T
    token_vectors = rms_norm(hidden, w["model.norm.weight"]) @ w["linear.weight"].T
    return numpy_pool(token_vectors.astype(np.float32), DIM)


@pytest.fixture
def weights() -> NumpyWeights:
    return tiny_weights()


@pytest.fixture
def engine(weights: NumpyWeights) -> MlxEngine:
    return MlxEngine({name: mx.array(value) for name, value in weights.items()}, TINY, DIM)


def text(length: int, start: int = 1) -> list[int]:
    return [(start + n) % (VOCABULARY - 1) + 1 for n in range(length)]


AttentionCalls = list[dict[str, Any]]


@pytest.fixture
def attention_calls(monkeypatch: pytest.MonkeyPatch) -> AttentionCalls:
    calls: AttentionCalls = []
    original = mx.fast.scaled_dot_product_attention

    def recording(q: mx.array, k: mx.array, v: mx.array, **options: Any) -> mx.array:
        calls.append({"length": q.shape[2], "mask": options.get("mask")})
        return original(q, k, v, **options)

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", recording)
    return calls


def test_one_text_matches_the_numpy_forward_pass(engine: MlxEngine, weights: NumpyWeights) -> None:
    [vector] = engine.embed([text(5)])

    assert vector.dtype == np.float32
    assert vector.shape == (DIM,)
    assert vector == pytest.approx(reference_vector(weights, text(5)), abs=1e-5)


def test_no_documents_never_load_the_engine(tmp_path: Path) -> None:
    def failing(_: LocalModel, __: Path) -> MlxEngine:
        raise AssertionError("nothing to embed, so nothing to load")

    assert LocalEmbedder(VOYAGE_NANO, tmp_path, failing).embed_documents([]).shape == (0, 1024)


def test_a_text_of_exactly_the_alignment_needs_no_mask(engine: MlxEngine, attention_calls: AttentionCalls) -> None:
    engine.embed([text(ALIGN)])

    assert {call["length"] for call in attention_calls} == {ALIGN}
    assert all(call["mask"] is None for call in attention_calls)


@pytest.mark.parametrize(("length", "padded"), [(31, 32), (32, 32), (33, 64)])
def test_lengths_pad_to_the_next_multiple_of_the_alignment_without_changing_the_vector(
    engine: MlxEngine, weights: NumpyWeights, attention_calls: AttentionCalls, length: int, padded: int
) -> None:
    [vector] = engine.embed([text(length)])

    assert {call["length"] for call in attention_calls} == {padded}
    assert (attention_calls[0]["mask"] is None) == (length == padded)
    assert vector == pytest.approx(reference_vector(weights, text(length)), abs=1e-5)


def test_each_text_in_a_mixed_batch_gets_the_vector_it_gets_alone(engine: MlxEngine) -> None:
    texts = [text(1 + (n * 7) % 70, start=n) for n in range(64)]

    together = engine.embed(texts)

    alone = np.stack([engine.embed([one])[0] for one in texts])
    assert together.shape == (64, DIM)
    assert np.abs(together - alone).max() < 1e-5


def test_the_mlx_pool_and_the_numpy_pool_agree() -> None:
    random = np.random.default_rng(1)
    token_vectors = random.normal(size=(3, 9, 12)).astype(np.float32)
    lengths = [9, 4, 1]
    keep = np.array([[n < length for n in range(9)] for length in lengths])

    pooled = np.array(mlx_engine.pool(mx.array(token_vectors), mx.array(keep), 5))

    expected = np.stack([numpy_pool(token_vectors[row, :length], 5) for row, length in enumerate(lengths)])
    assert pooled == pytest.approx(expected, abs=1e-6)


def test_an_mlx_failure_stops_with_one_line_naming_the_cpu_fallback(
    engine: MlxEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(*_: object) -> mx.array:
        raise RuntimeError("[METAL] Command buffer execution failed: Caused GPU Timeout Error\nmore detail")

    monkeypatch.setattr(engine, "_token_vectors", failing)

    with pytest.raises(ModelUnavailableError) as raised:
        engine.embed([text(3)])

    message = str(raised.value)
    assert "\n" not in message
    assert message.startswith(
        "MLX failed on the GPU: [METAL] Command buffer execution failed: Caused GPU Timeout Error"
    )
    assert "REPOLORE_DEVICE=cpu" in message


def stored_file(tmp_path: Path, weights: NumpyWeights, transposed: set[str]) -> tuple[Path, dict[str, Any]]:
    """Weights laid out the way the export stores them: some transposed, at offsets a manifest records."""
    tensors, chunks, offset = {}, [], 0
    for name, value in weights.items():
        stored = np.ascontiguousarray(value.T) if name in transposed else value
        tensors[name] = {"offset": offset, "shape": list(stored.shape), "transposed": name in transposed}
        chunks.append(stored.tobytes())
        offset += stored.nbytes
    path = tmp_path / "tiny.onnx.data"
    path.write_bytes(b"".join(chunks))
    return path, {"release": "tiny-1", "weights": path.name, "tensors": tensors}


def test_weights_load_in_the_model_layout_whatever_the_export_stored(tmp_path: Path, weights: NumpyWeights) -> None:
    transposed = {name for name in weights if name.endswith("_proj.weight") or name == "linear.weight"}
    path, stored = stored_file(tmp_path, weights, transposed)

    loaded = load_weights(path, stored)

    assert loaded.keys() == weights.keys()
    assert all(np.array_equal(np.array(loaded[name]), value) for name, value in weights.items())


def test_the_engine_reads_the_release_through_the_manifest(
    tmp_path: Path, weights: NumpyWeights, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, stored = stored_file(tmp_path, weights, {"linear.weight"})
    monkeypatch.setattr(mlx_engine, "manifest", lambda: stored)
    weights_file = ModelFile(path.name, path.stat().st_size, "unchecked")
    spec = replace(VOYAGE_NANO, dim=DIM, architecture=TINY, release=replace(VOYAGE_NANO.release, weights=weights_file))

    [vector] = mlx_engine.mlx_engine(spec, tmp_path).embed([text(6)])

    assert vector == pytest.approx(reference_vector(weights, text(6)), abs=1e-5)


def test_weights_that_dont_fit_the_gpu_stop_with_the_same_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def out_of_memory(*_: object) -> mlx_engine.Weights:
        raise RuntimeError("[metal::malloc] Attempting to allocate 1385807872 bytes which is greater than the maximum")

    monkeypatch.setattr(mlx_engine, "load_weights", out_of_memory)

    with pytest.raises(ModelUnavailableError, match=r"^MLX failed on the GPU: \[metal::malloc\] .*REPOLORE_DEVICE=cpu"):
        mlx_engine.mlx_engine(VOYAGE_NANO, tmp_path)


def expected_names(architecture: Architecture) -> set[str]:
    per_layer = [
        "input_layernorm",
        "post_attention_layernorm",
        *(f"self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")),
        *(f"mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")),
    ]
    layers = {f"model.layers.{n}.{name}.weight" for n in range(architecture.layers) for name in per_layer}
    return layers | {"model.embed_tokens.weight", "model.norm.weight", "linear.weight"}


def test_the_manifest_belongs_to_the_pinned_release_and_names_every_tensor() -> None:
    shipped = manifest()

    assert shipped["release"] == VOYAGE_NANO.release.name
    assert shipped["weights"] == VOYAGE_NANO.release.weights.name
    assert shipped["tensors"].keys() == expected_names(VOYAGE_NANO.architecture)


def test_the_manifests_ranges_fit_the_weights_file_without_overlapping() -> None:
    ranges = sorted(
        (entry["offset"], entry["offset"] + 4 * prod(entry["shape"])) for entry in manifest()["tensors"].values()
    )

    assert ranges[0][0] >= 0
    assert ranges[-1][1] <= VOYAGE_NANO.release.weights.size
    assert all(end <= start for (_, end), (start, _) in itertools.pairwise(ranges))


@pytest.fixture
def default_device() -> None:
    mx.set_default_device(mx.gpu if mx.is_available(mx.gpu) else mx.cpu)


@pytest.mark.live
@pytest.mark.usefixtures("default_device")
def test_the_real_model_on_mlx_gives_the_vectors_onnx_runtime_gives() -> None:
    models_dir = resolve_paths().models_dir
    onnx = LocalEmbedder(VOYAGE_NANO, models_dir)
    onnx.download()
    mlx = LocalEmbedder(VOYAGE_NANO, models_dir, mlx_engine.mlx_engine)
    texts = ["why was the release delayed?", "Retry with exponential backoff instead of a fixed delay. " * 40]

    for got, expected in zip(mlx.embed_documents(texts), onnx.embed_documents(texts), strict=True):
        assert float(got @ expected) >= 0.99999
