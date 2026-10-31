import sys
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from repolore.embedding import local
from repolore.embedding.local import LocalEmbedder, UnknownModelError, embedder_for


class StubModel:
    loads: ClassVar[list[dict[str, Any]]] = []
    cache_lookups: ClassVar[list[tuple[str, str, str]]] = []
    modules_json_cached: ClassVar[bool] = True
    weights_cached: ClassVar[bool] = True

    def __init__(self, name: str, **kwargs: Any) -> None:
        StubModel.loads.append({"name": name, **kwargs})
        if kwargs.get("local_files_only") and not StubModel.weights_cached:
            raise OSError("couldn't find them in the cached files")

    @staticmethod
    def try_to_load_from_cache(repo_id: str, filename: str, revision: str) -> str | None:
        StubModel.cache_lookups.append((repo_id, filename, revision))
        return f"/hf/{repo_id}/{revision}/{filename}" if StubModel.modules_json_cached else None

    def encode_document(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        assert kwargs == {"batch_size": local.BATCH_SIZE, "normalize_embeddings": True}
        return [[1.0, 0.0]] * len(texts)

    def encode_query(self, text: str, **kwargs: Any) -> list[float]:
        assert kwargs == {"normalize_embeddings": True}
        return [0.0, 1.0]


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    StubModel.loads = []
    StubModel.cache_lookups = []
    StubModel.modules_json_cached = True
    StubModel.weights_cached = True
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=StubModel))
    hub = types.SimpleNamespace(try_to_load_from_cache=StubModel.try_to_load_from_cache)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_model_is_loaded_once_pinned_and_in_float32() -> None:
    import torch

    embedder = LocalEmbedder()

    embedder.embed_query("a")
    embedder.embed_documents(["b", "c"])

    assert StubModel.loads == [
        {
            "name": "voyageai/voyage-4-nano",
            "revision": local.VOYAGE_NANO.revision,
            "trust_remote_code": True,
            "truncate_dim": 1024,
            "prompts": None,
            "device": "cpu",
            "model_kwargs": {"dtype": torch.float32},
            "local_files_only": True,
        }
    ]
    assert StubModel.cache_lookups == [("voyageai/voyage-4-nano", "modules.json", local.VOYAGE_NANO.revision)]


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_a_snapshot_without_modules_json_is_loaded_from_the_hub() -> None:
    StubModel.modules_json_cached = False

    assert LocalEmbedder().embed_query("a").tolist() == [0.0, 1.0]

    [only] = StubModel.loads
    assert "local_files_only" not in only


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_a_snapshot_missing_other_files_is_downloaded() -> None:
    StubModel.weights_cached = False
    embedder = LocalEmbedder()

    assert embedder.embed_query("a").tolist() == [0.0, 1.0]
    embedder.embed_query("b")

    first, second = StubModel.loads
    assert first["local_files_only"] is True
    assert "local_files_only" not in second
    assert {**second, "local_files_only": True} == first


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_vectors_are_float32_arrays() -> None:
    embedder = LocalEmbedder()

    documents = embedder.embed_documents(["b", "c"])
    query = embedder.embed_query("a")

    assert documents.dtype == np.float32
    assert documents.shape == (2, 2)
    assert query.tolist() == [0.0, 1.0]


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_model_id_names_the_pinned_revision() -> None:
    assert LocalEmbedder().model_id == f"voyageai/voyage-4-nano@{local.VOYAGE_NANO.revision[:12]}"
    assert LocalEmbedder().dim == 1024


@pytest.mark.live
def test_real_model_embeds_unit_vectors() -> None:
    embedder = LocalEmbedder()

    assert float(np.linalg.norm(embedder.embed_query("why was the release delayed?"))) == pytest.approx(1.0)


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_bge_small_gets_its_query_instruction_and_no_remote_code() -> None:
    LocalEmbedder(local.BGE_SMALL).embed_query("a")

    load = StubModel.loads[0]
    assert load["name"] == "BAAI/bge-small-en-v1.5"
    assert load["trust_remote_code"] is False
    assert load["truncate_dim"] == 384
    assert load["prompts"] == {"query": "Represent this sentence for searching relevant passages: "}


def test_embedder_for_matches_the_index_model_and_reuses_it() -> None:
    embedder = embedder_for(local.BGE_SMALL.model_id)

    assert embedder.model_id == "BAAI/bge-small-en-v1.5@5c38ec7c405e"
    assert embedder.dim == 384
    assert embedder_for(local.BGE_SMALL.model_id) is embedder


def test_embedder_for_an_unknown_model_says_to_rebuild() -> None:
    with pytest.raises(UnknownModelError, match="rebuild it"):
        embedder_for("someone/else@123")


def test_a_recorded_model_is_named_whichever_revision_it_was_pinned_to() -> None:
    assert local.model_named_in(local.BGE_SMALL.model_id) == local.ModelName.BGE_SMALL
    assert local.model_named_in("voyageai/voyage-4-nano@000000000000") == local.ModelName.VOYAGE_NANO
    assert local.model_named_in("voyageai/voyage-4-nano-v2@000000000000") is None
    assert local.model_named_in("") is None
