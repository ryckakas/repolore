import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from repolore.embedding import local
from repolore.embedding.local import LocalEmbedder, UnknownModelError, embedder_for
from repolore.hub import ModelUnavailableError


class StubModel:
    loads: ClassVar[list[dict[str, Any]]] = []
    encodes: ClassVar[list[dict[str, Any]]] = []
    cache_lookups: ClassVar[list[tuple[str, str, str]]] = []
    modules_json_cached: ClassVar[bool] = True
    weights_cached: ClassVar[bool] = True
    hub_reachable: ClassVar[bool] = True

    def __init__(self, name: str, **kwargs: Any) -> None:
        StubModel.loads.append({"name": name, **kwargs})
        if kwargs.get("local_files_only") and not StubModel.weights_cached:
            raise OSError("couldn't find them in the cached files")
        if not kwargs.get("local_files_only") and not StubModel.hub_reachable:
            raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    @staticmethod
    def try_to_load_from_cache(repo_id: str, filename: str, revision: str) -> str | None:
        StubModel.cache_lookups.append((repo_id, filename, revision))
        return f"/hf/{repo_id}/{revision}/{filename}" if StubModel.modules_json_cached else None

    def encode_document(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        StubModel.encodes.append(kwargs)
        return [[1.0, 0.0]] * len(texts)

    def encode_query(self, text: str, **kwargs: Any) -> list[float]:
        assert kwargs == {"normalize_embeddings": True, "show_progress_bar": False}
        return [0.0, 1.0]


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    StubModel.loads = []
    StubModel.encodes = []
    StubModel.cache_lookups = []
    StubModel.modules_json_cached = True
    StubModel.weights_cached = True
    StubModel.hub_reachable = True
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
def test_warm_up_loads_the_model_a_search_then_reuses() -> None:
    embedder = LocalEmbedder()

    embedder.warm_up()
    assert len(StubModel.loads) == 1
    embedder.embed_query("a")

    assert len(StubModel.loads) == 1


@pytest.mark.usefixtures("stub_sentence_transformers")
@pytest.mark.parametrize(
    ("use_gpu", "mps", "override", "device"),
    [
        (True, True, None, "mps"),
        (True, False, None, "cpu"),
        (True, True, "cpu", "cpu"),
        (True, True, "", "mps"),
        (True, False, "mps", "mps"),
        (False, True, None, "cpu"),
        (False, True, "mps", "cpu"),
    ],
)
def test_the_gpu_is_used_only_when_asked_for_and_available(
    monkeypatch: pytest.MonkeyPatch, use_gpu: bool, mps: bool, override: str | None, device: str
) -> None:
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    if override is None:
        monkeypatch.delenv(local.DEVICE_ENV, raising=False)
    else:
        monkeypatch.setenv(local.DEVICE_ENV, override)

    LocalEmbedder(use_gpu=use_gpu).embed_documents(["a"])

    [only] = StubModel.loads
    assert only["device"] == device
    assert only["model_kwargs"] == {"dtype": torch.float32}


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
def test_a_model_neither_cached_nor_downloadable_says_how_to_get_it() -> None:
    StubModel.weights_cached = False
    StubModel.hub_reachable = False

    with pytest.raises(ModelUnavailableError, match=r"voyageai/voyage-4-nano \(revision 5fb34e870f49\) isn't in the"):
        LocalEmbedder().embed_query("a")


@pytest.mark.usefixtures("stub_sentence_transformers")
def test_vectors_are_float32_arrays() -> None:
    embedder = LocalEmbedder()

    documents = embedder.embed_documents(["b", "c"])
    query = embedder.embed_query("a")

    assert documents.dtype == np.float32
    assert documents.shape == (2, 2)
    assert query.tolist() == [0.0, 1.0]
    assert StubModel.encodes == [{"batch_size": 2, "normalize_embeddings": True, "show_progress_bar": False}]


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


@dataclass
class TinyHub:
    tokenizer_file: Path
    cached: bool = True
    reachable: bool = True
    lookups: list[tuple[str, str, str]] = field(default_factory=list)
    downloads: list[tuple[str, str, str]] = field(default_factory=list)

    def try_to_load_from_cache(self, repo_id: str, filename: str, revision: str) -> str | None:
        self.lookups.append((repo_id, filename, revision))
        return str(self.tokenizer_file) if self.cached else None

    def hf_hub_download(self, repo_id: str, filename: str, revision: str) -> str:
        self.downloads.append((repo_id, filename, revision))
        if not self.reachable:
            raise OSError("An error happened while trying to locate the file on the Hub")
        return str(self.tokenizer_file)


@pytest.fixture
def tiny_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TinyHub:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "retry": 1, "upload": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=2)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    hub = TinyHub(tmp_path / "tokenizer.json")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return hub


def test_tokens_are_counted_with_the_cached_tokenizer_and_without_its_truncation(tiny_hub: TinyHub) -> None:
    embedder = LocalEmbedder()

    assert embedder.count_tokens(["retry the upload again", "", "retry"]) == [4, 0, 1]
    assert embedder.count_tokens(["upload"]) == [1]
    assert tiny_hub.lookups == [("voyageai/voyage-4-nano", "tokenizer.json", local.VOYAGE_NANO.revision)]
    assert tiny_hub.downloads == []


def test_a_tokenizer_not_in_the_cache_is_downloaded_at_the_pinned_revision(tiny_hub: TinyHub) -> None:
    tiny_hub.cached = False

    assert LocalEmbedder(local.BGE_SMALL).count_tokens(["retry upload"]) == [2]
    assert tiny_hub.downloads == [("BAAI/bge-small-en-v1.5", "tokenizer.json", local.BGE_SMALL.revision)]


def test_a_tokenizer_neither_cached_nor_downloadable_says_how_to_get_it(tiny_hub: TinyHub) -> None:
    tiny_hub.cached = False
    tiny_hub.reachable = False

    with pytest.raises(ModelUnavailableError, match=r"BAAI/bge-small-en-v1\.5 .* locate the file on the Hub\. Run"):
        LocalEmbedder(local.BGE_SMALL).count_tokens(["retry"])


def test_many_texts_are_counted_in_slices(tiny_hub: TinyHub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "COUNT_BATCH", 2)

    assert LocalEmbedder().count_tokens(["retry", "retry upload", "a b c", "", "upload"]) == [1, 2, 3, 0, 1]


def test_the_window_leaves_room_for_the_prompt() -> None:
    assert LocalEmbedder().max_tokens == 32_768 - local.PROMPT_MARGIN
    assert LocalEmbedder(local.BGE_SMALL).max_tokens == 512 - local.PROMPT_MARGIN
