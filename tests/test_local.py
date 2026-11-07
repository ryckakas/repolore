import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import onnxruntime
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from repolore.embedding import local
from repolore.embedding.local import VOYAGE_NANO, LocalEmbedder, LocalModel, UnknownModelError, embedder_for, pool
from repolore.embedding.model_files import ModelFile, ModelRelease
from repolore.hub import ModelUnavailableError
from repolore.paths import resolve_paths

FIXTURES = Path(__file__).parent / "fixtures"
VOCABULARY = {"[UNK]": 0, "retry": 1, "upload": 2, "doc": 3, "query": 4}
TOKEN_DIM = 6


class StubSession:
    created: ClassVar[list[dict[str, Any]]] = []
    runs: ClassVar[list[dict[str, Any]]] = []
    load_seconds: ClassVar[float] = 0

    def __init__(self, path: str, options: onnxruntime.SessionOptions, providers: list[str]) -> None:
        time.sleep(StubSession.load_seconds)
        StubSession.created.append({"path": path, "threads": options.intra_op_num_threads, "providers": providers})

    def run(self, _: None, feeds: dict[str, Any]) -> list[Any]:
        StubSession.runs.append(feeds)
        ids = feeds["input_ids"][0]
        return [np.array([[[token + 1, 1, 1, 1, 9, 9] for token in ids]], dtype=np.float64)]


@pytest.fixture(autouse=True)
def stub_session(monkeypatch: pytest.MonkeyPatch) -> None:
    StubSession.created = []
    StubSession.runs = []
    StubSession.load_seconds = 0
    monkeypatch.setattr(onnxruntime, "InferenceSession", StubSession)


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    return tmp_path / "models"


@pytest.fixture
def tiny(models_dir: Path) -> LocalModel:
    """A word-level model whose files are on disk; its saved truncation must not reach the counts."""
    folder = models_dir / "tiny-1"
    folder.mkdir(parents=True)
    tokenizer = Tokenizer(WordLevel(VOCABULARY, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=2)
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "tiny.onnx").write_bytes(b"graph")
    (folder / "tiny.onnx.data").write_bytes(b"weights")
    return replace(
        VOYAGE_NANO,
        dim=4,
        document_prompt="doc ",
        query_prompt="query ",
        release=ModelRelease(
            "tiny-1",
            "https://example.invalid/tiny-1",
            ModelFile("tiny.onnx", 5, "unchecked"),
            ModelFile("tiny.onnx.data", 7, "unchecked"),
            ModelFile("tokenizer.json", (folder / "tokenizer.json").stat().st_size, "unchecked"),
        ),
    )


def test_each_document_runs_alone_with_the_document_prompt(tiny: LocalModel, models_dir: Path) -> None:
    vectors = LocalEmbedder(tiny, models_dir).embed_documents(["retry", "upload retry"])

    assert [run["input_ids"].tolist() for run in StubSession.runs] == [[[3, 1]], [[3, 2, 1]]]
    assert [run["attention_mask"].tolist() for run in StubSession.runs] == [[[1, 1]], [[1, 1, 1]]]
    assert StubSession.runs[0]["input_ids"].dtype == np.int64
    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])


def test_a_query_gets_the_query_prompt_and_one_vector(tiny: LocalModel, models_dir: Path) -> None:
    vector = LocalEmbedder(tiny, models_dir).embed_query("retry")

    assert StubSession.runs[0]["input_ids"].tolist() == [[4, 1]]
    assert vector.shape == (4,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_no_documents_need_no_model(tmp_path: Path) -> None:
    vectors = LocalEmbedder(VOYAGE_NANO, tmp_path).embed_documents([])

    assert vectors.shape == (0, 1024)
    assert vectors.dtype == np.float32
    assert StubSession.created == []


def test_the_session_is_loaded_once_from_the_release_folder_on_the_cpu(tiny: LocalModel, models_dir: Path) -> None:
    embedder = LocalEmbedder(tiny, models_dir)

    embedder.embed_query("retry")
    embedder.embed_documents(["upload"])

    assert StubSession.created == [
        {"path": str(models_dir / "tiny-1" / "tiny.onnx"), "threads": 0, "providers": ["CPUExecutionProvider"]}
    ]


def test_warm_up_loads_what_a_search_then_reuses(tiny: LocalModel, models_dir: Path) -> None:
    embedder = LocalEmbedder(tiny, models_dir)

    embedder.warm_up()
    assert len(StubSession.created) == 1
    embedder.embed_query("retry")

    assert len(StubSession.created) == 1


def test_parallel_first_searches_load_the_model_once(tiny: LocalModel, models_dir: Path) -> None:
    StubSession.load_seconds = 0.1
    embedder = LocalEmbedder(tiny, models_dir)

    threads = [threading.Thread(target=embedder.embed_query, args=("retry",)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(StubSession.created) == 1
    assert len(StubSession.runs) == 4


@pytest.mark.parametrize(
    ("system", "machine", "threads"),
    [("darwin", "arm64", 10), ("darwin", "x86_64", 0), ("linux", "arm64", 0), ("win32", "AMD64", 0)],
)
def test_only_apple_silicon_sets_the_thread_count(
    tiny: LocalModel, models_dir: Path, monkeypatch: pytest.MonkeyPatch, system: str, machine: str, threads: int
) -> None:
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(local.platform, "machine", lambda: machine)
    monkeypatch.setattr(os, "cpu_count", lambda: 10)

    LocalEmbedder(tiny, models_dir).warm_up()

    assert StubSession.created[0]["threads"] == threads


def test_vectors_are_pooled_over_the_tokens_of_one_text(tiny: LocalModel, models_dir: Path) -> None:
    [vector] = LocalEmbedder(tiny, models_dir).embed_documents(["retry"])

    expected = np.array([(4 + 2) / 2, 1, 1, 1])
    assert vector == pytest.approx(expected / np.linalg.norm(expected))


def test_pooling_one_token_keeps_its_direction_at_the_cut() -> None:
    assert pool(np.array([[3.0, 4.0, 100.0]], dtype=np.float32), 2) == pytest.approx([0.6, 0.8])


def test_pooling_averages_the_tokens() -> None:
    tokens = np.array([[2.0, 0.0, 7.0], [0.0, 2.0, 7.0]], dtype=np.float32)

    assert pool(tokens, 2) == pytest.approx([2**-0.5, 2**-0.5])


def test_pooling_a_zero_vector_stays_zero_rather_than_nan() -> None:
    assert pool(np.zeros((1, 3), dtype=np.float32), 2).tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("retry upload", [3, 1, 2]),
        ("retry upload retry", [3, 1, 2, 1]),
        ("retry upload retry upload", [3, 1, 2, 1]),
    ],
)
def test_a_document_is_cut_at_the_window_with_its_prompt_included(
    tiny: LocalModel, models_dir: Path, text: str, ids: list[int]
) -> None:
    LocalEmbedder(replace(tiny, max_seq_length=4), models_dir).embed_documents([text])

    assert StubSession.runs[0]["input_ids"].tolist() == [ids]


def test_counts_ignore_the_window_and_any_saved_truncation(tiny: LocalModel, models_dir: Path) -> None:
    embedder = LocalEmbedder(replace(tiny, max_seq_length=4), models_dir)

    assert embedder.count_tokens(["retry the upload again now", "", "retry"]) == [5, 0, 1]


def test_many_texts_are_counted_in_slices(tiny: LocalModel, models_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "COUNT_BATCH", 2)

    counts = LocalEmbedder(tiny, models_dir).count_tokens(["retry", "retry upload", "a b c", "", "upload"])

    assert counts == [1, 2, 3, 0, 1]


@pytest.mark.parametrize("name", ["tiny.onnx", "tiny.onnx.data", "tokenizer.json"])
def test_a_search_without_the_files_fails_at_once_and_downloads_nothing(
    tiny: LocalModel, models_dir: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(local, "ensure", lambda *_: pytest.fail("a search must not download"))
    (models_dir / "tiny-1" / name).unlink()

    with pytest.raises(ModelUnavailableError, match=r"isn't downloaded: run `repolore refresh`\."):
        LocalEmbedder(tiny, models_dir).embed_query("retry")

    assert StubSession.created == []


def test_counting_without_the_tokenizer_says_what_to_run(tiny: LocalModel, models_dir: Path) -> None:
    (models_dir / "tiny-1" / "tokenizer.json").unlink()

    with pytest.raises(ModelUnavailableError, match="isn't downloaded"):
        LocalEmbedder(tiny, models_dir).count_tokens(["retry"])


def test_an_embedder_finds_files_a_refresh_downloaded_after_its_first_failure(
    tiny: LocalModel, models_dir: Path
) -> None:
    weights = models_dir / "tiny-1" / "tiny.onnx.data"
    weights.unlink()
    embedder = LocalEmbedder(tiny, models_dir)
    with pytest.raises(ModelUnavailableError):
        embedder.embed_query("retry")

    weights.write_bytes(b"weights")

    assert embedder.embed_query("retry").shape == (4,)


def test_download_asks_for_every_file_and_passes_progress_on(
    tiny: LocalModel, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(local, "ensure", lambda *args: calls.append(args))
    report = print

    LocalEmbedder(tiny, models_dir).download(report)

    assert calls == [(tiny.release, tiny.release.files, models_dir, report)]


def test_inspect_needs_only_the_tokenizer(tiny: LocalModel, models_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(local, "ensure", lambda *args: calls.append(args))

    LocalEmbedder(tiny, models_dir).download_tokenizer()

    assert calls == [(tiny.release, [tiny.release.tokenizer], models_dir)]


def test_a_model_is_downloaded_when_every_file_is_there(tiny: LocalModel, models_dir: Path) -> None:
    assert local.is_downloaded(tiny, models_dir)

    (models_dir / "tiny-1" / "tiny.onnx").unlink()

    assert not local.is_downloaded(tiny, models_dir)


def test_model_id_names_the_pinned_revision(tmp_path: Path) -> None:
    embedder = LocalEmbedder(VOYAGE_NANO, tmp_path)

    assert embedder.model_id == f"voyageai/voyage-4-nano@{VOYAGE_NANO.revision[:12]}"
    assert embedder.dim == 1024


def test_the_window_leaves_room_for_the_prompt(tmp_path: Path) -> None:
    assert LocalEmbedder(VOYAGE_NANO, tmp_path).max_tokens == 32_768 - local.PROMPT_MARGIN


def test_embedder_for_matches_the_index_model_and_reuses_it(tmp_path: Path) -> None:
    embedder = embedder_for(VOYAGE_NANO.model_id, tmp_path)

    assert embedder.model_id == VOYAGE_NANO.model_id
    assert embedder.dim == 1024
    assert embedder_for(VOYAGE_NANO.model_id, tmp_path) is embedder


@pytest.mark.parametrize("model_id", ["someone/else@123", "BAAI/bge-small-en-v1.5@5c38ec7c405e"])
def test_embedder_for_an_unknown_model_says_to_rebuild(tmp_path: Path, model_id: str) -> None:
    with pytest.raises(UnknownModelError, match="rebuild it"):
        embedder_for(model_id, tmp_path)


def test_a_recorded_model_id_names_its_spec() -> None:
    assert local.spec_for(VOYAGE_NANO.model_id) is VOYAGE_NANO


def test_a_model_id_of_another_revision_has_no_spec() -> None:
    assert local.spec_for("voyageai/voyage-4-nano@000000000000") is None


@pytest.fixture
def real_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()


@pytest.mark.live
@pytest.mark.usefixtures("real_session")
def test_the_release_gives_the_vectors_torch_gave() -> None:
    reference = json.loads((FIXTURES / "voyage-4-nano-reference.json").read_text(encoding="utf-8"))
    embedder = LocalEmbedder(VOYAGE_NANO, resolve_paths().models_dir)
    embedder.download()

    query = embedder.embed_query(reference["query"])
    [document] = embedder.embed_documents([reference["document"]])

    assert float(query @ np.array(reference["query_vector"])) >= 0.9999
    assert float(document @ np.array(reference["document_vector"])) >= 0.9999
