import sys
import types
from typing import Any, ClassVar

import pytest

from repolore import reranking
from repolore.hub import ModelUnavailableError
from repolore.reranking import CrossEncoderReranker
from repolore.store import Hit


class StubCrossEncoder:
    loads: ClassVar[list[dict[str, Any]]] = []
    cached: ClassVar[bool] = True
    hub_reachable: ClassVar[bool] = True

    def __init__(self, name: str, **kwargs: Any) -> None:
        StubCrossEncoder.loads.append({"name": name, **kwargs})
        if kwargs.get("local_files_only") and not StubCrossEncoder.cached:
            raise OSError("couldn't find them in the cached files")
        if not kwargs.get("local_files_only") and not StubCrossEncoder.hub_reachable:
            raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(passage)) for _, passage in pairs]


@pytest.fixture(autouse=True)
def stub_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    StubCrossEncoder.loads = []
    StubCrossEncoder.cached = True
    StubCrossEncoder.hub_reachable = True
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=StubCrossEncoder))


def hit(text: str) -> Hit:
    return Hit(
        text, f"https://github.com/octo/repo/pull/{len(text)}", "octo/repo", "pr", "t", None, None, None, "h", text, 0.1
    )


def test_hits_are_reordered_by_the_cross_encoder_score() -> None:
    reranked = CrossEncoderReranker().rerank("q", [hit("a"), hit("abc"), hit("ab")])

    assert [h.text for h in reranked] == ["abc", "ab", "a"]
    assert reranked[0].score == len("h\n\nabc")


def test_model_is_loaded_once_and_pinned() -> None:
    reranker = CrossEncoderReranker()

    reranker.rerank("q", [hit("a")])
    reranker.rerank("q", [hit("b")])

    assert StubCrossEncoder.loads == [
        {
            "name": reranking.MODEL,
            "revision": reranking.REVISION,
            "max_length": reranking.MAX_LENGTH,
            "device": "cpu",
            "local_files_only": True,
        }
    ]


def test_a_model_missing_from_the_cache_is_downloaded() -> None:
    StubCrossEncoder.cached = False
    reranker = CrossEncoderReranker()

    assert [h.text for h in reranker.rerank("q", [hit("a"), hit("ab")])] == ["ab", "a"]
    reranker.rerank("q", [hit("b")])

    first, second = StubCrossEncoder.loads
    assert first["local_files_only"] is True
    assert "local_files_only" not in second
    assert {**second, "local_files_only": True} == first


def test_a_model_neither_cached_nor_downloadable_says_how_to_get_it() -> None:
    StubCrossEncoder.cached = False
    StubCrossEncoder.hub_reachable = False

    with pytest.raises(ModelUnavailableError, match=f"{reranking.MODEL} .* couldn't be downloaded"):
        CrossEncoderReranker().rerank("q", [hit("a")])


def test_no_candidates_needs_no_model() -> None:
    assert CrossEncoderReranker().rerank("q", []) == []
    assert StubCrossEncoder.loads == []
