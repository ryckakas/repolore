import httpx
import pytest

from repolore.hub import ModelUnavailableError, downloading

REVISION = "5fb34e870f49f19e79c94d5e118f5ee730133279"


def test_a_download_that_works_passes_through() -> None:
    with downloading("voyageai/voyage-4-nano", REVISION):
        result = "loaded"

    assert result == "loaded"


def test_a_failed_download_names_the_model_its_revision_and_the_fix() -> None:
    reason = "We couldn't connect to 'https://huggingface.co' to load the files.\nCheck your connection."

    with pytest.raises(ModelUnavailableError) as raised, downloading("voyageai/voyage-4-nano", REVISION):
        raise OSError(reason)

    assert str(raised.value) == (
        "voyageai/voyage-4-nano (revision 5fb34e870f49) isn't in the Hugging Face cache and couldn't be downloaded: "
        "We couldn't connect to 'https://huggingface.co' to load the files. "
        "Run repolore once with access to huggingface.co, and the same HF_HOME, to download it."
    )
    assert isinstance(raised.value.__cause__, OSError)


def test_an_error_without_a_message_is_named_by_its_type() -> None:
    message = r"couldn't be downloaded: FileNotFoundError\. Run"
    with pytest.raises(ModelUnavailableError, match=message), downloading("BAAI/bge-small-en-v1.5", REVISION):
        raise FileNotFoundError


def test_a_proxy_that_refuses_huggingface_co_is_a_failed_download() -> None:
    message = "couldn't be downloaded: 403 Forbidden"
    with pytest.raises(ModelUnavailableError, match=message), downloading("BAAI/bge-small-en-v1.5", REVISION):
        raise httpx.ProxyError("403 Forbidden")


def test_other_errors_are_left_alone() -> None:
    with pytest.raises(ValueError, match="bad config"), downloading("BAAI/bge-small-en-v1.5", REVISION):
        raise ValueError("bad config")
