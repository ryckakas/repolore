import hashlib
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from repolore.embedding import model_files
from repolore.embedding.model_files import ModelFile, ModelRelease, ensure, missing
from repolore.hub import ModelUnavailableError
from repolore.progress import Phase, Progress

BASE = "https://github.com/someone/models/releases/download/tiny-1"
CONTENTS = {"tiny.onnx": b"graph" * 10, "tiny.onnx.data": b"weights" * 100, "tokenizer.json": b"{}" * 7}


def model_file(name: str) -> ModelFile:
    content = CONTENTS[name]
    return ModelFile(name, len(content), hashlib.sha256(content).hexdigest())


RELEASE = ModelRelease(
    "tiny-1", BASE, model_file("tiny.onnx"), model_file("tiny.onnx.data"), model_file("tokenizer.json")
)


class FakeGitHub:
    """Serves the release's files, as a real server would answer a range."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.honours_range = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        content = CONTENTS[request.url.path.rsplit("/", 1)[1]]
        ranged = request.headers.get("range")
        if ranged and self.honours_range:
            return httpx.Response(206, content=content[int(ranged.removeprefix("bytes=").rstrip("-")) :])
        return httpx.Response(200, content=content)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    @property
    def fetched(self) -> list[str]:
        return [request.url.path.rsplit("/", 1)[1] for request in self.requests]


@pytest.fixture
def github() -> FakeGitHub:
    return FakeGitHub()


def folder_of(models_dir: Path) -> Path:
    folder = RELEASE.folder(models_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_every_file_is_downloaded_under_its_final_name(tmp_path: Path, github: FakeGitHub) -> None:
    ensure(RELEASE, RELEASE.files, tmp_path, transport=github.transport)

    folder = RELEASE.folder(tmp_path)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == CONTENTS
    assert missing(folder, RELEASE.files) == []


def test_files_already_there_cost_no_request(tmp_path: Path, github: FakeGitHub) -> None:
    ensure(RELEASE, RELEASE.files, tmp_path, transport=github.transport)
    github.requests.clear()

    ensure(RELEASE, RELEASE.files, tmp_path, transport=github.transport)

    assert github.requests == []


def test_only_the_file_that_is_missing_is_fetched(tmp_path: Path, github: FakeGitHub) -> None:
    ensure(RELEASE, RELEASE.files, tmp_path, transport=github.transport)
    (RELEASE.folder(tmp_path) / "tiny.onnx").unlink()
    github.requests.clear()

    ensure(RELEASE, RELEASE.files, tmp_path, transport=github.transport)

    assert github.fetched == ["tiny.onnx"]


def test_only_the_files_asked_for_are_fetched(tmp_path: Path, github: FakeGitHub) -> None:
    ensure(RELEASE, [RELEASE.tokenizer], tmp_path, transport=github.transport)

    assert github.fetched == ["tokenizer.json"]


def test_a_file_of_the_wrong_size_counts_as_missing(tmp_path: Path) -> None:
    folder = folder_of(tmp_path)
    (folder / "tiny.onnx").write_bytes(b"cut off")

    assert missing(folder, [RELEASE.graph]) == [RELEASE.graph]


def test_a_redirect_to_the_asset_host_is_followed_and_carries_no_token(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def redirecting(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        name = request.url.path.rsplit("/", 1)[1]
        if request.url.host == "github.com":
            return httpx.Response(302, headers={"Location": f"https://assets.example/{name}"})
        return httpx.Response(200, content=CONTENTS[name])

    ensure(RELEASE, [RELEASE.graph], tmp_path, transport=httpx.MockTransport(redirecting))

    assert [request.url.host for request in seen] == ["github.com", "assets.example"]
    assert all("authorization" not in request.headers for request in seen)
    assert (RELEASE.folder(tmp_path) / "tiny.onnx").read_bytes() == CONTENTS["tiny.onnx"]


def test_a_wrong_checksum_removes_the_download_and_counts_nothing_as_present(tmp_path: Path) -> None:
    def corrupting(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * RELEASE.graph.size)

    with pytest.raises(ModelUnavailableError, match=r"tiny\.onnx from .* didn't match its checksum"):
        ensure(RELEASE, [RELEASE.graph], tmp_path, transport=httpx.MockTransport(corrupting))

    assert list(RELEASE.folder(tmp_path).iterdir()) == []


def test_a_not_found_fails_in_one_line_naming_the_url_and_the_folder(tmp_path: Path) -> None:
    def not_found(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    with pytest.raises(ModelUnavailableError) as raised:
        ensure(RELEASE, [RELEASE.graph], tmp_path, transport=httpx.MockTransport(not_found))

    message = str(raised.value)
    assert "\n" not in message
    assert f"{BASE}/tiny.onnx" in message
    assert "github.com" in message
    assert str(RELEASE.folder(tmp_path)) in message


def failing_after(content: bytes) -> Callable[[httpx.Request], httpx.Response]:
    def stream() -> Iterator[bytes]:
        yield content
        raise httpx.ReadError("connection reset")

    return lambda _: httpx.Response(200, content=stream())


def test_a_dropped_connection_keeps_what_arrived_and_the_next_run_resumes(
    tmp_path: Path, github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_files, "MB", 10)
    head = CONTENTS["tiny.onnx.data"][:100]

    with pytest.raises(ModelUnavailableError, match="connection reset"):
        ensure(RELEASE, [RELEASE.weights], tmp_path, transport=httpx.MockTransport(failing_after(head)))

    folder = RELEASE.folder(tmp_path)
    assert (folder / "tiny.onnx.data.partial").read_bytes() == head
    assert missing(folder, [RELEASE.weights]) == [RELEASE.weights]

    ensure(RELEASE, [RELEASE.weights], tmp_path, transport=github.transport)

    assert github.requests[0].headers["range"] == "bytes=100-"
    assert (folder / "tiny.onnx.data").read_bytes() == CONTENTS["tiny.onnx.data"]
    assert not (folder / "tiny.onnx.data.partial").exists()


def test_a_server_that_ignores_the_range_starts_the_file_over(tmp_path: Path, github: FakeGitHub) -> None:
    folder = folder_of(tmp_path)
    (folder / "tiny.onnx.data.partial").write_bytes(CONTENTS["tiny.onnx.data"][:100])
    github.honours_range = False

    ensure(RELEASE, [RELEASE.weights], tmp_path, transport=github.transport)

    assert (folder / "tiny.onnx.data").read_bytes() == CONTENTS["tiny.onnx.data"]


def test_a_partial_with_every_byte_needs_no_request(tmp_path: Path, github: FakeGitHub) -> None:
    folder = folder_of(tmp_path)
    (folder / "tiny.onnx.partial").write_bytes(CONTENTS["tiny.onnx"])

    ensure(RELEASE, [RELEASE.graph], tmp_path, transport=github.transport)

    assert github.requests == []
    assert (folder / "tiny.onnx").read_bytes() == CONTENTS["tiny.onnx"]


def test_a_partial_longer_than_the_file_starts_over(tmp_path: Path, github: FakeGitHub) -> None:
    folder = folder_of(tmp_path)
    (folder / "tiny.onnx.partial").write_bytes(b"x" * (RELEASE.graph.size + 1))

    ensure(RELEASE, [RELEASE.graph], tmp_path, transport=github.transport)

    assert "range" not in github.requests[0].headers
    assert (folder / "tiny.onnx").read_bytes() == CONTENTS["tiny.onnx"]


def test_a_corrupted_partial_is_removed_after_the_resume_fails_its_checksum(tmp_path: Path, github: FakeGitHub) -> None:
    folder = folder_of(tmp_path)
    (folder / "tiny.onnx.partial").write_bytes(b"x" * 10)

    with pytest.raises(ModelUnavailableError, match="didn't match its checksum"):
        ensure(RELEASE, [RELEASE.graph], tmp_path, transport=github.transport)

    assert list(folder.iterdir()) == []


def test_a_second_caller_waits_and_finds_the_files_there(tmp_path: Path, github: FakeGitHub) -> None:
    first_is_downloading = threading.Event()
    second_is_waiting = threading.Event()

    def slow(request: httpx.Request) -> httpx.Response:
        first_is_downloading.set()
        assert second_is_waiting.wait(5)
        return github(request)

    def second_caller() -> None:
        assert first_is_downloading.wait(5)
        second_is_waiting.set()
        ensure(RELEASE, [RELEASE.graph], tmp_path, transport=github.transport)

    thread = threading.Thread(target=second_caller)
    thread.start()
    ensure(RELEASE, [RELEASE.graph], tmp_path, transport=httpx.MockTransport(slow))
    thread.join(10)

    assert not thread.is_alive()
    assert github.fetched == ["tiny.onnx"]


def test_progress_is_reported_in_megabytes_up_to_the_total(
    tmp_path: Path, github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_files, "MB", 100)
    reports: list[Progress] = []

    ensure(RELEASE, [RELEASE.weights], tmp_path, reports.append, github.transport)

    assert {report.phase for report in reports} == {Phase.DOWNLOAD}
    assert reports[-1] == Progress(phase=Phase.DOWNLOAD, done=7, total=7)
    assert str(reports[-1]) == "download 7 of 7 MB"


def test_a_resumed_download_counts_what_it_already_has(
    tmp_path: Path, github: FakeGitHub, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_files, "MB", 100)
    (folder_of(tmp_path) / "tiny.onnx.data.partial").write_bytes(CONTENTS["tiny.onnx.data"][:300])
    reports: list[Progress] = []

    ensure(RELEASE, [RELEASE.weights], tmp_path, reports.append, github.transport)

    assert reports[0].done == 3
    assert reports[-1].done == 7
