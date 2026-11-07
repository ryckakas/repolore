import subprocess
import sys

import mlx.core
import pytest

from repolore.embedding import engines
from repolore.embedding.engines import DEVICE_ENV, build_engine
from repolore.embedding.local import onnx_engine
from repolore.embedding.mlx_engine import mlx_engine


@pytest.fixture
def apple_silicon(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(engines.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mlx.core, "is_available", lambda device: device == mlx.core.gpu)
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    return monkeypatch


def test_apple_silicon_with_a_gpu_builds_on_mlx(apple_silicon: pytest.MonkeyPatch) -> None:
    choice = build_engine()

    assert (choice.name, choice.load) == ("MLX on the GPU", mlx_engine)


@pytest.mark.parametrize("value", ["", "mps", "gpu", "CPU"])
def test_only_cpu_overrides_the_gpu(apple_silicon: pytest.MonkeyPatch, value: str) -> None:
    apple_silicon.setenv(DEVICE_ENV, value)

    assert build_engine().load is mlx_engine


def test_repolore_device_cpu_builds_on_onnx_runtime(apple_silicon: pytest.MonkeyPatch) -> None:
    apple_silicon.setenv(DEVICE_ENV, "cpu")

    assert (build_engine().name, build_engine().load) == ("ONNX Runtime on the CPU", onnx_engine)


def test_without_a_metal_device_builds_on_onnx_runtime(apple_silicon: pytest.MonkeyPatch) -> None:
    apple_silicon.setattr(mlx.core, "is_available", lambda _: False)

    assert build_engine().load is onnx_engine


def test_without_mlx_installed_builds_on_onnx_runtime(apple_silicon: pytest.MonkeyPatch) -> None:
    apple_silicon.setitem(sys.modules, "mlx.core", None)

    assert build_engine().load is onnx_engine


@pytest.mark.parametrize(("system", "machine"), [("darwin", "x86_64"), ("linux", "aarch64"), ("win32", "AMD64")])
def test_other_platforms_build_on_onnx_runtime(apple_silicon: pytest.MonkeyPatch, system: str, machine: str) -> None:
    apple_silicon.setattr(sys, "platform", system)
    apple_silicon.setattr(engines.platform, "machine", lambda: machine)

    assert build_engine().load is onnx_engine


@pytest.mark.skipif(sys.platform == "darwin", reason="Apple Silicon imports mlx to look for its GPU")
def test_other_platforms_never_import_mlx() -> None:
    script = (
        "import sys; from repolore.embedding.engines import build_engine; build_engine(); print('mlx' in sys.modules)"
    )

    imported = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout

    assert imported.strip() == "False"
