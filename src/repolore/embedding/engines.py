import os
import platform
import sys
from dataclasses import dataclass

from repolore.embedding.local import EngineFactory, onnx_engine

DEVICE_ENV = "REPOLORE_DEVICE"


@dataclass(frozen=True)
class EngineChoice:
    name: str
    load: EngineFactory


def build_engine() -> EngineChoice:
    """MLX on Apple Silicon's GPU unless REPOLORE_DEVICE is cpu, ONNX Runtime otherwise. Queries always use ONNX."""
    if _gpu_ready():
        from repolore.embedding.mlx_engine import mlx_engine

        return EngineChoice("MLX on the GPU", mlx_engine)
    return EngineChoice("ONNX Runtime on the CPU", onnx_engine)


def _gpu_ready() -> bool:
    if os.environ.get(DEVICE_ENV) == "cpu" or sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core as mx
    except ImportError:
        return False
    # Not mx.metal.is_available(), which the Metal build always answers true: this looks for an actual device.
    return mx.is_available(mx.gpu)
