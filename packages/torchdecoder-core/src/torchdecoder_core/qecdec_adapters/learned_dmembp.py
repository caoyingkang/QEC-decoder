from typing import ClassVar

from .base import TorchModelDecoder


class LearnedDMemBPDecoder(TorchModelDecoder, registry_name="LearnedDMemBP"):
    """DMemBP with learnable per-VN memory coefficients."""

    model_name: ClassVar[str] = "LearnedDMemBP"
