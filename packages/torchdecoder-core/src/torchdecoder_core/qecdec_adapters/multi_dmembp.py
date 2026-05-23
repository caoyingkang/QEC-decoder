from typing import ClassVar

from .base import TorchModelDecoder


class MultiDMemBPDecoder(TorchModelDecoder, registry_name="TorchModel(MultiDMemBP)"):
    """DMemBP variant with vector-valued messages, MLP transforms, and
    majority voting across message features."""

    model_name: ClassVar[str] = "MultiDMemBP"
