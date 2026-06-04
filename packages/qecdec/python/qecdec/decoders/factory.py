"""Factory for building decoders."""

import inspect

from ..types import Bit2DArray, Float1DArray
from .base import Decoder, DECODERS_REGISTRY


def create_decoder(
    name: str, pcm: Bit2DArray, prior: Float1DArray, **kwargs
) -> Decoder:
    """Create a decoder by name. Only the keyword-only arguments in the constructor
    will be passed through.

    Check out ``qecdec.decoders.DECODERS_REGISTRY`` for all available decoder names and
    the corresponding return classes.
    """
    if (cls := DECODERS_REGISTRY.get(name)) is None:
        raise ValueError(f"Invalid decoder name: {name!r}")
    sig = inspect.signature(cls.__init__)
    whitelist = [
        param_name
        for param_name, param in sig.parameters.items()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in whitelist}
    return cls(pcm, prior, **filtered_kwargs)
