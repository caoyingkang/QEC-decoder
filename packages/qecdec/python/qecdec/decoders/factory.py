"""Factory for building decoders."""

import inspect

from .base import Decoder, DECODERS_REGISTRY


def create_decoder(name: str, **kwargs) -> Decoder:
    """Create a decoder by name. Only the kwargs that are present in the constructor's
    signature will be passed to the constructor.

    Check out ``qecdec.decoders.DECODERS_REGISTRY`` for all available decoder names and
    the corresponding return classes.
    """
    cls = DECODERS_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Invalid decoder name: {name!r}")
    sig = inspect.signature(cls.__init__)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**filtered_kwargs)
