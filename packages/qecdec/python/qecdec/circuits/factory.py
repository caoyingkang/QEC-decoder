"""Factory for building QEC circuits."""

import inspect

from .base import CIRCUITS_REGISTRY, QECCircuit


def create_circuit(name: str, **kwargs) -> QECCircuit:
    """Create a QEC circuit by name. Only the kwargs that are present in the constructor's
    signature will be passed to the constructor.

    Check out ``qecdec.circuits.CIRCUITS_REGISTRY`` for all available circuit names and
    the corresponding return classes.
    """
    cls = CIRCUITS_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Invalid circuit name: {name!r}")
    sig = inspect.signature(cls.__init__)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**filtered_kwargs)
