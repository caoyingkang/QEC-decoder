"""Module for QEC experiments."""

import inspect
from typing import Callable, Literal

from .base import Experiment
from .repetition_code_mem import RepetitionCode_Memory
from .rotated_surface_code_mem import RotatedSurfaceCode_Memory
from .hex_color_code_phenom_mem import HexColorCode_Phenom_Memory
from .stim_file import StimFileExperiment
# from .hex_color_code_superdense_mem import HexColorCode_Superdense_Memory


CIRCUIT_NAME_TO_FACTORY: dict[str, Callable[..., Experiment]] = {}


def register_circuit(name: str, factory: Callable[..., Experiment]) -> None:
    """Register a circuit factory under `name` in the global circuit registry.

    The factory is a callable taking `(p: float, **kwargs)` and returning an
    `Experiment`. Only the kwargs that are present in the factory's signature
    are passed by `create_experiment`.

    Parameters
    ----------
    name : str
        Registry key. Used by `create_experiment(circuit_name=name, ...)`.
    factory : Callable[..., Experiment]
        Callable producing an `Experiment` from `(p, **circuit_params)`.
    overwrite : bool, default False
        If False, raises `ValueError` when `name` is already registered.
    """
    if name in CIRCUIT_NAME_TO_FACTORY:
        raise ValueError(f"Circuit name {name!r} is already registered. ")
    CIRCUIT_NAME_TO_FACTORY[name] = factory


def create_experiment(circuit_name: str, *, p: float, **kwargs) -> Experiment:
    """Create a QEC experiment by name.

    Looks up `circuit_name` in `CIRCUIT_NAME_TO_FACTORY` and calls the factory
    with `p` plus only those `kwargs` present in the factory's signature.

    Parameters
    ----------
    circuit_name : str
        Registered circuit name. See `ALL_CIRCUITS` for the list.
    p : float
        Physical error rate (sweep variable). The factory decides how this
        maps onto the underlying experiment's error-rate kwargs.
    **kwargs
        Additional structural/numeric kwargs (e.g. `d`, `rounds`, `basis`).
    """
    factory = CIRCUIT_NAME_TO_FACTORY.get(circuit_name)
    if factory is None:
        raise ValueError(f"Invalid circuit name: {circuit_name!r}")
    sig = inspect.signature(factory)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return factory(p, **filtered_kwargs)


# -- Built-in factories ------------------------------------------------------


def _rotated_surface_phenom_memory(
    p: float, *, d: int, rounds: int, basis: Literal["X", "Z"]
) -> Experiment:
    return RotatedSurfaceCode_Memory(
        d=d,
        rounds=rounds,
        basis=basis,
        data_qubit_error_rate=p,
        meas_error_rate=p,
    )


def _rotated_surface_circuit_level_memory(
    p: float, *, d: int, rounds: int, basis: Literal["X", "Z"]
) -> Experiment:
    return RotatedSurfaceCode_Memory(
        d=d,
        rounds=rounds,
        basis=basis,
        data_qubit_error_rate=p,
        prep_error_rate=p,
        meas_error_rate=p,
        gate1_error_rate=p,
        gate2_error_rate=p,
    )


def _hex_color_code_phenom_memory(
    p: float, *, d: int, rounds: int, basis: Literal["X", "Y", "Z"]
) -> Experiment:
    return HexColorCode_Phenom_Memory(
        d=d,
        rounds=rounds,
        basis=basis,
        depolarizing_error_rate=p,
        meas_error_rate=p,
    )


def _repetition_code_phenom_memory(p: float, *, d: int, rounds: int) -> Experiment:
    return RepetitionCode_Memory(
        d=d,
        rounds=rounds,
        data_qubit_error_rate=p,
        meas_error_rate=p,
    )


register_circuit("RotatedSurfaceCode_Phenom_Memory", _rotated_surface_phenom_memory)
register_circuit(
    "RotatedSurfaceCode_CircuitLevel_Memory", _rotated_surface_circuit_level_memory
)
register_circuit("HexColorCode_Phenom_Memory", _hex_color_code_phenom_memory)
register_circuit("RepetitionCode_Phenom_Memory", _repetition_code_phenom_memory)


__all__ = [
    "Experiment",
    "RepetitionCode_Memory",
    "RotatedSurfaceCode_Memory",
    "HexColorCode_Phenom_Memory",
    "StimFileExperiment",
    "CIRCUIT_NAME_TO_FACTORY",
    "register_circuit",
    "create_experiment",
]
