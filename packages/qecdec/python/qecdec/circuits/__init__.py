"""Module for QEC circuits."""

from .base import CIRCUITS_REGISTRY, QECCircuit
from .factory import create_circuit, create_circuit_with_uniform_error_rate
from .bb_144_12_12_circuit import BB_144_12_12_Circuit
from .hex_color_code_phenom import HexColorCode_Phenom
from .repetition_code_circuit import RepetitionCode_Circuit
from .rotated_surface_code_circuit import RotatedSurfaceCode_Circuit
from .rotated_surface_code_phenom import RotatedSurfaceCode_Phenom

__all__ = [
    "CIRCUITS_REGISTRY",
    "QECCircuit",
    "create_circuit",
    "create_circuit_with_uniform_error_rate",
    "BB_144_12_12_Circuit",
    "HexColorCode_Phenom",
    "RepetitionCode_Circuit",
    "RotatedSurfaceCode_Circuit",
    "RotatedSurfaceCode_Phenom",
]
