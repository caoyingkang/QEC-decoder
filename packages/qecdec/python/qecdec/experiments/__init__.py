"""Module for QEC experiments."""

from .base import Experiment
from .repetition_code_mem import RepetitionCode_Memory
from .rotated_surface_code_mem import RotatedSurfaceCode_Memory
from .hex_color_code_phenom_mem import HexColorCode_Phenom_Memory
from .hex_color_code_superdense_mem import HexColorCode_Superdense_Memory
from .stim_file import StimFileExperiment

__all__ = [
    "Experiment",
    "RepetitionCode_Memory",
    "RotatedSurfaceCode_Memory",
    "HexColorCode_Phenom_Memory",
    "HexColorCode_Superdense_Memory",
    "StimFileExperiment",
]
