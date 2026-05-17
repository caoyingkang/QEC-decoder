"""Numpy type aliases."""

from typing import TypeAlias

import numpy as np

Int1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int64]]
Bool1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
Float1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
Bit2DArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
