from typing import TypeAlias

import numpy as np


Bit1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.uint8]]
Bit2DArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
Bool1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.bool_]]
Bool2DArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
Bool3DArray: TypeAlias = np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]
Float1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
Float2DArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]
Int1DArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int64]]
Int2DArray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.int64]]
