import numpy as np
import stim

from ..types import Float2DArray


def extract_error_mechanisms_from_dem(
    dem: stim.DetectorErrorModel,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    """
    Extract error mechanisms from a stim.DetectorErrorModel object. Each error mechanism is identified by its effect,
    which includes the set of detectors and the set of observables that are flipped. Error mechanisms with identical
    effect will be combined into one.

    The output is a dict with each item representing an error mechanism. For each item, the key is a pair of tuples:
    the first tuple contains the flipped detectors (sorted in increasing order), and the second tuple contains the
    flipped observables (sorted in increasing order); the value is the net probability that the error mechanism occurs.
    """
    eff2prob = {}

    instruction: stim.DemInstruction
    for instruction in dem.flattened():
        if instruction.type == "error":
            p = instruction.args_copy()[0]  # probability of the error

            dets: set[int] = set()  # flipped detectors
            obsers: set[int] = set()  # flipped observables
            t: stim.DemTarget
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    if t.val in dets:
                        dets.remove(t.val)
                    else:
                        dets.add(t.val)
                elif t.is_logical_observable_id():
                    if t.val in obsers:
                        obsers.remove(t.val)
                    else:
                        obsers.add(t.val)
                elif t.is_separator():
                    pass
                else:
                    raise RuntimeError("Not supposed to be here")
            eff = (tuple(sorted(dets)), tuple(sorted(obsers)))
            if (
                eff in eff2prob
            ):  # this error has appeared earlier, let's update its probability
                eff2prob[eff] = (1 - eff2prob[eff]) * p + eff2prob[eff] * (1 - p)
            else:  # this error is new, let's register it
                eff2prob[eff] = p
        elif instruction.type == "detector" or instruction.type == "logical_observable":
            pass
        else:
            raise ValueError(f"Instruction type not expected: {instruction.type}")

    return eff2prob


def extract_detector_coords_from_dem(dem: stim.DetectorErrorModel) -> Float2DArray:
    """
    Extract detector coordinates from a stim.DetectorErrorModel object. The output
    is a numpy array that has num_detectors rows, with the i-th row being the vector
    of coordinates of the i-th detector.
    """
    coords: list[list[float]] = [None] * dem.num_detectors

    instruction: stim.DemInstruction
    for instruction in dem.flattened():
        if instruction.type == "detector":
            coo = instruction.args_copy()
            t: stim.DemTarget = instruction.targets_copy()[0]
            coords[t.val] = coo
        elif instruction.type == "error" or instruction.type == "logical_observable":
            pass
        else:
            raise ValueError(f"Instruction type not expected: {instruction.type}")

    if None in coords:
        raise ValueError("Some detector coordinates are not found.")

    if not all(
        len(coords[i]) == len(coords[i + 1]) for i in range(dem.num_detectors - 1)
    ):
        raise ValueError("Detector coordinates have different lengths.")

    return np.array(coords)
