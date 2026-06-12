from functools import cache
from pathlib import Path

import ldpc.mod2
import numpy as np
import pytest
import stim

from qecdec.circuits import BBCode_Circuit, CIRCUITS_REGISTRY
from qecdec.circuits.utils import _extract_error_mechanisms_from_dem

# Gross code [[144,12,12]]: A = x^3 + y + y^2, B = y^3 + x + x^2.
GROSS = dict(
    ell=12, m=6, a_poly=((3, 0), (0, 1), (0, 2)), b_poly=((0, 3), (1, 0), (2, 0))
)
# Same polynomials on a 6x6 torus: the [[72,12,6]] code (fast geometry tests).
SMALL = dict(
    ell=6, m=6, a_poly=((3, 0), (0, 1), (0, 2)), b_poly=((0, 3), (1, 0), (2, 0))
)
ERROR_RATE = 0.003
SHIPPED_DIR = Path(__file__).resolve().parents[3] / "circuits" / "BB_144_12_12_Circuit"


@cache
def make_small(basis: str) -> BBCode_Circuit:
    return BBCode_Circuit(**SMALL, basis=basis, rounds=3, error_rate=ERROR_RATE)


@cache
def make_gross_unfiltered(basis: str) -> BBCode_Circuit:
    return BBCode_Circuit(
        **GROSS, basis=basis, rounds=12, error_rate=ERROR_RATE, filter_detectors=False
    )


@cache
def load_shipped(basis: str) -> stim.Circuit:
    return stim.Circuit.from_file(
        SHIPPED_DIR / f"basis={basis}_rounds=12" / f"error_rate={ERROR_RATE}.stim"
    )


def test_registered() -> None:
    assert CIRCUITS_REGISTRY["BBCode_Circuit"] is BBCode_Circuit


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_code_parameters(basis: str) -> None:
    circuit = make_small(basis)
    n2 = circuit.n2
    assert n2 == 36
    assert circuit.num_observables == 12
    assert circuit.num_detectors == (circuit.rounds + 1) * n2
    assert circuit.grid_shape == (circuit.rounds + 1, circuit.ell, circuit.m)
    unfiltered = BBCode_Circuit(
        **SMALL, basis=basis, rounds=3, error_rate=ERROR_RATE, filter_detectors=False
    )
    assert unfiltered.num_detectors == 2 * unfiltered.rounds * n2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(basis="Y"),
        dict(rounds=1),
        dict(a_poly=((3, 0), (0, 1))),
        dict(b_poly=((0, 3), (1, 0), (1, 0))),
    ],
)
def test_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BBCode_Circuit(**{**SMALL, "basis": "Z", "rounds": 3, **kwargs}, error_rate=ERROR_RATE)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_detector_grid_sites_match_dem_coords(basis: str) -> None:
    """Ranking the DEM detector coordinates (x -> col, y -> row, t -> layer)
    must reproduce the construction-derived grid sites exactly."""
    circuit = make_small(basis)
    coords = circuit.detector_coords
    assert coords.shape == (circuit.num_detectors, 3)
    xs, ys, ts = coords.T
    expected = np.stack(
        [
            np.searchsorted(np.unique(ts), ts),
            np.searchsorted(np.unique(ys), ys),
            np.searchsorted(np.unique(xs), xs),
        ],
        axis=1,
    )
    assert np.array_equal(circuit.detector_grid_sites, expected)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_check_data_nbrs_match_stabilizers(basis: str) -> None:
    """Every in-basis check has exactly 6 distinct data-qubit neighbors, and
    they coincide with the supports of the in-basis check matrix."""
    circuit = make_small(basis)
    nbrs = circuit.check_data_nbrs
    assert nbrs.shape == (circuit.n2, 6)
    h = circuit.hz if basis == "Z" else circuit.hx
    assert np.all(h.sum(axis=1) == 6)
    for i in range(circuit.n2):
        assert len(set(nbrs[i].tolist())) == 6
        assert set(nbrs[i].tolist()) == set(np.nonzero(h[i])[0].tolist())


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("shift", [(1, 0), (0, 1), (4, 5)])
def test_check_data_nbrs_torus_translation(basis: str, shift: tuple[int, int]) -> None:
    """Translating a check on the torus translates its data neighbors (with
    wraparound), separately within each data plane."""
    circuit = make_small(basis)
    ell, m, n2 = circuit.ell, circuit.m, circuit.n2
    dl, dm = shift
    sites = np.arange(n2)
    translated = ((sites // m + dl) % ell) * m + (sites % m + dm) % m
    nbrs = circuit.check_data_nbrs
    plane, local = nbrs // n2, nbrs % n2
    translated_nbrs = plane * n2 + ((local // m + dl) % ell) * m + (local % m + dm) % m
    assert np.array_equal(nbrs[translated], translated_nbrs)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_observable_supports(basis: str) -> None:
    circuit = make_small(basis)
    supports = circuit.observable_data_supports
    assert supports.shape == (circuit.num_observables, 2 * circuit.n2)
    assert supports.any(axis=1).all()
    # Logical operators commute with the opposite-basis stabilizers and are
    # independent modulo the same-basis stabilizers.
    h_same, h_other = (
        (circuit.hz, circuit.hx) if basis == "Z" else (circuit.hx, circuit.hz)
    )
    assert not ((h_other @ supports.T) % 2).any()
    rank_same = ldpc.mod2.rank(h_same)
    assert (
        ldpc.mod2.rank(np.vstack([h_same, supports]))
        == rank_same + circuit.num_observables
    )
    # The grid masks are the supports reorganized onto the two data planes.
    masks = circuit.observable_grid_masks
    assert masks.shape == (circuit.num_observables, 2, circuit.ell, circuit.m)
    assert masks.dtype == np.bool_
    assert np.array_equal(
        masks.reshape(circuit.num_observables, -1), supports.astype(np.bool_)
    )


# --------------------------------------------------------------------------------------------
# Agreement with the pre-generated [[144,12,12]] circuits
# --------------------------------------------------------------------------------------------


def _measurement_sequence(circuit: stim.Circuit) -> list[tuple[str, int]]:
    """(gate, qubit) of every measurement, in measurement order."""
    return [
        (inst.name, t.qubit_value)
        for inst in circuit
        if inst.name in ("M", "MX")
        for t in inst.targets_copy()
    ]


def _detector_measurement_sets(circuit: stim.Circuit) -> list[tuple[int, ...]]:
    """Absolute measurement indices of each detector, in detector order."""
    out: list[tuple[int, ...]] = []
    count = 0
    for inst in circuit:
        if inst.name in ("M", "MX"):
            count += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            out.append(tuple(sorted(t.value + count for t in inst.targets_copy())))
    return out


def _observable_supports_from_circuit(circuit: stim.Circuit) -> np.ndarray:
    """Data-qubit supports of the observables (data qubit id = column index)."""
    measured_qubits = [q for _, q in _measurement_sequence(circuit)]
    supports = np.zeros((circuit.num_observables, 144), dtype=np.uint8)
    for inst in circuit:
        if inst.name == "OBSERVABLE_INCLUDE":
            j = int(inst.gate_args_copy()[0])
            for t in inst.targets_copy():
                supports[j, measured_qubits[t.value + len(measured_qubits)]] ^= 1
    return supports


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_gross_code_measurements_and_detectors_match_shipped(basis: str) -> None:
    shipped = load_shipped(basis)
    ours = make_gross_unfiltered(basis).stim_circuit
    assert _measurement_sequence(ours) == _measurement_sequence(shipped)
    assert _detector_measurement_sets(ours) == _detector_measurement_sets(shipped)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_gross_code_error_model_matches_shipped(basis: str) -> None:
    """With the shipped observable definitions substituted in, the generated
    circuit's detector error model is identical to the shipped one: same error
    mechanisms (detector + observable flips) with the same probabilities."""
    shipped = load_shipped(basis)
    ours = make_gross_unfiltered(basis).stim_circuit
    swapped = stim.Circuit()
    for inst in ours:
        if inst.name != "OBSERVABLE_INCLUDE":
            swapped.append(inst)
    for inst in shipped:
        if inst.name == "OBSERVABLE_INCLUDE":
            swapped.append(inst)

    ours_mechs = _extract_error_mechanisms_from_dem(swapped.detector_error_model())
    shipped_mechs = _extract_error_mechanisms_from_dem(shipped.detector_error_model())
    assert ours_mechs.keys() == shipped_mechs.keys()
    np.testing.assert_allclose(
        [ours_mechs[k] for k in ours_mechs],
        [shipped_mechs[k] for k in ours_mechs],
        rtol=1e-9,
    )


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_gross_code_observables_equivalent_to_shipped(basis: str) -> None:
    """Our observables and the shipped ones span the same logical group: each
    set lies in the span of the other plus the same-basis stabilizers."""
    circuit = make_gross_unfiltered(basis)
    ours = circuit.observable_data_supports
    shipped = _observable_supports_from_circuit(load_shipped(basis))
    h_same = circuit.hz if basis == "Z" else circuit.hx

    rank_ours = ldpc.mod2.rank(np.vstack([h_same, ours]))
    rank_shipped = ldpc.mod2.rank(np.vstack([h_same, shipped]))
    rank_joint = ldpc.mod2.rank(np.vstack([h_same, ours, shipped]))
    assert rank_ours == rank_shipped == rank_joint
    assert rank_ours == ldpc.mod2.rank(h_same) + circuit.num_observables
