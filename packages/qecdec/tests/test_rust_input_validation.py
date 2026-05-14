"""Validation errors raised by the Rust constructors.

Each test calls a Rust `*Rust` class directly (bypassing the Python wrappers,
which validate input separately) to verify that bad inputs surface as a clean
`ValueError` rather than a `PanicException`.
"""

import numpy as np
import pytest

from qecdec.qecdec import (
    BPDecoderRust,
    EnsSerialBPDecoderRust,
    UnionFindDecoderRust,
)


def _valid_pcm():
    # 3 checks, 5 variables; every CN has ≥2 nbrs, every VN has ≥1 nbr.
    return np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


# ---------- BPBase (via BPDecoderRust) ----------


def test_bpbase_check_with_one_neighbor_raises():
    pcm = _valid_pcm()
    pcm[0] = [1, 0, 0, 0, 0]  # CN 0 has 1 neighbor
    prior = np.full(pcm.shape[1], 0.1)
    with pytest.raises(ValueError, match="CN 0"):
        BPDecoderRust(pcm, prior, max_iter=5)


def test_bpbase_variable_with_zero_neighbors_raises():
    # 2 checks, 3 variables; column 2 is all-zero.
    pcm = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.uint8)
    prior = np.full(pcm.shape[1], 0.1)
    with pytest.raises(ValueError, match="VN 2"):
        BPDecoderRust(pcm, prior, max_iter=5)


def test_bpbase_prior_length_mismatch_raises():
    pcm = _valid_pcm()
    prior = np.full(pcm.shape[1] + 1, 0.1)  # wrong length
    with pytest.raises(ValueError, match="prior length"):
        BPDecoderRust(pcm, prior, max_iter=5)


# ---------- EnsSerialBPDecoderRust ----------


def test_ens_serial_vn_orders_wrong_ncols_raises():
    pcm = _valid_pcm()
    prior = np.full(pcm.shape[1], 0.1)
    bad_vn_orders = np.zeros((2, pcm.shape[1] + 1), dtype=np.int64)
    with pytest.raises(ValueError, match="vn_orders must have"):
        EnsSerialBPDecoderRust(
            pcm, prior, vn_orders=bad_vn_orders, max_iter=5, topk=1
        )


def test_ens_serial_topk_out_of_range_raises():
    pcm = _valid_pcm()
    prior = np.full(pcm.shape[1], 0.1)
    vn_orders = np.tile(np.arange(pcm.shape[1], dtype=np.int64), (3, 1))
    with pytest.raises(ValueError, match="topk"):
        EnsSerialBPDecoderRust(pcm, prior, vn_orders=vn_orders, max_iter=5, topk=0)
    with pytest.raises(ValueError, match="topk"):
        EnsSerialBPDecoderRust(pcm, prior, vn_orders=vn_orders, max_iter=5, topk=4)


# ---------- UnionFindDecoderRust ----------


def test_uf_check_with_one_variable_raises():
    pcm = np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )  # check 0 has 1 variable
    with pytest.raises(ValueError, match="Check 0"):
        UnionFindDecoderRust(pcm)


def test_uf_variable_in_zero_checks_raises():
    pcm = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.uint8,
    )  # variable 2 in no checks
    with pytest.raises(ValueError, match="Variable 2"):
        UnionFindDecoderRust(pcm)


def test_uf_variable_in_more_than_two_checks_raises():
    pcm = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )  # variable 0 is in 3 checks
    with pytest.raises(ValueError, match="Variable 0.*more than 2"):
        UnionFindDecoderRust(pcm)
