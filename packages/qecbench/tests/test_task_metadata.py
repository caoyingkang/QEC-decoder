"""Tests for ``qecbench.task.metadata.TaskMetadata``."""

import json

import pytest
from frozendict import frozendict

from qecbench.task import TaskMetadata
from qecbench.task.metadata import _METADATA_COLUMNS


def test_valid_construction_freezes_params():
    md = TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params={"d": 3, "rounds": 3},
        error_rate=0.05,
        decoder_name="BP",
        decoder_params={"max_iter": 10, "norm": 0.9},
    )
    assert isinstance(md.circuit_params, frozendict)
    assert isinstance(md.decoder_params, frozendict)
    assert md.circuit_params["d"] == 3
    assert md.decoder_params["max_iter"] == 10


def test_decoder_label_defaults_to_decoder_name():
    md = TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params={"d": 3, "rounds": 3},
        error_rate=0.05,
        decoder_name="BP",
        decoder_params={"max_iter": 10},
    )
    assert md.decoder_label == "BP"


def test_decoder_label_custom_preserved():
    md = TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params={"d": 3, "rounds": 3},
        error_rate=0.05,
        decoder_name="BP",
        decoder_params={"max_iter": 10},
        decoder_label="BP-tuned",
    )
    assert md.decoder_label == "BP-tuned"


def test_invalid_circuit_name_raises():
    with pytest.raises(ValueError, match="Invalid circuit name"):
        TaskMetadata(
            circuit_name="NotARealCircuit",
            circuit_params={},
            error_rate=0.05,
            decoder_name="BP",
            decoder_params={"max_iter": 10},
        )


def test_invalid_decoder_name_raises():
    with pytest.raises(ValueError, match="Invalid decoder name"):
        TaskMetadata(
            circuit_name="RepetitionCode_Circuit",
            circuit_params={"d": 3, "rounds": 3},
            error_rate=0.05,
            decoder_name="NotARealDecoder",
            decoder_params={},
        )


def test_is_iterative_true_for_bp(bp_metadata):
    assert bp_metadata.is_iterative is True


def test_is_iterative_false_for_mwpm(mwpm_metadata):
    assert mwpm_metadata.is_iterative is False


def test_max_iter_resolved_for_iterative(bp_metadata):
    assert bp_metadata.max_iter == 20


def test_max_iter_raises_for_noniterative(mwpm_metadata):
    with pytest.raises(ValueError, match="Non-iterative decoder"):
        _ = mwpm_metadata.max_iter


def test_equality_and_hashable_dict_vs_frozendict():
    md_a = TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params={"d": 3, "rounds": 3},
        error_rate=0.05,
        decoder_name="BP",
        decoder_params={"max_iter": 10, "norm": 0.9},
    )
    md_b = TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params=frozendict({"d": 3, "rounds": 3}),
        error_rate=0.05,
        decoder_name="BP",
        decoder_params=frozendict({"max_iter": 10, "norm": 0.9}),
    )
    assert md_a == md_b
    assert hash(md_a) == hash(md_b)


def test_to_csv_rowdict_keys_and_schema_version(bp_metadata):
    row = bp_metadata.to_csv_rowdict()
    for col in _METADATA_COLUMNS:
        assert col in row
    assert row["SCHEMA_VERSION"] == TaskMetadata.SCHEMA_VERSION
    assert row["circuit_name"] == "RepetitionCode_Circuit"
    assert row["decoder_name"] == "BP"
    assert row["decoder_label"] == "BP"
    assert row["error_rate"] == 0.05


def test_to_csv_rowdict_params_are_sorted_json():
    md_a = TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params={"d": 3, "rounds": 3},
        error_rate=0.05,
        decoder_name="BP",
        decoder_params={"max_iter": 10, "norm": 0.9},
    )
    md_b = TaskMetadata(
        circuit_name="RepetitionCode_Circuit",
        circuit_params={"rounds": 3, "d": 3},
        error_rate=0.05,
        decoder_name="BP",
        decoder_params={"norm": 0.9, "max_iter": 10},
    )
    row_a = md_a.to_csv_rowdict()
    row_b = md_b.to_csv_rowdict()
    assert row_a["circuit_params"] == row_b["circuit_params"]
    assert row_a["decoder_params"] == row_b["decoder_params"]
    # Confirm the serialization is actually sorted-key JSON
    assert json.loads(row_a["circuit_params"]) == {"d": 3, "rounds": 3}
