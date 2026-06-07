# qecdec

Core **q**uantum **e**rror **c**orrection **dec**oding library: Rust implementations of decoders (BP, MemBP, DMemBP, RelayBP, UnionFind, etc.) exposed to Python via PyO3/maturin, plus Python APIs for QEC circuits and `sinter` integration.

This package is a member of the repository’s [uv](https://docs.astral.sh/uv/) workspace; install it from the repository root with `uv sync` (see the root [README.md](../../README.md)).

## Package layout

| Path | Contents |
|------|----------|
| `Cargo.toml`, `pyproject.toml` | Rust and Python/maturin project metadata. |
| `src/` | Rust crate: decoder implementations (BP, DMemBP, UnionFind, etc.). |
| `python/qecdec/` | Source code for the Python package `qecdec`. |
| `python/qecdec/decoders/` | Python-facing decoder APIs backed by Rust extension and third-party libraries. |
| `python/qecdec/circuits/` | QEC circuit classes and helper functions. |
| `python/qecdec/sinter_utils/` | Helpers to plug decoders into [sinter](https://pypi.org/project/sinter/). |
| `notebooks/` | Example Jupyter notebooks. |
| `tests/` | Unit tests. |

Extra dependencies for `notebooks/` are listed under `[dependency-groups]` in `pyproject.toml`. To include them, run the following command from the repository root:
```bash
uv sync --all-packages --group qecdec-notebooks
```

## Running tests

The test suite under `tests/` uses [pytest](https://pytest.org). From the repository root, install the `test` dependency group and run the suite:

```bash
uv sync --group test
uv run pytest packages/qecdec/tests -v
```

## Usage examples

- Sample syndrome-observable pairs from a repetition code memory circuit under circuit-level noise, and decode the syndromes using a BP decoder:

  ```python
  import numpy as np
  from qecdec.circuits import RepetitionCode_Circuit
  from qecdec.decoders import BPDecoder

  circuit = RepetitionCode_Circuit(
      d=5,
      rounds=5,
      data_qubit_error_rate=0.01,
      meas_error_rate=0.01,
      prep_error_rate=0.01,
      cnot_error_rate=0.01,
  )
  sampler = circuit.stim_circuit.compile_detector_sampler(seed=42)
  syndromes, observables = sampler.sample(shots=10_000, separate_observables=True)

  decoder = BPDecoder(circuit.chkmat, circuit.prior, max_iter=50)
  decoded_errors = decoder.decode_batch(syndromes.astype(np.uint8))
  ```

- Build a circuit by name (with a uniform error rate) and decode with a decoder built by name:

  ```python
  import numpy as np
  from qecdec.circuits import create_circuit_with_uniform_error_rate
  from qecdec.decoders import create_decoder

  circuit = create_circuit_with_uniform_error_rate(
      "RotatedSurfaceCode_Phenom", 0.01, d=5, rounds=5, basis="Z"
  )
  sampler = circuit.stim_dem.compile_sampler(seed=42)
  syndromes, observables, _ = sampler.sample(shots=10_000)

  decoder = create_decoder("MWPM", circuit.chkmat, circuit.prior)
  decoded_errors = decoder.decode_batch(syndromes.astype(np.uint8))
  ```

- Use `sinter` to collect the decoding results of a DMemBP decoder (with randomly selected memory coefficients) for a rotated surface code Z-basis memory circuit under phenomenological noise:

  ```python
  import numpy as np
  from qecdec.circuits import RotatedSurfaceCode_Phenom
  from qecdec.decoders import DMemBPDecoder
  from qecdec.sinter_utils import QecdecSinterDecoder
  import sinter
  import os

  circuit = RotatedSurfaceCode_Phenom(
      d=5,
      rounds=5,
      basis="Z",
      data_qubit_error_rate=0.01,
      meas_error_rate=0.01,
  )

  decoder = DMemBPDecoder(
      circuit.chkmat,
      circuit.prior,
      gamma=np.random.uniform(0, 1, size=(circuit.num_error_mechanisms,)),
      max_iter=50,
  )
  sinter_decoder = QecdecSinterDecoder(decoder, circuit.obsmat)
  custom_decoders = {"dmembp": sinter_decoder}

  tasks = [sinter.Task(circuit=circuit.stim_circuit, decoder="dmembp")]

  stats = sinter.collect(
      num_workers=os.cpu_count() - 1,
      max_shots=10_000_000,
      max_errors=100,
      tasks=tasks,
      custom_decoders=custom_decoders,
      print_progress=True,
  )
  ```
