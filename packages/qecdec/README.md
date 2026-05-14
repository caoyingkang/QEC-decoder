# qecdec

Core **q**uantum **e**rror **c**orrection **dec**oding library: Rust implementations of decoders (BP, DMemBP, UnionFind, etc.) exposed to Python via PyO3/maturin, plus Python APIs for memory experiments, sliding-window decoding, and `sinter` integration.

This package is a member of the repository’s [uv](https://docs.astral.sh/uv/) workspace; install it from the repository root with `uv sync` (see the root [README.md](../../README.md)).

## Package layout

| Path | Contents |
|------|----------|
| **`Cargo.toml`**, **`pyproject.toml`** | Rust and Python/maturin project metadata. |
| **`src/`** | Rust crate: decoder implementations (BP, DMemBP, UnionFind, etc.). |
| **`python/qecdec/`** | Source code for the Python package `qecdec`. |
| **`python/qecdec/decoders/`** | Python-facing decoder APIs backed by the Rust module. |
| **`python/qecdec/experiments/`** | Circuit for memory experiments. |
| **`python/qecdec/sinter_utils/`** | Helpers to plug decoders into [sinter](https://pypi.org/project/sinter/). |
| **`python/qecdec/slwin/`** | Helpers to use decoders in sliding-window decoding. |
| **`notebooks/`** | Example Jupyter notebooks. |
| **`tests/`** | Unit tests. |

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

- Sample syndrome-observable pairs from a repetition code memory experiment under circuit-level noise, and decode the syndromes using a BP decoder:

  ```python
  import numpy as np
  from qecdec.experiments import RepetitionCode_Memory
  from qecdec.decoders import BPDecoder

  expmt = RepetitionCode_Memory(
      d=5,
      rounds=5,
      data_qubit_error_rate=0.01,
      meas_error_rate=0.01,
      prep_error_rate=0.01,
      cnot_error_rate=0.01,
  )
  sampler = expmt.circuit.compile_detector_sampler(seed=42)
  syndromes, observables = sampler.sample(shots=10_000, separate_observables=True)

  decoder = BPDecoder(expmt.chkmat, expmt.prior, max_iter=50)
  decoded_errors = decoder.decode_batch(syndromes.astype(np.uint8))
  ```

- Sample syndrome-observable pairs from a rotated surface code Z-basis memory experiment under phenomenological noise, and decode the syndromes in sliding window, with MWPM as the inner decoder:

  ```python
  from qecdec.experiments import RotatedSurfaceCode_Memory
  from qecdec.slwin import SlidingWindowDecoder

  expmt = RotatedSurfaceCode_Memory(
      d=5,
      rounds=50,
      basis="Z",
      data_qubit_error_rate=0.01,
      meas_error_rate=0.01,
  )
  sampler = expmt.circuit.compile_detector_sampler(seed=42)
  syndromes, observables = sampler.sample(shots=10_000, separate_observables=True)

  decoder = SlidingWindowDecoder.from_pcm_prior(
      expmt.chkmat,
      expmt.prior,
      detectors_per_layer=expmt.num_detectors_per_layer,
      window_size=5,
      commit_size=1,
  )
  decoder.configure_inner_decoders("MWPM")
  decoded_errors = decoder.decode_batch(syndromes.astype(np.uint8))
  ```

- Use `sinter` to collect the decoding results of DMemBP decoder (with randomly selected memory coefficients) for a rotated surface code Z-basis memory experiment under phenomenological noise:

  ```python
  import numpy as np
  from qecdec.experiments import RotatedSurfaceCode_Memory
  from qecdec.decoders import DMemBPDecoder
  from qecdec.sinter_utils import QecdecSinterDecoder
  import sinter
  import os

  expmt = RotatedSurfaceCode_Memory(
      d=5,
      rounds=50,
      basis="Z",
      data_qubit_error_rate=0.01,
      meas_error_rate=0.01,
  )

  decoder = DMemBPDecoder(
      expmt.chkmat,
      expmt.prior,
      gamma=np.random.uniform(0, 1, size=(expmt.num_error_mechanisms,)),
      max_iter=50,
  )
  sinter_decoder = QecdecSinterDecoder(decoder, expmt.obsmat)
  custom_decoders = {"dmembp": sinter_decoder}

  tasks = [
      sinter.Task(
          circuit=expmt.circuit,
          decoder="dmembp",
          json_metadata={"d": 5, "rounds": 5, "p": 0.01},
      )
  ]

  stats = sinter.collect(
      num_workers=os.cpu_count() - 1,
      max_shots=10_000_000,
      max_errors=100,
      tasks=tasks,
      custom_decoders=custom_decoders,
      print_progress=True,
  )
  ```
