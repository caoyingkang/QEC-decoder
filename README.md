# QEC-decoder

### Dependencies and installation

- Python >= 3.8
- [Install rust](https://www.rust-lang.org/tools/install).
- Download the repository and create a Python virtual environment.
  ```
  $ git clone https://github.com/caoyingkang/QEC-decoder.git
  $ cd QEC-decoder
  $ python3 -m venv .env
  $ source .env/bin/activate
  ```
- Install maturin:
  ```
  (.env) $ pip install maturin
  ```
- Build the Rust-based Python module in the current virtual environment:
  ```
  (.env) $ maturin develop --release
  ```


### Notations and conventions
The two figures below show the coordinates and indexing convention for the physical qubits (including data qubits and measure qubits) in a rotated surface code:

![Rotated Surface Code Coordinates](figs/rotated_surface_code_coords.png)

![Rotated Surface Code Indices](figs/rotated_surface_code_indices.png)