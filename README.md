# QEC-decoder


- `decoder.py` contains the implementation of the following decoders:

    - BP decoder (min-sum version)

    - RelayBP decoder from the paper ["Improved belief propagation is sufficient for real-time decoding of quantum memory." arXiv:2506.01779 (2025)](https://arxiv.org/abs/2506.01779).


- `rotated_surface_code.py` constructs data structures to represent the rotated surface code, including the stabilizer matrices and logical matrices used for decoding in the perfect stabilizer measurement scenario, as well as check matrices and action matrices used for decoding in the noisy stabilizer measurement scenario.

    The coordinates and indexing convention of the data qubits and ancilla qubits are shown below:

    ![Rotated Surface Code Coordinates](figs/rotated_surface_code_coords.png)

    ![Rotated Surface Code Indices](figs/rotated_surface_code_indices.png)


- The notebook `test.ipynb` contains Monte Carlo simulation of the decoding performance.