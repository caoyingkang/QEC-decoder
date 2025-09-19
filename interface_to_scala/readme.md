## Parameters

- code distance `d=5`
- number of rounds of stabilizer measurements `rounds=5`
- maximum number of BP iterations: `max_iter=50`


## Description of files in this directory

- `chkmat.npy`: numpy array of parity-check matrix, shape=(num_detectors, num_error_mechanisms), dtype=np.uint8. The entry at row $i$ and column $j$ is 1 if the $i$-th detector is flipped by the $j$-th error mechanism (i.e., the $i$-th check node is connected to the $j$-th variable node in the Tanner graph). Otherwise that entry is 0.
- `prior.npy`: numpy array of prior probabilities, shape=(num_error_mechanisms,), dtype=np.float64. The $j$-th entry is the prior probability that the $j$-th error mechanism occurs.
- `syndromes.npy`: numpy array of input syndromes, shape=(num_shots, num_detectors), dtype=np.uint8. The $i$-th row is the syndrome vector for the $i$-th sample.
- `ehat_bp.npy`: numpy array of estimated error vectors output by the vanilla BP decoder, shape=(num_shots, num_error_mechanisms), dtype=np.uint8. The $i$-th row is the estimated error vector for the $i$-th sample.
- `ehat_dmembp.npy`: Not here yet. Will be uploaded later.
- `decoding_graph.html`: Interactive picture of the decoding Tanner graph. To view it, open the html file in your browser. Hover over a node in the graph to see the index of that node. You can zoom, drag, and rotate the picture.
- `notebook.ipynb`: the notebook used to generate the above files.


## Other
My implementations of BP decoder and DMemBP decoder can be found in the file `QEC-decoder/src/lib.rs`.