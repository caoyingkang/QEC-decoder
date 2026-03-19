# PyTorch training and benchmarking

This subproject provides training scripts and a web interface for Monte Carlo benchmarking of learned decoder models.

## Training

Examples: (Assuming run from `pytorch/scripts` directory)
- `python train.py --config configs/train_LearnedDMemBP_d=5.yaml`
- `python train.py --config configs/train_MultiDMemBP_d=5.yaml loss.beta=0.0 model.mlp.activation=ReLU`
- `python train.py --config configs/train_MultiDMemBP_d=5.yaml --profile`

## Monte Carlo benchmark app

To launch the Streamlit app for Monte Carlo benchmarking of the PyTorch decoders:

```bash
streamlit run pytorch/benchmark/app.py
```

## Decoder metrics:

- Convergence Rate: The fraction of decoding attempts where the decoder converged (i.e. the estimated error satisfied the input syndrome).
- Logical Success Rate: The fraction of decoding attempts where the decoder predicted logical observables correctly.
- Strict Success Rate: The fraction of decoding attempts where the decoder converged and predicted logical observables correctly.
- Accidental Success Rate: The fraction of decoding attempts where the decoder failed to converge but luckily predicted logical observables correctly.
- Success Rate on Convergence: The fraction of decoding attempts where the decoder predicted logical observables correctly, given that the decoder converged.
- Average Iterations: The average number of decoding iterations.
- Average Iterations on Convergence: The average number of decoding iterations, given that the decoder converged.
