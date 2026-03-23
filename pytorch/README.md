# PyTorch decoders

Neural-network-based QEC decoders and training framework.

## Training

From `pytorch/scripts`:

```bash
cd pytorch/scripts
python train.py --config configs/train_LearnedDMemBP_UniformIterationLoss_d=5.yaml
python train.py --config configs/train_MultiDMemBP_UniformIterationLoss_d=5.yaml loss.beta=0.0 model.mlp.activation=ReLU
python train.py --config configs/train_MultiDMemBP_ConvergenceAwareLoss_d=5.yaml --profile
```

With uv (from repo root):

```bash
uv run --project pytorch python pytorch/scripts/train.py --config pytorch/scripts/configs/train_LearnedDMemBP_UniformIterationLoss_d=5.yaml
```

## Decoder metrics:

- Convergence Rate: The fraction of decoding attempts where the decoder converged (i.e. the estimated error satisfied the input syndrome).
- Logical Success Rate: The fraction of decoding attempts where the decoder predicted logical observables correctly.
- Strict Success Rate: The fraction of decoding attempts where the decoder converged and predicted logical observables correctly.
- Accidental Success Rate: The fraction of decoding attempts where the decoder failed to converge but luckily predicted logical observables correctly.
- Success Rate on Convergence: The fraction of decoding attempts where the decoder predicted logical observables correctly, given that the decoder converged.
- Average Iterations: The average number of decoding iterations.
- Average Iterations on Convergence: The average number of decoding iterations, given that the decoder converged.
