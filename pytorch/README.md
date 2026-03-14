# PyTorch training and benchmarking

This subproject provides training scripts and a web interface for Monte Carlo benchmarking of learned decoder models.

## Training

Examples: (Assuming run from `pytorch/scripts` directory)
- `python train.py --config configs/train_LearnedDMemBP.yaml`
- `python train.py --config configs/train_MultiDMemBP.yaml qec.d=11 qec.rounds=11 model.mlp.activation=ReLU`

## Monte Carlo benchmark app

To launch the Streamlit app for Monte Carlo benchmarking of the PyTorch decoders:

```bash
streamlit run pytorch/benchmark/app.py
```
