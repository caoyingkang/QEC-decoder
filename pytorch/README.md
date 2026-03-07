# PyTorch training and benchmarking

This subproject provides training scripts and a web interface for Monte Carlo benchmarking of learned decoder models.

## Training

```bash
cd pytorch/scripts
python train_dmembp.py qec.d=5 qec.rounds=5 model.num_iters=5
```

## Monte Carlo benchmark app

Run the Streamlit app to select trained runs, run or load cached benchmarks, and plot logical error rate vs physical error rate:

```bash
streamlit run pytorch/app/benchmark_app.py
```
