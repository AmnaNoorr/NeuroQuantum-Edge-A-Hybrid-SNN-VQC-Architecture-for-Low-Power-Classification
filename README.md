# NeuroQuantum-Edge

Research scaffold for comparing three binary ECG classifiers on the same data split:

- a classical multilayer perceptron (MLP);
- a pure spiking neural network (SNN) built with [snnTorch](https://snntorch.readthedocs.io/);
- a hybrid SNN with a PennyLane variational quantum circuit (VQC) head.

The classical MLP, pure SNN, and hybrid SNN + VQC model modules are available.
Training evaluates the held-out test split and writes one JSON report per seed.

## Dataset contract

Provide one CSV file with 141 columns:

- 140 floating-point ECG features per row;
- one binary target column with `0` for normal and `1` for abnormal.

The loader uses `data.header: auto` and `data.label_column: auto` to detect a
header and the unique column containing only both binary values, rather than
assuming a target name or position. If a CSV is ambiguous, set
`data.label_column` to a header name or zero-based column index explicitly.
Place the file at `data/ecg.csv` when running the experiments. Dataset files
are intentionally ignored by Git.

## Repository layout

```text
src/
  data.py              # dataset schema and future loading entry point
  models/               # reserved for MLP, SNN, and hybrid model modules
  train.py              # configuration-driven training and test evaluation
  evaluate.py           # shared evaluation metrics and timing utilities
  utils.py              # reproducibility helpers
configs/
  classical_mlp.yaml
  snn.yaml
  hybrid_snn_vqc.yaml
results/                # metrics, figures, and checkpoints produced by runs
notebooks/              # Colab notebooks and notebook notes
tests/                  # lightweight unit tests
scripts/                # result aggregation utilities
```

## Running experiments

Run one seed, or use `--smoke` for 200 training samples and one epoch:

```bash
python -m src.train --config configs/classical_mlp.yaml --seed 42 --smoke
```

Reports are saved under `results/{experiment}/seed_{seed}.json`. Aggregate all
available seeds into a Markdown table with:

```bash
python scripts/aggregate_results.py --results-dir results --output results/summary.md
```

## Colab setup

1. Clone or upload this repository to Colab.
2. Put the ECG CSV in `data/ecg.csv` (or update the configuration path).
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Select a GPU runtime when one is available. The project will remain device-agnostic, although all experiment runs are designed to be launched from Colab.

Every eventual training notebook or script should call `src.utils.set_seed` before creating splits, data loaders, or models. The default seed is recorded in each experiment configuration. Use `src.data.create_ecg_dataloaders(config_path)` to create the shared, stratified 70/15/15 DataLoaders; it fits feature scaling on training rows only and records the resulting row indices in `results/splits.json`.

## Experiment plan

Use the same preprocessing, stratified split, seed, and reporting metrics for all three configurations. Report at least accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, parameter count, and an inference-cost proxy appropriate to the target edge device.

