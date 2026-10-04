# NeuroQuantum-Edge

Research scaffold for comparing three binary ECG classifiers on the same data split:

- a classical multilayer perceptron (MLP);
- a pure spiking neural network (SNN) built with [snnTorch](https://snntorch.readthedocs.io/);
- a hybrid SNN with a PennyLane variational quantum circuit (VQC) head.

Model implementations are intentionally not included yet. This repository establishes a reproducible layout, shared experiment settings, and a Colab-ready environment first.

## Dataset contract

Provide one CSV file with 141 columns:

- 140 floating-point ECG features per row;
- one binary target column with `0` for normal and `1` for abnormal.

The starter configurations assume the target is named `label`. Change `data.label_column` in the relevant YAML file if your data uses a different name. Place the file at `data/ecg.csv` when running the experiments. Dataset files are intentionally ignored by Git.

## Repository layout

```text
src/
  data.py              # dataset schema and future loading entry point
  models/               # reserved for MLP, SNN, and hybrid model modules
  train.py              # training entry point placeholder
  evaluate.py           # evaluation entry point placeholder
  utils.py              # reproducibility helpers
configs/
  classical_mlp.yaml
  snn.yaml
  hybrid_snn_vqc.yaml
results/                # metrics, figures, and checkpoints produced by runs
notebooks/              # Colab notebooks and notebook notes
tests/                  # lightweight unit tests
```

## Colab setup

1. Clone or upload this repository to Colab.
2. Put the ECG CSV in `data/ecg.csv` (or update the configuration path).
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Select a GPU runtime when one is available. The project will remain device-agnostic, although all experiment runs are designed to be launched from Colab.

Every eventual training notebook or script should call `src.utils.set_seed` before creating splits, data loaders, or models. The default seed is recorded in each experiment configuration.

## Experiment plan

Use the same preprocessing, stratified split, seed, and reporting metrics for all three configurations. Report at least accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, parameter count, and an inference-cost proxy appropriate to the target edge device.

