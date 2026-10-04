# NeuroQuantum-Edge

**A hybrid spiking neural network + variational quantum circuit (SNN-VQC) study on binary ECG classification.**

[![Open demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AmnaNoorr/NeuroQuantum-Edge-A-Hybrid-SNN-VQC-Architecture-for-Low-Power-Classification/blob/main/notebooks/demo.ipynb)

This repository compares three classifiers on the same ECG data, with the same stratified split, seeds, and metrics:

| Model | Idea |
| --- | --- |
| **MLP** | Classical fully connected baseline |
| **SNN** | Spiking network of leaky integrate-and-fire neurons ([snnTorch](https://snntorch.readthedocs.io/)) |
| **Hybrid SNN+VQC** | Spiking feature extractor, a linear bottleneck to a few qubits, and a variational quantum circuit ([PennyLane](https://pennylane.ai/)) |

The question is whether spiking and hybrid quantum-classical models can **match** a classical baseline on a real biomedical signal task, and what they cost. It is a small, reproducible feasibility study, not a claim of superiority.

## Key results

> **Status:** the MLP and Hybrid SNN+VQC results below are current for 5 seeds (0-4). The pure SNN is being re-run after an input-encoding fix, so its row is not reported yet.

Held-out test results, pooled over the test sets of the 5 seeds (3,750 beats per model: 1,560 abnormal, 2,190 normal). Per-run JSON reports are in [`results/`](results/) and the plots are in [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

| Model | Accuracy | Macro F1 | Abnormal recall | Normal recall | Parameters |
| --- | --- | --- | --- | --- | --- |
| MLP | 99.36% | 0.9934 | 99.17% (1547/1560) | 99.50% (2179/2190) | 26,434 |
| Hybrid SNN+VQC | 99.25% | 0.9923 | 99.17% (1547/1560) | 99.32% (2175/2190) | 26,598 |
| SNN | pending re-run | pending | pending | pending | 26,434 |

These figures are computed from the pooled confusion matrices; per-seed means ± std and ROC-AUC are in the demo notebook.

What the results support:

- The MLP and the hybrid both reach about 99.3% accuracy. They differ by 4 errors out of 3,750 beats (24 vs. 28), far below what the seed-to-seed spread can resolve. **Neither model is claimed to be more accurate than the other.**
- The hybrid adds only 164 parameters over the MLP; the quantum circuit itself has 24 trainable weights (4 qubits x 2 layers x 3 rotation angles).
- Simulated inference is about two orders of magnitude slower for the hybrid than for the MLP on CPU (roughly 0.8 vs. 0.004 ms per sample; see [Limitations](#limitations)), so this repository makes **no efficiency claim**.

## Dataset

- A public Kaggle ECG dataset: 4,998 heartbeats, 140 samples each, plus a binary label (the page is by Devavrata Tripathy, license listed as unknown, so the CSV is **not** included in this repository). The Kaggle page does not state its original source; its shape and class counts match the widely used ECG5000 benchmark, but verify the provenance before citing it.
- **Labels:** `0` = abnormal (2,079 rows, 41.6%), `1` = normal (2,919 rows, 58.4%).
- Place the file at `data/ecg.csv`. The loader detects a header and the binary label column automatically (override with `data.header` / `data.label_column` in a config).

## Method

**Data handling.** Stratified 70/15/15 train/validation/test split (3,498 / 750 / 750 rows). A new split is drawn per seed. The `StandardScaler` is fitted on training rows only. Split indices are saved to `results/splits.json`.

**MLP.** Small fully connected network, 26,434 trainable parameters.

**SNN.** 140 -> 128 -> 64 -> 2 leaky integrate-and-fire layers (beta = 0.9, fast-sigmoid surrogate gradient with slope 25), simulated for 25 timesteps. The default `direct` encoding feeds the real-valued 140-point beat to the first layer at every timestep, so the first layer is a dense layer on real values and the spiking happens in the layers after it. Per-feature `rate` (Bernoulli) and `delta` (threshold-crossing) encodings are also implemented. The prediction is the output-layer spike count.

**Hybrid SNN+VQC.**

```
ECG (140) -> SNN feature extractor (140 -> 128 -> 64, spike counts over 25 steps)
          -> linear bottleneck (64 -> n_qubits)
          -> angle scaling  pi * tanh(x / timesteps)
          -> AngleEmbedding (RY) + StronglyEntanglingLayers (2 layers), 4 qubits
          -> PauliZ expectation values (4)
          -> linear readout (4 -> 2 logits)
```

The bottleneck is needed because a 64-wide SNN output cannot be loaded into a circuit with only a handful of qubits. The circuit runs on PennyLane's `default.qubit` simulator with backpropagation.

**Training.** Learning rate 1e-3, weight decay 1e-4, up to 50 epochs with early stopping (patience 10) on validation macro F1. Batch size 64 (MLP, SNN) and 32 (hybrid). All settings live in `configs/`.

**Metrics.** Accuracy, macro F1, ROC-AUC, confusion matrix, abnormal-class recall and precision, parameter count, and inference time per sample, all on the held-out test split.

## Repository layout

```
src/
  data.py          # CSV loading, stratified split, train-only scaling
  models/          # mlp.py, snn.py, hybrid.py
  train.py         # config-driven training and test evaluation
  evaluate.py      # metrics and timing
  utils.py         # seeding helpers
configs/           # classical_mlp.yaml, snn.yaml, hybrid_snn_vqc.yaml
scripts/           # aggregate_results.py
notebooks/         # demo.ipynb (plots and tables from saved results)
results/           # per-run JSON reports and summary.md
tests/             # lightweight unit tests
```

## Reproduce

Training is designed to be run on Colab (no GPU is required; the models are small).

```bash
git clone https://github.com/AmnaNoorr/NeuroQuantum-Edge-A-Hybrid-SNN-VQC-Architecture-for-Low-Power-Classification.git
cd NeuroQuantum-Edge-A-Hybrid-SNN-VQC-Architecture-for-Low-Power-Classification
pip install -r requirements.txt
# put the dataset at data/ecg.csv

# quick check that everything runs (1 epoch, 200 samples; delete the smoke outputs afterwards)
python -m src.train --config configs/snn.yaml --seed 42 --smoke

# full runs, seeds 0-4
for s in 0 1 2 3 4; do
  python -m src.train --config configs/classical_mlp.yaml --seed $s
  python -m src.train --config configs/snn.yaml --seed $s
  python -m src.train --config configs/hybrid_snn_vqc.yaml --seed $s
done

python scripts/aggregate_results.py --results-dir results --output results/summary.md
```

Each run writes `results/<experiment>/seed_<n>.json`. Smoke runs are saved with `"smoke": true` and are ignored by the demo notebook; remove them before aggregating with the script (`grep -l '"smoke": true' results/*/*.json | xargs rm -f`). A full run of the hybrid takes about two minutes on Colab's CPU.

Then open [`notebooks/demo.ipynb`](notebooks/demo.ipynb) to see the metrics table, confusion matrices, training curves, and cost comparison. It only reads saved results and does no training.

## Limitations

- **One dataset, one recording source.** Results may not transfer to other patients, devices, or lead configurations, and the test set (750 beats) is small: one beat is about 0.13% accuracy.
- **Near-ceiling performance.** All models score close to 100%, so this dataset cannot separate them; small differences are noise.
- **The quantum circuit is simulated** on a classical CPU with 4 qubits, so the timings say nothing about quantum hardware, and no quantum advantage is claimed or tested.
- **No energy claim.** The SNN runs as dense PyTorch code, so it gets no speed or energy benefit from spike sparsity here. Spike counts and an operations-based energy estimate are future work. Note that with `direct` encoding the first layer still performs full multiply-accumulates on real values.
- **The "Edge" and "low-power" framing is a motivation, not a result.**

## Planned work

- Re-run the pure SNN for seeds 0-4 after the encoding fix and add it to the results table.
- Ablation: replace the VQC with a classical layer of the same width to test whether the quantum circuit contributes anything.
- Sweep over the number of qubits and circuit depth.
- Low-data and noisy-signal experiments, where the models may separate.
- Spike-sparsity and operation-count analysis for an edge-energy estimate.
- Compare spike encodings (`direct`, `rate`, `delta`).

## References

- J. K. Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning," *Proceedings of the IEEE*, 2023 (snnTorch).
- V. Bergholm et al., "PennyLane: Automatic differentiation of hybrid quantum-classical computations," arXiv:1811.04968, 2018.
