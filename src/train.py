"""Configuration-driven training entry point for ECG experiments."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

from .data import create_ecg_dataloaders
from .evaluate import evaluate_torch_model
from .models.mlp import MLPClassifier
from .models.snn import SpikingECGClassifier
from .utils import set_seed


def run_training(
    config_path: str | Path,
    *,
    seed: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one configured model, evaluate its test split, and save JSON results."""
    config_file = Path(config_path)
    config = _load_config(config_file)
    experiment = _mapping(config, "experiment")
    effective_seed = int(experiment.get("seed", 42) if seed is None else seed)
    set_seed(effective_seed)

    results_dir = _results_dir(config_file) / str(experiment.get("name", config_file.stem))
    loaders = create_ecg_dataloaders(config_file, results_dir=_results_dir(config_file), seed=effective_seed)
    if smoke:
        loaders.train = _smoke_loader(loaders.train, effective_seed)

    device = _resolve_device(experiment.get("device", "cpu"))
    model = _build_model(_mapping(config, "model")).to(device)
    training = _mapping(config, "training")
    epochs = 1 if smoke else _positive_int(training.get("epochs", 1), "training.epochs")
    learning_rate = float(training.get("learning_rate", 0.001))
    weight_decay = float(training.get("weight_decay", 0.0))
    patience = _positive_int(
        training.get("early_stopping_patience", 10),
        "training.early_stopping_patience",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for features, targets in loaders.train:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1

        if batch_count == 0:
            raise ValueError("Training split is empty")
        val_metrics = evaluate_torch_model(
            model, loaders.val, device, measure_inference=False
        )
        epoch_record = {
            "epoch": epoch,
            "loss": total_loss / batch_count,
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch}/{epochs} loss={epoch_record['loss']:.6f} "
            f"val_macro_f1={epoch_record['val_macro_f1']:.6f}"
        )

        if epoch_record["val_macro_f1"] > best_f1:
            best_f1 = float(epoch_record["val_macro_f1"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state")
    model.load_state_dict(best_state)
    test_metrics = evaluate_torch_model(model, loaders.test, device)
    payload = {
        "experiment": str(experiment.get("name", config_file.stem)),
        "seed": effective_seed,
        "config": config,
        "smoke": smoke,
        "device": str(device),
        "training_samples": len(cast(Sized, loaders.train.dataset)),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "test": test_metrics,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"seed_{effective_seed}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved={output_path}")
    return payload


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    return config


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration requires a '{key}' mapping")
    return value


def _build_model(model_config: dict[str, Any]) -> nn.Module:
    family = model_config.get("family")
    if family == "mlp":
        kwargs = {key: value for key, value in model_config.items() if key != "family"}
        return MLPClassifier(**kwargs)
    if family == "snn":
        kwargs = {key: value for key, value in model_config.items() if key not in {"family", "neuron"}}
        return SpikingECGClassifier(**kwargs)
    if family == "hybrid_snn_vqc":
        from .models.hybrid import HybridSNNVQCClassifier

        vqc = model_config.get("vqc")
        if not isinstance(vqc, dict):
            raise ValueError("hybrid_snn_vqc model requires a vqc mapping")
        kwargs = {key: value for key, value in model_config.items() if key not in {"family", "neuron", "vqc"}}
        kwargs.update(
            {
                "backend": vqc.get("backend", "default.qubit"),
                "n_qubits": vqc.get("n_qubits", 4),
                "n_layers": vqc.get("n_layers", 2),
                "output_classes": vqc.get("output_classes", 2),
            }
        )
        return HybridSNNVQCClassifier(**kwargs)
    raise ValueError(f"Unsupported model.family: {family!r}")


def _smoke_loader(loader: DataLoader, seed: int) -> DataLoader:
    sample_count = min(200, len(cast(Sized, loader.dataset)))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(loader.dataset, range(sample_count)),
        batch_size=loader.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


def _resolve_device(configured: object) -> torch.device:
    requested = str(configured).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("experiment.device must be 'cpu' or 'cuda'")
    return torch.device("cpu")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _results_dir(config_path: Path) -> Path:
    return config_path.parent.parent / "results" if config_path.parent.name == "configs" else config_path.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one ECG experiment configuration")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_training(args.config, seed=args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
