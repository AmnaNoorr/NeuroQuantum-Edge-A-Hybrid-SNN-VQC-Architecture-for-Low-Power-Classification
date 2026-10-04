"""Shared binary-classification evaluation and inference-timing utilities."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .utils import count_params


class ProbabilityEstimator(Protocol):
    """Subset of the sklearn classifier protocol required for evaluation."""

    classes_: np.ndarray

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return class probabilities for feature rows."""


def evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    measure_inference: bool = True,
) -> dict[str, Any]:
    """Evaluate a PyTorch binary classifier and optionally time its forward pass.

    Inference timing covers model forward passes only: batches are already in
    memory and data-loader iteration or metric computation is excluded. CUDA is
    synchronized around each timed forward pass for an accurate wall-clock
    measurement.

    Args:
        model: Neural classifier that returns ``[batch, 2]`` logits.
        loader: Evaluation DataLoader.
        device: Device on which the model and inputs reside.
        measure_inference: Whether to include inference milliseconds per sample.

    Returns:
        JSON-serializable binary metrics and the model parameter count.
    """
    was_training = model.training
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    positive_scores: list[np.ndarray] = []
    elapsed_seconds = 0.0
    sample_count = 0

    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if measure_inference:
                _synchronize_if_cuda(device)
                start = time.perf_counter()
            logits = model(features)
            if measure_inference:
                _synchronize_if_cuda(device)
                elapsed_seconds += time.perf_counter() - start

            _validate_logits(logits)
            probabilities = torch.softmax(logits, dim=1)
            labels.append(targets.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            positive_scores.append(probabilities[:, 1].cpu().numpy())
            sample_count += targets.numel()

    if was_training:
        model.train()
    return binary_classification_metrics(
        y_true=np.concatenate(labels),
        y_pred=np.concatenate(predictions),
        positive_scores=np.concatenate(positive_scores),
        parameter_count=count_params(model),
        inference_time_ms_per_sample=(elapsed_seconds * 1_000 / sample_count)
        if measure_inference
        else None,
    )


def evaluate_sklearn_model(
    model: ProbabilityEstimator,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    parameter_count: int,
    measure_inference: bool = True,
) -> dict[str, Any]:
    """Evaluate a fitted sklearn probabilistic classifier on binary ECG data."""
    if measure_inference:
        start = time.perf_counter()
    probabilities = np.asarray(model.predict_proba(features))
    if measure_inference:
        elapsed_seconds = time.perf_counter() - start
    else:
        elapsed_seconds = 0.0

    classes = np.asarray(model.classes_)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("predict_proba must return [samples, classes] probabilities")
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError("Binary classifier must expose a class labeled 1")

    predictions = classes[probabilities.argmax(axis=1)]
    return binary_classification_metrics(
        y_true=labels,
        y_pred=predictions,
        positive_scores=probabilities[:, positive_indices[0]],
        parameter_count=parameter_count,
        inference_time_ms_per_sample=(elapsed_seconds * 1_000 / len(labels))
        if measure_inference
        else None,
    )


def binary_classification_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_scores: np.ndarray,
    parameter_count: int,
    inference_time_ms_per_sample: float | None,
) -> dict[str, Any]:
    """Compute JSON-serializable metrics for normal-vs-abnormal classification."""
    true_labels = np.asarray(y_true, dtype=np.int64)
    predicted_labels = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(positive_scores, dtype=float)
    if not (len(true_labels) == len(predicted_labels) == len(scores)):
        raise ValueError("Labels, predictions, and probability scores must have equal length")
    if len(true_labels) == 0:
        raise ValueError("Cannot evaluate an empty split")

    try:
        auroc: float | None = float(roc_auc_score(true_labels, scores))
    except ValueError:
        # A tiny or custom split can contain only one class; retain a valid JSON
        # result rather than failing the entire experiment report.
        auroc = None

    return {
        "num_samples": int(len(true_labels)),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "auroc": auroc,
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=[0, 1]
        ).tolist(),
        "parameter_count": int(parameter_count),
        "inference_time_ms_per_sample": (
            float(inference_time_ms_per_sample)
            if inference_time_ms_per_sample is not None
            else None
        ),
    }


def loader_to_numpy(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a CPU DataLoader as feature and label NumPy arrays.

    This is used only by the sklearn baseline; neural models should stream
    batches through :func:`evaluate_torch_model` instead.
    """
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_features, batch_labels in loader:
        features.append(batch_features.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())
    if not features:
        raise ValueError("Cannot convert an empty DataLoader to NumPy arrays")
    return np.concatenate(features), np.concatenate(labels)


def _validate_logits(logits: Tensor) -> None:
    """Fail clearly if a model does not implement the binary-logit contract."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            "Expected model logits with shape [batch, 2], received "
            f"{tuple(logits.shape)}"
        )


def _synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize asynchronous CUDA work before or after a timing boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
