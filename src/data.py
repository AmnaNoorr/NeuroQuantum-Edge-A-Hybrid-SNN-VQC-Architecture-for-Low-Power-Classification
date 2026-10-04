"""Configuration-driven data preparation for binary ECG classification.

The loader accepts CSV files with or without a header and locates the binary
label column without assuming a particular name or position. It writes the
row-level train, validation, and test membership to ``results/splits.json`` so
Colab runs can be audited and reproduced.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ECG_FEATURE_COUNT: Final[int] = 140
TRAIN_FRACTION: Final[float] = 0.70
VALIDATION_FRACTION: Final[float] = 0.15
TEST_FRACTION: Final[float] = 0.15

HeaderSetting: TypeAlias = bool | Literal["auto"]
LabelColumn: TypeAlias = str | int | None


@dataclass(frozen=True, slots=True)
class ECGDatasetSpec:
    """Describe an ECG CSV and its optional label metadata.

    A ``None`` label column enables automatic discovery. An integer selects a
    zero-based CSV column position; a string selects a named header column.
    """

    csv_path: Path
    label_column: LabelColumn = None
    header: HeaderSetting = "auto"
    feature_count: int = ECG_FEATURE_COUNT

    def __post_init__(self) -> None:
        """Validate static dataset assumptions before reading the CSV."""
        if self.feature_count != ECG_FEATURE_COUNT:
            raise ValueError(
                f"NeuroQuantum-Edge expects exactly {ECG_FEATURE_COUNT} ECG features"
            )
        if isinstance(self.label_column, str) and not self.label_column.strip():
            raise ValueError("label_column must not be empty when explicitly provided")
        if isinstance(self.label_column, bool):
            raise ValueError("label_column must be a name, integer position, or auto")
        if self.header not in (True, False, "auto"):
            raise ValueError("header must be true, false, or 'auto'")


@dataclass(frozen=True, slots=True)
class ECGDataConfig:
    """Data-related settings read from one experiment YAML file."""

    dataset: ECGDatasetSpec
    seed: int
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = False


@dataclass(slots=True)
class ECGDataLoaders:
    """PyTorch loaders and metadata created from one stratified data split."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    scaler: StandardScaler
    split_indices: dict[str, list[int]]
    split_labels: dict[str, np.ndarray]
    label_column: str
    has_header: bool


def load_data_config(config_path: str | Path) -> ECGDataConfig:
    """Load the data, training, and seed settings required for ECG loading.

    Relative dataset paths are resolved from the project root when the config
    is stored in a ``configs/`` directory; otherwise they are resolved from
    the config's own directory.

    Args:
        config_path: Path to an experiment YAML configuration.

    Returns:
        Validated settings ready for :func:`create_ecg_dataloaders`.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, Mapping):
        raise ValueError("Experiment configuration must be a YAML mapping")

    data_section = _required_mapping(config, "data")
    training_section = _required_mapping(config, "training")
    experiment_section = _required_mapping(config, "experiment")

    raw_csv_path = data_section.get("csv_path")
    if not isinstance(raw_csv_path, str) or not raw_csv_path.strip():
        raise ValueError("data.csv_path must be a non-empty string")

    root = path.parent.parent if path.parent.name == "configs" else path.parent
    csv_path = Path(raw_csv_path)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    feature_count = _positive_int(
        data_section.get("feature_count", ECG_FEATURE_COUNT), "data.feature_count"
    )
    label_column = _parse_label_column(data_section.get("label_column", "auto"))
    header = _parse_header_setting(data_section.get("header", "auto"))
    _validate_fixed_split(data_section.get("split"))
    seed = _non_negative_int(experiment_section.get("seed", 42), "experiment.seed")
    batch_size = _positive_int(training_section.get("batch_size", 64), "training.batch_size")
    num_workers = _non_negative_int(data_section.get("num_workers", 0), "data.num_workers")
    pin_memory = bool(data_section.get("pin_memory", torch.cuda.is_available()))

    return ECGDataConfig(
        dataset=ECGDatasetSpec(
            csv_path=csv_path,
            label_column=label_column,
            header=header,
            feature_count=feature_count,
        ),
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_ecg_dataloaders(
    config_path: str | Path,
    *,
    results_dir: str | Path | None = None,
    seed: int | None = None,
) -> ECGDataLoaders:
    """Load, stratify, scale, and package one ECG experiment dataset.

    The split is always 70% train, 15% validation, and 15% test. The
    ``StandardScaler`` is fitted on train rows only, then applied unchanged to
    validation and test rows. The split row indices are saved as JSON before
    the loaders are returned.

    Args:
        config_path: Path to an experiment YAML file containing ``data.csv_path``.
        results_dir: Optional output directory override. Defaults to the
            project-level ``results/`` directory.
        seed: Optional seed override for a multi-seed experiment run. When
            omitted, uses ``experiment.seed`` from the YAML configuration.

    Returns:
        A bundle containing train, validation, and test PyTorch DataLoaders.
    """
    config_file = Path(config_path)
    config = load_data_config(config_file)
    effective_seed = config.seed if seed is None else _non_negative_int(seed, "seed")
    features, labels, label_column, has_header = _read_ecg_csv(config.dataset)
    split_indices = _make_stratified_split_indices(labels, effective_seed)

    train_indices = np.asarray(split_indices["train"], dtype=np.int64)
    val_indices = np.asarray(split_indices["val"], dtype=np.int64)
    test_indices = np.asarray(split_indices["test"], dtype=np.int64)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices]).astype(np.float32)
    val_features = scaler.transform(features[val_indices]).astype(np.float32)
    test_features = scaler.transform(features[test_indices]).astype(np.float32)

    split_labels = {
        "train": labels[train_indices],
        "val": labels[val_indices],
        "test": labels[test_indices],
    }
    output_dir = Path(results_dir) if results_dir is not None else _default_results_dir(config_file)
    _save_split_indices(
        output_dir / "splits.json",
        split_indices=split_indices,
        seed=effective_seed,
        csv_path=config.dataset.csv_path,
        label_column=label_column,
        has_header=has_header,
    )

    generator = torch.Generator().manual_seed(effective_seed)
    worker_init_fn = _seed_worker if config.num_workers else None
    return ECGDataLoaders(
        train=DataLoader(
            _to_tensor_dataset(train_features, split_labels["train"]),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            worker_init_fn=worker_init_fn,
        ),
        val=DataLoader(
            _to_tensor_dataset(val_features, split_labels["val"]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            worker_init_fn=worker_init_fn,
        ),
        test=DataLoader(
            _to_tensor_dataset(test_features, split_labels["test"]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            worker_init_fn=worker_init_fn,
        ),
        scaler=scaler,
        split_indices=split_indices,
        split_labels=split_labels,
        label_column=label_column,
        has_header=has_header,
    )


def print_class_balance(loaders: ECGDataLoaders) -> None:
    """Print the sample count and normal/abnormal balance of every split.

    Args:
        loaders: Bundle returned by :func:`create_ecg_dataloaders`.
    """
    for split_name in ("train", "val", "test"):
        labels = loaders.split_labels[split_name]
        counts = np.bincount(labels, minlength=2)
        total = len(labels)
        normal_pct = counts[0] / total * 100
        abnormal_pct = counts[1] / total * 100
        print(
            f"{split_name}: total={total}, normal(0)={counts[0]} "
            f"({normal_pct:.1f}%), abnormal(1)={counts[1]} ({abnormal_pct:.1f}%)"
        )


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required YAML section while preserving a useful error message."""
    section = config.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"Configuration requires a '{key}' mapping")
    return section


def _parse_label_column(value: object) -> LabelColumn:
    """Translate the YAML label-column option to its typed representation."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("data.label_column cannot be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("data.label_column position must be non-negative")
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        return None if cleaned.lower() == "auto" else cleaned
    raise ValueError("data.label_column must be 'auto', a column name, or an index")


def _parse_header_setting(value: object) -> HeaderSetting:
    """Validate a true/false/auto CSV header option from YAML."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    raise ValueError("data.header must be true, false, or 'auto'")


def _validate_fixed_split(value: object) -> None:
    """Reject config drift from the shared 70/15/15 comparison protocol."""
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("data.split must be [0.70, 0.15, 0.15]")

    expected = (TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION)
    try:
        matches_protocol = all(
            np.isclose(float(actual), expected_fraction)
            for actual, expected_fraction in zip(value, expected, strict=True)
        )
    except (TypeError, ValueError):
        matches_protocol = False
    if not matches_protocol:
        raise ValueError("data.split must be [0.70, 0.15, 0.15]")


def _positive_int(value: object, name: str) -> int:
    """Validate an integer setting that must be greater than zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    """Validate an integer setting that may be zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _read_ecg_csv(spec: ECGDatasetSpec) -> tuple[np.ndarray, np.ndarray, str, bool]:
    """Read an ECG CSV, infer the label, and validate the tabular contract."""
    if not spec.csv_path.is_file():
        raise FileNotFoundError(f"ECG CSV does not exist: {spec.csv_path}")

    has_header = _detect_header(spec.csv_path) if spec.header == "auto" else spec.header
    frame = pd.read_csv(spec.csv_path, header=0 if has_header else None)
    if frame.empty:
        raise ValueError("ECG CSV must contain at least one data row")
    if not has_header:
        frame.columns = [f"column_{index}" for index in range(frame.shape[1])]

    expected_columns = spec.feature_count + 1
    if frame.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} CSV columns ({spec.feature_count} features "
            f"and one label), found {frame.shape[1]}"
        )

    label_column = _resolve_label_column(frame, spec.label_column)
    label_values = _validate_labels(frame[label_column], label_column)
    feature_frame = frame.drop(columns=label_column)
    features = _validate_features(feature_frame)

    return features, label_values, str(label_column), has_header


def _detect_header(csv_path: Path) -> bool:
    """Infer a header when the first non-empty row is not entirely numeric."""
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        first_row = next((row for row in reader if any(cell.strip() for cell in row)), None)

    if first_row is None:
        raise ValueError("ECG CSV is empty")
    return not all(_is_numeric(cell) for cell in first_row)


def _is_numeric(value: str) -> bool:
    """Return whether one raw CSV cell can be parsed as a floating-point value."""
    try:
        float(value)
    except ValueError:
        return False
    return True


def _resolve_label_column(frame: pd.DataFrame, configured: LabelColumn) -> str:
    """Resolve an explicit label selector or infer the unique binary column."""
    if isinstance(configured, int):
        if configured >= frame.shape[1]:
            raise ValueError(
                f"data.label_column index {configured} exceeds {frame.shape[1]} columns"
            )
        return str(frame.columns[configured])
    if isinstance(configured, str):
        if configured not in frame.columns:
            raise ValueError(f"Configured label column does not exist: {configured!r}")
        return configured

    candidates = [
        str(column)
        for column in frame.columns
        if _is_binary_label_series(frame[column])
    ]
    if len(candidates) != 1:
        candidate_description = ", ".join(candidates) if candidates else "none"
        raise ValueError(
            "Could not infer one binary label column; found "
            f"{candidate_description}. Set data.label_column explicitly."
        )
    return candidates[0]


def _is_binary_label_series(series: pd.Series) -> bool:
    """Return whether a series contains both numeric class labels 0 and 1 only."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        return False
    return set(numeric.astype(float).unique()) == {0.0, 1.0}


def _validate_labels(series: pd.Series, label_column: str) -> np.ndarray:
    """Convert the target column to validated binary PyTorch class indices."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or set(numeric.astype(float).unique()) != {0.0, 1.0}:
        raise ValueError(
            f"Label column {label_column!r} must contain both numeric classes 0 and 1"
        )
    return numeric.to_numpy(dtype=np.int64, copy=True)


def _validate_features(feature_frame: pd.DataFrame) -> np.ndarray:
    """Convert feature columns to a finite float32 matrix."""
    numeric = feature_frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        invalid_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"ECG feature columns contain non-numeric or missing values: {invalid_columns}")

    features = numeric.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(features).all():
        raise ValueError("ECG feature columns must contain only finite values")
    return features


def _make_stratified_split_indices(labels: np.ndarray, seed: int) -> dict[str, list[int]]:
    """Create reproducible 70/15/15 row indices while preserving class ratios."""
    all_indices = np.arange(labels.shape[0])
    try:
        train_indices, held_out_indices = train_test_split(
            all_indices,
            test_size=VALIDATION_FRACTION + TEST_FRACTION,
            random_state=seed,
            stratify=labels,
        )
        val_indices, test_indices = train_test_split(
            held_out_indices,
            test_size=0.5,
            random_state=seed,
            stratify=labels[held_out_indices],
        )
    except ValueError as error:
        raise ValueError(
            "Could not make a stratified 70/15/15 split. Ensure every class has "
            "enough examples for all three splits."
        ) from error

    return {
        "train": [int(index) for index in train_indices],
        "val": [int(index) for index in val_indices],
        "test": [int(index) for index in test_indices],
    }


def _to_tensor_dataset(features: np.ndarray, labels: np.ndarray) -> TensorDataset:
    """Create a float32-feature, int64-label TensorDataset."""
    return TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python state inside each optional DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _default_results_dir(config_path: Path) -> Path:
    """Resolve the standard project-level results directory for a config file."""
    if config_path.parent.name == "configs":
        return config_path.parent.parent / "results"
    return config_path.parent / "results"


def _save_split_indices(
    output_path: Path,
    *,
    split_indices: Mapping[str, list[int]],
    seed: int,
    csv_path: Path,
    label_column: str,
    has_header: bool,
) -> None:
    """Persist split membership and enough metadata to trace its source."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "csv_path": str(csv_path.resolve()),
        "has_header": has_header,
        "label_column": label_column,
        "fractions": {
            "train": TRAIN_FRACTION,
            "val": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "indices": dict(split_indices),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
