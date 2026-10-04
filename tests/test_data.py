"""Tests for configuration-driven ECG data preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import ECG_FEATURE_COUNT, create_ecg_dataloaders, print_class_balance


@pytest.mark.parametrize(
    ("with_header", "label_index"),
    [(True, 17), (False, 93)],
)
def test_loader_infers_header_and_nonterminal_label_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], with_header: bool, label_index: int
) -> None:
    """A 140-feature synthetic CSV should load identically in both layouts."""
    sample_count = 200
    rng = np.random.default_rng(123)
    features = rng.normal(size=(sample_count, ECG_FEATURE_COUNT))
    labels = np.tile(np.array([0, 1], dtype=np.int64), sample_count // 2)
    csv_values = np.insert(features, label_index, labels, axis=1)
    csv_path = tmp_path / "synthetic_ecg.csv"

    if with_header:
        columns = [f"ecg_{index}" for index in range(ECG_FEATURE_COUNT + 1)]
        columns[label_index] = "diagnosis"
        pd.DataFrame(csv_values, columns=columns).to_csv(csv_path, index=False)
        expected_label_column = "diagnosis"
    else:
        pd.DataFrame(csv_values).to_csv(csv_path, index=False, header=False)
        expected_label_column = f"column_{label_index}"

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "synthetic.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"seed": 7},
                "data": {
                    "csv_path": "synthetic_ecg.csv",
                    "header": "auto",
                    "label_column": "auto",
                    "feature_count": ECG_FEATURE_COUNT,
                },
                "training": {"batch_size": 32},
            }
        ),
        encoding="utf-8",
    )

    loaders = create_ecg_dataloaders(config_path)

    assert loaders.has_header is with_header
    assert loaders.label_column == expected_label_column
    assert len(loaders.train.dataset) == 140
    assert len(loaders.val.dataset) == 30
    assert len(loaders.test.dataset) == 30
    assert np.allclose(
        loaders.train.dataset.tensors[0].mean(dim=0).numpy(), 0.0, atol=1e-6
    )
    expected_train_mean = features[loaders.split_indices["train"]].mean(axis=0)
    assert np.allclose(loaders.scaler.mean_, expected_train_mean, atol=1e-6)

    for split_name, expected_per_class in (("train", 70), ("val", 15), ("test", 15)):
        assert np.array_equal(
            np.bincount(loaders.split_labels[split_name], minlength=2),
            [expected_per_class, expected_per_class],
        )

    all_indices = set().union(*[set(indices) for indices in loaders.split_indices.values()])
    assert all_indices == set(range(sample_count))
    assert sum(len(indices) for indices in loaders.split_indices.values()) == sample_count

    splits_path = tmp_path / "results" / "splits.json"
    saved_splits = json.loads(splits_path.read_text(encoding="utf-8"))
    assert saved_splits["indices"] == loaders.split_indices
    assert saved_splits["label_column"] == expected_label_column

    print_class_balance(loaders)
    output = capsys.readouterr().out
    assert "train:" in output
    assert "val:" in output
    assert "test:" in output
