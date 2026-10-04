"""Tests for static ECG dataset assumptions."""

from pathlib import Path

import pytest

from src.data import ECGDatasetSpec, ECG_FEATURE_COUNT


def test_dataset_spec_uses_expected_defaults() -> None:
    """The binary ECG contract should have 140 features and a label target."""
    spec = ECGDatasetSpec(Path("data/ecg.csv"))

    assert spec.feature_count == ECG_FEATURE_COUNT
    assert spec.label_column == "label"


def test_dataset_spec_rejects_the_wrong_feature_count() -> None:
    """The research protocol must not silently accept an incompatible CSV."""
    with pytest.raises(ValueError, match="exactly"):
        ECGDatasetSpec(Path("data/ecg.csv"), feature_count=139)
