"""Tests for repository-wide utility helpers."""

import random

import numpy as np
import pytest
import torch

from src.utils import set_seed


def test_set_seed_repeats_random_sequences() -> None:
    """The same seed should reproduce Python, NumPy, and PyTorch values."""
    set_seed(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    set_seed(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_set_seed_rejects_negative_values() -> None:
    """Negative seed values should fail with a clear validation error."""
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)
