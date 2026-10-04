"""Dataset schema definitions for binary ECG classification.

CSV parsing, feature validation, splitting, and DataLoader construction will be
implemented here in a later milestone.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final


ECG_FEATURE_COUNT: Final[int] = 140
DEFAULT_LABEL_COLUMN: Final[str] = "label"


@dataclass(frozen=True, slots=True)
class ECGDatasetSpec:
    """Describe the expected tabular ECG input without loading it.

    Attributes:
        csv_path: Location of the dataset CSV.
        label_column: Name of the binary target column.
        feature_count: Number of floating-point ECG features per row.
    """

    csv_path: Path
    label_column: str = DEFAULT_LABEL_COLUMN
    feature_count: int = ECG_FEATURE_COUNT

    def __post_init__(self) -> None:
        """Validate static dataset assumptions early."""
        if not self.label_column.strip():
            raise ValueError("label_column must not be empty")
        if self.feature_count != ECG_FEATURE_COUNT:
            raise ValueError(
                f"NeuroQuantum-Edge expects exactly {ECG_FEATURE_COUNT} ECG features"
            )
