"""Evaluation entry point placeholder.

Metric computation and comparison plots will be added after model training is
available. No model inference code is implemented yet.
"""

from pathlib import Path


def run_evaluation(config_path: Path) -> None:
    """Reserve the public evaluation entry point for an experiment config.

    Args:
        config_path: Path to the YAML configuration used for the run.

    Raises:
        NotImplementedError: Until trained-model evaluation is implemented.
    """
    raise NotImplementedError(
        f"Evaluation is not implemented yet; received configuration: {config_path}"
    )
