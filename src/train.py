"""Training entry point placeholder.

This module will later construct models from YAML configuration and run the
shared training protocol. No model or optimization code is implemented yet.
"""

from pathlib import Path


def run_training(config_path: Path) -> None:
    """Reserve the public training entry point for a named experiment config.

    Args:
        config_path: Path to one YAML file in ``configs/``.

    Raises:
        NotImplementedError: Until data loading and model implementations exist.
    """
    raise NotImplementedError(
        f"Training is not implemented yet; received configuration: {config_path}"
    )
