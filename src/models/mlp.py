"""Compact configurable multilayer perceptron for ECG classification."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class MLPClassifier(nn.Module):
    """A small fully connected classifier that returns unnormalized logits.

    The default output size of two is compatible with integer normal/abnormal
    labels and ``torch.nn.CrossEntropyLoss``. Hidden-layer widths and dropout
    are intentionally configurable through the experiment YAML files.

    Args:
        input_features: Number of ECG values per input row.
        hidden_features: Width of each hidden linear layer. An empty sequence
            creates a single linear classifier.
        output_classes: Number of output logits.
        dropout: Probability used after each hidden activation.
    """

    def __init__(
        self,
        input_features: int = 140,
        hidden_features: Sequence[int] = (128, 64),
        output_classes: int = 2,
        dropout: float = 0.20,
    ) -> None:
        """Build the configurable sequence of linear layers."""
        super().__init__()
        _validate_positive_int(input_features, "input_features")
        _validate_positive_int(output_classes, "output_classes")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0.0, 1.0)")

        widths = tuple(hidden_features)
        for index, width in enumerate(widths):
            _validate_positive_int(width, f"hidden_features[{index}]")

        layers: list[nn.Module] = []
        previous_width = input_features
        for width in widths:
            layers.extend((nn.Linear(previous_width, width), nn.ReLU()))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            previous_width = width
        layers.append(nn.Linear(previous_width, output_classes))

        self.input_features = input_features
        self.output_classes = output_classes
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class logits for a batch of shape ``[batch, input_features]``.

        Args:
            inputs: Standardized ECG feature batch.

        Raises:
            ValueError: If the input is not a two-dimensional ECG batch with
                the configured feature count.
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.input_features:
            raise ValueError(
                "Expected inputs with shape "
                f"[batch, {self.input_features}], received {tuple(inputs.shape)}"
            )
        return self.network(inputs)


def _validate_positive_int(value: int, name: str) -> None:
    """Reject non-positive layer dimensions with an actionable error."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
