"""Hybrid SNN + variational quantum-circuit classifier for ECG data.

The SNN produces a temporal spike-count representation, which is reduced by a
linear bottleneck to ``n_qubits`` values before quantum processing. This
dimensionality-reduction step is necessary because the SNN feature width (for
example, 64 neurons) is usually much larger than the feasible qubit count; the
projected values supply exactly one angle per qubit to ``AngleEmbedding``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import pennylane as qml
import snntorch as snn
from snntorch import surrogate
import torch
from torch import Tensor, nn

from .snn import SpikeEncoding, delta_encode, rate_encode


FeatureReadout: TypeAlias = Literal["spike_count", "membrane"]


class SNNFeatureExtractor(nn.Module):
    """Extract an ECG feature vector from final hidden-layer LIF activity.

    The input row is converted to a time-indexed rate or delta spike train by
    the shared encoders from :mod:`src.models.snn`. The output has one value per
    neuron in the last SNN hidden layer.
    """

    def __init__(
        self,
        *,
        input_features: int = 140,
        hidden_features: Sequence[int] = (128, 64),
        timesteps: int = 25,
        beta: float = 0.90,
        encoding: SpikeEncoding = "rate",
        delta_threshold: float = 0.10,
        surrogate_slope: float = 25.0,
        feature_readout: FeatureReadout = "spike_count",
    ) -> None:
        """Build the feed-forward Leaky SNN feature stack."""
        super().__init__()
        _validate_positive_int(input_features, "input_features")
        _validate_positive_int(timesteps, "timesteps")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in the interval (0.0, 1.0]")
        if encoding not in ("rate", "delta"):
            raise ValueError("encoding must be 'rate' or 'delta'")
        if delta_threshold <= 0.0:
            raise ValueError("delta_threshold must be greater than zero")
        if surrogate_slope <= 0.0:
            raise ValueError("surrogate_slope must be greater than zero")
        if feature_readout not in ("spike_count", "membrane"):
            raise ValueError("feature_readout must be 'spike_count' or 'membrane'")

        widths = tuple(hidden_features)
        if not widths:
            raise ValueError("hidden_features must contain at least one SNN layer")
        for index, width in enumerate(widths):
            _validate_positive_int(width, f"hidden_features[{index}]")

        self.input_features = input_features
        self.timesteps = timesteps
        self.encoding: SpikeEncoding = encoding
        self.delta_threshold = delta_threshold
        self.feature_readout: FeatureReadout = feature_readout
        self.output_features = widths[-1]

        encoded_features = 1 if encoding == "rate" else 2
        spike_gradient = surrogate.fast_sigmoid(slope=surrogate_slope)
        self.linears = nn.ModuleList()
        self.neurons = nn.ModuleList()
        previous_width = encoded_features
        for width in widths:
            self.linears.append(nn.Linear(previous_width, width))
            self.neurons.append(snn.Leaky(beta=beta, spike_grad=spike_gradient))
            previous_width = width

    def forward(self, inputs: Tensor) -> Tensor:
        """Return final hidden spike counts or membrane features per ECG row."""
        self._validate_inputs(inputs)
        encoded = self._encode(inputs)
        batch_size = inputs.shape[0]
        membranes = [
            inputs.new_zeros((batch_size, linear.out_features))
            for linear in self.linears
        ]
        final_spikes: list[Tensor] = []
        final_membranes: list[Tensor] = []

        for encoded_step in encoded:
            spikes = encoded_step
            for index, (linear, neuron) in enumerate(
                zip(self.linears, self.neurons, strict=True)
            ):
                spikes, membranes[index] = neuron(linear(spikes), membranes[index])
            final_spikes.append(spikes)
            final_membranes.append(membranes[-1])

        if self.feature_readout == "spike_count":
            return torch.stack(final_spikes).sum(dim=0)
        return final_membranes[-1]

    def _encode(self, inputs: Tensor) -> Tensor:
        """Apply the configured one-dimensional ECG spike encoding."""
        if self.encoding == "rate":
            return rate_encode(inputs, self.timesteps)
        return delta_encode(inputs, self.timesteps, self.delta_threshold)

    def _validate_inputs(self, inputs: Tensor) -> None:
        """Ensure the batch is a floating-point ECG feature matrix."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input_features:
            raise ValueError(
                "Expected inputs with shape "
                f"[batch, {self.input_features}], received {tuple(inputs.shape)}"
            )
        if not torch.is_floating_point(inputs):
            raise TypeError("SNN inputs must be a floating-point tensor")


class HybridSNNVQCClassifier(nn.Module):
    """SNN feature extractor followed by a PennyLane variational quantum head.

    Args:
        input_features: Number of ECG samples per input row.
        snn_hidden_features: Widths of the SNN feature-extraction layers.
        snn_timesteps: ECG simulation length for the SNN encoder.
        beta: Leaky neuron decay factor in ``(0, 1]``.
        encoding: Spike encoding: ``"rate"`` or ``"delta"``.
        delta_threshold: Delta-event threshold for ``"delta"`` encoding.
        surrogate_slope: Fast-sigmoid surrogate-gradient slope for LIF neurons.
        n_qubits: Number of VQC wires and size of the linear bottleneck.
        n_layers: Number of ``StronglyEntanglingLayers`` blocks in the VQC.
        backend: PennyLane simulator device, defaulting to ``"default.qubit"``.
        output_classes: Number of unnormalized output logits.
    """

    def __init__(
        self,
        *,
        input_features: int = 140,
        snn_hidden_features: Sequence[int] = (128, 64),
        snn_timesteps: int = 25,
        beta: float = 0.90,
        encoding: SpikeEncoding = "rate",
        delta_threshold: float = 0.10,
        surrogate_slope: float = 25.0,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend: str = "default.qubit",
        output_classes: int = 2,
    ) -> None:
        """Build the SNN, quantum bottleneck, variational circuit, and readout."""
        super().__init__()
        _validate_positive_int(n_qubits, "n_qubits")
        _validate_positive_int(n_layers, "n_layers")
        _validate_positive_int(output_classes, "output_classes")
        if not backend.strip():
            raise ValueError("backend must be a non-empty PennyLane device name")

        self.snn_features = SNNFeatureExtractor(
            input_features=input_features,
            hidden_features=snn_hidden_features,
            timesteps=snn_timesteps,
            beta=beta,
            encoding=encoding,
            delta_threshold=delta_threshold,
            surrogate_slope=surrogate_slope,
        )
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend = backend
        self.feature_reduction = nn.Linear(self.snn_features.output_features, n_qubits)
        self.quantum_layer = _build_quantum_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            backend=backend,
        )
        self.classifier = nn.Linear(n_qubits, output_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return two-class logits from SNN features and VQC expectations."""
        snn_features = self.snn_features(inputs)
        qubit_angles = self.feature_reduction(snn_features)
        quantum_features = self.quantum_layer(qubit_angles)
        if quantum_features.ndim == 1:
            quantum_features = quantum_features.unsqueeze(0)
        return self.classifier(quantum_features.to(dtype=qubit_angles.dtype))


def _build_quantum_layer(*, n_qubits: int, n_layers: int, backend: str) -> nn.Module:
    """Create a Torch-interface AngleEmbedding + strongly-entangled VQC layer."""
    wires = tuple(range(n_qubits))
    device = qml.device(backend, wires=n_qubits)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(inputs: Tensor, weights: Tensor) -> list[qml.measurements.MeasurementProcess]:
        qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=wires)
        return [qml.expval(qml.PauliZ(wire)) for wire in wires]

    weight_shapes = {
        "weights": qml.StronglyEntanglingLayers.shape(
            n_layers=n_layers,
            n_wires=n_qubits,
        )
    }
    return qml.qnn.TorchLayer(circuit, weight_shapes)


def _validate_positive_int(value: int, name: str) -> None:
    """Reject invalid layer sizes and VQC dimensions early."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
