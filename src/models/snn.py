"""snnTorch Leaky SNN for binary ECG waveform classification.

Each tabular ECG row is interpreted as a one-dimensional signal. It is resampled
to the configured simulation length, spike-encoded, and passed through a stack
of fully connected Leaky Integrate-and-Fire (LIF) layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import snntorch as snn
from snntorch import surrogate
import torch
from torch import Tensor, nn
from torch.nn import functional as functional


SpikeEncoding: TypeAlias = Literal["rate", "delta"]
Readout: TypeAlias = Literal["spike_count", "membrane"]


@dataclass(frozen=True, slots=True)
class SNNForwardResult:
    """Logits and spike activity recorded during one SNN forward pass.

    Attributes:
        logits: Batch of unnormalized class scores.
        output_spikes: Final-layer spikes with shape
            ``[timesteps, batch, output_classes]``.
        output_membranes: Final-layer membrane potentials with the same shape.
        mean_spikes_per_neuron_per_sample: Mean count across every hidden and
            output LIF neuron for each sample in the batch. Lower values denote
            sparser activity.
    """

    logits: Tensor
    output_spikes: Tensor
    output_membranes: Tensor
    mean_spikes_per_neuron_per_sample: Tensor


def rate_encode(signal: Tensor, timesteps: int) -> Tensor:
    """Encode a batch of 1D ECG signals as stochastic rate-coded spikes.

    Standardized ECG samples may be negative, so a sigmoid maps every resampled
    amplitude to a valid Bernoulli probability. The returned tensor has shape
    ``[timesteps, batch, 1]``.

    Args:
        signal: ECG batch with shape ``[batch, input_features]``.
        timesteps: Number of SNN simulation steps.
    """
    waveform = _resample_signal(signal, timesteps)
    probabilities = torch.sigmoid(waveform)
    return torch.bernoulli(probabilities).transpose(0, 1).unsqueeze(-1)


def delta_encode(signal: Tensor, timesteps: int, threshold: float) -> Tensor:
    """Encode 1D ECG signals as signed threshold-crossing spikes.

    At each resampled signal point, the encoder emits one event for a rise of
    at least ``threshold`` and a separate event for a fall of at least the same
    magnitude. The first signal value is compared with a zero baseline. This
    preserves direction while remaining compatible with non-negative spike
    inputs. The result has shape ``[timesteps, batch, 2]`` where channel zero
    is positive and channel one is negative.

    Args:
        signal: ECG batch with shape ``[batch, input_features]``.
        timesteps: Number of SNN simulation steps.
        threshold: Absolute difference needed to emit a delta spike.
    """
    if threshold <= 0.0:
        raise ValueError("delta threshold must be greater than zero")

    waveform = _resample_signal(signal, timesteps)
    changes = torch.diff(waveform, dim=1, prepend=torch.zeros_like(waveform[:, :1]))
    positive_spikes = (changes >= threshold).to(dtype=signal.dtype)
    negative_spikes = (changes <= -threshold).to(dtype=signal.dtype)
    signed_spikes = torch.stack((positive_spikes, negative_spikes), dim=-1)
    return signed_spikes.transpose(0, 1)


def mean_spikes_per_neuron_per_sample(spike_record: Tensor) -> Tensor:
    """Calculate mean spikes per neuron per sample from a spike-time record.

    Args:
        spike_record: Spike tensor whose leading dimensions are time and batch
            and whose final dimension is neuron count.

    Returns:
        Scalar tensor equal to total spikes divided by batch size and neuron
        count. A lower result indicates greater spike sparsity.
    """
    if spike_record.ndim != 3:
        raise ValueError(
            "spike_record must have shape [timesteps, batch, neurons], received "
            f"{tuple(spike_record.shape)}"
        )
    return spike_record.sum(dim=0).mean()


class SpikingECGClassifier(nn.Module):
    """Fully connected Leaky SNN with selectable ECG spike encoding.

    Args:
        input_features: Number of sequential ECG samples in each input row.
        hidden_features: Width of each hidden LIF layer.
        output_classes: Number of output class neurons.
        timesteps: Number of simulation steps; the 1D ECG is resampled to this
            length before encoding.
        beta: Leaky neuron membrane decay factor in ``(0, 1]``.
        encoding: ``"rate"`` for Bernoulli rate coding or ``"delta"`` for
            signed threshold-crossing coding.
        delta_threshold: Minimum waveform difference that produces a delta
            event. Used only for ``"delta"`` encoding.
        surrogate_slope: Fast-sigmoid surrogate-gradient slope passed to
            snnTorch's Leaky neurons.
        readout: Use final-layer spike counts or final membrane potentials as
            unnormalized class logits.
    """

    def __init__(
        self,
        input_features: int = 140,
        hidden_features: Sequence[int] = (128, 64),
        output_classes: int = 2,
        timesteps: int = 25,
        beta: float = 0.90,
        encoding: SpikeEncoding = "rate",
        delta_threshold: float = 0.10,
        surrogate_slope: float = 25.0,
        readout: Readout = "spike_count",
    ) -> None:
        """Build linear projections and their paired snnTorch Leaky neurons."""
        super().__init__()
        _validate_positive_int(input_features, "input_features")
        _validate_positive_int(output_classes, "output_classes")
        _validate_positive_int(timesteps, "timesteps")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in the interval (0.0, 1.0]")
        if encoding not in ("rate", "delta"):
            raise ValueError("encoding must be 'rate' or 'delta'")
        if delta_threshold <= 0.0:
            raise ValueError("delta_threshold must be greater than zero")
        if surrogate_slope <= 0.0:
            raise ValueError("surrogate_slope must be greater than zero")
        if readout not in ("spike_count", "membrane"):
            raise ValueError("readout must be 'spike_count' or 'membrane'")

        widths = tuple(hidden_features)
        for index, width in enumerate(widths):
            _validate_positive_int(width, f"hidden_features[{index}]")

        self.input_features = input_features
        self.timesteps = timesteps
        self.encoding: SpikeEncoding = encoding
        self.delta_threshold = delta_threshold
        self.readout: Readout = readout

        encoded_features = 1 if encoding == "rate" else 2
        spike_gradient = surrogate.fast_sigmoid(slope=surrogate_slope)
        self.hidden_linears = nn.ModuleList()
        self.hidden_neurons = nn.ModuleList()
        previous_width = encoded_features
        for width in widths:
            self.hidden_linears.append(nn.Linear(previous_width, width))
            self.hidden_neurons.append(snn.Leaky(beta=beta, spike_grad=spike_gradient))
            previous_width = width

        self.output_linear = nn.Linear(previous_width, output_classes)
        self.output_neuron = snn.Leaky(beta=beta, spike_grad=spike_gradient)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class logits from final spike counts or membrane potentials."""
        return self.forward_with_metrics(inputs).logits

    def forward_with_metrics(self, inputs: Tensor) -> SNNForwardResult:
        """Run the SNN and return logits together with output activity metrics.

        The sparsity metric covers every hidden and output LIF neuron, not only
        the two final class neurons.
        """
        self._validate_inputs(inputs)
        encoded = self._encode(inputs)
        batch_size = inputs.shape[0]

        hidden_membranes: list[Tensor] = [
            inputs.new_zeros((batch_size, layer.out_features))
            for layer in self.hidden_linears
        ]
        output_membrane = inputs.new_zeros((batch_size, self.output_linear.out_features))
        output_spike_record: list[Tensor] = []
        output_membrane_record: list[Tensor] = []
        total_spikes = inputs.new_zeros(())
        neuron_count = self.output_linear.out_features + sum(
            linear.out_features for linear in self.hidden_linears
        )

        for encoded_step in encoded:
            spikes = encoded_step
            for index, (linear, neuron) in enumerate(
                zip(self.hidden_linears, self.hidden_neurons, strict=True)
            ):
                current = linear(spikes)
                spikes, hidden_membranes[index] = neuron(
                    current, hidden_membranes[index]
                )
                total_spikes = total_spikes + spikes.sum()

            output_current = self.output_linear(spikes)
            output_spikes, output_membrane = self.output_neuron(
                output_current, output_membrane
            )
            total_spikes = total_spikes + output_spikes.sum()
            output_spike_record.append(output_spikes)
            output_membrane_record.append(output_membrane)

        spike_record = torch.stack(output_spike_record)
        membrane_record = torch.stack(output_membrane_record)
        logits = (
            spike_record.sum(dim=0)
            if self.readout == "spike_count"
            else membrane_record[-1]
        )
        sparsity = total_spikes / (batch_size * neuron_count)

        return SNNForwardResult(
            logits=logits,
            output_spikes=spike_record,
            output_membranes=membrane_record,
            mean_spikes_per_neuron_per_sample=sparsity,
        )

    def _encode(self, inputs: Tensor) -> Tensor:
        """Apply the configured rate or delta spike encoder."""
        if self.encoding == "rate":
            return rate_encode(inputs, self.timesteps)
        return delta_encode(inputs, self.timesteps, self.delta_threshold)

    def _validate_inputs(self, inputs: Tensor) -> None:
        """Ensure input batches follow the fixed ECG waveform contract."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input_features:
            raise ValueError(
                "Expected inputs with shape "
                f"[batch, {self.input_features}], received {tuple(inputs.shape)}"
            )
        if not torch.is_floating_point(inputs):
            raise TypeError("SNN inputs must be a floating-point tensor")


def _resample_signal(signal: Tensor, timesteps: int) -> Tensor:
    """Resample a ``[batch, features]`` waveform to the simulation duration."""
    if signal.ndim != 2:
        raise ValueError(
            "signal must have shape [batch, input_features], received "
            f"{tuple(signal.shape)}"
        )
    if timesteps <= 0:
        raise ValueError("timesteps must be a positive integer")
    if signal.shape[1] == timesteps:
        return signal
    return functional.interpolate(
        signal.unsqueeze(1),
        size=timesteps,
        mode="linear",
        align_corners=True,
    ).squeeze(1)


def _validate_positive_int(value: int, name: str) -> None:
    """Reject non-positive shape and duration values with a clear error."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
