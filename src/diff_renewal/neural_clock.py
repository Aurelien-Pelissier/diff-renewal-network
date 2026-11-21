"""Conditional mixed-Erlang neural clock with exact age-conditioned survival."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


class ClockNetwork(NamedTuple):
    hidden_weight: jax.Array
    hidden_bias: jax.Array
    output_weight: jax.Array
    output_bias: jax.Array


def init_clock_network(key, features: int, *, stages: int = 16, width: int = 8):
    """Initialize near a broad stage-count distribution and a phase rate of four."""
    if min(features, stages, width) < 1:
        raise ValueError("features, stages and width must be positive")
    k1, k2 = jax.random.split(key)
    counts = jnp.arange(stages, 0, -1)
    bias = jnp.r_[
        jnp.zeros(1), -0.5 * ((counts - (stages + 1) / 2.0) / (stages / 4.0)) ** 2
    ]
    return ClockNetwork(
        jax.random.normal(k1, (features, width)) / jnp.sqrt(features),
        jnp.zeros(width),
        0.08 * jax.random.normal(k2, (width, stages + 1)),
        bias,
    )


def neural_clock(network: ClockNetwork, features):
    """Return positive phase rate and normalized entry probabilities."""
    hidden = jnp.tanh(jnp.asarray(features) @ network.hidden_weight + network.hidden_bias)
    raw = hidden @ network.output_weight + network.output_bias
    rate = 4.0 * jnp.exp(1.2 * jnp.tanh(raw[..., 0]))
    return rate, jax.nn.softmax(raw[..., 1:], axis=-1)


def survivor_phases(rate, entry, age):
    """Unnormalized phase masses conditional on no exit up to elapsed age."""
    rate, entry, age = map(jnp.asarray, (rate, entry, age))
    stages = entry.shape[-1]
    shifts = jnp.arange(stages)[:, None] - jnp.arange(stages)[None, :]
    order = jnp.maximum(shifts, 0)
    elapsed = (rate * age)[..., None, None]
    scale = jnp.maximum(elapsed, 1.0)
    powers = jnp.where(order == 0, 1.0, (elapsed / scale) ** jnp.maximum(order, 1))
    poisson = powers * jnp.exp(-elapsed + order * jnp.log(scale) - gammaln(order + 1.0))
    return jnp.einsum("...ij,...j->...i", jnp.where(shifts >= 0, poisson, 0.0), entry)
