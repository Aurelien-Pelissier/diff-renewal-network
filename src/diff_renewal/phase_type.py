"""Positive phase-type states for refractory renewal dwell times.

Q is a transient CTMC generator in row convention; occupancy is a column.
This gives exact linear population means for the specified independent-particle
phase process.
"""

import jax.nn as jnn
import jax.numpy as jnp


def erlang_generator(stages: int, mean):
    """Return (Q, alpha) for an Erlang dwell time of fixed integer shape."""
    if not isinstance(stages, int) or isinstance(stages, bool) or stages < 1:
        raise ValueError("stages must be a positive integer")
    rate = stages / jnp.asarray(mean)
    q = rate * (-jnp.eye(stages) + jnp.eye(stages, k=1))
    return q, jnp.eye(stages)[0]


def phase_type_dynamics(q, alpha, occupancy, inflow, clock=1.0):
    """Return (occupancy derivative, nonnegative absorption flux).

    Requires Metzler Q with nonpositive row sums, alpha >= 0 with sum 1,
    and nonnegative operational-time clock multiplier.
    """
    exit_rates = -jnp.sum(q, axis=1)
    return alpha * inflow + clock * (q.T @ occupancy), clock * (exit_rates @ occupancy)


def mixed_erlang_components(mean, logits):
    """Common-rate Erlang mixture represented as a single shared suffix chain.

    logits[n-1] controls the probability of n remaining exponential stages.
    Rate = E[N] / mean matches the target mean exactly while keeping stage
    dimension fixed at K.
    """
    logits = jnp.asarray(logits)
    if logits.ndim != 1 or logits.size < 1:
        raise ValueError("logits must be a nonempty 1D array")
    weights = jnn.softmax(logits)
    counts = jnp.arange(1, logits.size + 1, dtype=weights.dtype)
    return jnp.dot(counts, weights) / mean, weights[::-1]


def smooth_erlang_logits(center, *, stages: int = 64, width: float = 0.5):
    """Smooth Gaussian logits over integer stage counts 1..stages."""
    if not isinstance(stages, int) or isinstance(stages, bool) or stages < 1:
        raise ValueError("stages must be a positive integer")
    if width <= 0:
        raise ValueError("width must be positive")
    counts = jnp.arange(1, stages + 1)
    return -0.5 * ((counts - center) / width) ** 2


def mixed_erlang_moments(rate, alpha):
    """Return (mean, variance, equivalent Gamma shape E[T]^2 / Var[T])."""
    alpha = jnp.asarray(alpha)
    counts = jnp.arange(alpha.size, 0, -1, dtype=alpha.dtype)
    count_mean = jnp.dot(alpha, counts)
    count_variance = jnp.dot(alpha, (counts - count_mean) ** 2)
    mean = count_mean / rate
    variance = (count_mean + count_variance) / rate**2
    return mean, variance, mean**2 / variance


def coxian_generator(log_rates, continuation_logits):
    """Unconstrained trainable Coxian generator with fixed phase budget K."""
    log_rates = jnp.asarray(log_rates)
    continuation_logits = jnp.asarray(continuation_logits)
    if (
        log_rates.ndim != 1
        or log_rates.size < 1
        or continuation_logits.shape != (log_rates.size - 1,)
    ):
        raise ValueError("require K log rates and K-1 continuation logits")
    rates = jnp.exp(log_rates)
    q = -jnp.diag(rates) + jnp.diag(rates[:-1] * jnn.sigmoid(continuation_logits), 1)
    return q, jnp.eye(rates.size, dtype=rates.dtype)[0]
