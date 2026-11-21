"""Rate-preserving phase capping with coupling and latent-Fisher gradient bounds."""

import jax.nn as jnn
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


def cap_mixed_erlang(mean, logits, *, cap: int):
    """Replace latent stage count N by min(N, cap) while preserving exponential rate."""
    logits = jnp.asarray(logits)
    if logits.ndim != 1 or logits.size < 1:
        raise ValueError("logits must be a nonempty vector")
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= logits.size:
        raise ValueError("cap must be an integer between 1 and the original stage count")
    log_weights = jnn.log_softmax(logits)
    weights = jnp.exp(log_weights)
    counts = jnp.arange(1, logits.size + 1, dtype=weights.dtype)
    rate = jnp.dot(counts, weights) / mean
    capped_logits = jnp.concatenate(
        [log_weights[: cap - 1], logsumexp(log_weights[cap - 1 :])[None]]
    )
    capped_mean = jnp.dot(jnp.minimum(counts, cap), weights) / rate
    tail = (
        jnp.exp(logsumexp(log_weights[cap:]))
        if cap < logits.size
        else jnp.zeros((), dtype=weights.dtype)
    )
    return capped_mean, capped_logits, tail


def capped_network_tv_bound(tail_probabilities, clock_rates, edges, horizon: float):
    """Conservative single-particle path-TV bound under fresh independent resets."""
    tail, rates = np.asarray(tail_probabilities, float), np.asarray(clock_rates, float)
    if tail.ndim != 1 or rates.shape != tail.shape or len(edges) != len(tail):
        raise ValueError("One tail probability and rate are required per edge")
    if not np.all(np.isfinite(tail)) or np.any(tail < 0) or np.any(tail > 1):
        raise ValueError("Tail probabilities must be in [0,1]")
    if (
        not np.all(np.isfinite(rates))
        or np.any(rates <= 0)
        or not np.isfinite(horizon)
        or horizon < 0
    ):
        raise ValueError("Rates must be positive and the horizon finite and nonnegative")
    sources = [edge[0] for edge in edges]
    nodes = []
    for source in sorted(set(sources)):
        selected = np.array([s == source for s in sources])
        tails = tail[selected]
        failure = (
            1.0 if np.any(tails == 1) else float(-np.expm1(np.log1p(-tails).sum()))
        )
        nodes.append(
            dict(
                source=int(source),
                preparation_failure_probability=failure,
                exit_rate_upper_bound=float(rates[selected].sum()),
            )
        )
    delta = max((row["preparation_failure_probability"] for row in nodes), default=0.0)
    hazard = max((row["exit_rate_upper_bound"] for row in nodes), default=0.0)
    return dict(
        path_tv_bound=min(1.0, delta * (1.0 + hazard * horizon)),
        max_preparation_failure_probability=delta,
        max_exit_rate=hazard,
        expected_preparations_upper_bound=1.0 + hazard * horizon,
        nodes=nodes,
    )


def capped_clock_bank_gradient_bound(
    weights, log_rate_derivatives, log_weight_derivatives, *, cap: int
):
    """Loewner-order bound on expectation gradient bias for a finite clock bank."""
    w, dr, dw = [
        np.asarray(v, float)
        for v in (weights, log_rate_derivatives, log_weight_derivatives)
    ]
    if (
        w.ndim != 2
        or dr.ndim != 2
        or dw.shape != w.shape + (dr.shape[-1],)
        or dr.shape[0] != w.shape[0]
    ):
        raise ValueError("Expected weights (E,K), rate scores (E,P), weight scores (E,K,P)")
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= w.shape[1]:
        raise ValueError("cap must be an integer in [1,K]")
    if not all(np.all(np.isfinite(v)) for v in (w, dr, dw)) or np.any(w <= 0):
        raise ValueError("Weights must be positive and inputs finite")
    if not np.allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-9):
        raise ValueError("Weights must sum to one")
    if not np.allclose(np.einsum("en,enp->ep", w, dw), 0.0, rtol=0, atol=1e-8):
        raise ValueError("Weight scores must be centered")
    tails = w[:, cap:].sum(axis=1)
    delta = 1.0 if np.any(tails >= 1) else float(-np.expm1(np.log1p(-tails).sum()))
    expected_counts = w @ np.arange(1, w.shape[1] + 1)
    information = np.einsum("e,ei,ej->ij", expected_counts, dr, dr) + np.einsum(
        "en,eni,enj->ij", w, dw, dw
    )
    matrix = delta * information
    return dict(
        cap_failure_probability=delta,
        latent_fisher=information,
        gradient_outer_product_upper_bound=matrix,
        coordinate_absolute_bounds=np.sqrt(np.maximum(np.diag(matrix), 0.0)),
    )
