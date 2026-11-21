"""Observed, time-invariant individual traits in renewal networks."""

from functools import partial

import jax
import jax.numpy as jnp

from .phase_type import smooth_erlang_logits
from .race import solve_compact_renewal_network


def trait_mean_shape(coefficients, traits):
    """Log-linear positive mean and shape functions of observed features."""
    coefficients, traits = jnp.asarray(coefficients), jnp.asarray(traits)
    if traits.ndim != 2 or coefficients.ndim != 3 or coefficients.shape[0] != 2:
        raise ValueError("Require coefficients (2,E,D+1) and traits (G,D)")
    if coefficients.shape[-1] != traits.shape[-1] + 1:
        raise ValueError("Trait and coefficient dimensions disagree")
    basis = jnp.concatenate((jnp.ones((traits.shape[0], 1)), traits), axis=-1)
    values = jnp.exp(jnp.einsum("ged,xd->gxe", coefficients, basis))
    return values[0], values[1]


def endpoint_coefficients(log_endpoints):
    """Map endpoint values at scalar traits -1, +1 to intercept and slope."""
    x = jnp.asarray(log_endpoints)
    if x.ndim != 3 or x.shape[0] != 2 or x.shape[-1] != 2:
        raise ValueError("Require endpoint array (2,E,2)")
    return jnp.stack(
        ((x[..., 0] + x[..., 1]) / 2.0, (x[..., 1] - x[..., 0]) / 2.0), axis=-1
    )


def _center_shape(center, stages: int, width: float):
    weights = jax.nn.softmax(
        smooth_erlang_logits(center, stages=stages, width=width)
    )
    counts = jnp.arange(1, stages + 1)
    mean = weights @ counts
    variance = weights @ (counts - mean) ** 2
    return mean**2 / (mean + variance)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _shape_center(shape, stages: int, width: float):
    def step(_, interval):
        lo, hi = interval
        mid = (lo + hi) / 2.0
        below = _center_shape(mid, stages, width) < shape
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0, 60, step, (jnp.ones_like(shape), jnp.full_like(shape, stages - 1.0))
    )
    return (lo + hi) / 2.0


@_shape_center.defjvp
def _shape_center_jvp(stages: int, width: float, primals, tangents):
    (shape,), (dshape,) = primals, tangents
    center = _shape_center(shape, stages, width)
    derivative = jax.grad(_center_shape, argnums=0)(center, stages, width)
    return center, dshape / derivative


def moment_matched_logits(shapes):
    """Differentiable moment matching for the validated 16-phase family on [3, 12]."""
    shapes = jnp.asarray(shapes)
    centers = jax.vmap(lambda k: _shape_center(k, 16, 0.5))(shapes.reshape(-1))
    logits = jax.vmap(lambda c: smooth_erlang_logits(c, stages=16, width=0.5))(centers)
    return logits.reshape(shapes.shape + (16,))


def trait_network_probabilities(
    coefficients, traits, edges, initials, times, *, adjoint=None
):
    """Conditional node probabilities (trait, preparation, time, species)."""
    means, shapes = trait_mean_shape(coefficients, traits)
    logits = moment_matched_logits(shapes)

    def conditional(mean, logit):
        return jax.vmap(
            lambda initial: solve_compact_renewal_network(
                mean, logit, edges, initial, times, adjoint=adjoint
            ).ys
        )(jnp.asarray(initials))

    return jax.vmap(conditional)(means, logits)
