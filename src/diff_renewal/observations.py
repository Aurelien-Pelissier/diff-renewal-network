"""Differentiable count-weighted multinomial losses for snapshot observations."""

import jax.numpy as jnp


def multinomial_nll(probabilities, counts, *, probability_floor: float = 1e-12):
    """Constant-free multinomial negative log likelihood per observed cell."""
    p, n = jnp.asarray(probabilities), jnp.asarray(counts)
    if p.ndim < 1 or p.shape != n.shape or p.shape[-1] < 1:
        raise ValueError("Probabilities and counts must have matching category arrays")
    if not 0 < probability_floor < 1:
        raise ValueError("probability_floor must be between zero and one")
    p = jnp.maximum(p, probability_floor)
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    return -jnp.sum(n * jnp.log(p)) / jnp.sum(n)
