"""Differentiable population means for sequential independent renewal stages."""

import diffrax
import jax
import jax.numpy as jnp

from .phase_type import mixed_erlang_components


def solve_erlang_chain(means, logits, ts, *, rtol: float = 1e-8, atol: float = 1e-10):
    """Solve a fresh cohort through C sequential mixed-Erlang compartments.

    means: (C,) positive mean dwell times; logits: (C, K) mixture logits for
    1..K exponential stages; ts: sorted times >= 0 with positive final time.
    Returns Diffrax Solution with ys shape (len(ts), C + 1) containing the C
    transient population fractions and final absorbing fraction.
    """
    means, logits, ts = jnp.asarray(means), jnp.asarray(logits), jnp.asarray(ts)
    if (
        means.ndim != 1
        or means.size < 1
        or logits.ndim != 2
        or logits.shape[0] != means.size
        or logits.shape[1] < 1
    ):
        raise ValueError("require means shape (C,) and logits shape (C,K), with C,K >= 1")
    if ts.ndim != 1 or ts.size < 1:
        raise ValueError("ts must be a nonempty 1D array")
    rate, alpha = jax.vmap(mixed_erlang_components)(means, logits)
    initial = jnp.zeros_like(alpha).at[0].set(alpha[0])

    def field(t, state, args):
        rates, entry = args
        p, completed = state
        stage_flux = rates[:, None] * p
        exits = stage_flux[:, -1]
        inflows = jnp.concatenate((jnp.zeros(1, dtype=p.dtype), exits[:-1]))
        dp = (
            jnp.pad(stage_flux[:, :-1], ((0, 0), (1, 0)))
            - stage_flux
            + entry * inflows[:, None]
        )
        return dp, exits[-1]

    def observe(t, state, args):
        p, completed = state
        return jnp.concatenate((jnp.sum(p, axis=1), jnp.atleast_1d(completed)))

    return diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=ts[-1],
        dt0=None,
        y0=(initial, jnp.array(0.0, dtype=initial.dtype)),
        args=(rate, alpha),
        saveat=diffrax.SaveAt(ts=ts, fn=observe),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=100_000,
    )
