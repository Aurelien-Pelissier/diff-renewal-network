"""Exact reduction of independent common-rate mixed-Erlang competing races."""

import diffrax
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

from .network import network_layout
from .phase_type import mixed_erlang_components


def _log_convolve(a, b):
    """Positive polynomial multiplication in log space without factorial overflow."""
    n = jnp.arange(a.size + b.size - 1)[:, None]
    j = jnp.arange(a.size)[None, :]
    index = n - j
    terms = a[None, :] + b[jnp.clip(index, 0, b.size - 1)]
    return logsumexp(jnp.where((index >= 0) & (index < b.size), terms, -jnp.inf), axis=1)


def mixed_erlang_race(means, logits):
    """Compile d competing mixed-Erlang clocks into an exact L = 1 + d*(K-1) chain.

    Preserves the complete joint distribution of exit time and winning edge mark
    together with all parameter derivatives.
    """
    means, logits = jnp.asarray(means), jnp.asarray(logits)
    if (
        means.ndim != 1
        or means.size < 1
        or logits.ndim != 2
        or logits.shape[0] != means.size
        or logits.shape[1] < 1
    ):
        raise ValueError("require means (d,), logits (d,K), d,K >= 1")
    rates, _ = jax.vmap(mixed_erlang_components)(means, logits)
    total_rate = jnp.sum(rates)
    log_ratio = jnp.log(rates) - jnp.log(total_rate)
    log_weights = jax.nn.log_softmax(logits, axis=1)
    log_tail = jax.vmap(
        lambda w: jax.lax.associative_scan(jnp.logaddexp, w, reverse=True)
    )(log_weights)
    counts = jnp.arange(logits.shape[1], dtype=logits.dtype)
    log_factorial = gammaln(counts + 1.0)
    log_survival = log_tail + log_ratio[:, None] * counts - log_factorial
    log_density = log_weights + log_ratio[:, None] * (counts + 1) - log_factorial

    causes = []
    for e in range(means.size):
        polynomial = log_density[e]
        for other in range(means.size):
            if other != e:
                polynomial = _log_convolve(polynomial, log_survival[other])
        causes.append(
            polynomial + gammaln(jnp.arange(polynomial.size, dtype=logits.dtype) + 1.0)
        )
    log_joint = jnp.stack(causes)
    log_joint = log_joint - logsumexp(log_joint)
    log_count = logsumexp(log_joint, axis=0)
    tail = jax.lax.associative_scan(jnp.logaddexp, log_count, reverse=True)
    cause_probabilities = jnp.exp(log_joint - tail[None, :])
    continuations = jnp.exp(tail[1:] - tail[:-1])
    normalization = jnp.sum(cause_probabilities, axis=0) + jnp.pad(
        continuations, (0, 1)
    )
    cause_probabilities = cause_probabilities / normalization[None, :]
    continuations = continuations / normalization[:-1]
    return total_rate, continuations, cause_probabilities


def compact_network_state_size(edges, n_nodes: int, stages: int) -> int:
    """Exact marked-race compiled network dimension: V + (K - 1) * E."""
    if not isinstance(stages, int) or isinstance(stages, bool) or stages < 1:
        raise ValueError("stages must be a positive integer")
    edges, _ = network_layout(edges, n_nodes)
    return n_nodes + (stages - 1) * len(edges)


def solve_compact_renewal_network(
    means,
    logits,
    edges,
    initial,
    ts,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    adjoint=None,
):
    """Exact fresh-clock renewal network solver using marked-race compilation."""
    means, logits = jnp.asarray(means), jnp.asarray(logits)
    initial, ts = jnp.asarray(initial), jnp.asarray(ts)
    if initial.ndim != 1 or initial.size < 1:
        raise ValueError("initial must be a nonempty 1D array")
    edges, outgoing = network_layout(edges, initial.size)
    if (
        not edges
        or means.shape != (len(edges),)
        or logits.ndim != 2
        or logits.shape[0] != len(edges)
        or logits.shape[1] < 1
    ):
        raise ValueError("require E edges, means (E,), and logits (E,K), E,K >= 1")
    if ts.ndim != 1 or ts.size < 1:
        raise ValueError("ts must be a nonempty 1D array")
    components = []
    for group in outgoing:
        if not group:
            components.append(
                (
                    jnp.array(0.0, dtype=logits.dtype),
                    jnp.zeros(0, dtype=logits.dtype),
                    jnp.zeros((0, 1), dtype=logits.dtype),
                    jnp.ones(1, dtype=logits.dtype),
                )
            )
        elif len(group) == 1:
            rate, entry = mixed_erlang_components(means[group[0]], logits[group[0]])
            k = logits.shape[1]
            components.append(
                (
                    rate,
                    jnp.ones(k - 1, dtype=entry.dtype),
                    jnp.zeros((1, k), dtype=entry.dtype).at[0, -1].set(1.0),
                    entry,
                )
            )
        else:
            idx = jnp.array(group)
            rate, continuation, causes = mixed_erlang_race(means[idx], logits[idx])
            entry = jnp.zeros(causes.shape[1], dtype=causes.dtype).at[0].set(1.0)
            components.append((rate, continuation, causes, entry))
    components = tuple(components)
    y0 = tuple(initial[i] * component[3] for i, component in enumerate(components))

    def field(t, state, args):
        incoming = [jnp.array(0.0, dtype=initial.dtype) for _ in outgoing]
        derivatives = []
        for occupancy, group, (rate, continuation, causes, _) in zip(
            state, outgoing, args
        ):
            flux = rate * occupancy
            dp = jnp.pad(continuation * flux[:-1], (1, 0)) - flux
            exit_fluxes = causes @ flux
            for j, e in enumerate(group):
                dst = edges[e][1]
                incoming[dst] = incoming[dst] + exit_fluxes[j]
            derivatives.append(dp)
        return tuple(
            dp + arrival * component[3]
            for dp, arrival, component in zip(derivatives, incoming, args)
        )

    return diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=ts[-1],
        dt0=None,
        y0=y0,
        args=components,
        saveat=diffrax.SaveAt(
            ts=ts, fn=lambda t, y, args: jnp.stack([jnp.sum(p) for p in y])
        ),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=100_000,
    )
