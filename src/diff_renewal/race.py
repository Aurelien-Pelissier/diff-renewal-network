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
    k = logits.shape[1]
    n_nodes = int(initial.size)
    lengths = [1 + len(group) * (k - 1) for group in outgoing]
    offsets = []
    cur = 0
    for ln in lengths:
        offsets.append(cur)
        cur += ln
    total_dim = cur
    max_len = max(lengths)

    # Batch local clock / marked-race compilation by out-degree to avoid O(V) XLA unrolling
    by_deg: dict[int, list[int]] = {}
    for node_i, grp in enumerate(outgoing):
        by_deg.setdefault(len(grp), []).append(node_i)

    node_rate: dict[int, jnp.ndarray] = {}
    node_cont: dict[int, jnp.ndarray] = {}
    node_causes: dict[int, jnp.ndarray] = {}
    node_entry: dict[int, jnp.ndarray] = {}

    for deg, nodes_d in by_deg.items():
        if deg == 0:
            z_r = jnp.array(0.0, dtype=logits.dtype)
            z_c = jnp.zeros(0, dtype=logits.dtype)
            z_p = jnp.zeros((0, 1), dtype=logits.dtype)
            o_e = jnp.ones(1, dtype=logits.dtype)
            for ni in nodes_d:
                node_rate[ni], node_cont[ni], node_causes[ni], node_entry[ni] = z_r, z_c, z_p, o_e
        elif deg == 1:
            e_ids = jnp.array([outgoing[ni][0] for ni in nodes_d])
            rates_1, entries_1 = jax.vmap(mixed_erlang_components)(means[e_ids], logits[e_ids])
            c_1 = jnp.ones(k - 1, dtype=logits.dtype)
            p_1 = jnp.zeros((1, k), dtype=logits.dtype).at[0, -1].set(1.0)
            for idx_loc, ni in enumerate(nodes_d):
                node_rate[ni] = rates_1[idx_loc]
                node_cont[ni] = c_1
                node_causes[ni] = p_1
                node_entry[ni] = entries_1[idx_loc]
        else:
            grp_mat = jnp.array([outgoing[ni] for ni in nodes_d])
            rates_d, conts_d, causes_d = jax.vmap(
                lambda idx: mixed_erlang_race(means[idx], logits[idx])
            )(grp_mat)
            ln_d = 1 + deg * (k - 1)
            e_d = jnp.zeros(ln_d, dtype=logits.dtype).at[0].set(1.0)
            for idx_loc, ni in enumerate(nodes_d):
                node_rate[ni] = rates_d[idx_loc]
                node_cont[ni] = conts_d[idx_loc]
                node_causes[ni] = causes_d[idx_loc]
                node_entry[ni] = e_d

    rate_vec = jnp.concatenate(
        [jnp.repeat(node_rate[i], lengths[i]) for i in range(n_nodes)]
    )
    cont_parts = []
    for i in range(n_nodes):
        cont_parts.append(node_cont[i])
        if i < n_nodes - 1:
            cont_parts.append(jnp.zeros(1, dtype=logits.dtype))
    cont_vec = jnp.concatenate(cont_parts) if cont_parts else jnp.zeros(0, dtype=logits.dtype)
    entry_vec = jnp.concatenate([node_entry[i] for i in range(n_nodes)])

    stage_owner = jnp.array(
        [i for i in range(n_nodes) for _ in range(lengths[i])], dtype=jnp.int32
    )
    edge_stage_rows = []
    cause_rows = []
    for u_i, grp in enumerate(outgoing):
        ln_u = lengths[u_i]
        off_u = offsets[u_i]
        idx_row = jnp.pad(
            jnp.arange(off_u, off_u + ln_u, dtype=jnp.int32), (0, max_len - ln_u)
        )
        causes_u = jnp.pad(node_causes[u_i], ((0, 0), (0, max_len - ln_u)))
        for j_loc, e_id in enumerate(grp):
            edge_stage_rows.append((e_id, idx_row, causes_u[j_loc]))
    edge_stage_rows.sort(key=lambda item: item[0])
    edge_stage_idx = jnp.stack([r[1] for r in edge_stage_rows])
    cause_mat = jnp.stack([r[2] for r in edge_stage_rows])
    dst_nodes = jnp.array([dst for _, dst in edges], dtype=jnp.int32)

    y0 = initial[stage_owner] * entry_vec
    assert y0.size == total_dim

    def field(t, y, args):
        r_v, c_v, e_v, p_m = args
        flux = r_v * y
        subdiag = jnp.pad(c_v * flux[:-1], (1, 0))
        edge_fluxes = jnp.sum(p_m * flux[edge_stage_idx], axis=1)
        arrivals = jnp.zeros(n_nodes, dtype=y.dtype).at[dst_nodes].add(edge_fluxes)
        return subdiag - flux + arrivals[stage_owner] * e_v

    return diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=ts[-1],
        dt0=None,
        y0=y0,
        args=(rate_vec, cont_vec, entry_vec, cause_mat),
        saveat=diffrax.SaveAt(
            ts=ts,
            fn=lambda t, y, args: jnp.zeros(n_nodes, dtype=y.dtype).at[stage_owner].add(y),
        ),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=100_000,
    )
