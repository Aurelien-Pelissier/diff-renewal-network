"""Incoming-edge conditioned & Hyper-Erlang semi-Markov renewal networks (EdgeMemoryHyperDRN).

Lifts node-reset renewal dynamics onto directed incoming-transition fibres (u -> v)
with two-scale Hyper-Erlang competing exit hazards (capturing CV > 1 bursty micro-arousals
alongside multi-stage refractory consolidation) while preserving:
1. Strict Metzler mass conservation (global probability simplex invariance for all t >= 0), and
2. Exact analytical survivor-age conditioning P(X(a+t) | T_0 > a) for zero-shot cohort transfer.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl

from .race import mixed_erlang_race


def build_edge_memory_hyper_generator(
    params,
    edges: tuple[tuple[int, int], ...],
    n_nodes: int,
    *,
    stages: int = 6,
    aged_horizon: float = 0.5,
):
    """Construct incoming-edge lifted two-scale Hyper-Erlang network generator.

    State space consists of:
    - V fresh-preparation root fibres (for initial cohort entry into node v at t=0), and
    - E directed transition fibres e_in = (u -> v) (entered whenever transition u -> v fires!).
    Wait: to keep compiled state count ultra-compact, each node v has:
      * 1 fast exponential micro-exit branch (rate lambda_fast[f], routing softmax), and
      * 1 compiled mixed-Erlang refractory race (K=stages per outgoing edge via Proposition 1),
    mixed at arrival into fibre f by learned fast-vs-consolidated probability p_fast[f] = sigmoid(g_f),
    with outgoing edge means/logits modulated by incoming-edge context embedding!

    Returns:
        gen: (M_tot, M_tot) strictly Metzler column-conserving generator.
        fresh_mat: (M_tot, V) non-negative fresh cohort initializations.
        aged_mat: (M_tot, V) exact analytical >= aged_horizon survivor initializations.
        read_mat: (V, M_tot) node-occupancy readout matrix.
    """
    n_edges = len(edges)
    outgoing = [tuple(e for e, (u, _) in enumerate(edges) if u == i) for i in range(n_nodes)]
    # Total fibres: 0..V-1 are initial root preparations at node v;
    # V..V+E-1 are post-transition fibres after traversing edge e_in = (u -> v).
    fibre_nodes = list(range(n_nodes)) + [dst for _, dst in edges]
    n_fibres = len(fibre_nodes)

    # Unpack parameter vector cleanly
    # Base per-edge means & stage logits: n_edges * (1 + stages)
    idx = 0
    base_log_means = params[idx : idx + n_edges]
    idx += n_edges
    base_logits = params[idx : idx + n_edges * stages].reshape(n_edges, stages)
    idx += n_edges * stages

    # Incoming-fibre modulation of clock speed & fast-burst weight:
    # per fibre f in 0..n_fibres-1: (speed_shift_f, fast_mix_logit_f, fast_log_rate_f) -> 3 * n_fibres
    fibre_mod = params[idx : idx + 3 * n_fibres].reshape(n_fibres, 3)
    idx += 3 * n_fibres

    # Incoming-fibre routing bias on outgoing edges: (n_fibres, rank=3) @ (3, n_edges)
    rank = 3
    f_emb = params[idx : idx + n_fibres * rank].reshape(n_fibres, rank)
    idx += n_fibres * rank
    e_emb = params[idx : idx + rank * n_edges].reshape(rank, n_edges)
    idx += rank * n_edges
    route_bias = f_emb @ e_emb  # (n_fibres, n_edges)

    # Group fibres by node out-degree so mixed_erlang_race & expm are batched via jax.vmap!
    by_deg: dict[int, list[int]] = {}
    for f_id, v_node in enumerate(fibre_nodes):
        by_deg.setdefault(len(outgoing[v_node]), []).append(f_id)

    fibre_blk: dict[int, jnp.ndarray] = {}
    fibre_ent: dict[int, jnp.ndarray] = {}
    fibre_surv: dict[int, jnp.ndarray] = {}
    fibre_fluxes: dict[int, tuple[tuple[int, jnp.ndarray], ...]] = {}

    for d_v, f_ids in by_deg.items():
        if d_v == 0:
            for fi in f_ids:
                fibre_blk[fi] = jnp.zeros((1, 1), dtype=params.dtype)
                fibre_ent[fi] = jnp.ones((1,), dtype=params.dtype)
                fibre_surv[fi] = jnp.ones((1,), dtype=params.dtype)
                fibre_fluxes[fi] = ()
            continue
        f_arr = jnp.array(f_ids, dtype=jnp.int32)
        grp_mat = jnp.array([outgoing[fibre_nodes[fi]] for fi in f_ids], dtype=jnp.int32)

        def _compile_fibre(fi, grp_idx):
            sp_sc = jnp.exp(0.8 * jnp.tanh(fibre_mod[fi, 0]))
            pf = jax.nn.sigmoid(fibre_mod[fi, 1] - 1.0)
            rf = jnp.exp(0.5 + 1.8 * jnp.tanh(fibre_mod[fi, 2]))
            b_g = 0.6 * jnp.tanh(route_bias[fi, grp_idx])
            m_f = jnp.exp(base_log_means[grp_idx] - b_g) / sp_sc
            l_f = base_logits[grp_idx]
            rs, cs, causes_s = mixed_erlang_race(m_f, l_f)
            dim_s = causes_s.shape[1]
            dim_tot = 1 + dim_s
            blk_s = -rs * jnp.eye(dim_s) + jnp.diag(rs * cs, k=1).T
            blk_tot = jl.block_diag(jnp.array([[-rf]]), blk_s)
            ent_tot = jnp.zeros(dim_tot, dtype=params.dtype).at[0].set(pf).at[1].set(1.0 - pf)
            surv_tot = jl.expm(aged_horizon * blk_tot) @ ent_tot
            surv_tot = surv_tot / jnp.maximum(jnp.sum(surv_tot), 1e-15)
            fast_p = jax.nn.softmax(route_bias[fi, grp_idx])
            flx = jnp.concatenate([(rf * fast_p)[:, None], rs * causes_s], axis=1)
            return blk_tot, ent_tot, surv_tot, flx

        blks_d, ents_d, survs_d, flxs_d = jax.vmap(_compile_fibre)(f_arr, grp_mat)
        for loc_i, fi in enumerate(f_ids):
            fibre_blk[fi] = blks_d[loc_i]
            fibre_ent[fi] = ents_d[loc_i]
            fibre_surv[fi] = survs_d[loc_i]
            grp = outgoing[fibre_nodes[fi]]
            fibre_fluxes[fi] = tuple((grp[j], flxs_d[loc_i, j]) for j in range(d_v))

    blocks = []
    offsets = []
    exit_cols_by_edge = []
    fresh_fibre_entries = []
    aged_fibre_entries = []
    cur = 0
    for f_id in range(n_fibres):
        blk_f = fibre_blk[f_id]
        dim_f = blk_f.shape[0]
        offsets.append((cur, cur + dim_f))
        blocks.append(blk_f)
        fresh_fibre_entries.append(fibre_ent[f_id])
        aged_fibre_entries.append(fibre_surv[f_id])
        for e_out, flux_row in fibre_fluxes[f_id]:
            exit_cols_by_edge.append((cur, cur + dim_f, e_out, flux_row))
        cur += dim_f

    total_states = cur
    gen = jl.block_diag(*blocks)
    # Route exit via edge e_out into post-transition fibre (n_nodes + e_out)!
    for c0, c1, e_out, flux_row in exit_cols_by_edge:
        dst_fibre = n_nodes + e_out
        d0, d1 = offsets[dst_fibre]
        gen = gen.at[d0:d1, c0:c1].add(fresh_fibre_entries[dst_fibre][:, None] * flux_row[None, :])

    # Initial preparations at t=0 enter root fibres 0..V-1 (or stationary incoming mixture)
    fresh_mat = jnp.zeros((total_states, n_nodes), dtype=params.dtype)
    aged_mat = jnp.zeros((total_states, n_nodes), dtype=params.dtype)
    read_mat = jnp.zeros((n_nodes, total_states), dtype=params.dtype)

    for f_id, v_node in enumerate(fibre_nodes):
        s0, s1 = offsets[f_id]
        read_mat = read_mat.at[v_node, s0:s1].set(1.0)
        if f_id < n_nodes:
            fresh_mat = fresh_mat.at[s0:s1, v_node].set(fresh_fibre_entries[f_id])
            aged_mat = aged_mat.at[s0:s1, v_node].set(aged_fibre_entries[f_id])

    return gen, fresh_mat, aged_mat, read_mat


def edge_memory_param_size(n_nodes: int, n_edges: int, *, stages: int = 6, rank: int = 3) -> int:
    """Return total parameter count of EdgeMemoryHyperDRN."""
    n_fibres = n_nodes + n_edges
    return n_edges * (1 + stages) + 3 * n_fibres + rank * (n_fibres + n_edges)
