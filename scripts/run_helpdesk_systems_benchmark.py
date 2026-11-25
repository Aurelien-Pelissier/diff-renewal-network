"""Real Non-Biological Systems ML Benchmark: IT Helpdesk Service-Queueing Network.

Evaluates Differentiable Renewal Networks (DRN & EdgeMemoryHyperDRN) on the
official 4TU.ResearchData IT Helpdesk Ticketing Management Event Log
(Polato, 2017; DOI: 10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb;
4,580 real software-support tickets, 21,348 timestamped events, 5 lifecycle
states: Assign -> Active <-> Wait -> Resolved -> Closed):

1. Five-fold ticket-grouped CV for fresh-ticket evolution and ZERO-SHOT
   aged-ticket backlog transfer (tickets already waiting >= 7 days in Active,
   Wait, or Resolved) comparing EdgeMemoryHyperDRN, Renewal DRN, Memoryless
   CTMC, Time-Inhomogeneous Neural-CME(t), and Latent ODE.
2. Coarsening panel-snapshot interval sweep (Delta_t in {0.5, 2.0, 7.0, 14.0}
   days) comparing multi-hop DRN against a monotone cumulative-hazard Neural
   Competing-Risks / Neural Semi-Markov TPP baseline (Omi et al., 2019;
   Lee et al., 2018 DeepHit style), isolating how single-jump survival models
   break down when unobserved multi-hop transitions occur inside coarse panels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    build_edge_memory_hyper_generator,
    compact_network_state_size,
    edge_memory_param_size,
    init_edge_memory_params,
    mixed_erlang_race,
    network_state_size,
)

HELPDESK_URL = (
    "https://data.4tu.nl/file/94ee26c8-78f6-4387-b32b-f028f2103a2c/"
    "697ff9af-fca0-4363-bd84-e81735d27e9f"
)
EXPECTED_MD5 = "a91d74377c55be5cbdfda05dae69db30"
CORE_STATES = [
    "Assign seriousness",
    "Take in charge ticket",
    "Wait",
    "Resolve ticket",
    "Closed",
]
STATE_MAP = {s: i for i, s in enumerate(CORE_STATES)}
# 9 directed edges with >1% exit share (V=5, E=9)
EDGES = (
    (0, 1),  # Assign -> Active
    (0, 2),  # Assign -> Wait
    (0, 3),  # Assign -> Resolved
    (1, 2),  # Active -> Wait
    (1, 3),  # Active -> Resolved
    (2, 1),  # Wait -> Active
    (2, 3),  # Wait -> Resolved
    (3, 1),  # Resolved -> Active (reopen)
    (3, 4),  # Resolved -> Closed (auto/manual close)
)
V_NODES = 5
E_EDGES = len(EDGES)

HORIZONS_DAYS = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 35.0, 45.0], dtype=float)
AGED_THRESHOLD_DAYS = 7.0
# Active non-absorbing starting nodes for fresh & aged cohorts
FRESH_STARTS = (0, 1, 2, 3)
AGED_STARTS = (1, 2, 3)


def ensure_helpdesk_csv(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "finale.csv"
    if not csv_path.exists():
        urllib.request.urlretrieve(HELPDESK_URL, csv_path)
    md5 = hashlib.md5(csv_path.read_bytes()).hexdigest()
    if md5 != EXPECTED_MD5:
        raise ValueError(f"Helpdesk MD5 mismatch: got {md5}, expected {EXPECTED_MD5}")
    return csv_path


def load_ticket_trajectories(csv_path: Path):
    """Parse 4,580 tickets into right-continuous piecewise-constant state trajectories."""
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["Complete Timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    df = df.sort_values(["Case ID", "ts"])
    sub = df[df["Activity"].isin(CORE_STATES)].copy()
    sub = sub[
        ~((sub["Case ID"] == sub["Case ID"].shift(1)) & (sub["Activity"] == sub["Activity"].shift(1)))
    ].copy()
    sub["state"] = sub["Activity"].map(STATE_MAP).astype(int)

    trajs = []
    sojourns = {s: [] for s in range(4)}
    for case_id, grp in sub.groupby("Case ID", sort=False):
        sts = grp["state"].to_numpy(dtype=np.int32)
        t0 = grp["ts"].iloc[0]
        times = (grp["ts"] - t0).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        # Ensure strictly non-decreasing times
        if len(sts) < 2 or np.any(np.diff(times) < 0):
            continue
        trajs.append((str(case_id), times, sts))
        for i in range(len(sts) - 1):
            if sts[i] < 4:
                sojourns[int(sts[i])].append(float(times[i + 1] - times[i]))
    return trajs, sojourns


def extract_snapshot_tables(trajs, horizons=HORIZONS_DAYS, age_thr=AGED_THRESHOLD_DAYS):
    """Extract fresh-entry and aged-survivor (>= age_thr days) transition counts."""
    n_h = len(horizons)
    fresh_counts = np.zeros((4, n_h, V_NODES), dtype=float)
    aged_counts = np.zeros((3, n_h, V_NODES), dtype=float)  # starts 1, 2, 3

    for _, times, sts in trajs:
        # Each visit segment [times[i], times[i+1]) in state sts[i]
        for i in range(len(sts) - 1):
            s0 = int(sts[i])
            if s0 >= 4:
                continue
            t_enter = float(times[i])
            dur = float(times[i + 1] - t_enter)
            # Fresh entry at t_enter -> evaluate right-continuous state at t_enter + h
            q_fresh = t_enter + horizons
            idx_f = np.searchsorted(times, q_fresh, side="right") - 1
            idx_f = np.clip(idx_f, 0, len(sts) - 1)
            for hi, st_h in enumerate(sts[idx_f]):
                fresh_counts[s0, hi, int(st_h)] += 1.0

            # Aged survivor: still in s0 at t_enter + age_thr (so dur >= age_thr)
            if s0 in AGED_STARTS and dur >= age_thr:
                q_aged = t_enter + age_thr + horizons
                idx_a = np.searchsorted(times, q_aged, side="right") - 1
                idx_a = np.clip(idx_a, 0, len(sts) - 1)
                a_row = s0 - 1
                for hi, st_h in enumerate(sts[idx_a]):
                    aged_counts[a_row, hi, int(st_h)] += 1.0

    return fresh_counts, aged_counts


def extract_panel_pairs(trajs, delta_t: float):
    """Sample coarsened panel pairs (s(t), s(t + delta_t)) on a uniform grid of step delta_t."""
    counts = np.zeros((V_NODES, V_NODES), dtype=float)
    for _, times, sts in trajs:
        t_max = float(times[-1])
        # Sample on uniform grid k * delta_t up to last transition + delta_t
        grid = np.arange(0.0, t_max + delta_t + 1e-9, delta_t)
        if len(grid) < 2:
            continue
        idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(sts) - 1)
        obs = sts[idx]
        for a, b in zip(obs[:-1], obs[1:]):
            if a < 4:
                counts[int(a), int(b)] += 1.0
    return counts


def empirical_kl(pred_probs: np.ndarray, counts: np.ndarray) -> float:
    """Weighted average KL(emp || pred) across all (start, horizon) strata with >0 mass."""
    tot = counts.sum(axis=-1, keepdims=True)
    mask = tot[..., 0] > 0
    emp = np.where(tot > 0, counts / np.maximum(tot, 1.0), 0.0)
    pred = np.clip(pred_probs, 1e-12, 1.0)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    kl_mat = np.sum(np.where(emp > 0, emp * np.log(np.maximum(emp, 1e-30) / pred), 0.0), axis=-1)
    weights = tot[..., 0] * mask
    return float(np.sum(kl_mat * weights) / np.maximum(np.sum(weights), 1.0))


# ---------------------------------------------------------------------------
# Model 1: Memoryless CTMC (V=5, E=9 parameters)
# ---------------------------------------------------------------------------
def build_ctmc_generator(theta: jax.Array) -> jax.Array:
    rates = jax.nn.softplus(theta) + 1e-4
    gen = jnp.zeros((V_NODES, V_NODES), dtype=jnp.float64)
    for e_idx, (u, v) in enumerate(EDGES):
        r = rates[e_idx]
        gen = gen.at[v, u].add(r)
        gen = gen.at[u, u].add(-r)
    return gen


def predict_ctmc(theta: jax.Array, horizons: jax.Array):
    gen = build_ctmc_generator(theta)

    def one_h(h):
        return jl.expm(h * gen)

    mats = jax.vmap(one_h)(horizons)  # (H, V, V) where col u -> row v
    # mats has shape (H, V_dst, V_src) -> transpose to (V_src, H, V_dst)
    all_p = jnp.transpose(mats, (2, 0, 1))
    return all_p[:4], all_p[1:4]


# ---------------------------------------------------------------------------
# Model 2: Strict Renewal DRN (Proposition 1 Marked-Race Compiler, K=6)
# ---------------------------------------------------------------------------
K_RENEWAL = 6


def _build_renewal_drn_matrices(theta: jax.Array, stages: int = K_RENEWAL, age_thr: float = AGED_THRESHOLD_DAYS):
    """Compile V=5, E=9 branching cyclic graph via Proposition 1."""
    # Parameters: E log-means + E * stages stage logits
    means = jnp.exp(jnp.clip(theta[:E_EDGES], -5.0, 5.0))
    logits = theta[E_EDGES:].reshape(E_EDGES, stages)

    out_edges = [[] for _ in range(V_NODES)]
    for e_idx, (u, v) in enumerate(EDGES):
        out_edges[u].append((e_idx, v))

    offsets = []
    cur = 0
    for u in range(V_NODES):
        deg = len(out_edges[u])
        sz = 1 if deg == 0 else 1 + deg * (stages - 1)
        offsets.append((cur, sz))
        cur += sz
    n_tot = cur

    gen = jnp.zeros((n_tot, n_tot), dtype=jnp.float64)
    fresh = jnp.zeros((n_tot, V_NODES), dtype=jnp.float64)
    aged = jnp.zeros((n_tot, 3), dtype=jnp.float64)
    read = jnp.zeros((V_NODES, n_tot), dtype=jnp.float64)

    for u in range(V_NODES):
        off, sz = offsets[u]
        read = read.at[u, off : off + sz].set(1.0)
        fresh = fresh.at[off, u].set(1.0)
        if not out_edges[u]:
            continue
        e_ids = [x[0] for x in out_edges[u]]
        dsts = [x[1] for x in out_edges[u]]
        m_u = means[jnp.asarray(e_ids)]
        l_u = logits[jnp.asarray(e_ids)]
        lam_u, cont_u, causes_u = mixed_erlang_race(m_u, l_u)
        blk = -lam_u * jnp.eye(sz) + jnp.diag(lam_u * cont_u, k=-1)
        gen = gen.at[off : off + sz, off : off + sz].set(blk)
        for k_i, dst in enumerate(dsts):
            dst_off = offsets[dst][0]
            gen = gen.at[dst_off, off : off + sz].add(lam_u * causes_u[k_i])
        if u in AGED_STARTS:
            # Analytical survivor conditioning at age_thr via transient block expm!
            st_a = jl.expm(age_thr * blk)[:, 0]
            st_a = jnp.maximum(st_a, 1e-30)
            st_a = st_a / jnp.sum(st_a)
            aged = aged.at[off : off + sz, u - 1].set(st_a)
    return gen, fresh, aged, read


def predict_renewal_drn(theta: jax.Array, horizons: jax.Array, stages: int = K_RENEWAL):
    gen, fresh, aged, read = _build_renewal_drn_matrices(theta, stages=stages)
    # Single unit-step matrix exponential + fast binary power / direct expm per horizon
    def eval_one(h):
        e_h = jl.expm(h * gen)
        pf = read @ (e_h @ fresh[:, :4])  # (V, 4)
        pa = read @ (e_h @ aged)  # (V, 3)
        return pf.T, pa.T

    pf_all, pa_all = jax.vmap(eval_one)(horizons)  # (H, 4, V), (H, 3, V)
    return jnp.swapaxes(pf_all, 0, 1), jnp.swapaxes(pa_all, 0, 1)


# ---------------------------------------------------------------------------
# Model 3: EdgeMemoryHyperDRN (Incoming-edge Hyper-Erlang fibres, K=6)
# ---------------------------------------------------------------------------
K_EDGE = 6


def init_edge_memory_drn(seed: int = 0, stages: int = K_EDGE) -> np.ndarray:
    """Symmetry-breaking initializer avoiding the bilinear f_emb @ e_emb = 0 saddle."""
    return init_edge_memory_params(
        V_NODES, E_EDGES, stages=stages, base_mean=8.0, emb_scale=0.08, seed=seed
    )


def predict_edge_memory_drn(theta: jax.Array, horizons: jax.Array, stages: int = K_EDGE):
    gen, fresh, aged, read = build_edge_memory_hyper_generator(
        theta, EDGES, V_NODES, stages=stages, aged_horizon=AGED_THRESHOLD_DAYS
    )

    def eval_one(h):
        e_h = jl.expm(h * gen)
        pf = read @ (e_h @ fresh[:, :4])  # (V, 4)
        pa = read @ (e_h @ aged[:, 1:4])  # (V, 3)
        return pf.T, pa.T

    pf_all, pa_all = jax.vmap(eval_one)(horizons)
    return jnp.swapaxes(pf_all, 0, 1), jnp.swapaxes(pa_all, 0, 1)


# ---------------------------------------------------------------------------
# Model 4: Time-Inhomogeneous Neural-CME(t) (clock-time rates A_phi(t))
# ---------------------------------------------------------------------------
def init_helpdesk_neural_cme(seed: int = 0):
    rng = np.random.default_rng(seed)
    w_dim = 20
    # inputs: [t_norm, sin(2pi t_norm), cos(2pi t_norm), is_aged] -> 4 features
    # 4*20 + 20 + 20*9 + 9 = 289 params
    return np.concatenate(
        [
            rng.normal(0.0, 0.25, size=4 * w_dim),
            np.zeros(w_dim),
            rng.normal(0.0, 0.15, size=w_dim * E_EDGES),
            np.full(E_EDGES, -2.0),
        ]
    )


_HDK_SRC = jnp.asarray([u for u, _ in EDGES], dtype=jnp.int32)
_HDK_DST = jnp.asarray([v for _, v in EDGES], dtype=jnp.int32)


def _exp_euler_helpdesk_step(p: jax.Array, rates: jax.Array, dt: float) -> jax.Array:
    """Unconditionally positive, simplex-conserving exponential-Euler flux update."""
    # rates: (B, E) or (E,)
    if rates.ndim == 1:
        rates_b = jnp.broadcast_to(rates[None, :], (p.shape[0], E_EDGES))
    else:
        rates_b = rates
    tot_out = jax.vmap(lambda r: jax.ops.segment_sum(r, _HDK_SRC, num_segments=V_NODES))(rates_b)
    frac_leave = -jnp.expm1(-jnp.minimum(tot_out * dt, 50.0))
    scale = frac_leave / jnp.maximum(tot_out, 1e-30)
    edge_mass = p[:, _HDK_SRC] * scale[:, _HDK_SRC] * rates_b
    p_next = p * (1.0 - frac_leave)
    p_next = p_next.at[:, _HDK_DST].add(edge_mass)
    p_next = jnp.maximum(p_next, 1e-12)
    return p_next / jnp.sum(p_next, axis=-1, keepdims=True)


def predict_helpdesk_neural_cme(theta: jax.Array, horizons: jax.Array, *, clean_age0: bool = False):
    w_dim = 20
    p1 = 4 * w_dim
    w1 = theta[:p1].reshape(4, w_dim)
    b1 = theta[p1 : p1 + w_dim]
    p2 = p1 + w_dim
    w2 = theta[p2 : p2 + w_dim * E_EDGES].reshape(w_dim, E_EDGES)
    b2 = theta[p2 + w_dim * E_EDGES :]

    # Integrate on uniform 0.25-day steps up to 45.0 days (180 steps) with unconditionally positive exp-Euler
    n_steps = 180
    dt = 0.25
    h_idx = jnp.asarray(np.round(np.asarray(horizons) / dt).astype(int) - 1)

    def run_cohort(is_aged_flag, init_p):
        def step(carry, _k):
            p, t_cur = carry
            t_mid = (t_cur + 0.5 * dt) / 45.0
            feat = jnp.array(
                [t_mid, jnp.sin(2.0 * jnp.pi * t_mid), jnp.cos(2.0 * jnp.pi * t_mid), is_aged_flag]
            )
            rates = 1e-4 + 4.0 * jax.nn.sigmoid(jax.nn.gelu(feat @ w1 + b1) @ w2 + b2)
            p_next = _exp_euler_helpdesk_step(p, rates, dt)
            return (p_next, t_cur + dt), p_next

        _, traj = jax.lax.scan(step, (init_p, 0.0), jnp.arange(n_steps))
        return jnp.swapaxes(traj[h_idx], 0, 1)  # (N_starts, H, V)

    pf = run_cohort(0.0, jnp.eye(V_NODES)[:4])
    pa = pf[1:4] if clean_age0 else run_cohort(1.0, jnp.eye(V_NODES)[1:4])
    return pf, pa


# ---------------------------------------------------------------------------
# Model 5: Population Latent ODE (simplex-conserving graph flux)
# ---------------------------------------------------------------------------
def init_helpdesk_latent_ode(seed: int = 0):
    rng = np.random.default_rng(seed)
    z_dim, w_dim = 6, 16
    # enc: (5+1)*6 + 6 = 42; w1: (5+6)*16 + 16 = 192; w2: 16*(9+6) + 15 = 255 -> 489 params
    return np.concatenate(
        [
            rng.normal(0.0, 0.25, size=6 * z_dim),
            np.zeros(z_dim),
            rng.normal(0.0, 0.20, size=(V_NODES + z_dim) * w_dim),
            np.zeros(w_dim),
            rng.normal(0.0, 0.12, size=w_dim * (E_EDGES + z_dim)),
            np.concatenate([np.full(E_EDGES, -2.0), np.zeros(z_dim)]),
        ]
    )


def predict_helpdesk_latent_ode(theta: jax.Array, horizons: jax.Array, *, clean_age0: bool = False):
    z_dim, w_dim = 6, 16
    i0 = 6 * z_dim
    enc = theta[:i0].reshape(6, z_dim)
    eb = theta[i0 : i0 + z_dim]
    i1 = i0 + z_dim
    w1 = theta[i1 : i1 + (V_NODES + z_dim) * w_dim].reshape(V_NODES + z_dim, w_dim)
    i2 = i1 + (V_NODES + z_dim) * w_dim
    b1 = theta[i2 : i2 + w_dim]
    i3 = i2 + w_dim
    w2 = theta[i3 : i3 + w_dim * (E_EDGES + z_dim)].reshape(w_dim, E_EDGES + z_dim)
    b2 = theta[i3 + w_dim * (E_EDGES + z_dim) :]

    n_steps = 180
    dt = 0.25
    h_idx = jnp.asarray(np.round(np.asarray(horizons) / dt).astype(int) - 1)

    def run_cohort(is_aged_flag, init_p):
        age_col = jnp.full((init_p.shape[0], 1), is_aged_flag)
        z0 = jnp.tanh(jnp.concatenate([init_p, age_col], axis=-1) @ enc + eb)
        y0 = jnp.concatenate([init_p, z0], axis=-1)

        def step(y, _):
            p, z = y[:, :V_NODES], y[:, V_NODES:]
            raw = jnp.tanh(y @ w1 + b1) @ w2 + b2
            rates = 1e-4 + 4.0 * jax.nn.sigmoid(raw[:, :E_EDGES])
            p_c = _exp_euler_helpdesk_step(p, rates, dt)
            # Midpoint latent update with damping
            dz1 = jnp.tanh(raw[:, E_EDGES:]) - 0.1 * z
            y_mid = jnp.concatenate([0.5 * (p + p_c), z + 0.5 * dt * dz1], axis=-1)
            raw_mid = jnp.tanh(y_mid @ w1 + b1) @ w2 + b2
            dz2 = jnp.tanh(raw_mid[:, E_EDGES:]) - 0.1 * (z + 0.5 * dt * dz1)
            z_n = z + dt * dz2
            return jnp.concatenate([p_c, z_n], axis=-1), p_c

        _, traj = jax.lax.scan(step, y0, None, length=n_steps)
        return jnp.swapaxes(traj[h_idx], 0, 1)

    pf = run_cohort(0.0, jnp.eye(V_NODES)[:4])
    pa = pf[1:4] if clean_age0 else run_cohort(1.0, jnp.eye(V_NODES)[1:4])
    return pf, pa


# ---------------------------------------------------------------------------
# Model 6: Monotone Cumulative-Hazard Neural Competing-Risks / Neural TPP
# (Omi et al., NeurIPS 2019; Lee et al., AAAI 2018 DeepHit single-jump model)
# ---------------------------------------------------------------------------
# All 16 non-self competing destinations (u -> v, u in 0..3, v != u) so coarsened multi-hop
# observations never hit hardcoded 0.0 / 1e-12 dead-gradient clips!
NCR_PAIRS = tuple((u, v) for u in range(4) for v in range(V_NODES) if u != v)


def init_neural_competing_risks(seed: int = 0):
    rng = np.random.default_rng(seed)
    h_dim = 8
    n_pairs = len(NCR_PAIRS)  # 16 directed non-self pairs -> 3 * 16 * 8 = 384 params
    return np.concatenate(
        [
            rng.normal(-1.0, 0.3, size=n_pairs * h_dim),
            rng.normal(-0.5, 0.3, size=n_pairs * h_dim),
            rng.normal(-1.5, 0.3, size=n_pairs * h_dim),
        ]
    )


def _cum_hazard_edge(theta_e, tau):
    """Strictly increasing cumulative hazard Lambda_e(tau) with Lambda_e(0) = 0 (Omi et al., 2019)."""
    h_dim = theta_e.size // 3
    w1 = jax.nn.softplus(theta_e[:h_dim])
    b1 = theta_e[h_dim : 2 * h_dim]
    w2 = jax.nn.softplus(theta_e[2 * h_dim :])
    return jnp.sum(w2 * (jax.nn.softplus(w1 * tau + b1) - jax.nn.softplus(b1)))


def predict_neural_competing_risks_single_step(theta: jax.Array, delta_t: float) -> jax.Array:
    """Exact single-jump competing-risks transition matrix P_1step(v | u, delta_t) via 32-pt Gauss-Legendre."""
    theta_edges = theta.reshape(len(NCR_PAIRS), -1)
    gl_x, gl_w = np.polynomial.legendre.leggauss(32)
    ts_q = jnp.asarray(0.5 * delta_t * (gl_x + 1.0), dtype=jnp.float64)
    ws_q = jnp.asarray(0.5 * delta_t * gl_w, dtype=jnp.float64)

    def edge_lam_and_h(th_e, t_val):
        lam = _cum_hazard_edge(th_e, t_val)
        haz = jax.grad(lambda s: _cum_hazard_edge(th_e, s))(t_val)
        return lam, haz

    lams_q, hazs_q = jax.vmap(
        lambda th_e: jax.vmap(lambda t_val: edge_lam_and_h(th_e, t_val))(ts_q)
    )(theta_edges)
    lams_end = jax.vmap(lambda th_e: _cum_hazard_edge(th_e, jnp.asarray(delta_t, dtype=jnp.float64)))(
        theta_edges
    )

    out_edges = [[] for _ in range(V_NODES)]
    for e_idx, (u, v) in enumerate(NCR_PAIRS):
        out_edges[u].append((e_idx, v))

    p_mat = jnp.zeros((V_NODES, V_NODES), dtype=jnp.float64)
    p_mat = p_mat.at[4, 4].set(1.0)
    for u in range(4):
        e_ids = [x[0] for x in out_edges[u]]
        dsts = [x[1] for x in out_edges[u]]
        tot_lam_q = jnp.sum(lams_q[jnp.asarray(e_ids)], axis=0)
        surv_q = jnp.exp(-jnp.minimum(tot_lam_q, 60.0))
        surv_end = jnp.exp(-jnp.minimum(jnp.sum(lams_end[jnp.asarray(e_ids)]), 60.0))
        p_mat = p_mat.at[u, u].set(surv_end)
        for e_i, dst in zip(e_ids, dsts):
            cif = jnp.sum(ws_q * hazs_q[e_i] * surv_q)
            p_mat = p_mat.at[u, dst].add(cif)
        row_sum = jnp.sum(p_mat[u])
        p_mat = p_mat.at[u].set(p_mat[u] / jnp.maximum(row_sum, 1e-12))
    return p_mat


def fit_model(init_theta, predict_fn, fresh_tr, aged_tr, *, joint: bool, reg: float = 1e-4, maxiter: int = 160):
    h_jax = jnp.asarray(HORIZONS_DAYS, dtype=jnp.float64)
    f_tr = jnp.asarray(fresh_tr, dtype=jnp.float64)
    a_tr = jnp.asarray(aged_tr, dtype=jnp.float64)
    n_f = float(np.maximum(fresh_tr.sum(), 1.0))
    n_a = float(np.maximum(aged_tr.sum(), 1.0))

    @jax.jit
    def val_and_grad(w):
        def loss_fn(ww):
            pf, pa = predict_fn(ww, h_jax)
            lf = -jnp.sum(f_tr * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_f
            if joint:
                la = -jnp.sum(a_tr * jnp.log(jnp.clip(pa, 1e-12, 1.0))) / n_a
                tot = 0.5 * (lf + la)
            else:
                tot = lf
            return tot + reg * jnp.mean(ww**2)

        return jax.value_and_grad(loss_fn)(w)

    def obj(x):
        v, g = val_and_grad(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(obj, np.asarray(init_theta, dtype=np.float64), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x


def run_coarsening_panel_experiment(trajs_train, trajs_test):
    """Compare Multi-Hop DRN / EdgeMemoryHyperDRN vs Single-Jump Neural Competing-Risks
    across coarsening panel observation grids Delta_t in {0.5, 2.0, 7.0, 14.0} days.
    """
    deltas = [0.5, 2.0, 7.0, 14.0]
    eval_horizons = np.array([1.0, 3.0, 7.0, 15.0, 30.0], dtype=float)
    fresh_te, aged_te = extract_snapshot_tables(trajs_test, horizons=eval_horizons)
    h_eval_jax = jnp.asarray(eval_horizons, dtype=jnp.float64)

    rows = []
    for dt in deltas:
        # Uniform coarsened sampling grid {dt, 2*dt, ..., <= 45d} (cap at 12 snapshots for speed)
        grid_full = np.arange(dt, 45.0 + 1e-9, dt)
        if len(grid_full) > 12:
            pick = np.unique(np.round(np.geomspace(1, len(grid_full), 12)).astype(int) - 1)
            grid_obs = grid_full[pick]
        else:
            grid_obs = grid_full
        grid_jax = jnp.asarray(grid_obs, dtype=jnp.float64)
        f_grid_tr, _ = extract_snapshot_tables(trajs_train, horizons=grid_obs)
        f_grid_jax = jnp.asarray(f_grid_tr, dtype=jnp.float64)
        n_grid_obs = float(np.maximum(f_grid_tr.sum(), 1.0))

        # 1. Single-Jump Neural Competing-Risks trained on the EXACT SAME coarsened snapshot grid {m * dt}
        # (matched multi-horizon supervision, zero dead 0.0 clips!)
        panel_tr = extract_panel_pairs(trajs_train, dt)
        init_ncr = init_neural_competing_risks(seed=17)

        @jax.jit
        def ncr_vg(w):
            def loss_fn(ww):
                mats = jnp.stack(
                    [predict_neural_competing_risks_single_step(ww, float(h))[:4] for h in grid_obs],
                    axis=1,
                )
                nll = -jnp.sum(f_grid_jax * jnp.log(jnp.clip(mats, 1e-12, 1.0))) / n_grid_obs
                return nll + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(loss_fn)(w)

        res_ncr = minimize(
            lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, np.float64), ncr_vg(jnp.asarray(x)))),
            init_ncr,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 140},
        )
        p_ncr_direct = np.stack(
            [np.asarray(predict_neural_competing_risks_single_step(jnp.asarray(res_ncr.x), float(h))[:4]) for h in eval_horizons],
            axis=1,
        )
        kl_ncr_direct = empirical_kl(p_ncr_direct, fresh_te)
        p_step = np.asarray(predict_neural_competing_risks_single_step(jnp.asarray(res_ncr.x), dt))
        p_ncr_comp = np.stack(
            [
                np.linalg.matrix_power(p_step, max(1, int(round(h / dt))))[:4]
                if h >= dt
                else np.asarray(predict_neural_competing_risks_single_step(jnp.asarray(res_ncr.x), float(h))[:4])
                for h in eval_horizons
            ],
            axis=1,
        )
        kl_ncr_comp = empirical_kl(p_ncr_comp, fresh_te)

        # 2. Multi-Hop EdgeMemoryHyperDRN trained on the EXACT coarsened snapshot grid {m * dt}
        init_em = init_edge_memory_drn(seed=0)

        @jax.jit
        def em_vg(w):
            def loss_fn(ww):
                pf, _ = predict_edge_memory_drn(ww, grid_jax)
                nll = -jnp.sum(f_grid_jax * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_grid_obs
                return nll + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(loss_fn)(w)

        res_em = minimize(
            lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, np.float64), em_vg(jnp.asarray(x)))),
            init_em,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 150},
        )
        pf_em, pa_em = predict_edge_memory_drn(jnp.asarray(res_em.x), h_eval_jax)
        kl_em = empirical_kl(np.asarray(pf_em), fresh_te)
        kl_em_aged = empirical_kl(np.asarray(pa_em), aged_te)

        # 3. Multi-Hop Renewal DRN trained on the EXACT coarsened snapshot grid {m * dt}
        init_drn = np.concatenate([np.log(np.full(E_EDGES, 8.0)), np.zeros(E_EDGES * K_RENEWAL)])

        @jax.jit
        def drn_vg(w):
            def loss_fn(ww):
                pf, _ = predict_renewal_drn(ww, grid_jax)
                nll = -jnp.sum(f_grid_jax * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_grid_obs
                return nll + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(loss_fn)(w)

        res_drn = minimize(
            lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, np.float64), drn_vg(jnp.asarray(x)))),
            init_drn,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 150},
        )
        pf_drn, pa_drn = predict_renewal_drn(jnp.asarray(res_drn.x), h_eval_jax)
        kl_drn = empirical_kl(np.asarray(pf_drn), fresh_te)
        kl_drn_aged = empirical_kl(np.asarray(pa_drn), aged_te)

        # 4. Memoryless CTMC trained on the EXACT same coarsened snapshot grid {m * dt}
        init_ctmc = np.full(E_EDGES, -2.0)

        @jax.jit
        def ctmc_vg(w):
            def loss_fn(ww):
                pf, _ = predict_ctmc(ww, grid_jax)
                nll = -jnp.sum(f_grid_jax * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_grid_obs
                return nll + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(loss_fn)(w)

        res_ctmc = minimize(
            lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, np.float64), ctmc_vg(jnp.asarray(x)))),
            init_ctmc,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 150},
        )
        pf_ctmc, pa_ctmc = predict_ctmc(jnp.asarray(res_ctmc.x), h_eval_jax)
        kl_ctmc = empirical_kl(np.asarray(pf_ctmc), fresh_te)

        # 5. Multi-hop Neural-CME(t) and Latent ODE trained on the EXACT same coarsened grid {m * dt}
        def fit_grid_neural(init_w, pred_fn):
            @jax.jit
            def n_vg(w):
                def loss_fn(ww):
                    pf, _ = pred_fn(ww, grid_jax)
                    nll = -jnp.sum(f_grid_jax * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_grid_obs
                    return nll + 1e-4 * jnp.mean(ww**2)

                return jax.value_and_grad(loss_fn)(w)

            res_n = minimize(
                lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, np.float64), n_vg(jnp.asarray(x)))),
                init_w,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 150},
            )
            pf_n, pa_n = pred_fn(jnp.asarray(res_n.x), h_eval_jax)
            return empirical_kl(np.asarray(pf_n), fresh_te), empirical_kl(np.asarray(pa_n), aged_te)

        kl_cme, kl_cme_aged = fit_grid_neural(init_helpdesk_neural_cme(42), predict_helpdesk_neural_cme)
        kl_lode, kl_lode_aged = fit_grid_neural(init_helpdesk_latent_ode(42), predict_helpdesk_latent_ode)

        # Fraction of direct-to-Closed transitions from Assign/Active/Wait inside a single delta_t step (multi-hop!)
        # In EDGES, only node 3 (Resolved) has a direct edge to node 4 (Closed)!
        off_diag = panel_tr[:4].copy()
        for u in range(4):
            off_diag[u, u] = 0.0
        direct_mask = np.zeros((4, V_NODES), dtype=bool)
        for u, v in EDGES:
            direct_mask[u, v] = True
        multi_hop_frac = float(off_diag[~direct_mask].sum() / np.maximum(off_diag.sum(), 1.0))

        row = {
            "delta_t_days": dt,
            "n_grid_snapshots": int(len(grid_obs)),
            "panel_transitions": int(panel_tr[:4].sum()),
            "observed_unconnected_multihop_fraction": multi_hop_frac,
            "neural_competing_risks_direct_kl": kl_ncr_direct,
            "neural_competing_risks_composed_kl": kl_ncr_comp,
            "neural_competing_risks_best_kl": min(kl_ncr_direct, kl_ncr_comp),
            "ctmc_panel_kl": kl_ctmc,
            "drn_multihop_kl": kl_drn,
            "drn_multihop_aged_kl": kl_drn_aged,
            "edge_memory_drn_multihop_kl": kl_em,
            "edge_memory_drn_multihop_aged_kl": kl_em_aged,
            "neural_cme_multihop_kl": kl_cme,
            "neural_cme_multihop_aged_kl": kl_cme_aged,
            "latent_ode_multihop_kl": kl_lode,
            "latent_ode_multihop_aged_kl": kl_lode_aged,
        }
        print(
            f"Coarsening dt={dt:4.1f}d ({len(grid_obs):2d} snaps) | "
            f"Neural-CR={row['neural_competing_risks_best_kl']:.4f} | CTMC={kl_ctmc:.4f} | "
            f"DRN={kl_drn:.4f} | EdgeMem={kl_em:.4f} (0-shot aged={kl_em_aged:.4f})"
        )
        rows.append(row)
    return rows


def main():
    csv_path = ensure_helpdesk_csv(Path("/tmp/drn_helpdesk"))
    trajs, sojourns = load_ticket_trajectories(csv_path)
    print(f"Loaded {len(trajs)} IT Helpdesk tickets (MD5 verified: {EXPECTED_MD5})")

    sojourn_summary = {}
    for s_idx, name in enumerate(CORE_STATES[:-1]):
        arr = np.asarray(sojourns[s_idx], dtype=float)
        sojourn_summary[name] = {
            "n": int(len(arr)),
            "mean_days": float(arr.mean()),
            "median_days": float(np.median(arr)),
            "sd_days": float(arr.std()),
            "cv": float(arr.std() / max(arr.mean(), 1e-12)),
        }

    # 5-fold ticket-grouped CV (deterministic shuffle)
    rng = np.random.default_rng(2026)
    order = rng.permutation(len(trajs))
    folds = np.array_split(order, 5)
    h_jax = jnp.asarray(HORIZONS_DAYS, dtype=jnp.float64)

    n_em_params = edge_memory_param_size(V_NODES, E_EDGES, stages=K_EDGE)
    tensor_states = network_state_size(EDGES, V_NODES, K_RENEWAL)
    race_states = compact_network_state_size(EDGES, V_NODES, K_RENEWAL)
    em_states = int(
        build_edge_memory_hyper_generator(jnp.zeros(n_em_params), EDGES, V_NODES, stages=K_EDGE)[0].shape[0]
    )

    models_spec = [
        ("ctmc", lambda s: np.full(E_EDGES, -2.0), predict_ctmc, False),
        ("renewal_drn_fresh_only", lambda s: np.concatenate([np.log(np.full(E_EDGES, 8.0)), np.zeros(E_EDGES * K_RENEWAL)]), predict_renewal_drn, False),
        ("renewal_drn_joint", lambda s: np.concatenate([np.log(np.full(E_EDGES, 8.0)), np.zeros(E_EDGES * K_RENEWAL)]), predict_renewal_drn, True),
        ("edge_memory_drn_fresh_only", init_edge_memory_drn, predict_edge_memory_drn, False),
        ("edge_memory_drn_joint", init_edge_memory_drn, predict_edge_memory_drn, True),
        ("neural_cme_fresh_only", init_helpdesk_neural_cme, predict_helpdesk_neural_cme, False),
        ("neural_cme_joint", init_helpdesk_neural_cme, predict_helpdesk_neural_cme, True),
        ("latent_ode_fresh_only", init_helpdesk_latent_ode, predict_helpdesk_latent_ode, False),
        ("latent_ode_joint", init_helpdesk_latent_ode, predict_helpdesk_latent_ode, True),
    ]

    fold_results = {name: {"fresh_kl": [], "aged_kl": []} for name, *_ in models_spec}
    for fi in range(5):
        te_idx = set(folds[fi].tolist())
        tr_trajs = [trajs[i] for i in range(len(trajs)) if i not in te_idx]
        te_trajs = [trajs[i] for i in range(len(trajs)) if i in te_idx]
        f_tr, a_tr = extract_snapshot_tables(tr_trajs)
        f_te, a_te = extract_snapshot_tables(te_trajs)
        print(f"Fold {fi+1}/5: train_tickets={len(tr_trajs)}, test_tickets={len(te_trajs)}, aged_obs={int(a_te.sum())}")

        for mname, init_fn, pred_fn, is_joint in models_spec:
            w_hat = fit_model(init_fn(100 + fi), pred_fn, f_tr, a_tr, joint=is_joint, maxiter=150)
            pf, pa = pred_fn(jnp.asarray(w_hat), h_jax)
            # Clean age=0 zero-shot evaluation for black-box neural ODEs trained fresh_only
            if (not is_joint) and ("neural_cme" in mname or "latent_ode" in mname):
                pa = pf[1:4]
            f_kl = empirical_kl(np.asarray(pf), f_te)
            a_kl = empirical_kl(np.asarray(pa), a_te)
            fold_results[mname]["fresh_kl"].append(f_kl)
            fold_results[mname]["aged_kl"].append(a_kl)
            print(f"  {mname:28s}: fresh_kl={f_kl:.5f} | aged_kl={a_kl:.5f}")

    summary_cv = {}
    for mname, d in fold_results.items():
        summary_cv[mname] = {
            "fresh_kl_mean": float(np.mean(d["fresh_kl"])),
            "fresh_kl_sd": float(np.std(d["fresh_kl"], ddof=1)),
            "aged_kl_mean": float(np.mean(d["aged_kl"])),
            "aged_kl_sd": float(np.std(d["aged_kl"], ddof=1)),
            "folds": d,
        }

    # Part 2: Coarsening panel sweep on fold 0 split (80/20 tickets)
    te_idx0 = set(folds[0].tolist())
    tr0 = [trajs[i] for i in range(len(trajs)) if i not in te_idx0]
    te0 = [trajs[i] for i in range(len(trajs)) if i in te_idx0]
    coarsening_rows = run_coarsening_panel_experiment(tr0, te0)

    out = {
        "dataset": {
            "name": "IT Helpdesk Ticketing Management Event Log (Polato, 2017)",
            "doi": "10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb",
            "md5": EXPECTED_MD5,
            "tickets_used": len(trajs),
            "nodes": V_NODES,
            "edges": E_EDGES,
            "aged_threshold_days": AGED_THRESHOLD_DAYS,
            "horizons_days": HORIZONS_DAYS.tolist(),
            "sojourn_statistics": sojourn_summary,
        },
        "state_counts": {
            "ctmc_states": V_NODES,
            "ctmc_params": E_EDGES,
            "renewal_drn_tensor_states_K6": tensor_states,
            "renewal_drn_race_states_K6": race_states,
            "renewal_drn_params_K6": E_EDGES * (1 + K_RENEWAL),
            "edge_memory_drn_states_K6": em_states,
            "edge_memory_drn_params_K6": n_em_params,
            "neural_cme_params": int(init_helpdesk_neural_cme(0).size),
            "latent_ode_params": int(init_helpdesk_latent_ode(0).size),
            "neural_competing_risks_params": int(init_neural_competing_risks(0).size),
        },
        "cv_5fold_summary": summary_cv,
        "coarsening_panel_sweep": coarsening_rows,
    }
    out_path = ROOT / "results" / "archived" / "helpdesk_systems_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
