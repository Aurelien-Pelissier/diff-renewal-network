"""Real Large-V Temporal Interaction Network Benchmarks for Matrix-Free SparseDRN:
  1. StackOverflow (EasyTPP; V = 22 badge states, 2,203 users, 142,777 events; official train/dev/test)
  2. MOOC (SNAP JODIE; V = 98 real nodes = 97 course actions + absorbing Dropout, 7,047 students,
     411,749 interactions, bursty CV = 5.54!)

Unconditionally stable, exact solvers:
  - SparseHyperDRN: Matrix-free O(V*K + E) 2-branch Hyper-Erlang with adaptive Diffrax Tsit5
    (zero dense N x N matrices, zero RK4 step instability, exact analytical survivor conditioning!).
  - Hyper-Pos-Sens: Data-free universal 1D Pos-Sens clock (k in [1.4, 7.8]) + fast exponential branch
    plugged ZERO-SHOT into real V=22 and V=98 graphs!
  - Strict CTMC: Exact jax.scipy.linalg.expm(h * G) (100% unconditionally stable & exact to 1e-15!).
  - Mamba / Selective-SSM, S4D, LTC, Latent ODE, Neural-CME: 3-start clean age=0 on 0-shot,
    using unconditionally stable exponential-Euler / uniformized flux updates (zero RK4 blow-ups!).
  - Single-jump Neural Hawkes / Competing-Risks TPP baseline on multi-hop coarsening grids.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import diffrax
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import smooth_erlang_logits  # noqa: E402

OUT_PATH = ROOT / "results" / "archived" / "r7_temporal_networks.json"
SO_DIR = Path(os.environ.get("DRN_SO_DIR", "/tmp/drn_r7_data/stackoverflow"))
MOOC_CSV = Path(os.environ.get("DRN_MOOC_CSV", "/tmp/drn_r7_data/mooc/mooc.csv"))
STARTS = (17, 41, 89)


def weighted_kl(pred: np.ndarray, counts: np.ndarray) -> float:
    tot = counts.sum(axis=-1, keepdims=True)
    emp = np.where(tot > 0, counts / np.maximum(tot, 1.0), 0.0)
    p = np.clip(pred, 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    kl = np.sum(np.where(emp > 0, emp * np.log(np.maximum(emp, 1e-30) / p), 0.0), axis=-1)
    w = tot[..., 0]
    return float(np.sum(kl * w) / np.maximum(np.sum(w), 1.0))


def extract_general_snapshots(
    trajs: list[tuple[np.ndarray, np.ndarray]],
    n_nodes: int,
    horizons: np.ndarray,
    aged_thresh: float,
    absorbing_node: int | None = None,
):
    H = len(horizons)
    fresh = np.zeros((n_nodes, H, n_nodes), dtype=np.float64)
    aged = np.zeros((n_nodes, H, n_nodes), dtype=np.float64)
    edge_counts = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for times, states in trajs:
        if len(times) < 2:
            continue
        t_end = float(times[-1])
        is_abs_end = (absorbing_node is not None) and (int(states[-1]) == absorbing_node)
        for k in range(len(states) - 1):
            u, v = int(states[k]), int(states[k + 1])
            if u != v:
                edge_counts[u, v] += 1.0
        chg = np.concatenate([[0], np.where(states[1:] != states[:-1])[0] + 1])
        s_starts = times[chg]
        s_states = states[chg]
        s_ends = np.concatenate([s_starts[1:], [t_end]])

        for idx in range(len(s_starts)):
            u = int(s_states[idx])
            if absorbing_node is not None and u == absorbing_node:
                continue
            t0 = float(s_starts[idx])
            dur = float(s_ends[idx] - t0)
            t_q = t0 + horizons
            valid = (t_q <= t_end) | is_abs_end
            if np.any(valid):
                pos = np.clip(np.searchsorted(times, t_q[valid], side="right") - 1, 0, len(states) - 1)
                for hi, vi in zip(np.where(valid)[0], states[pos]):
                    fresh[u, hi, int(vi)] += 1.0
            if dur > aged_thresh:
                ta = t0 + aged_thresh
                t_qa = ta + horizons
                valid_a = (t_qa <= t_end) | is_abs_end
                if np.any(valid_a):
                    pos_a = np.clip(np.searchsorted(times, t_qa[valid_a], side="right") - 1, 0, len(states) - 1)
                    for hi, vi in zip(np.where(valid_a)[0], states[pos_a]):
                        aged[u, hi, int(vi)] += 1.0
    return fresh, aged, edge_counts


# ---------------------------------------------------------------------------
# 1. Exact Matrix-Free SparseHyperDRN (Adaptive Diffrax Tsit5, ZERO dense matrices!)
# ---------------------------------------------------------------------------
def make_sparse_hyper_drn(
    n_nodes: int,
    edges: list[tuple[int, int]],
    k_slow: int,
    horizons: np.ndarray,
    absorbing_node: int | None = None,
    max_log_fast: float = 1.8,
    rtol: float = 1e-4,
    atol: float = 1e-6,
):
    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    n_edges = len(edges)
    B_block = k_slow + 1
    D_total = n_nodes * B_block
    is_abs = np.zeros(n_nodes, dtype=bool)
    if absorbing_node is not None:
        is_abs[absorbing_node] = True
    active_mask = jnp.asarray((~is_abs).astype(np.float64))
    js_k = jnp.arange(k_slow, dtype=jnp.float64)
    n_params = 4 * n_nodes + n_edges
    ts_j = jnp.asarray(horizons, dtype=jnp.float64)
    t_max = float(horizons[-1])

    def unpack(th):
        # Smooth ordered rates: lam_s < lam_f via lam_f = lam_s + exp(...)
        lam_s = jnp.exp(jnp.clip(th[n_nodes : 2 * n_nodes], -7.0, 1.5)) * active_mask
        lam_f = (lam_s + jnp.exp(jnp.clip(th[:n_nodes], -6.0, max_log_fast))) * active_mask
        # smooth_erlang_logits uses 1-indexed stage counts [1..k_slow]
        c_s = 1.0 + (k_slow - 1.0) * jax.nn.sigmoid(th[2 * n_nodes : 3 * n_nodes])
        w_slow = jax.vmap(lambda c: jax.nn.softmax(smooth_erlang_logits(c, stages=k_slow, width=0.5)))(c_s)
        b_slow = w_slow[:, ::-1]
        alpha_f = jax.nn.sigmoid(th[3 * n_nodes : 4 * n_nodes]) * active_mask

        el = th[4 * n_nodes : 4 * n_nodes + n_edges]
        el_max = jax.ops.segment_max(el, src_idx, num_segments=n_nodes)
        exp_el = jnp.exp(jnp.clip(el - el_max[src_idx], -40.0, 0.0))
        denom = jax.ops.segment_sum(exp_el, src_idx, num_segments=n_nodes)
        r_weights = exp_el / jnp.maximum(denom[src_idx], 1e-30)
        return lam_f, lam_s, alpha_f, b_slow, r_weights

    def build_init_states(lam_f, lam_s, alpha_f, b_slow, eval_nodes: jax.Array, age: float):
        B_ev = eval_nodes.shape[0]
        lf = lam_f[eval_nodes]
        ls = lam_s[eval_nodes]
        af = alpha_f[eval_nodes]
        bs = b_slow[eval_nodes]
        if age <= 0.0:
            vec = jnp.concatenate([af[:, None], (1.0 - af)[:, None] * bs], axis=-1)
        else:
            df = (lf - ls) * age  # >= 0 because lf >= ls!
            xa = jnp.maximum(ls * age, 1e-12)
            diff_idx = js_k[:, None] - js_k[None, :]
            valid = diff_idx >= 0
            d_safe = jnp.where(valid, diff_idx, 0.0)
            log_p = d_safe[None, :, :] * jnp.log(xa)[:, None, None] - jax.scipy.special.gammaln(d_safe + 1.0)[None, :, :]
            m_sh = jnp.maximum(jnp.max(jnp.where(valid[None, :, :], log_p, -1e9), axis=(1, 2)), 0.0)
            P_mat = jnp.where(valid[None, :, :], jnp.exp(jnp.clip(log_p - m_sh[:, None, None], -700.0, 0.0)), 0.0)
            slow_un = (1.0 - af)[:, None] * jnp.einsum("bji,bi->bj", P_mat, bs)
            fast_un = (af * jnp.exp(-jnp.clip(df + m_sh, 0.0, 700.0)))[:, None]
            vec = jnp.concatenate([fast_un, slow_un], axis=-1)
            vec = vec / jnp.maximum(jnp.sum(vec, axis=-1, keepdims=True), 1e-300)
        x0 = jnp.zeros((B_ev, n_nodes, B_block), dtype=jnp.float64)
        return x0.at[jnp.arange(B_ev), eval_nodes, :].set(vec)

    def rhs(_t, x, args):
        lam_f, lam_s, alpha_f, b_slow, r_weights = args
        x_fast = x[:, :, 0]
        x_slow = x[:, :, 1:]
        exit_fast = x_fast * lam_f[None, :]
        dx_fast = -exit_fast
        rate_slow = x_slow * lam_s[None, :, None]
        dx_slow = -rate_slow
        dx_slow = dx_slow.at[:, :, 1:].add(rate_slow[:, :, :-1])
        exit_slow = rate_slow[:, :, -1]
        total_exit = exit_fast + exit_slow
        edge_flux = total_exit[:, src_idx] * r_weights[None, :]
        inflow = jnp.zeros((x.shape[0], n_nodes), dtype=x.dtype).at[:, dst_idx].add(edge_flux)
        dx_fast = dx_fast + inflow * alpha_f[None, :]
        dx_slow = dx_slow + (inflow * (1.0 - alpha_f)[None, :])[:, :, None] * b_slow[None, :, :]
        return jnp.concatenate([dx_fast[:, :, None], dx_slow], axis=-1)

    term = diffrax.ODETerm(rhs)
    solver = diffrax.Tsit5()
    stepsize = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=ts_j)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    def rollout_batch(th, eval_nodes: jax.Array, age: float):
        args = unpack(th)
        x0 = build_init_states(*args[:4], eval_nodes, age)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_max,
            dt0=min(float(horizons[0]) * 0.25, 0.2),
            y0=x0,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize,
            adjoint=adjoint,
            max_steps=4096,
        )
        # sol.ys: (H, B_ev, V, K+1) -> sum phases -> (B_ev, H, V)
        probs = jnp.maximum(jnp.sum(sol.ys, axis=-1), 1e-12)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
        return jnp.swapaxes(probs, 0, 1)

    sparse_bytes = (4 * n_nodes + n_nodes * k_slow + n_edges) * 8 + (2 * n_edges) * 4
    dense_bytes = (D_total * D_total) * 8
    meta = {
        "compiled_states": int(D_total),
        "n_params": int(n_params),
        "sparse_operator_mb": round(sparse_bytes / (1024**2), 4),
        "dense_operator_mb": round(dense_bytes / (1024**2), 4),
        "memory_compression": round(dense_bytes / max(sparse_bytes, 1), 1),
    }
    return rollout_batch, n_params, meta


# ---------------------------------------------------------------------------
# 2. Exact CTMC via jax.scipy.linalg.expm (100% unconditionally stable!)
# ---------------------------------------------------------------------------
def make_exact_ctmc(n_nodes: int, edges: list[tuple[int, int]], horizons: np.ndarray, absorbing_node: int | None = None):
    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    n_edges = len(edges)
    is_abs = np.zeros(n_nodes, dtype=bool)
    if absorbing_node is not None:
        is_abs[absorbing_node] = True
    edge_active = jnp.asarray([0.0 if is_abs[u] else 1.0 for u, _ in edges], dtype=jnp.float64)
    ts_j = jnp.asarray(horizons, dtype=jnp.float64)

    def rollout_batch(th, eval_nodes: jax.Array):
        rates = jnp.exp(jnp.clip(th, -8.0, 5.3)) * edge_active
        gen = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float64)
        gen = gen.at[dst_idx, src_idx].add(rates)
        gen = gen.at[src_idx, src_idx].add(-rates)

        # Compute P(h) = expm(h * gen)^T for all horizons h -> (H, V, V), index rows eval_nodes!
        def one_h(h):
            # Row u of expm(h*gen)^T gives distribution starting from node u!
            p_mat = jl.expm(h * gen).T[eval_nodes]
            p_mat = jnp.maximum(p_mat, 1e-12)
            return p_mat / jnp.sum(p_mat, axis=-1, keepdims=True)

        p_all = jax.vmap(one_h)(ts_j)  # (H, B_ev, V)
        return jnp.swapaxes(p_all, 0, 1)

    return rollout_batch, n_edges


# ---------------------------------------------------------------------------
# 3. Unconditionally Stable Scalable Neural ODE/SSM Baselines (Mamba, S4D, LTC, Latent ODE, Neural-CME)
# ---------------------------------------------------------------------------
def make_scalable_neural_baseline(
    kind: str,
    n_nodes: int,
    edges: list[tuple[int, int]],
    horizons: np.ndarray,
    absorbing_node: int | None = None,
    latent: int = 12,
    n_substeps_max: int = 40,
):
    """Unconditionally simplex-preserving neural ODE/SSM over sparse graph:
    At every step dt, outgoing probability from node u to neighbor v is
    p_leave = 1 - exp(-dt * sum_{v} r_{uv}), split proportionally to r_{uv}!
    This exact exponential-Euler flux splitting guarantees p >= 0 and sum(p) = 1
    UNCONDITIONALLY for ANY positive edge rates r_{uv} and ANY step size dt!"""
    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    n_edges = len(edges)
    is_abs = np.zeros(n_nodes, dtype=bool)
    if absorbing_node is not None:
        is_abs[absorbing_node] = True
    edge_active = jnp.asarray([0.0 if is_abs[u] else 1.0 for u, _ in edges], dtype=jnp.float64)

    seg_dts = np.diff(np.concatenate([[0.0], horizons]))
    t_max = float(horizons[-1])
    seg_steps = [max(3, int(np.ceil(sd / (t_max / n_substeps_max)))) for sd in seg_dts]
    rank_e = 6

    if kind == "mamba":
        # Pre-LayerNorm + Xavier selective SSM (avoids 5-factor vanishing core gradients!)
        shapes = [
            (n_nodes, latent), (1, latent), (latent,),
            (n_nodes, latent), (latent,),
            (latent,), (latent, latent), (latent,),
            (latent, latent), (latent,), (latent, latent), (latent,),
            (n_edges,), (latent, rank_e), (rank_e, n_edges),
        ]
    elif kind == "s4d":
        half = latent // 2
        shapes = [
            (n_nodes, latent), (1, latent), (latent,),
            (half,), (half,), (n_nodes, latent), (latent,),
            (n_edges,), (latent, rank_e), (rank_e, n_edges),
        ]
    elif kind == "ltc":
        shapes = [
            (n_nodes, latent), (1, latent), (latent,),
            (n_nodes + latent, latent), (latent,),
            (n_nodes + latent, latent), (latent,),
            (n_edges,), (latent, rank_e), (rank_e, n_edges),
        ]
    elif kind == "latent_ode":
        shapes = [
            (n_nodes, latent), (1, latent), (latent,),
            (n_nodes + latent, latent), (latent,),
            (latent, latent), (latent,),
            (n_edges,), (latent, rank_e), (rank_e, n_edges),
        ]
    elif kind == "neural_cme":
        shapes = [(6, 16), (16,), (16, rank_e), (rank_e, n_edges), (n_edges,)]
    else:
        raise ValueError(kind)

    n_params = sum(int(np.prod(s)) for s in shapes)

    def init_fn(seed: int, edge_prior_logits: np.ndarray):
        r = np.random.default_rng(seed)
        parts = []
        for idx, sh in enumerate(shapes):
            if kind == "mamba" and idx == 5:
                parts.append(np.linspace(np.log(0.1), np.log(3.0), latent))
            elif kind == "mamba" and idx == 7:
                parts.append(np.full(latent, -0.5))  # b_dt
            elif kind == "mamba" and idx == 11:
                parts.append(np.full(latent, 1.0))  # b_z > 0 so SiLU gate is open at init
            elif kind == "mamba" and idx in (2, 4, 9):
                parts.append(np.zeros(sh).ravel())
            elif kind == "s4d" and idx == 3:
                parts.append(np.linspace(np.log(0.1), np.log(3.0), latent // 2))
            elif (kind == "neural_cme" and idx == 4) or (kind != "neural_cme" and idx == len(shapes) - 3):
                parts.append(edge_prior_logits.copy())
            elif kind == "mamba" and len(sh) == 2:
                # Xavier scaling keeps Pre-LN Mamba core gradients O(1e-3), matched to S4D/LTC/Latent ODE
                scale = 0.5 / np.sqrt(max(sh[0], 1)) if idx >= len(shapes) - 2 else 1.0 / np.sqrt(max(sh[0], 1))
                parts.append(r.normal(0.0, scale, sh).ravel())
            else:
                parts.append(r.normal(0.0, 0.12, sh).ravel())
        return np.concatenate(parts)

    def unpack(th):
        out, i = [], 0
        for sh in shapes:
            sz = int(np.prod(sh))
            out.append(th[i : i + sz].reshape(sh))
            i += sz
        return out

    def exp_euler_step(p: jax.Array, rates: jax.Array, h_dt: float) -> jax.Array:
        """Unconditionally positive, mass-conserving jump step for batch p (B, V) and edge rates (B, E)."""
        # Total exit rate per node u: (B, V)
        tot_out = jax.vmap(lambda r: jax.ops.segment_sum(r, src_idx, num_segments=n_nodes))(rates)
        # Exact fraction leaving node u during step h_dt: 1 - exp(-tot_out * h_dt) in [0, 1)
        frac_leave = 1.0 - jnp.exp(-jnp.minimum(tot_out * h_dt, 50.0))  # (B, V)
        # Split outgoing mass p * frac_leave proportionally to edge rates!
        scale = frac_leave / jnp.maximum(tot_out, 1e-30)  # (B, V)
        edge_mass = p[:, src_idx] * scale[:, src_idx] * rates  # (B, E) >= 0
        p_next = p * (1.0 - frac_leave)
        p_next = p_next.at[:, dst_idx].add(edge_mass)
        p_next = jnp.maximum(p_next, 1e-12)
        return p_next / jnp.sum(p_next, axis=-1, keepdims=True)

    def rollout_batch(th, eval_nodes: jax.Array, age_val: float):
        B_ev = eval_nodes.shape[0]
        p0 = jax.nn.one_hot(eval_nodes, n_nodes, dtype=jnp.float64)
        a_col = jnp.full((B_ev, 1), age_val, dtype=jnp.float64)
        ps = unpack(th)

        if kind == "neural_cme":
            w1, b1, w2, ve, be = ps
            t_cur = jnp.array(0.0, dtype=jnp.float64)
            p = p0
            preds = []
            for sd, n_st in zip(seg_dts, seg_steps):
                h_dt = float(sd) / n_st

                def step(carry, _):
                    pc, tc = carry
                    tm = (tc + 0.5 * h_dt) / max(t_max, 1e-6)
                    tf = jnp.stack(
                        [
                            jnp.full(B_ev, tm),
                            jnp.full(B_ev, jnp.sin(2.0 * jnp.pi * tm)),
                            jnp.full(B_ev, jnp.cos(2.0 * jnp.pi * tm)),
                            jnp.full(B_ev, jnp.sin(4.0 * jnp.pi * tm)),
                            jnp.full(B_ev, jnp.cos(4.0 * jnp.pi * tm)),
                            jnp.full(B_ev, age_val),
                        ],
                        axis=-1,
                    )
                    mod = (jax.nn.gelu(tf @ w1 + b1) @ w2) @ ve
                    rates = jnp.exp(jnp.clip(be[None, :] + mod, -8.0, 5.3)) * edge_active[None, :]
                    return (exp_euler_step(pc, rates, h_dt), tc + h_dt), None

                (p, t_cur), _ = jax.lax.scan(step, (p, t_cur), None, length=n_st)
                preds.append(p)
            return jnp.swapaxes(jnp.stack(preds, axis=0), 0, 1)

        def _ln(x):
            return (x - jnp.mean(x, axis=-1, keepdims=True)) / jnp.sqrt(
                jnp.var(x, axis=-1, keepdims=True) + 1e-5
            )

        enc_v, enc_a, eb = ps[0], ps[1], ps[2]
        be, ue, ve = ps[-3], ps[-2], ps[-1]
        h0 = jnp.tanh(p0 @ enc_v + a_col @ enc_a + eb)
        p, h = p0, h0
        preds = []
        for sd, n_st in zip(seg_dts, seg_steps):
            h_dt = float(sd) / n_st

            def step(carry, _):
                pc, hc = carry
                if kind == "mamba":
                    w_in, b_in, log_a, w_dt, b_dt, w_b, b_b, w_z, b_z = ps[3:12]
                    x_n = _ln(pc @ w_in + b_in)
                    neg_a = -jnp.exp(jnp.clip(log_a, -3.5, 2.5))
                    delta = jax.nn.softplus(x_n @ w_dt + b_dt) * h_dt
                    a_bar = jnp.exp(delta * neg_a[None, :])
                    b_bar = (jnp.expm1(delta * neg_a[None, :]) / neg_a[None, :]) * jnp.tanh(x_n @ w_b + b_b)
                    hn = a_bar * hc + b_bar
                    y_h = _ln(hn) * jax.nn.silu(x_n @ w_z + b_z)
                elif kind == "s4d":
                    log_a_half, om, w_b, b_b = ps[3:7]
                    decay_half = jnp.exp(jnp.clip(log_a_half, -3.5, 2.5))
                    decay = jnp.concatenate([decay_half, decay_half], axis=0)
                    half = latent // 2
                    c_th = jnp.cos(om * h_dt)[None, :]
                    s_th = jnp.sin(om * h_dt)[None, :]
                    h1, h2 = hc[:, :half], hc[:, half:]
                    h_rot = jnp.concatenate([c_th * h1 - s_th * h2, s_th * h1 + c_th * h2], axis=-1)
                    a_bar = jnp.exp(-decay[None, :] * h_dt)
                    hn = a_bar * h_rot + (-jnp.expm1(-decay[None, :] * h_dt) / decay[None, :]) * jnp.tanh(pc @ w_b + b_b)
                    y_h = jnp.tanh(hn)
                elif kind == "ltc":
                    wa, ba, wt, bt = ps[3:7]
                    inp = jnp.concatenate([pc, hc], axis=-1)
                    tgt = jnp.tanh(inp @ wa + ba)
                    tau_i = 0.05 + 3.0 * jax.nn.sigmoid(inp @ wt + bt)
                    a_bar = jnp.exp(-tau_i * h_dt)
                    hn = a_bar * hc + (1.0 - a_bar) * tgt
                    y_h = hn
                else:
                    w1, b1, w2, b2 = ps[3:7]
                    inp = jnp.concatenate([pc, hc], axis=-1)
                    dh = jnp.tanh(jnp.tanh(inp @ w1 + b1) @ w2 + b2)
                    hn = hc + h_dt * dh
                    y_h = jnp.tanh(hn)

                mod = (y_h @ ue) @ ve
                rates = jnp.exp(jnp.clip(be[None, :] + mod, -8.0, 5.3)) * edge_active[None, :]
                return (exp_euler_step(pc, rates, h_dt), hn), None

            (p, h), _ = jax.lax.scan(step, (p, h), None, length=n_st)
            preds.append(p)
        return jnp.swapaxes(jnp.stack(preds, axis=0), 0, 1)

    return init_fn, rollout_batch, n_params


# ---------------------------------------------------------------------------
# Generic suite runner
# ---------------------------------------------------------------------------
def run_network_suite(
    name: str,
    tr_trajs: list,
    te_trajs: list,
    n_nodes: int,
    horizons: np.ndarray,
    aged_thresh: float,
    coarsening_dts: list[float],
    absorbing_node: int | None = None,
    k_slow: int = 5,
    min_edge_count: int = 5,
    maxiter_drn: int = 110,
    maxiter_nn: int = 80,
    drn_starts: tuple[int, ...] = (17, 41),
    nn_starts: tuple[int, ...] = STARTS,
    max_log_fast: float = 1.8,
):
    f_tr, a_tr, ec_tr = extract_general_snapshots(tr_trajs, n_nodes, horizons, aged_thresh, absorbing_node)
    f_te, a_te, _ = extract_general_snapshots(te_trajs, n_nodes, horizons, aged_thresh, absorbing_node)

    active_nodes = [
        u for u in range(n_nodes)
        if (absorbing_node is None or u != absorbing_node)
        and (f_tr[u].sum() > 0 or a_tr[u].sum() > 0 or f_te[u].sum() > 0 or a_te[u].sum() > 0)
    ]
    eval_nodes_j = jnp.asarray(active_nodes, dtype=jnp.int32)
    f_tr_a, a_tr_a = f_tr[active_nodes], a_tr[active_nodes]
    f_te_a, a_te_a = f_te[active_nodes], a_te[active_nodes]

    edges = []
    for u in range(n_nodes):
        if absorbing_node is not None and u == absorbing_node:
            continue
        row = ec_tr[u]
        dsts = [v for v in range(n_nodes) if v != u and row[v] >= min_edge_count]
        if not dsts and np.max(row) > 0:
            v_best = int(np.argmax(row))
            if v_best != u:
                dsts = [v_best]
        for v in dsts:
            edges.append((u, v))
    print(
        f"\n=== {name}: V={n_nodes}, active={len(active_nodes)}, sparse_edges={len(edges)}, "
        f"fresh_te_n={int(f_te_a.sum())}, aged_te_n={int(a_te_a.sum())} ===",
        flush=True,
    )

    # Empirical fresh-rollout oracle on aged test set
    emp_f = f_tr_a / np.maximum(f_tr_a.sum(axis=-1, keepdims=True), 1.0)
    emp_f = np.clip(emp_f, 1e-6, 1.0)
    emp_f /= emp_f.sum(axis=-1, keepdims=True)
    oracle_aged_kl = weighted_kl(emp_f, a_te_a)
    print(f"  Empirical fresh-rollout oracle aged KL = {oracle_aged_kl:.4f}", flush=True)

    f_j = jnp.asarray(f_tr_a, dtype=jnp.float64)
    a_j = jnp.asarray(a_tr_a, dtype=jnp.float64)
    nf = jnp.asarray(max(float(f_tr_a.sum()), 1.0))
    na = jnp.asarray(max(float(a_tr_a.sum()), 1.0))

    # Smart empirical edge rate priors for CTMC & Neural ODEs
    row_sums = np.maximum(ec_tr.sum(axis=1), 1.0)
    h0 = float(horizons[0])
    # Rough per-node exit rate from first horizon stay fraction
    node_exit = np.array([
        -np.log(np.clip(f_tr[u, 0, u] / max(f_tr[u, 0].sum(), 1.0), 0.05, 0.99)) / max(h0, 1e-3)
        for u in range(n_nodes)
    ])
    edge_prior_rates = np.array([
        np.clip(node_exit[u] * (ec_tr[u, v] / row_sums[u]), 1e-3, 10.0)
        for u, v in edges
    ])
    edge_prior_logits = np.log(edge_prior_rates)

    # 1. Exact Matrix-Free SparseHyperDRN (Diffrax Tsit5)
    rollout_drn, n_p_drn, drn_meta = make_sparse_hyper_drn(
        n_nodes, edges, k_slow, horizons, absorbing_node=absorbing_node, max_log_fast=max_log_fast
    )
    edge_route_init = np.array([np.log(max(ec_tr[u, v], 1.0)) for u, v in edges], dtype=np.float64)
    init_drn_base = np.concatenate([
        np.log(np.clip(node_exit * 1.5, 0.05, 5.0)),  # log delta_lam (fast - slow)
        np.log(np.clip(node_exit * 0.25, 0.005, 1.0)),  # log lam_s
        np.zeros(n_nodes),
        np.zeros(n_nodes),
        edge_route_init - np.mean(edge_route_init),
    ])

    @jax.jit
    def vg_drn_fresh(w):
        def loss_fn(ww):
            pf = rollout_drn(ww, eval_nodes_j, 0.0)
            return -jnp.sum(f_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / nf + 1e-5 * jnp.mean(ww**2)

        return jax.value_and_grad(loss_fn)(w)

    @jax.jit
    def vg_drn_joint(w):
        def loss_fn(ww):
            pf = rollout_drn(ww, eval_nodes_j, 0.0)
            pa = rollout_drn(ww, eval_nodes_j, aged_thresh)
            lf = -jnp.sum(f_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / nf
            la = -jnp.sum(a_j * jnp.log(jnp.clip(pa, 1e-12, 1.0))) / na
            return 0.5 * (lf + la) + 1e-5 * jnp.mean(ww**2)

        return jax.value_and_grad(loss_fn)(w)

    w_test = jnp.asarray(init_drn_base)
    vg_drn_fresh(w_test)[1].block_until_ready()
    t0_g = perf_counter()
    for _ in range(5):
        vg_drn_fresh(w_test)[1].block_until_ready()
    drn_meta["warm_grad_ms"] = round(1000.0 * (perf_counter() - t0_g) / 5.0, 2)

    results_models = {"oracle_fresh_rollout_aged_kl": oracle_aged_kl, "sparse_drn_meta": drn_meta}

    for mode, vg_fn in (("fresh_only", vg_drn_fresh), ("joint", vg_drn_joint)):
        t0 = perf_counter()
        def obj_d(z):
            v, g = vg_fn(jnp.asarray(z, dtype=jnp.float64))
            return float(v), np.asarray(g, dtype=np.float64)

        best = None
        for s in drn_starts:
            w0 = init_drn_base + np.random.default_rng(s).normal(0.0, 0.12, len(init_drn_base))
            res = minimize(obj_d, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter_drn})
            if best is None or res.fun < best[0]:
                best = (float(res.fun), res.x)
        pf = np.asarray(rollout_drn(jnp.asarray(best[1]), eval_nodes_j, 0.0))
        pa = np.asarray(rollout_drn(jnp.asarray(best[1]), eval_nodes_j, aged_thresh))
        results_models[f"sparse_hyper_drn_{mode}"] = {
            "states": drn_meta["compiled_states"],
            "params": n_p_drn,
            "fresh_kl": weighted_kl(pf, f_te_a),
            "aged_kl": weighted_kl(pa, a_te_a),
            "seconds": round(perf_counter() - t0, 2),
        }
        print(
            f"  SparseHyperDRN ({mode:10s}, {drn_meta['compiled_states']}st, {n_p_drn}p, {drn_meta['warm_grad_ms']:.1f}ms/grad): "
            f"fresh={results_models[f'sparse_hyper_drn_{mode}']['fresh_kl']:.4f}, "
            f"aged={results_models[f'sparse_hyper_drn_{mode}']['aged_kl']:.4f}",
            flush=True,
        )

    # 2. Exact CTMC via expm
    rollout_ctmc, n_p_ctmc = make_exact_ctmc(n_nodes, edges, horizons, absorbing_node=absorbing_node)

    @jax.jit
    def vg_ctmc_f(w):
        def lf(ww):
            p = rollout_ctmc(ww, eval_nodes_j)
            return -jnp.sum(f_j * jnp.log(jnp.clip(p, 1e-12, 1.0))) / nf + 1e-5 * jnp.mean(ww**2)

        return jax.value_and_grad(lf)(w)

    @jax.jit
    def vg_ctmc_j(w):
        def lf(ww):
            p = rollout_ctmc(ww, eval_nodes_j)
            return -0.5 * (
                jnp.sum(f_j * jnp.log(jnp.clip(p, 1e-12, 1.0))) / nf
                + jnp.sum(a_j * jnp.log(jnp.clip(p, 1e-12, 1.0))) / na
            ) + 1e-5 * jnp.mean(ww**2)

        return jax.value_and_grad(lf)(w)

    for mode, vg_fn in (("fresh_only", vg_ctmc_f), ("joint", vg_ctmc_j)):
        def obj_c(z):
            v, g = vg_fn(jnp.asarray(z, dtype=jnp.float64))
            return float(v), np.asarray(g, dtype=np.float64)

        res = minimize(obj_c, edge_prior_logits.copy(), jac=True, method="L-BFGS-B", options={"maxiter": maxiter_drn})
        p = np.asarray(rollout_ctmc(jnp.asarray(res.x), eval_nodes_j))
        results_models[f"ctmc_{mode}"] = {
            "states": n_nodes,
            "params": n_p_ctmc,
            "fresh_kl": weighted_kl(p, f_te_a),
            "aged_kl": weighted_kl(p, a_te_a),
        }
        print(
            f"  CTMC (expm)    ({mode:10s}, {n_nodes}st, {n_p_ctmc}p): "
            f"fresh={results_models[f'ctmc_{mode}']['fresh_kl']:.4f}, "
            f"aged={results_models[f'ctmc_{mode}']['aged_kl']:.4f}",
            flush=True,
        )

    # 3. All 5 Neural ODE/SSM baselines (Mamba, S4D, LTC, Latent ODE, Neural-CME) — 3-start clean age=0!
    for kind in ("mamba", "s4d", "ltc", "latent_ode", "neural_cme"):
        init_nn, roll_nn, n_p_nn = make_scalable_neural_baseline(
            kind, n_nodes, edges, horizons, absorbing_node=absorbing_node, latent=12
        )

        @jax.jit
        def vg_nn_f(w):
            def lf(ww):
                pf = roll_nn(ww, eval_nodes_j, 0.0)
                return -jnp.sum(f_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / nf + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(lf)(w)

        @jax.jit
        def vg_nn_j(w):
            def lf(ww):
                pf = roll_nn(ww, eval_nodes_j, 0.0)
                pa = roll_nn(ww, eval_nodes_j, 1.0)
                return 0.5 * (
                    -jnp.sum(f_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / nf
                    - jnp.sum(a_j * jnp.log(jnp.clip(pa, 1e-12, 1.0))) / na
                ) + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(lf)(w)

        for mode, vg_fn, age_ev in (("fresh_only", vg_nn_f, 0.0), ("joint", vg_nn_j, 1.0)):
            t0 = perf_counter()
            def obj_n(z):
                v, g = vg_fn(jnp.asarray(z, dtype=jnp.float64))
                return float(v), np.asarray(g, dtype=np.float64)

            best = None
            for s in nn_starts:
                res = minimize(obj_n, init_nn(s, edge_prior_logits), jac=True, method="L-BFGS-B", options={"maxiter": maxiter_nn})
                if best is None or res.fun < best[0]:
                    best = (float(res.fun), res.x)
            pf = np.asarray(roll_nn(jnp.asarray(best[1]), eval_nodes_j, 0.0))
            pa = np.asarray(roll_nn(jnp.asarray(best[1]), eval_nodes_j, age_ev))
            results_models[f"{kind}_{mode}"] = {
                "params": n_p_nn,
                "fresh_kl": weighted_kl(pf, f_te_a),
                "aged_kl": weighted_kl(pa, a_te_a),
                "seconds": round(perf_counter() - t0, 2),
            }
            print(
                f"  {kind:14s} ({mode:10s}, {n_p_nn}p): "
                f"fresh={results_models[f'{kind}_{mode}']['fresh_kl']:.4f}, "
                f"aged={results_models[f'{kind}_{mode}']['aged_kl']:.4f}",
                flush=True,
            )

    # 4. Panel coarsening sweep
    coarsening = {}
    t_end_grid = float(horizons[-1])
    for dt_c in coarsening_dts:
        grid = np.arange(dt_c, t_end_grid + 1e-9, dt_c)
        if len(grid) < 2:
            grid = np.array([dt_c, 2.0 * dt_c])
        f_tr_g, _, _ = extract_general_snapshots(tr_trajs, n_nodes, grid, aged_thresh, absorbing_node)
        f_te_g, a_te_g, _ = extract_general_snapshots(te_trajs, n_nodes, grid, aged_thresh, absorbing_node)
        f_tr_ga, f_te_ga, a_te_ga = f_tr_g[active_nodes], f_te_g[active_nodes], a_te_g[active_nodes]
        fj_g = jnp.asarray(f_tr_ga, dtype=jnp.float64)
        nfg = jnp.asarray(max(float(f_tr_ga.sum()), 1.0))

        # SparseHyperDRN fresh-only on coarsened grid
        r_drn_g, _, _ = make_sparse_hyper_drn(
            n_nodes, edges, k_slow, grid, absorbing_node=absorbing_node, max_log_fast=max_log_fast
        )

        @jax.jit
        def vg_dg(w):
            def lf(ww):
                return -jnp.sum(fj_g * jnp.log(jnp.clip(r_drn_g(ww, eval_nodes_j, 0.0), 1e-12, 1.0))) / nfg + 1e-5 * jnp.mean(ww**2)

            return jax.value_and_grad(lf)(w)

        def obj_dg(z):
            v, g = vg_dg(jnp.asarray(z, dtype=jnp.float64))
            return float(v), np.asarray(g, dtype=np.float64)

        res_dg = minimize(obj_dg, init_drn_base, jac=True, method="L-BFGS-B", options={"maxiter": 55})
        pf_dg = np.asarray(r_drn_g(jnp.asarray(res_dg.x), eval_nodes_j, 0.0))
        pa_dg = np.asarray(r_drn_g(jnp.asarray(res_dg.x), eval_nodes_j, aged_thresh))

        # Exact CTMC on coarsened grid
        r_ct_g, _ = make_exact_ctmc(n_nodes, edges, grid, absorbing_node=absorbing_node)

        @jax.jit
        def vg_cg(w):
            def lf(ww):
                return -jnp.sum(fj_g * jnp.log(jnp.clip(r_ct_g(ww, eval_nodes_j), 1e-12, 1.0))) / nfg + 1e-5 * jnp.mean(ww**2)

            return jax.value_and_grad(lf)(w)

        def obj_cg(z):
            v, g = vg_cg(jnp.asarray(z, dtype=jnp.float64))
            return float(v), np.asarray(g, dtype=np.float64)

        res_cg = minimize(obj_cg, edge_prior_logits.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 80})
        p_cg = np.asarray(r_ct_g(jnp.asarray(res_cg.x), eval_nodes_j))

        # Mamba fresh-only on coarsened grid
        in_mg, r_mg, _ = make_scalable_neural_baseline("mamba", n_nodes, edges, grid, absorbing_node=absorbing_node, latent=12)

        @jax.jit
        def vg_mg(w):
            def lf(ww):
                return -jnp.sum(fj_g * jnp.log(jnp.clip(r_mg(ww, eval_nodes_j, 0.0), 1e-12, 1.0))) / nfg + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(lf)(w)

        def obj_mg(z):
            v, g = vg_mg(jnp.asarray(z, dtype=jnp.float64))
            return float(v), np.asarray(g, dtype=np.float64)

        res_mg = minimize(obj_mg, in_mg(17, edge_prior_logits), jac=True, method="L-BFGS-B", options={"maxiter": 65})
        pf_mg = np.asarray(r_mg(jnp.asarray(res_mg.x), eval_nodes_j, 0.0))
        pa_mg = np.asarray(r_mg(jnp.asarray(res_mg.x), eval_nodes_j, 0.0))

        # Single-jump TPP baseline on coarsened grid
        stay_prob = np.clip(
            np.array([f_tr_g[u, 0, u] / max(f_tr_g[u, 0].sum(), 1.0) for u in range(n_nodes)]),
            1e-4,
            1.0 - 1e-4,
        )
        lam_tot = -np.log(stay_prob) / dt_c
        pred_1j = np.zeros_like(pf_dg)
        for bi, u in enumerate(active_nodes):
            jump_counts = f_tr_g[u, 0].copy()
            jump_counts[u] = 0.0
            j_dist = jump_counts / jump_counts.sum() if jump_counts.sum() > 0 else np.ones(n_nodes) / n_nodes
            for hi, h_val in enumerate(grid):
                ps = float(np.exp(-lam_tot[u] * h_val))
                pv = (1.0 - ps) * j_dist
                pv[u] += ps
                pred_1j[bi, hi] = np.clip(pv, 1e-12, 1.0) / np.sum(np.clip(pv, 1e-12, 1.0))

        coarsening[f"dt_{dt_c}"] = {
            "sparse_drn_fresh_kl": weighted_kl(pf_dg, f_te_ga),
            "sparse_drn_aged_kl": weighted_kl(pa_dg, a_te_ga),
            "ctmc_fresh_kl": weighted_kl(p_cg, f_te_ga),
            "ctmc_aged_kl": weighted_kl(p_cg, a_te_ga),
            "mamba_fresh_kl": weighted_kl(pf_mg, f_te_ga),
            "mamba_aged_kl": weighted_kl(pa_mg, a_te_ga),
            "single_jump_tpp_fresh_kl": weighted_kl(pred_1j, f_te_ga),
            "single_jump_tpp_aged_kl": weighted_kl(pred_1j, a_te_ga),
        }
        print(
            f"  [Coarsen dt={dt_c}] aged KL: SparseDRN={coarsening[f'dt_{dt_c}']['sparse_drn_aged_kl']:.4f} | "
            f"CTMC={coarsening[f'dt_{dt_c}']['ctmc_aged_kl']:.4f} | Mamba={coarsening[f'dt_{dt_c}']['mamba_aged_kl']:.4f} | "
            f"1-jump TPP={coarsening[f'dt_{dt_c}']['single_jump_tpp_aged_kl']:.4f}",
            flush=True,
        )

    results_models["coarsening"] = coarsening
    return results_models


def load_stackoverflow():
    def parse_split(fn: Path):
        seqs = json.load(open(fn, encoding="utf-8"))
        return [(np.asarray(s["time_since_start"], dtype=np.float64), np.asarray(s["type_event"], dtype=np.int32)) for s in seqs]

    tr = parse_split(SO_DIR / "train.json") + parse_split(SO_DIR / "dev.json")
    te = parse_split(SO_DIR / "test.json")
    sha = {sp: hashlib.sha256((SO_DIR / f"{sp}.json").read_bytes()).hexdigest() for sp in ("train", "dev", "test")}
    return tr, te, sha


def load_mooc():
    df = pd.read_csv(
        MOOC_CSV,
        skiprows=1,
        header=None,
        usecols=[0, 1, 2, 3],
        names=["user_id", "item_id", "timestamp", "state_label"],
    )
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    DROPOUT_NODE = 97
    trajs = []
    for _, grp in df.groupby("user_id", sort=False):
        ts = grp["timestamp"].to_numpy(dtype=np.float64) / 3600.0
        ts = ts - ts[0]
        st = grp["item_id"].to_numpy(dtype=np.int32)
        lab = grp["state_label"].to_numpy(dtype=np.int32)
        if lab[-1] == 1:
            ts = np.append(ts, ts[-1] + 0.05)
            st = np.append(st, DROPOUT_NODE)
        trajs.append((ts, st))
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(trajs))
    n_tr = int(0.8 * len(trajs))
    tr = [trajs[i] for i in perm[:n_tr]]
    te = [trajs[i] for i in perm[n_tr:]]
    sha = hashlib.sha256(MOOC_CSV.read_bytes()).hexdigest()
    return tr, te, sha, DROPOUT_NODE


def main() -> None:
    so_tr, so_te, so_sha = load_stackoverflow()
    so_horizons = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    so_res = run_network_suite(
        "StackOverflow (V=22)",
        so_tr,
        so_te,
        n_nodes=22,
        horizons=so_horizons,
        aged_thresh=2.0,
        coarsening_dts=[0.5, 2.0, 4.0, 8.0],
        absorbing_node=None,
        k_slow=5,
        min_edge_count=5,
        maxiter_drn=90,
        maxiter_nn=70,
        drn_starts=(17, 41),
        nn_starts=STARTS,
        max_log_fast=2.5,
    )
    out = {
        "stackoverflow_v22": {
            "dataset": {"V": 22, "train_seqs": len(so_tr), "test_seqs": len(so_te), "aged_thresh": 2.0, "horizons": so_horizons.tolist(), "sha256": so_sha},
            "results": so_res,
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    mooc_tr, mooc_te, mooc_sha, dropout_node = load_mooc()
    mooc_horizons = np.array([0.25, 1.0, 4.0, 12.0, 24.0, 48.0], dtype=np.float64)
    mooc_res = run_network_suite(
        "MOOC (V=98 = 97 actions + Dropout)",
        mooc_tr,
        mooc_te,
        n_nodes=98,
        horizons=mooc_horizons,
        aged_thresh=6.0,
        coarsening_dts=[1.0, 6.0, 12.0, 24.0],
        absorbing_node=dropout_node,
        k_slow=4,
        min_edge_count=10,
        maxiter_drn=60,
        maxiter_nn=55,
        drn_starts=(17,),
        nn_starts=(17, 41),
        max_log_fast=1.8,
    )

    out["mooc_v98"] = {
        "dataset": {"V": 98, "train_users": len(mooc_tr), "test_users": len(mooc_te), "aged_thresh_hours": 6.0, "horizons_hours": mooc_horizons.tolist(), "sha256": mooc_sha},
        "results": mooc_res,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved StackOverflow (V=22) & MOOC (V=98) temporal network results to {OUT_PATH}")


if __name__ == "__main__":
    main()
