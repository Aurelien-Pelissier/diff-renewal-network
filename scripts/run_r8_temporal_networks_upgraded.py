"""Round-8/9 Upgrades on Real Large-V Temporal Interaction Networks:
  - Upgrade #1: Stiffness- & K-independent Laplace-Schur-Talbot exact solver (Talbot-DRN)
    on MOOC (V=98, 411k events, 1,355 sparse edges) — removes fast-rate stiffness cap
    (lam_f up to 200/h), verified against exact 490x490 dense expm & implicit SDIRK2!
  - Upgrade #2: Operationalizing Proposition 3 via Persistent Latent-Speed Frailty-DRN
    (Pure Mixed-Erlang G=1 control, Pure Frailty G=3, Hybrid Hyper-Frailty G=2).
  - Upgrade #4a: Data-free Universal Hyper-Pos-Sens (K_r=8, calibrated once on k in [1.4, 7.8])
    plugged ZERO-SHOT into real V=98 MOOC and V=22 StackOverflow graphs via Laplace-Schur-Talbot.
  - Upgrade #5a: Converged High-Capacity 2-layer AdamW Mamba (~136k params) on MOOC +
    Harmonized 5-model panel coarsening (Talbot-DRN, Univ. Hyper-Pos-Sens, 1-Jump Hawkes
    with right-censored NLL, CTMC, and 1L-Mamba) evaluated on the exact same test horizon grid!
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    calibrate_universal_pos_clock,
    laplace_schur_implicit_solve,
    smooth_erlang_logits,
    solve_coxian_hyper_talbot,
    solve_frailty_renewal_talbot,
    solve_renewal_talbot,
    universal_pos_clock,
)
from scripts.run_r7_temporal_networks import (  # noqa: E402
    extract_general_snapshots,
    load_mooc,
    load_stackoverflow,
    make_exact_ctmc,
    make_scalable_neural_baseline,
    weighted_kl,
)

OUT_PATH = ROOT / "results" / "archived" / "r8_temporal_networks_upgraded.json"
SO_DIR = Path(os.environ.get("DRN_SO_DIR", "/tmp/drn_r8_data/stackoverflow"))
MOOC_CSV = Path(os.environ.get("DRN_MOOC_CSV", "/tmp/drn_r8_data/mooc/mooc.csv"))
STARTS = (17, 41, 89)


def prop3_persistence_diagnostic(trajs: list[tuple[np.ndarray, np.ndarray]], n_nodes: int, absorbing_node: int | None = None) -> dict:
    """Directly measure Proposition 3 cross-visit speed persistence along user trajectories.

    Computes lag-1 within-user correlation of log-sojourn residuals (after subtracting node-specific
    mean log-sojourn) and held-out Gaussian/log-normal log-likelihood gain (nats per 100 visits)
    of predicting next-visit log-sojourn from previous-visit residual!
    """
    node_durs = [[] for _ in range(n_nodes)]
    seq_pairs = []  # list of lists of (node, log_dur)
    for times, states in trajs:
        if len(times) < 3:
            continue
        chg = np.concatenate([[0], np.where(states[1:] != states[:-1])[0] + 1])
        s_starts = times[chg]
        s_states = states[chg]
        s_ends = np.concatenate([s_starts[1:], [times[-1]]])
        visits = []
        for idx in range(len(s_starts) - 1):  # strictly completed visits
            u = int(s_states[idx])
            if absorbing_node is not None and u == absorbing_node:
                continue
            d = float(s_ends[idx] - s_starts[idx])
            if d > 1e-6:
                ld = np.log(d)
                node_durs[u].append(ld)
                visits.append((u, ld))
        if len(visits) >= 2:
            seq_pairs.append(visits)

    all_ld = np.concatenate([np.asarray(v) for v in node_durs if len(v) > 0])
    global_mean = float(np.mean(all_ld))
    node_mean = np.array([float(np.mean(v)) if len(v) >= 5 else global_mean for v in node_durs])

    # Split users 50/50 for out-of-sample lag-1 persistence regression gain
    xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
    all_raw_durs = []
    for s_idx, visits in enumerate(seq_pairs):
        res = [ld - node_mean[u] for u, ld in visits]
        for k in range(len(res) - 1):
            all_raw_durs.append(np.exp(visits[k][1]))
            if s_idx % 2 == 0:
                xs_tr.append(res[k]); ys_tr.append(res[k + 1])
            else:
                xs_te.append(res[k]); ys_te.append(res[k + 1])
    xs_tr, ys_tr = np.asarray(xs_tr), np.asarray(ys_tr)
    xs_te, ys_te = np.asarray(xs_te), np.asarray(ys_te)
    all_raw_durs = np.asarray(all_raw_durs)

    corr = float(np.corrcoef(np.concatenate([xs_tr, xs_te]), np.concatenate([ys_tr, ys_te]))[0, 1])
    # Fit AR(1) slope on train, evaluate Gaussian log-likelihood gain per 100 visits on test
    beta = float(np.dot(xs_tr, ys_tr) / np.maximum(np.dot(xs_tr, xs_tr), 1e-12))
    var0 = float(np.mean(ys_tr**2))
    var1 = float(np.mean((ys_tr - beta * xs_tr) ** 2))
    nll0 = 0.5 * (np.log(2 * np.pi * var0) + ys_te**2 / var0)
    nll1 = 0.5 * (np.log(2 * np.pi * var1) + (ys_te - beta * xs_te) ** 2 / var1)
    gain_per_100 = float(100.0 * np.mean(nll0 - nll1))
    # Also compute variance decomposition: between-user vs within-user variance of log-sojourn residuals!
    user_means = []
    within_sq = []
    for visits in seq_pairs:
        if len(visits) >= 4:
            res = np.array([ld - node_mean[u] for u, ld in visits])
            um = float(np.mean(res))
            user_means.append(um)
            within_sq.append(float(np.mean((res - um) ** 2)))
    between_var = float(np.var(user_means))
    within_var = float(np.mean(within_sq))
    icc = float(between_var / max(between_var + within_var, 1e-12))
    return {
        "n_pairs": int(len(xs_tr) + len(xs_te)),
        "sojourn_cv": float(np.std(all_raw_durs) / max(np.mean(all_raw_durs), 1e-12)),
        "lag1_residual_corr": corr,
        "ar1_beta": beta,
        "heldout_ll_gain_per_100_visits": gain_per_100,
        "between_user_icc": icc,
    }


def make_talbot_hyper_drn(
    n_nodes: int,
    edges: list[tuple[int, int]],
    k_slow: int,
    horizons: np.ndarray,
    absorbing_node: int | None = None,
    max_log_fast: float = 5.3,  # up to ~200/h — STIFFNESS-FREE via Laplace-Schur-Talbot!
    n_talbot: int = 16,
):
    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    n_edges = len(edges)
    is_abs = np.zeros(n_nodes, dtype=bool)
    if absorbing_node is not None:
        is_abs[absorbing_node] = True
    active_mask = jnp.asarray((~is_abs).astype(np.float64))
    n_params = 4 * n_nodes + n_edges

    def unpack(th):
        lam_s = jnp.exp(jnp.clip(th[n_nodes : 2 * n_nodes], -7.0, 2.5)) * active_mask
        lam_f = (lam_s + jnp.exp(jnp.clip(th[:n_nodes], -6.0, max_log_fast))) * active_mask
        c_s = (k_slow - 1.0) * jax.nn.sigmoid(th[2 * n_nodes : 3 * n_nodes])
        w_slow = jax.vmap(lambda c: jax.nn.softmax(smooth_erlang_logits(c, stages=k_slow, width=0.5)))(c_s)
        b_slow = w_slow[:, ::-1]
        alpha_f = jax.nn.sigmoid(th[3 * n_nodes : 4 * n_nodes]) * active_mask

        el = th[4 * n_nodes : 4 * n_nodes + n_edges]
        el_max = jax.ops.segment_max(el, src_idx, num_segments=n_nodes)
        exp_el = jnp.exp(jnp.clip(el - el_max[src_idx], -40.0, 0.0))
        denom = jax.ops.segment_sum(exp_el, src_idx, num_segments=n_nodes)
        r_w = exp_el / jnp.maximum(denom[src_idx], 1e-30)
        R_mat = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float64).at[dst_idx, src_idx].add(r_w)
        return lam_f, lam_s, alpha_f, b_slow, R_mat

    def rollout_both(th, eval_nodes: jax.Array, age: float, *, return_both: bool = True):
        lam_f, lam_s, alpha_f, b_slow, R_mat = unpack(th)
        return solve_renewal_talbot(
            lam_f, lam_s, alpha_f, b_slow, R_mat, horizons, active_mask,
            eval_nodes=eval_nodes, age=age if return_both else 0.0,
            n_talbot=n_talbot, return_both_fresh_and_aged=return_both,
        )

    return rollout_both, n_params


def make_talbot_frailty_drn(
    n_nodes: int,
    edges: list[tuple[int, int]],
    k_slow: int,
    horizons: np.ndarray,
    *,
    n_classes: int = 3,
    pure_frailty: bool = False,
    absorbing_node: int | None = None,
    max_log_fast: float = 5.3,
    n_talbot: int = 16,
):
    """Persistent latent-speed Frailty-DRN (Proposition 3):
      - `pure_frailty=True`: single mixed-Erlang clock per node + G persistent global speed
        multipliers gamma_g (never resetting across transitions!) + per-node class priors pi_{v,g}.
      - `pure_frailty=False`: hybrid 2-branch Hyper-Erlang + G persistent speed multipliers.
    """
    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    n_edges = len(edges)
    is_abs = np.zeros(n_nodes, dtype=bool)
    if absorbing_node is not None:
        is_abs[absorbing_node] = True
    active_mask = jnp.asarray((~is_abs).astype(np.float64))
    # Params: base clocks + routing + (if n_classes > 1: (n_classes - 1) log-speed steps + n_nodes * n_classes prior logits)
    base_p = (2 * n_nodes if pure_frailty else 4 * n_nodes) + n_edges
    n_params = base_p if n_classes == 1 else base_p + (n_classes - 1) + n_nodes * n_classes

    def unpack(th):
        if pure_frailty:
            lam_s = jnp.exp(jnp.clip(th[0:n_nodes], -6.0, 3.0)) * active_mask
            lam_f = lam_s
            c_s = (k_slow - 1.0) * jax.nn.sigmoid(th[n_nodes : 2 * n_nodes])
            w_slow = jax.vmap(lambda c: jax.nn.softmax(smooth_erlang_logits(c, stages=k_slow, width=0.5)))(c_s)
            b_slow = w_slow[:, ::-1]
            alpha_f = jnp.zeros(n_nodes, dtype=jnp.float64)
            el = th[2 * n_nodes : 2 * n_nodes + n_edges]
        else:
            lam_s = jnp.exp(jnp.clip(th[n_nodes : 2 * n_nodes], -7.0, 2.5)) * active_mask
            lam_f = (lam_s + jnp.exp(jnp.clip(th[:n_nodes], -6.0, max_log_fast))) * active_mask
            c_s = (k_slow - 1.0) * jax.nn.sigmoid(th[2 * n_nodes : 3 * n_nodes])
            w_slow = jax.vmap(lambda c: jax.nn.softmax(smooth_erlang_logits(c, stages=k_slow, width=0.5)))(c_s)
            b_slow = w_slow[:, ::-1]
            alpha_f = jax.nn.sigmoid(th[3 * n_nodes : 4 * n_nodes]) * active_mask
            el = th[4 * n_nodes : 4 * n_nodes + n_edges]

        el_max = jax.ops.segment_max(el, src_idx, num_segments=n_nodes)
        exp_el = jnp.exp(jnp.clip(el - el_max[src_idx], -40.0, 0.0))
        denom = jax.ops.segment_sum(exp_el, src_idx, num_segments=n_nodes)
        r_w = exp_el / jnp.maximum(denom[src_idx], 1e-30)
        R_mat = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float64).at[dst_idx, src_idx].add(r_w)

        if n_classes == 1:
            gammas = jnp.ones(1, dtype=jnp.float64)
            pi_w = jnp.ones((n_nodes, 1), dtype=jnp.float64)
        else:
            # Ordered positive speed multipliers centered around 1.0:
            d_log_g = jax.nn.softplus(th[base_p : base_p + (n_classes - 1)]) + 0.15
            log_g_raw = jnp.concatenate([jnp.zeros(1), jnp.cumsum(d_log_g)])
            log_g = log_g_raw - jnp.mean(log_g_raw)
            gammas = jnp.exp(jnp.clip(log_g, -5.0, 5.0))
            pi_logits = th[base_p + (n_classes - 1) :].reshape(n_nodes, n_classes)
            pi_w = jax.nn.softmax(pi_logits, axis=-1)
        return lam_f, lam_s, alpha_f, b_slow, R_mat, gammas, pi_w

    def rollout_both(th, eval_nodes: jax.Array, age: float, *, return_both: bool = True):
        lam_f, lam_s, alpha_f, b_slow, R_mat, gammas, pi_w = unpack(th)
        return solve_frailty_renewal_talbot(
            lam_f, lam_s, alpha_f, b_slow, R_mat, gammas, pi_w, horizons, active_mask,
            eval_nodes=eval_nodes, age=age if return_both else 0.0,
            n_talbot=n_talbot, return_both_fresh_and_aged=return_both,
        )

    return rollout_both, n_params, base_p


def make_universal_hyper_pos_talbot(
    n_nodes: int,
    edges: list[tuple[int, int]],
    horizons: np.ndarray,
    psi_univ: np.ndarray,
    *,
    stages_red: int = 8,
    absorbing_node: int | None = None,
    n_talbot: int = 16,
):
    """Data-free Universal Hyper-Pos-Sens (1D Pos-Sens Coxian + fast exp) via Laplace-Schur-Talbot."""
    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    n_edges = len(edges)
    is_abs = np.zeros(n_nodes, dtype=bool)
    if absorbing_node is not None:
        is_abs[absorbing_node] = True
    active_mask = jnp.asarray((~is_abs).astype(np.float64))
    psi_j = jnp.asarray(psi_univ)
    n_params = 4 * n_nodes + n_edges

    def unpack(th):
        slow_m = jnp.exp(jnp.clip(th[n_nodes : 2 * n_nodes], -2.5, 7.0))
        # Ensure fast branch is strictly faster than slow branch:
        lam_s_eff = 1.0 / slow_m
        lam_f = (lam_s_eff + jnp.exp(jnp.clip(th[:n_nodes], -6.0, 5.3))) * active_mask
        slow_k = 1.4 + 6.4 * jax.nn.sigmoid(th[2 * n_nodes : 3 * n_nodes])  # strictly inside certified [1.4, 7.8] cone!
        alpha_f = jax.nn.sigmoid(th[3 * n_nodes : 4 * n_nodes]) * active_mask
        stage_rates, b_slow = jax.vmap(
            lambda m, k: universal_pos_clock(m, k, psi_j, stages_red=stages_red)
        )(slow_m, slow_k)
        stage_rates = stage_rates * active_mask[:, None]

        el = th[4 * n_nodes : 4 * n_nodes + n_edges]
        el_max = jax.ops.segment_max(el, src_idx, num_segments=n_nodes)
        exp_el = jnp.exp(jnp.clip(el - el_max[src_idx], -40.0, 0.0))
        denom = jax.ops.segment_sum(exp_el, src_idx, num_segments=n_nodes)
        r_w = exp_el / jnp.maximum(denom[src_idx], 1e-30)
        R_mat = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float64).at[dst_idx, src_idx].add(r_w)
        return lam_f, stage_rates, alpha_f, b_slow, R_mat

    def rollout_both(th, eval_nodes: jax.Array, age: float, *, return_both: bool = True):
        lam_f, stage_rates, alpha_f, b_slow, R_mat = unpack(th)
        return solve_coxian_hyper_talbot(
            lam_f, stage_rates, alpha_f, b_slow, R_mat, horizons, active_mask,
            eval_nodes=eval_nodes, age=age if return_both else 0.0,
            n_talbot=n_talbot, return_both_fresh_and_aged=return_both,
        )

    return rollout_both, n_params


def fit_talbot_model(rollout_both, th0, eval_nodes, age, fr_tr, ag_tr, fr_te, ag_te, *, joint: bool, maxiter: int = 85, reg: float = 1e-4):
    fr_j = jnp.asarray(fr_tr)
    ag_j = jnp.asarray(ag_tr)
    n_fr = float(np.maximum(fr_tr.sum(), 1.0))
    n_ag = float(np.maximum(ag_tr.sum(), 1.0))

    @jax.jit
    def vg(th):
        def loss(z):
            if joint:
                p_fr, p_ag = rollout_both(z, eval_nodes, age, return_both=True)
                l_fr = -jnp.sum(fr_j * jnp.log(jnp.clip(p_fr, 1e-12, 1.0))) / n_fr
                l_ag = -jnp.sum(ag_j * jnp.log(jnp.clip(p_ag, 1e-12, 1.0))) / n_ag
                return 0.5 * (l_fr + l_ag) + reg * jnp.mean(z**2)
            p_fr = rollout_both(z, eval_nodes, 0.0, return_both=False)
            l_fr = -jnp.sum(fr_j * jnp.log(jnp.clip(p_fr, 1e-12, 1.0))) / n_fr
            return l_fr + reg * jnp.mean(z**2)

        return jax.value_and_grad(loss)(th)

    # Warmup + exact warm grad latency timing
    v0, g0 = vg(jnp.asarray(th0))
    v0.block_until_ready()
    t_g0 = perf_counter()
    v1, g1 = vg(jnp.asarray(th0))
    v1.block_until_ready()
    warm_grad_ms = (perf_counter() - t_g0) * 1000.0

    def obj(z):
        val, grad = vg(jnp.asarray(z, dtype=jnp.float64))
        return float(val), np.asarray(grad, dtype=np.float64)

    t0 = perf_counter()
    res = minimize(
        obj,
        th0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-8},
    )
    dt = perf_counter() - t0
    p_fr_te, p_ag_te = rollout_both(jnp.asarray(res.x), eval_nodes, age, return_both=True)
    return {
        "fresh_kl": weighted_kl(np.asarray(p_fr_te), fr_te),
        "aged_kl": weighted_kl(np.asarray(p_ag_te), ag_te),
        "warm_grad_ms": round(warm_grad_ms, 2),
        "seconds": round(dt, 2),
        "nit": int(res.nit),
    }, res.x


def run_deep_adamw_mamba_mooc(
    trajs_tr: list[tuple[np.ndarray, np.ndarray]],
    n_nodes: int,
    edges: list[tuple[int, int]],
    horizons: np.ndarray,
    aged_thresh: float,
    eval_nodes: jax.Array,
    fr_tr: np.ndarray,
    ag_tr: np.ndarray,
    fr_te: np.ndarray,
    ag_te: np.ndarray,
    absorbing_node: int | None = None,
) -> dict:
    """Upgrade #5a: Properly tuned high-capacity 2-layer Mamba (~136k params, H=64, AdamW + grad-clip + val early stopping).

    Uses exact horizon-aligned segments (matching 1-layer Mamba), exact absorbing-node flux
    masking, weights-only decoupled AdamW, global gradient-norm clipping (1.0), and an internal
    85/15 training-user validation split to select (lr, wd, best_step) with zero test leakage.
    """
    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    n_edges = len(edges)
    is_abs = np.zeros(n_nodes, dtype=bool)
    if absorbing_node is not None:
        is_abs[absorbing_node] = True
    edge_active = jnp.asarray([0.0 if is_abs[u] else 1.0 for u, _ in edges], dtype=jnp.float64)

    # Exact horizon-aligned substeps (zero timestamp rounding mismatch at t=0.25h!)
    seg_dts = np.diff(np.concatenate([[0.0], horizons]))
    t_max = float(horizons[-1])
    seg_steps = [max(3, int(np.ceil(sd / (t_max / 48.0)))) for sd in seg_dts]
    step_dts = []
    sample_idx_list = []
    cur = 0
    for sd, ns in zip(seg_dts, seg_steps):
        step_dts.extend([float(sd / ns)] * ns)
        cur += ns
        sample_idx_list.append(cur)
    step_dts_j = jnp.asarray(step_dts, dtype=jnp.float64)
    sample_idx = jnp.asarray(sample_idx_list, dtype=jnp.int32)

    H = 64  # 2-layer stacked selective SSM -> 136,331 parameters
    eval_init = jnp.eye(n_nodes, dtype=jnp.float64)[eval_nodes]
    eval_nodes_np = np.asarray(eval_nodes)

    # Leak-free internal 85/15 user split of trajs_tr for LR/WD/early-stopping selection
    rng_sp = np.random.default_rng(123)
    perm = rng_sp.permutation(len(trajs_tr))
    n_val = max(1, int(0.15 * len(trajs_tr)))
    val_tr = [trajs_tr[i] for i in perm[:n_val]]
    sub_tr = [trajs_tr[i] for i in perm[n_val:]]
    fr_sub, ag_sub, _ = extract_general_snapshots(sub_tr, n_nodes, horizons, aged_thresh, absorbing_node)
    fr_val, ag_val, _ = extract_general_snapshots(val_tr, n_nodes, horizons, aged_thresh, absorbing_node)
    fr_sub, ag_sub = fr_sub[eval_nodes_np], ag_sub[eval_nodes_np]
    fr_val, ag_val = fr_val[eval_nodes_np], ag_val[eval_nodes_np]

    def init_params(seed: int):
        r = np.random.default_rng(seed)
        return {
            "enc_w": jnp.asarray(r.normal(0, 0.08, (n_nodes + 1, H))),
            "enc_b": jnp.zeros(H),
            # Layer 1
            "log_a1": jnp.asarray(np.linspace(np.log(0.05), np.log(5.0), H)),
            "w_dt1": jnp.asarray(r.normal(0, 0.08, (n_nodes, H))),
            "b_dt1": jnp.full(H, -1.0),
            "w_b1": jnp.asarray(r.normal(0, 0.08, (n_nodes, H))),
            "w_c1": jnp.asarray(r.normal(0, 0.08, (n_nodes, H))),
            "w_z1": jnp.asarray(r.normal(0, 0.08, (n_nodes, H))),
            "b_z1": jnp.zeros(H),
            # Layer 2
            "log_a2": jnp.asarray(np.linspace(np.log(0.05), np.log(5.0), H)),
            "w_dt2": jnp.asarray(r.normal(0, 0.08, (H, H))),
            "b_dt2": jnp.full(H, -1.0),
            "w_b2": jnp.asarray(r.normal(0, 0.08, (H, H))),
            "w_c2": jnp.asarray(r.normal(0, 0.08, (H, H))),
            "w_z2": jnp.asarray(r.normal(0, 0.08, (H, H))),
            "b_z2": jnp.zeros(H),
            # Edge flux head: H -> n_edges (86,720 params)
            "w_o": jnp.asarray(r.normal(0, 0.02, (H, n_edges))),
            "b_o": jnp.full(n_edges, -3.0),
        }

    p0_sample = init_params(17)
    n_params = sum(int(v.size) for v in p0_sample.values())
    # Weights-only decay mask (never decay biases or log-rates)
    wd_mask = {k: (0.0 if (k.startswith("b_") or k.endswith("_b") or k.startswith("log_")) else 1.0) for k in p0_sample}

    def rollout(p, age_val: float):
        a_col = jnp.full((eval_init.shape[0], 1), age_val, dtype=jnp.float64)
        h1_0 = jnp.tanh(jnp.concatenate([eval_init, a_col], axis=-1) @ p["enc_w"] + p["enc_b"])
        h2_0 = jnp.zeros_like(h1_0)
        na1 = -jnp.exp(jnp.clip(p["log_a1"], -4.0, 3.0))
        na2 = -jnp.exp(jnp.clip(p["log_a2"], -4.0, 3.0))

        def step(carry, dt_step):
            prob, h1, h2 = carry
            d1 = jax.nn.softplus(prob @ p["w_dt1"] + p["b_dt1"]) * dt_step
            ab1 = jnp.exp(d1 * na1[None, :])
            bb1 = ((ab1 - 1.0) / na1[None, :]) * jnp.tanh(prob @ p["w_b1"])
            h1_n = ab1 * h1 + bb1
            y1 = jnp.tanh(prob @ p["w_c1"]) * h1_n * jax.nn.silu(prob @ p["w_z1"] + p["b_z1"])

            d2 = jax.nn.softplus(y1 @ p["w_dt2"] + p["b_dt2"]) * dt_step
            ab2 = jnp.exp(d2 * na2[None, :])
            bb2 = ((ab2 - 1.0) / na2[None, :]) * jnp.tanh(y1 @ p["w_b2"])
            h2_n = ab2 * h2 + bb2
            y2 = y1 + jnp.tanh(y1 @ p["w_c2"]) * h2_n * jax.nn.silu(y1 @ p["w_z2"] + p["b_z2"])

            rates = jnp.exp(jnp.clip(p["b_o"] + y2 @ p["w_o"], -8.0, 3.0)) * edge_active
            tot_out = jax.vmap(lambda r: jax.ops.segment_sum(r, src_idx, num_segments=n_nodes))(rates)
            frac_leave = 1.0 - jnp.exp(-jnp.minimum(tot_out * dt_step, 50.0))
            scale = frac_leave / jnp.maximum(tot_out, 1e-30)
            edge_mass = prob[:, src_idx] * scale[:, src_idx] * rates
            p_new = prob * (1.0 - frac_leave)
            p_new = p_new.at[:, dst_idx].add(edge_mass)
            p_new = jnp.maximum(p_new, 1e-12)
            p_new = p_new / jnp.sum(p_new, axis=-1, keepdims=True)
            return (p_new, h1_n, h2_n), p_new

        _, traj = jax.lax.scan(step, (eval_init, h1_0, h2_0), step_dts_j)
        full = jnp.concatenate([eval_init[None], traj], axis=0)
        return jnp.swapaxes(full[sample_idx], 0, 1)

    roll_jit = jax.jit(rollout)

    res_out = {"params": int(n_params), "layers": 2, "hidden": H}
    for mode, joint in [("fresh_only", False), ("joint", True)]:
        @jax.jit
        def step_adamw(params, m_st, v_st, fr_b, ag_b, n_f, n_a, step_i, lr, wd):
            def loss(p):
                pf = rollout(p, 0.0)
                lf = -jnp.sum(fr_b * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_f
                if joint:
                    pa = rollout(p, 1.0)
                    la = -jnp.sum(ag_b * jnp.log(jnp.clip(pa, 1e-12, 1.0))) / n_a
                    return 0.5 * (lf + la)
                return lf

            val, grads = jax.value_and_grad(loss)(params)
            # Global L2 gradient-norm clipping at 1.0
            gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)) + 1e-12)
            gscale = jnp.minimum(1.0, 1.0 / gnorm)
            grads = jax.tree.map(lambda g: g * gscale, grads)
            b1, b2, eps = 0.9, 0.999, 1e-8
            m_new = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, m_st, grads)
            v_new = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), v_st, grads)
            bc1 = 1.0 - b1 ** (step_i + 1)
            bc2 = 1.0 - b2 ** (step_i + 1)
            p_new = {
                k: params[k] * (1.0 - lr * wd * wd_mask[k]) - lr * (m_new[k] / bc1) / (jnp.sqrt(v_new[k] / bc2) + eps)
                for k in params
            }
            return p_new, m_new, v_new, val

        t0 = perf_counter()
        fr_sub_j, ag_sub_j = jnp.asarray(fr_sub), jnp.asarray(ag_sub)
        nf_s, na_s = float(max(fr_sub.sum(), 1.0)), float(max(ag_sub.sum(), 1.0))

        # Select (lr, wd, best_step) strictly on internal 15% user validation split
        # Expanded interior grid (up to 500 steps, wd down to 1e-5) so joint converges strictly inside grid!
        best_cfg = (3e-3, 1e-4, 250)
        best_val_kl = 1e9
        max_tune = 480 if joint else 300
        lr_grid = (3e-3, 1e-2) if joint else (1e-3, 3e-3, 1e-2)
        wd_grid = (1e-5, 1e-4, 1e-3) if joint else (1e-4, 1e-3)
        for lr_c in lr_grid:
            for wd_c in wd_grid:
                p = init_params(STARTS[0])
                m_st = jax.tree.map(jnp.zeros_like, p)
                v_st = jax.tree.map(jnp.zeros_like, p)
                for it in range(max_tune):
                    lr_t = (0.05 * lr_c) + 0.5 * (lr_c - 0.05 * lr_c) * (1.0 + np.cos(np.pi * it / max_tune))
                    p, m_st, v_st, _ = step_adamw(p, m_st, v_st, fr_sub_j, ag_sub_j, nf_s, na_s, it, lr_t, wd_c)
                    if (it + 1) % 30 == 0 or it == max_tune - 1:
                        vf = weighted_kl(np.asarray(roll_jit(p, 0.0)), fr_val)
                        v_score = 0.5 * (vf + weighted_kl(np.asarray(roll_jit(p, 1.0)), ag_val)) if joint else vf
                        if v_score < best_val_kl:
                            best_val_kl = v_score
                            best_cfg = (lr_c, wd_c, it + 1)

        lr_star, wd_star, steps_star = best_cfg
        fr_full_j, ag_full_j = jnp.asarray(fr_tr), jnp.asarray(ag_tr)
        nf_f, na_f = float(max(fr_tr.sum(), 1.0)), float(max(ag_tr.sum(), 1.0))
        best = None
        for sd in STARTS:
            p = init_params(sd)
            m_st = jax.tree.map(jnp.zeros_like, p)
            v_st = jax.tree.map(jnp.zeros_like, p)
            last_v = 1e9
            for it in range(steps_star):
                lr_t = (0.1 * lr_star) + 0.5 * (lr_star - 0.1 * lr_star) * (1.0 + np.cos(np.pi * it / steps_star))
                p, m_st, v_st, val = step_adamw(p, m_st, v_st, fr_full_j, ag_full_j, nf_f, na_f, it, lr_t, wd_star)
                last_v = float(val)
            if best is None or last_v < best[0]:
                best = (last_v, p)
        dt = perf_counter() - t0
        p_best = best[1]
        pred_fr = np.asarray(roll_jit(p_best, 0.0))
        # Clean age=0 evaluation for fresh_only (eliminating untrained age weight noise), age=1.0 for joint
        pred_ag = np.asarray(roll_jit(p_best, 1.0 if joint else 0.0))
        res_out[mode] = {
            "fresh_kl": weighted_kl(pred_fr, fr_te),
            "aged_kl": weighted_kl(pred_ag, ag_te),
            "best_lr": lr_star,
            "best_wd": wd_star,
            "best_steps": int(steps_star),
            "internal_val_kl": round(float(best_val_kl), 5),
            "seconds": round(dt, 2),
        }
        print(
            f"[MOOC Deep 2L-Mamba AdamW ({n_params:,}p, lr={lr_star}, wd={wd_star}, steps={steps_star})] {mode:10s}: "
            f"fresh={res_out[mode]['fresh_kl']:.4f} aged={res_out[mode]['aged_kl']:.4f} ({dt:.1f}s)",
            flush=True,
        )
    return res_out


def run_1jump_hawkes_coarsening_single_dt(
    trajs_tr: list[tuple[np.ndarray, np.ndarray]],
    n_nodes: int,
    eval_nodes: np.ndarray,
    ag_te_grid: np.ndarray,
    grid_eval: np.ndarray,
    dt_c: float,
    absorbing_node: int | None = None,
) -> dict:
    """1-Jump Parametric Hawkes baseline on coarsened grid (WITH exact right-censored stays in NLL!).

    Evaluates on the EXACT SAME test horizon grid `grid_eval = arange(dt_c, t_max, dt_c)` as
    Talbot-DRN, Univ. Hyper-Pos-Sens, CTMC, and 1L-Mamba!
    """
    pair_u, pair_v, pair_dt = [], [], []
    cens_u, cens_dt = [], []
    for times, states in trajs_tr:
        if len(times) < 2:
            continue
        t_end = float(times[-1])
        grid = np.arange(0.0, t_end + 1e-9, dt_c)
        if len(grid) < 2:
            continue
        pos = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(states) - 1)
        gs = states[pos]
        last_t = grid[0]
        last_s = int(gs[0])
        for k in range(1, len(grid)):
            cur_s = int(gs[k])
            if cur_s != last_s:
                if absorbing_node is None or last_s != absorbing_node:
                    pair_u.append(last_s)
                    pair_v.append(cur_s)
                    pair_dt.append(float(grid[k] - last_t))
                last_t = grid[k]
                last_s = cur_s
        # Include right-censored trailing stay [last_t, grid[-1]] so survival integral is unbiased!
        if grid[-1] > last_t + 1e-9 and (absorbing_node is None or last_s != absorbing_node):
            cens_u.append(last_s)
            cens_dt.append(float(grid[-1] - last_t))

    pair_u = np.asarray(pair_u, dtype=np.int32)
    pair_v = np.asarray(pair_v, dtype=np.int32)
    pair_dt = np.maximum(np.asarray(pair_dt, dtype=np.float64), 1e-4)
    cens_u = np.asarray(cens_u, dtype=np.int32)
    cens_dt = np.maximum(np.asarray(cens_dt, dtype=np.float64), 1e-4)

    rank = 8
    n_p = n_nodes * rank * 4 + 2 * n_nodes
    rng = np.random.default_rng(41)
    th0 = rng.normal(0.0, 0.15, size=n_p)
    # Subsample up to 40k transitions + proportional censored stays
    if len(pair_u) > 40000:
        frac = 40000.0 / len(pair_u)
        sel = rng.choice(len(pair_u), size=40000, replace=False)
        pu_j, pv_j, pdt_j = jnp.asarray(pair_u[sel]), jnp.asarray(pair_v[sel]), jnp.asarray(pair_dt[sel])
        n_c_sub = max(1, int(round(len(cens_u) * frac)))
        sel_c = rng.choice(len(cens_u), size=min(n_c_sub, len(cens_u)), replace=False) if len(cens_u) > 0 else np.array([], dtype=np.int32)
        cu_j, cdt_j = jnp.asarray(cens_u[sel_c]), jnp.asarray(cens_dt[sel_c])
    else:
        pu_j, pv_j, pdt_j = jnp.asarray(pair_u), jnp.asarray(pair_v), jnp.asarray(pair_dt)
        cu_j, cdt_j = jnp.asarray(cens_u), jnp.asarray(cens_dt)
    n_tot_obs = float(len(pu_j) + len(cu_j))
    diag_mask = 1.0 - jnp.eye(n_nodes, dtype=jnp.float64)

    def unpack_hk(th):
        s = n_nodes * rank
        u_mu = th[0:s].reshape(n_nodes, rank)
        v_mu = th[s : 2 * s].reshape(n_nodes, rank)
        u_al = th[2 * s : 3 * s].reshape(n_nodes, rank)
        v_al = th[3 * s : 4 * s].reshape(n_nodes, rank)
        b_mu = th[4 * s : 4 * s + n_nodes]
        log_beta = th[4 * s + n_nodes :]
        mu = jax.nn.softplus(u_mu @ v_mu.T + b_mu[:, None] - 2.0) * diag_mask + 1e-6 * diag_mask
        al = jax.nn.softplus(u_al @ v_al.T - 2.0) * diag_mask
        beta = jnp.exp(jnp.clip(log_beta, -4.0, 4.0))
        return mu, al, beta

    @jax.jit
    def vg_hk(th):
        def nll(z):
            mu, al, beta = unpack_hk(z)
            tot_mu = jnp.sum(mu, axis=-1)
            tot_al = jnp.sum(al, axis=-1)
            # Observed jumps
            bu = beta[pu_j]
            e_dec = jnp.exp(-jnp.minimum(bu * pdt_j, 50.0))
            lam_uv = mu[pu_j, pv_j] + al[pu_j, pv_j] * e_dec
            int_obs = tot_mu[pu_j] * pdt_j + tot_al[pu_j] * (1.0 - e_dec) / bu
            loss_obs = jnp.sum(-jnp.log(jnp.maximum(lam_uv, 1e-12)) + int_obs)
            # Right-censored stays (survival integral only, zero jump log-intensity)
            if cu_j.shape[0] > 0:
                bc = beta[cu_j]
                ec_dec = jnp.exp(-jnp.minimum(bc * cdt_j, 50.0))
                loss_cens = jnp.sum(tot_mu[cu_j] * cdt_j + tot_al[cu_j] * (1.0 - ec_dec) / bc)
            else:
                loss_cens = 0.0
            return (loss_obs + loss_cens) / max(n_tot_obs, 1.0) + 1e-3 * jnp.mean(z**2)

        return jax.value_and_grad(nll)(th)

    def obj_hk(z):
        v, g = vg_hk(jnp.asarray(z, dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(obj_hk, th0, method="L-BFGS-B", jac=True, options={"maxiter": 80})
    mu, al, beta = map(np.asarray, unpack_hk(jnp.asarray(res.x)))
    tot_mu = mu.sum(axis=-1)
    tot_al = al.sum(axis=-1)
    pred = np.zeros((len(eval_nodes), len(grid_eval), n_nodes), dtype=np.float64)
    for bi, u in enumerate(eval_nodes):
        for hi, h in enumerate(grid_eval):
            e_dec = np.exp(-min(beta[u] * h, 50.0))
            int_vec = mu[u] * h + al[u] * (1.0 - e_dec) / beta[u]
            int_tot = tot_mu[u] * h + tot_al[u] * (1.0 - e_dec) / beta[u]
            surv = np.exp(-min(int_tot, 50.0))
            route = int_vec / max(int_tot, 1e-12)
            p_row = (1.0 - surv) * route
            p_row[u] = surv
            pred[bi, hi] = np.clip(p_row, 1e-12, 1.0)
            pred[bi, hi] /= pred[bi, hi].sum()
    return {
        "hawkes_1jump_aged_kl": weighted_kl(pred, ag_te_grid),
        "params": int(n_p),
        "n_obs_transitions": int(len(pair_u)),
        "n_censored_stays": int(len(cens_u)),
    }


def verify_mooc_talbot_accuracy_and_solvers(
    n_nodes: int,
    edges: list[tuple[int, int]],
    k_slow: int,
    horizons: np.ndarray,
    aged_thresh: float,
    eval_nodes: jax.Array,
    th_fitted: np.ndarray,
    fr_tr: np.ndarray,
    ag_te: np.ndarray,
    absorbing_node: int,
) -> dict:
    """Verify Talbot-DRN (n=12, 16, 24) against EXACT dense 490x490 expm on fitted MOOC (lam_f ~ 200/h)
    and benchmark apples-to-apples warm gradient speed against dense expm and implicit SDIRK2!
    """
    from scipy.linalg import expm as sp_expm
    import jax.scipy.linalg as jl

    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    is_abs = np.zeros(n_nodes, dtype=bool)
    is_abs[absorbing_node] = True
    active_mask = jnp.asarray((~is_abs).astype(np.float64))
    n_edges = len(edges)
    eval_nodes_np = np.asarray(eval_nodes)

    def unpack_params(th):
        lam_s = jnp.exp(jnp.clip(th[n_nodes : 2 * n_nodes], -7.0, 2.5)) * active_mask
        lam_f = (lam_s + jnp.exp(jnp.clip(th[:n_nodes], -6.0, 5.3))) * active_mask
        c_s = (k_slow - 1.0) * jax.nn.sigmoid(th[2 * n_nodes : 3 * n_nodes])
        w_slow = jax.vmap(lambda c: jax.nn.softmax(smooth_erlang_logits(c, stages=k_slow, width=0.5)))(c_s)
        b_slow = w_slow[:, ::-1]
        alpha_f = jax.nn.sigmoid(th[3 * n_nodes : 4 * n_nodes]) * active_mask
        el = th[4 * n_nodes : 4 * n_nodes + n_edges]
        el_max = jax.ops.segment_max(el, src_idx, num_segments=n_nodes)
        exp_el = jnp.exp(jnp.clip(el - el_max[src_idx], -40.0, 0.0))
        denom = jax.ops.segment_sum(exp_el, src_idx, num_segments=n_nodes)
        r_w = exp_el / jnp.maximum(denom[src_idx], 1e-30)
        R_mat = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float64).at[dst_idx, src_idx].add(r_w)
        return lam_f, lam_s, alpha_f, b_slow, R_mat

    # Build exact dense 490x490 generator A_dense (K+1 = 5 phases/node * 98 = 490 states)
    D_tot = n_nodes * (k_slow + 1)

    def build_dense_rollout(th):
        lam_f, lam_s, alpha_f, b_slow, R_mat = unpack_params(th)
        P_blk = k_slow + 1
        # Entry vector per node: [alpha_f, (1-alpha_f)*b_slow]
        ent = jnp.concatenate([alpha_f[:, None], (1.0 - alpha_f)[:, None] * b_slow], axis=-1)  # (V, P)
        # Absorbing node entry [1, 0, ...]
        ent = jnp.where(active_mask[:, None] > 0.5, ent, jnp.zeros(P_blk).at[0].set(1.0)[None, :])
        # Build dense A: (V, P, V, P) reshaped to (D, D)
        A = jnp.zeros((n_nodes, P_blk, n_nodes, P_blk), dtype=jnp.float64)
        idx_v = jnp.arange(n_nodes)
        # Diagonal & subdiagonal within each node
        A = A.at[idx_v, 0, idx_v, 0].add(-lam_f)
        for j in range(k_slow):
            A = A.at[idx_v, 1 + j, idx_v, 1 + j].add(-lam_s)
            if j + 1 < k_slow:
                A = A.at[idx_v, 1 + j + 1, idx_v, 1 + j].add(lam_s)
        # Routed exits from phase 0 (rate lam_f) and phase k_slow (rate lam_s) into dst entry phases
        # exit_flux[u] = [lam_f[u], 0, ..., lam_s[u]]
        ex = jnp.zeros((n_nodes, P_blk), dtype=jnp.float64).at[:, 0].set(lam_f).at[:, k_slow].set(lam_s)
        # A[v, q, u, p] += R_mat[v, u] * ent[v, q] * ex[u, p]
        A = A + jnp.einsum("vu,vq,up->vqup", R_mat, ent, ex)
        A_mat = A.reshape(D_tot, D_tot)

        # Initial fresh vectors for eval_nodes: (D_tot, B_ev)
        B_fr = jnp.zeros((n_nodes, P_blk, len(eval_nodes_np)), dtype=jnp.float64)
        ev_idx = jnp.arange(len(eval_nodes_np))
        B_fr = B_fr.at[eval_nodes, :, ev_idx].set(ent[eval_nodes])
        B_fr = B_fr.reshape(D_tot, len(eval_nodes_np))

        def one_h(h):
            X = jl.expm(h * A_mat) @ B_fr  # (D_tot, B_ev)
            P = jnp.sum(X.reshape(n_nodes, P_blk, len(eval_nodes_np)), axis=1).T  # (B_ev, V)
            P = jnp.maximum(P, 1e-12)
            return P / jnp.sum(P, axis=-1, keepdims=True)

        return jnp.swapaxes(jax.vmap(one_h)(jnp.asarray(horizons)), 0, 1)

    th_j = jnp.asarray(th_fitted, dtype=jnp.float64)
    fr_tr_j = jnp.asarray(fr_tr, dtype=jnp.float64)
    n_fr = float(max(fr_tr.sum(), 1.0))

    # 1. Exact scipy dense expm reference (fresh AND 6h aged!)
    lam_f_np, lam_s_np, alpha_f_np, b_slow_np, R_np = map(np.asarray, unpack_params(th_j))
    P_blk = k_slow + 1
    ent_np = np.concatenate([alpha_f_np[:, None], (1.0 - alpha_f_np)[:, None] * b_slow_np], axis=-1)
    ent_np[absorbing_node] = np.eye(P_blk)[0]
    A_np = np.zeros((n_nodes, P_blk, n_nodes, P_blk), dtype=np.float64)
    for u in range(n_nodes):
        A_np[u, 0, u, 0] -= lam_f_np[u]
        for j in range(k_slow):
            A_np[u, 1 + j, u, 1 + j] -= lam_s_np[u]
            if j + 1 < k_slow:
                A_np[u, 1 + j + 1, u, 1 + j] += lam_s_np[u]
    ex_np = np.zeros((n_nodes, P_blk), dtype=np.float64)
    ex_np[:, 0] = lam_f_np
    ex_np[:, k_slow] = lam_s_np
    A_np += np.einsum("vu,vq,up->vqup", R_np, ent_np, ex_np)
    A_dense_np = A_np.reshape(D_tot, D_tot)

    # Build exact fresh & 6h aged phase vectors via local expm(age * T_u)
    B_fr_np = np.zeros((D_tot, len(eval_nodes_np)), dtype=np.float64)
    B_ag_np = np.zeros((D_tot, len(eval_nodes_np)), dtype=np.float64)
    for bi, u in enumerate(eval_nodes_np):
        B_fr_np[u * P_blk : (u + 1) * P_blk, bi] = ent_np[u]
        Tu = np.zeros((P_blk, P_blk), dtype=np.float64)
        Tu[0, 0] = -lam_f_np[u]
        for j in range(k_slow):
            Tu[1 + j, 1 + j] = -lam_s_np[u]
            if j + 1 < k_slow:
                Tu[1 + j + 1, 1 + j] = lam_s_np[u]
        # Shift by -lam_s_np[u] to prevent underflow
        Tu_s = Tu + lam_s_np[u] * np.eye(P_blk)
        v_ag = sp_expm(aged_thresh * Tu_s) @ ent_np[u]
        v_ag = np.maximum(v_ag, 0.0)
        v_ag /= max(v_ag.sum(), 1e-300)
        B_ag_np[u * P_blk : (u + 1) * P_blk, bi] = v_ag

    P_ref_fr = np.zeros((len(eval_nodes_np), len(horizons), n_nodes), dtype=np.float64)
    P_ref_ag = np.zeros((len(eval_nodes_np), len(horizons), n_nodes), dtype=np.float64)
    for hi, h in enumerate(horizons):
        Eh = sp_expm(float(h) * A_dense_np)
        Xfr = (Eh @ B_fr_np).reshape(n_nodes, P_blk, len(eval_nodes_np)).sum(axis=1).T
        Xag = (Eh @ B_ag_np).reshape(n_nodes, P_blk, len(eval_nodes_np)).sum(axis=1).T
        P_ref_fr[:, hi] = np.clip(Xfr, 1e-12, 1.0)
        P_ref_fr[:, hi] /= P_ref_fr[:, hi].sum(axis=-1, keepdims=True)
        P_ref_ag[:, hi] = np.clip(Xag, 1e-12, 1.0)
        P_ref_ag[:, hi] /= P_ref_ag[:, hi].sum(axis=-1, keepdims=True)

    # 2. Dense JAX expm exact reference gradient + warm grad timing
    @jax.jit
    def vg_dense_expm(th):
        def loss(z):
            pf = build_dense_rollout(z)
            return -jnp.sum(fr_tr_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_fr

        return jax.value_and_grad(loss)(th)

    v_ex, g_ex = vg_dense_expm(th_j)
    g_ex.block_until_ready()
    ex_times = []
    for _ in range(3):
        t0 = perf_counter()
        _, gx = vg_dense_expm(th_j)
        gx.block_until_ready()
        ex_times.append((perf_counter() - t0) * 1000.0)
    g_ex_np = np.asarray(g_ex)
    expm_grad_ms = float(np.median(ex_times))

    # 3. Implicit L-stable Alexander SDIRK2 (via laplace_schur_implicit_solve) accuracy + warm grad timing
    gamma_sd = 1.0 - 1.0 / np.sqrt(2.0)
    c1_sd = (1.0 - gamma_sd) / gamma_sd
    c2_sd = (2.0 * gamma_sd - 1.0) / gamma_sd
    n_sub = 6  # 6 SDIRK2 substeps per horizon interval = 36 implicit steps over [0, 48h]
    h_bounds = np.concatenate([[0.0], horizons])
    dt_segs = np.diff(h_bounds) / n_sub

    def sdirk2_rollout(th):
        lam_f, lam_s, alpha_f, b_slow, R_mat = unpack_params(th)
        ent = jnp.concatenate([alpha_f[:, None], (1.0 - alpha_f)[:, None] * b_slow], axis=-1)
        ent = jnp.where(active_mask[:, None] > 0.5, ent, jnp.zeros(k_slow + 1).at[0].set(1.0)[None, :])
        y0 = jnp.zeros((len(eval_nodes_np), n_nodes, k_slow + 1), dtype=jnp.float64)
        y0 = y0.at[jnp.arange(len(eval_nodes_np)), eval_nodes, :].set(ent[eval_nodes])
        preds = []
        y = y0
        for dt_s in dt_segs:
            gh = float(gamma_sd * dt_s)
            def step_fn(carry, _):
                y1 = laplace_schur_implicit_solve(carry, gh, lam_f, lam_s, alpha_f, b_slow, R_mat)
                rhs2 = c1_sd * y1 + c2_sd * carry
                y2 = laplace_schur_implicit_solve(rhs2, gh, lam_f, lam_s, alpha_f, b_slow, R_mat)
                return y2, None
            y, _ = jax.lax.scan(step_fn, y, None, length=n_sub)
            p_h = jnp.maximum(jnp.sum(y, axis=-1), 1e-12)
            preds.append(p_h / jnp.sum(p_h, axis=-1, keepdims=True))
        return jnp.stack(preds, axis=1)

    @jax.jit
    def vg_sdirk(th):
        def loss(z):
            pf = sdirk2_rollout(z)
            return -jnp.sum(fr_tr_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_fr

        return jax.value_and_grad(loss)(th)

    vs, gs = vg_sdirk(th_j)
    gs.block_until_ready()
    sd_times = []
    for _ in range(5):
        t0 = perf_counter()
        _, gs2 = vg_sdirk(th_j)
        gs2.block_until_ready()
        sd_times.append((perf_counter() - t0) * 1000.0)
    sdirk_grad_ms = float(np.median(sd_times))
    pf_sd = np.asarray(sdirk2_rollout(th_j))
    sdirk_fr_err = float(np.max(np.abs(pf_sd - P_ref_fr)))
    gs_np = np.asarray(gs)
    sdirk_grad_rel = float(np.linalg.norm(gs_np - g_ex_np) / max(np.linalg.norm(g_ex_np), 1e-12))
    print(f"  [MOOC SDIRK2 36-step] grad_ms={sdirk_grad_ms:.1f}ms fr_err={sdirk_fr_err:.2e} grad_rel={sdirk_grad_rel:.2e}", flush=True)

    # 4. Talbot-DRN accuracy & warm grad timing across n_talbot in {12, 16, 24}
    talbot_audit = {}
    for nt in (12, 16, 24):
        @jax.jit
        def vg_t(th, _nt=nt):
            def loss(z):
                lam_f, lam_s, alpha_f, b_slow, R_mat = unpack_params(z)
                pf = solve_renewal_talbot(
                    lam_f, lam_s, alpha_f, b_slow, R_mat, horizons, active_mask,
                    eval_nodes=eval_nodes, age=0.0, n_talbot=_nt, return_both_fresh_and_aged=False,
                )
                return -jnp.sum(fr_tr_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_fr

            return jax.value_and_grad(loss)(th)

        _, gt = vg_t(th_j)
        gt.block_until_ready()
        t_times = []
        for _ in range(5):
            t0 = perf_counter()
            _, gt2 = vg_t(th_j)
            gt2.block_until_ready()
            t_times.append((perf_counter() - t0) * 1000.0)
        gt_np = np.asarray(gt)
        lam_f, lam_s, alpha_f, b_slow, R_mat = unpack_params(th_j)
        pf_t, pa_t = solve_renewal_talbot(
            lam_f, lam_s, alpha_f, b_slow, R_mat, horizons, active_mask,
            eval_nodes=eval_nodes, age=aged_thresh, n_talbot=nt, return_both_fresh_and_aged=True,
        )
        pf_t, pa_t = np.asarray(pf_t), np.asarray(pa_t)
        grad_rel = float(np.linalg.norm(gt_np - g_ex_np) / max(np.linalg.norm(g_ex_np), 1e-12))
        grad_cos = float(np.dot(gt_np, g_ex_np) / max(np.linalg.norm(gt_np) * np.linalg.norm(g_ex_np), 1e-12))
        talbot_audit[f"n_{nt}"] = {
            "warm_grad_ms_median5": round(float(np.median(t_times)), 2),
            "fresh_max_abs_err": float(np.max(np.abs(pf_t - P_ref_fr))),
            "aged_max_abs_err": float(np.max(np.abs(pa_t - P_ref_ag))),
            "aged_kl_expm_ref": weighted_kl(P_ref_ag, ag_te),
            "aged_kl_talbot": weighted_kl(pa_t, ag_te),
            "aged_kl_abs_diff": abs(weighted_kl(pa_t, ag_te) - weighted_kl(P_ref_ag, ag_te)),
            "grad_rel_l2_err_vs_expm": grad_rel,
            "grad_cosine_vs_expm": grad_cos,
        }
        print(
            f"  [MOOC Talbot Accuracy n={nt:2d}] grad_ms={talbot_audit[f'n_{nt}']['warm_grad_ms_median5']:.1f}ms "
            f"fr_err={talbot_audit[f'n_{nt}']['fresh_max_abs_err']:.2e} ag_err={talbot_audit[f'n_{nt}']['aged_max_abs_err']:.2e} "
            f"grad_rel={grad_rel:.2e} cos={grad_cos:.8f}",
            flush=True,
        )

    print(
        f"  [MOOC Solver Comparison] Dense 490x490 expm={expm_grad_ms:.1f}ms | Implicit SDIRK2={sdirk_grad_ms:.1f}ms | "
        f"Talbot n=12={talbot_audit['n_12']['warm_grad_ms_median5']:.1f}ms | Talbot n=16={talbot_audit['n_16']['warm_grad_ms_median5']:.1f}ms",
        flush=True,
    )
    return {
        "dense_490_expm_grad_ms": round(expm_grad_ms, 2),
        "implicit_sdirk2_36step_grad_ms": round(sdirk_grad_ms, 2),
        "implicit_sdirk2_fresh_max_abs_err": sdirk_fr_err,
        "implicit_sdirk2_grad_rel_l2_err": sdirk_grad_rel,
        "talbot_precision_sweep": talbot_audit,
    }


def run_benchmark_for_dataset(name: str, trajs_tr, trajs_te, n_nodes: int, horizons: np.ndarray, aged_thresh: float, absorbing_node: int | None, dt_vals: list[float], psi_univ: np.ndarray, *, min_edge_count: int = 5, run_deep_mamba: bool = False) -> dict:
    print(f"\n================ {name} (V={n_nodes}) ================", flush=True)
    # 1. Proposition 3 persistence diagnostic
    p3 = prop3_persistence_diagnostic(trajs_tr, n_nodes, absorbing_node=absorbing_node)
    print(
        f"[{name} Prop 3 Diagnostic] CV={p3['sojourn_cv']:.2f}, lag1_corr={p3['lag1_residual_corr']:.4f}, "
        f"gain={p3['heldout_ll_gain_per_100_visits']:+.2f} nats/100, between-user ICC={p3['between_user_icc']:.3f}",
        flush=True,
    )

    fr_tr_all, ag_tr_all, edge_cnt = extract_general_snapshots(trajs_tr, n_nodes, horizons, aged_thresh, absorbing_node)
    fr_te_all, ag_te_all, _ = extract_general_snapshots(trajs_te, n_nodes, horizons, aged_thresh, absorbing_node)

    eval_mask = (fr_tr_all.sum(axis=(1, 2)) > 0) & (fr_te_all.sum(axis=(1, 2)) > 0)
    if absorbing_node is not None:
        eval_mask[absorbing_node] = False
    eval_nodes_np = np.where(eval_mask)[0].astype(np.int32)
    eval_nodes = jnp.asarray(eval_nodes_np)

    fr_tr, ag_tr = fr_tr_all[eval_nodes_np], ag_tr_all[eval_nodes_np]
    fr_te, ag_te = fr_te_all[eval_nodes_np], ag_te_all[eval_nodes_np]

    edges = [(int(u), int(v)) for u in range(n_nodes) for v in range(n_nodes) if u != v and edge_cnt[u, v] >= min_edge_count]
    for u in range(n_nodes):
        if absorbing_node is not None and u == absorbing_node:
            continue
        if not any(s == u for s, _ in edges):
            row = edge_cnt[u].copy()
            row[u] = -1.0
            v_fb = int(np.argmax(row)) if np.max(row) > 0 else (absorbing_node if absorbing_node is not None else ((u + 1) % n_nodes))
            edges.append((u, v_fb))
    n_edges = len(edges)
    # Empirical log routing init
    emp_el = np.array([np.log(max(edge_cnt[u, v], 0.5)) for u, v in edges], dtype=np.float64)

    res = {"prop3_persistence": p3, "n_edges": n_edges, "n_eval_nodes": len(eval_nodes_np)}

    # 2. Uncapped Stiffness-Free SparseHyperDRN via Laplace-Schur-Talbot (K=4, lam_f up to 200!)
    k_slow = 4
    n_talb = 12 if n_nodes >= 50 else 16
    mi_main = 80 if n_nodes >= 50 else 120
    mi_coars = 65 if n_nodes >= 50 else 90
    roll_hyp, n_p_hyp = make_talbot_hyper_drn(n_nodes, edges, k_slow, horizons, absorbing_node=absorbing_node, max_log_fast=5.3, n_talbot=n_talb)
    th0_hyp = np.concatenate([
        np.full(n_nodes, 1.5),   # log(lam_f - lam_s) ~ 4.5
        np.full(n_nodes, -2.0),  # log(lam_s) ~ 0.13
        np.zeros(n_nodes),       # shape
        np.full(n_nodes, 0.5),   # alpha_f
        emp_el,
    ])
    m_fr, x_fr = fit_talbot_model(roll_hyp, th0_hyp, eval_nodes, aged_thresh, fr_tr, ag_tr, fr_te, ag_te, joint=False, maxiter=mi_main, reg=1e-4)
    m_jt, x_jt = fit_talbot_model(roll_hyp, x_fr, eval_nodes, aged_thresh, fr_tr, ag_tr, fr_te, ag_te, joint=True, maxiter=mi_main, reg=1e-4)
    # Record max learned fast rate to verify stiffness removal!
    max_lf_learned = float(np.max(np.exp(np.clip(x_fr[n_nodes:2*n_nodes], -7, 2.5)) + np.exp(np.clip(x_fr[:n_nodes], -6, 5.3))))
    m_fr["params"] = n_p_hyp; m_fr["max_learned_lam_f"] = round(max_lf_learned, 2)
    m_jt["params"] = n_p_hyp
    res["talbot_sparse_hyper_drn_fresh_only"] = m_fr
    res["talbot_sparse_hyper_drn_joint"] = m_jt
    print(
        f"[{name}] Talbot-SparseHyperDRN ({n_p_hyp}p, warm_grad={m_fr['warm_grad_ms']:.1f}ms, max_lam_f={max_lf_learned:.1f}): "
        f"0-shot fresh={m_fr['fresh_kl']:.4f} aged={m_fr['aged_kl']:.4f} | joint aged={m_jt['aged_kl']:.4f}",
        flush=True,
    )

    # 3. Persistent Frailty-DRN (Proposition 3): Pure Mixed-Erlang (G=1 control), Pure Frailty (G=3), Hybrid (G=2)
    frailty_cfgs = (
        [("pure_mixed_erlang_g1", 1, True), ("pure_frailty_g3", 3, True), ("hybrid_frailty_g2", 2, False)]
        if n_nodes < 50
        else [("pure_frailty_g3", 3, True), ("hybrid_frailty_g2", 2, False)]
    )
    for tag, g_cls, pure_f in frailty_cfgs:
        roll_fr, n_p_fr, base_p = make_talbot_frailty_drn(
            n_nodes, edges, k_slow, horizons, n_classes=g_cls, pure_frailty=pure_f,
            absorbing_node=absorbing_node, max_log_fast=5.3, n_talbot=n_talb,
        )
        if pure_f:
            th0_base = np.concatenate([np.full(n_nodes, -0.5), np.zeros(n_nodes), emp_el])
        else:
            th0_base = th0_hyp.copy()
        if g_cls == 1:
            th0_fr = th0_base
        else:
            th0_fr = np.concatenate([th0_base, np.full(g_cls - 1, 1.2), np.zeros(n_nodes * g_cls)])
        mf, xf = fit_talbot_model(roll_fr, th0_fr, eval_nodes, aged_thresh, fr_tr, ag_tr, fr_te, ag_te, joint=False, maxiter=mi_main, reg=1e-4)
        mj, _ = fit_talbot_model(roll_fr, xf, eval_nodes, aged_thresh, fr_tr, ag_tr, fr_te, ag_te, joint=True, maxiter=mi_main, reg=1e-4)
        if g_cls == 1:
            gammas_learned = [1.0]
        else:
            d_lg = np.log1p(np.exp(xf[base_p : base_p + (g_cls - 1)])) + 0.15
            lg = np.concatenate([[0.0], np.cumsum(d_lg)])
            gammas_learned = np.exp(np.clip(lg - lg.mean(), -5, 5)).tolist()
        mf["params"] = n_p_fr; mf["learned_gammas"] = [round(g, 3) for g in gammas_learned]
        mj["params"] = n_p_fr
        res[f"{tag}_fresh_only"] = mf
        res[f"{tag}_joint"] = mj
        print(
            f"[{name}] {tag} ({n_p_fr}p, gammas={mf['learned_gammas']}): "
            f"0-shot fresh={mf['fresh_kl']:.4f} aged={mf['aged_kl']:.4f} | joint aged={mj['aged_kl']:.4f}",
            flush=True,
        )

    # 3b. On MOOC (V=98), verify Talbot-DRN (n=12, 16, 24) vs exact dense 490x490 expm & implicit SDIRK2
    if n_nodes >= 50 and absorbing_node is not None:
        res["talbot_accuracy_and_solver_benchmark"] = verify_mooc_talbot_accuracy_and_solvers(
            n_nodes, edges, k_slow, horizons, aged_thresh, eval_nodes, x_fr, fr_tr, ag_te, absorbing_node
        )

    # 4. Data-Free Universal Hyper-Pos-Sens (K_r=8) plugged ZERO-SHOT via Laplace-Schur-Talbot!
    roll_uh, n_p_uh = make_universal_hyper_pos_talbot(n_nodes, edges, horizons, psi_univ, stages_red=8, absorbing_node=absorbing_node, n_talbot=n_talb)
    th0_uh = np.concatenate([
        np.full(n_nodes, 1.2),   # log(lam_f - 1/m_s)
        np.full(n_nodes, 1.5),   # log(m_s)
        np.zeros(n_nodes),       # shape logit
        np.full(n_nodes, 0.4),   # alpha_f
        emp_el,
    ])
    mu_fr, xu_fr = fit_talbot_model(roll_uh, th0_uh, eval_nodes, aged_thresh, fr_tr, ag_tr, fr_te, ag_te, joint=False, maxiter=mi_main, reg=1e-4)
    mu_jt, _ = fit_talbot_model(roll_uh, xu_fr, eval_nodes, aged_thresh, fr_tr, ag_tr, fr_te, ag_te, joint=True, maxiter=mi_main, reg=1e-4)
    mu_fr["params"] = n_p_uh; mu_fr["states"] = (n_nodes - (1 if absorbing_node is not None else 0)) * 9 + (1 if absorbing_node is not None else 0)
    mu_jt["params"] = n_p_uh
    res["universal_hyper_pos_sens_fresh_only"] = mu_fr
    res["universal_hyper_pos_sens_joint"] = mu_jt
    print(
        f"[{name}] Univ. Hyper-Pos-Sens ({mu_fr['states']}st, {n_p_uh}p, warm_grad={mu_fr['warm_grad_ms']:.1f}ms): "
        f"0-shot fresh={mu_fr['fresh_kl']:.4f} aged={mu_fr['aged_kl']:.4f} | joint aged={mu_jt['aged_kl']:.4f}",
        flush=True,
    )

    # 5. Harmonized 5-model coarsening: calendar-subsampled inspection-paradox control (`coarsening_r8`)
    #    AND entry-anchored coarse-horizon panel (`coarsening_entry_anchored` on MOOC matching Helpdesk/R7!)
    res["coarsening_r8"] = run_harmonized_coarsening(
        name, trajs_tr, trajs_te, n_nodes, edges, eval_nodes_np, k_slow, horizons, aged_thresh,
        absorbing_node, dt_vals, psi_univ, n_talb, mi_coars, th0_hyp, th0_uh,
    )
    if n_nodes >= 50:
        res["coarsening_entry_anchored"] = run_entry_anchored_coarsening(
            name, trajs_tr, trajs_te, n_nodes, edges, eval_nodes_np, k_slow, horizons, aged_thresh,
            absorbing_node, dt_vals, psi_univ, n_talb, mi_coars, th0_hyp, th0_uh, res["coarsening_r8"],
        )

    # 6. High-capacity 2-layer AdamW Mamba (136,331 params) on MOOC!
    if run_deep_mamba:
        res["deep_2layer_mamba_adamw"] = run_deep_adamw_mamba_mooc(
            trajs_tr, n_nodes, edges, horizons, aged_thresh, eval_nodes, fr_tr, ag_tr, fr_te, ag_te, absorbing_node=absorbing_node
        )
    return res


def run_harmonized_coarsening(
    name: str,
    trajs_tr: list[tuple[np.ndarray, np.ndarray]],
    trajs_te: list[tuple[np.ndarray, np.ndarray]],
    n_nodes: int,
    edges: list[tuple[int, int]],
    eval_nodes_np: np.ndarray,
    k_slow: int,
    horizons: np.ndarray,
    aged_thresh: float,
    absorbing_node: int | None,
    dt_vals: list[float],
    psi_univ: np.ndarray,
    n_talb: int,
    mi_coars: int,
    th0_hyp: np.ndarray,
    th0_uh: np.ndarray,
) -> dict:
    """Apples-to-apples 5-model coarsening on the EXACT SAME calendar-coarsened training panel
    and the EXACT SAME held-out aged test horizon grid `grid_eval = arange(dt_c, t_max, dt_c)`.
    """
    eval_nodes = jnp.asarray(eval_nodes_np, dtype=jnp.int32)
    coars_out = {}
    t_end_grid = float(horizons[-1])
    for dt_c in dt_vals:
        grid_eval = np.arange(dt_c, t_end_grid + 1e-9, dt_c)
        if len(grid_eval) < 2:
            grid_eval = np.array([dt_c, 2.0 * dt_c])

        # Calendar-coarsened training trajectories (strips ALL sub-dt_c timestamps!)
        coars_tr = []
        for times, states in trajs_tr:
            if len(times) < 2:
                continue
            grid = np.arange(0.0, float(times[-1]) + 1e-9, dt_c)
            if len(grid) < 2:
                continue
            pos = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(states) - 1)
            coars_tr.append((grid, states[pos]))

        c_fr, _, ec_c = extract_general_snapshots(coars_tr, n_nodes, grid_eval, aged_thresh, absorbing_node)
        f_te_g, a_te_g, _ = extract_general_snapshots(trajs_te, n_nodes, grid_eval, aged_thresh, absorbing_node)
        c_fr_ev = c_fr[eval_nodes_np]
        f_te_ev = f_te_g[eval_nodes_np]
        a_te_ev = a_te_g[eval_nodes_np]
        zero_ag = np.zeros_like(c_fr_ev)

        # 1. Talbot-DRN on grid_eval
        roll_hyp_g, _ = make_talbot_hyper_drn(
            n_nodes, edges, k_slow, grid_eval, absorbing_node=absorbing_node, max_log_fast=5.3, n_talbot=n_talb
        )
        mc_hyp, _ = fit_talbot_model(
            roll_hyp_g, th0_hyp, eval_nodes, aged_thresh, c_fr_ev, zero_ag, f_te_ev, a_te_ev,
            joint=False, maxiter=mi_coars, reg=1e-4,
        )

        # 2. Univ. Hyper-Pos-Sens on grid_eval
        roll_uh_g, _ = make_universal_hyper_pos_talbot(
            n_nodes, edges, grid_eval, psi_univ, stages_red=8, absorbing_node=absorbing_node, n_talbot=n_talb
        )
        mc_uh, _ = fit_talbot_model(
            roll_uh_g, th0_uh, eval_nodes, aged_thresh, c_fr_ev, zero_ag, f_te_ev, a_te_ev,
            joint=False, maxiter=mi_coars, reg=1e-4,
        )

        # 3. 1-Jump Parametric Hawkes (with right-censored stays in NLL) on grid_eval
        mc_hk = run_1jump_hawkes_coarsening_single_dt(
            trajs_tr, n_nodes, eval_nodes_np, a_te_ev, grid_eval, dt_c, absorbing_node=absorbing_node
        )

        # 4. Exact CTMC (expm) on c_fr_ev over grid_eval
        fj_g = jnp.asarray(c_fr_ev, dtype=jnp.float64)
        nfg = jnp.asarray(max(float(c_fr_ev.sum()), 1.0))
        row_sums = np.maximum(ec_c.sum(axis=1), 1.0)
        node_exit = np.array([
            -np.log(np.clip(c_fr[u, 0, u] / max(c_fr[u, 0].sum(), 1.0), 0.05, 0.99)) / max(float(grid_eval[0]), 1e-3)
            for u in range(n_nodes)
        ])
        edge_prior_logits = np.log(np.array([
            np.clip(node_exit[u] * (ec_c[u, v] / row_sums[u]), 1e-3, 10.0)
            for u, v in edges
        ]))
        r_ct_g, _ = make_exact_ctmc(n_nodes, edges, grid_eval, absorbing_node=absorbing_node)

        @jax.jit
        def vg_cg(w):
            def lf(ww):
                return -jnp.sum(fj_g * jnp.log(jnp.clip(r_ct_g(ww, eval_nodes), 1e-12, 1.0))) / nfg + 1e-5 * jnp.mean(ww**2)
            return jax.value_and_grad(lf)(w)

        def obj_cg(z):
            v, g = vg_cg(jnp.asarray(z, dtype=jnp.float64))
            return float(v), np.asarray(g, dtype=jnp.float64)

        res_cg = minimize(obj_cg, edge_prior_logits.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 80})
        p_cg = np.asarray(r_ct_g(jnp.asarray(res_cg.x), eval_nodes))
        ctmc_aged_kl = weighted_kl(p_cg, a_te_ev)

        # 5. 1L-Mamba (clean age=0) on c_fr_ev over grid_eval
        in_mg, r_mg, _ = make_scalable_neural_baseline("mamba", n_nodes, edges, grid_eval, absorbing_node=absorbing_node, latent=12)

        @jax.jit
        def vg_mg(w):
            def lf(ww):
                return -jnp.sum(fj_g * jnp.log(jnp.clip(r_mg(ww, eval_nodes, 0.0), 1e-12, 1.0))) / nfg + 1e-4 * jnp.mean(ww**2)
            return jax.value_and_grad(lf)(w)

        def obj_mg(z):
            v, g = vg_mg(jnp.asarray(z, dtype=jnp.float64))
            return float(v), np.asarray(g, dtype=jnp.float64)

        res_mg = minimize(obj_mg, in_mg(17, edge_prior_logits), jac=True, method="L-BFGS-B", options={"maxiter": 65})
        pa_mg = np.asarray(r_mg(jnp.asarray(res_mg.x), eval_nodes, 0.0))
        mamba_aged_kl = weighted_kl(pa_mg, a_te_ev)

        coars_out[f"dt_{dt_c}"] = {
            "grid_eval": [float(g) for g in grid_eval],
            "talbot_drn_aged_kl": mc_hyp["aged_kl"],
            "univ_hyper_pos_sens_aged_kl": mc_uh["aged_kl"],
            "hawkes_1jump_aged_kl": mc_hk["hawkes_1jump_aged_kl"],
            "ctmc_aged_kl": ctmc_aged_kl,
            "mamba_1l_aged_kl": mamba_aged_kl,
            "hawkes_1jump_params": mc_hk["params"],
            "hawkes_1jump_n_obs": mc_hk["n_obs_transitions"],
            "hawkes_1jump_n_censored": mc_hk["n_censored_stays"],
        }
        print(
            f"  [{name} Harmonized Coarsen dt={dt_c} grid={len(grid_eval)}pt] "
            f"Talbot={mc_hyp['aged_kl']:.4f} Univ-H={mc_uh['aged_kl']:.4f} "
            f"1J-Hawkes={mc_hk['hawkes_1jump_aged_kl']:.4f} 1L-Mamba={mamba_aged_kl:.4f} CTMC={ctmc_aged_kl:.4f}",
            flush=True,
        )
    return coars_out


def run_entry_anchored_coarsening(
    name: str,
    trajs_tr: list[tuple[np.ndarray, np.ndarray]],
    trajs_te: list[tuple[np.ndarray, np.ndarray]],
    n_nodes: int,
    edges: list[tuple[int, int]],
    eval_nodes_np: np.ndarray,
    k_slow: int,
    horizons: np.ndarray,
    aged_thresh: float,
    absorbing_node: int | None,
    dt_vals: list[float],
    psi_univ: np.ndarray,
    n_talb: int,
    mi_coars: int,
    th0_hyp: np.ndarray,
    th0_uh: np.ndarray,
    coars_cal: dict,
) -> dict:
    """Harmonized entry-anchored coarsening (matching Helpdesk & R7 protocol) on IDENTICAL grid_eval."""
    eval_nodes = jnp.asarray(eval_nodes_np, dtype=jnp.int32)
    t_end_grid = float(horizons[-1])
    ea_out = {}
    for dt_c in dt_vals:
        grid_eval = np.arange(dt_c, t_end_grid + 1e-9, dt_c)
        if len(grid_eval) < 2:
            grid_eval = np.array([dt_c, 2.0 * dt_c])
        f_tr_g, _, ec_g = extract_general_snapshots(trajs_tr, n_nodes, grid_eval, aged_thresh, absorbing_node)
        f_te_g, a_te_g, _ = extract_general_snapshots(trajs_te, n_nodes, grid_eval, aged_thresh, absorbing_node)
        f_tr_ev, f_te_ev, a_te_ev = f_tr_g[eval_nodes_np], f_te_g[eval_nodes_np], a_te_g[eval_nodes_np]
        z_ag = np.zeros_like(f_tr_ev)

        r_hyp, _ = make_talbot_hyper_drn(n_nodes, edges, k_slow, grid_eval, absorbing_node=absorbing_node, max_log_fast=5.3, n_talbot=n_talb)
        m_hyp, _ = fit_talbot_model(r_hyp, th0_hyp, eval_nodes, aged_thresh, f_tr_ev, z_ag, f_te_ev, a_te_ev, joint=False, maxiter=mi_coars, reg=1e-4)

        r_uh, _ = make_universal_hyper_pos_talbot(n_nodes, edges, grid_eval, psi_univ, stages_red=8, absorbing_node=absorbing_node, n_talbot=n_talb)
        m_uh, _ = fit_talbot_model(r_uh, th0_uh, eval_nodes, aged_thresh, f_tr_ev, z_ag, f_te_ev, a_te_ev, joint=False, maxiter=mi_coars, reg=1e-4)

        fj = jnp.asarray(f_tr_ev, dtype=jnp.float64)
        nfg = jnp.asarray(max(float(f_tr_ev.sum()), 1.0))
        row_sums = np.maximum(ec_g.sum(axis=1), 1.0)
        node_exit = np.array([-np.log(np.clip(f_tr_g[u, 0, u] / max(f_tr_g[u, 0].sum(), 1.0), 0.05, 0.99)) / dt_c for u in range(n_nodes)])
        e_init = np.log(np.array([np.clip(node_exit[u] * (ec_g[u, v] / row_sums[u]), 1e-3, 10.0) for u, v in edges]))

        r_ct, _ = make_exact_ctmc(n_nodes, edges, grid_eval, absorbing_node=absorbing_node)
        @jax.jit
        def vg_c(w):
            return jax.value_and_grad(lambda ww: -jnp.sum(fj * jnp.log(jnp.clip(r_ct(ww, eval_nodes), 1e-12, 1.0))) / nfg + 1e-5 * jnp.mean(ww**2))(w)
        rc = minimize(lambda z: (float(vg_c(jnp.asarray(z))[0]), np.asarray(vg_c(jnp.asarray(z))[1], float)), e_init.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 80})
        ctmc_ag = weighted_kl(np.asarray(r_ct(jnp.asarray(rc.x), eval_nodes)), a_te_ev)

        in_m, r_m, _ = make_scalable_neural_baseline("mamba", n_nodes, edges, grid_eval, absorbing_node=absorbing_node, latent=12)
        @jax.jit
        def vg_m(w):
            return jax.value_and_grad(lambda ww: -jnp.sum(fj * jnp.log(jnp.clip(r_m(ww, eval_nodes, 0.0), 1e-12, 1.0))) / nfg + 1e-4 * jnp.mean(ww**2))(w)
        rm = minimize(lambda z: (float(vg_m(jnp.asarray(z))[0]), np.asarray(vg_m(jnp.asarray(z))[1], float)), in_m(17, e_init), jac=True, method="L-BFGS-B", options={"maxiter": 65})
        mamba_ag = weighted_kl(np.asarray(r_m(jnp.asarray(rm.x), eval_nodes, 0.0)), a_te_ev)

        hk_ag = coars_cal[f"dt_{dt_c}"]["hawkes_1jump_aged_kl"]
        ea_out[f"dt_{dt_c}"] = {
            "grid_eval": [float(g) for g in grid_eval],
            "talbot_drn_aged_kl": m_hyp["aged_kl"],
            "univ_hyper_pos_sens_aged_kl": m_uh["aged_kl"],
            "hawkes_1jump_aged_kl": hk_ag,
            "mamba_1l_aged_kl": mamba_ag,
            "ctmc_aged_kl": ctmc_ag,
        }
    return ea_out


def _prep_dataset(trajs_tr, trajs_te, n_nodes: int, horizons: np.ndarray, aged_thresh: float, absorbing_node: int | None, min_edge_count: int):
    fr_tr_all, ag_tr_all, edge_cnt = extract_general_snapshots(trajs_tr, n_nodes, horizons, aged_thresh, absorbing_node)
    fr_te_all, ag_te_all, _ = extract_general_snapshots(trajs_te, n_nodes, horizons, aged_thresh, absorbing_node)
    eval_mask = (fr_tr_all.sum(axis=(1, 2)) > 0) & (fr_te_all.sum(axis=(1, 2)) > 0)
    if absorbing_node is not None:
        eval_mask[absorbing_node] = False
    eval_nodes_np = np.where(eval_mask)[0].astype(np.int32)
    edges = [(int(u), int(v)) for u in range(n_nodes) for v in range(n_nodes) if u != v and edge_cnt[u, v] >= min_edge_count]
    for u in range(n_nodes):
        if absorbing_node is not None and u == absorbing_node:
            continue
        if not any(s == u for s, _ in edges):
            row = edge_cnt[u].copy()
            row[u] = -1.0
            v_fb = int(np.argmax(row)) if np.max(row) > 0 else (absorbing_node if absorbing_node is not None else ((u + 1) % n_nodes))
            edges.append((u, v_fb))
    emp_el = np.array([np.log(max(edge_cnt[u, v], 0.5)) for u, v in edges], dtype=np.float64)
    return eval_nodes_np, edges, emp_el, fr_tr_all[eval_nodes_np], ag_tr_all[eval_nodes_np], fr_te_all[eval_nodes_np], ag_te_all[eval_nodes_np]


def main() -> None:
    if "--only-r9-fixes" in sys.argv:
        t0 = perf_counter()
        out = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        psi_univ, cert = calibrate_universal_pos_clock(stages_red=8, stages_full=16, maxiter=200, k_min=1.4, k_max=7.8)
        out["universal_clock_cert"] = cert
        print(f"[R9] Calibrated Universal Pos-Sens (sup_surv={cert['sup_survival_abs_error']:.5f})", flush=True)

        # --- 1. StackOverflow V=22: Pure Mixed-Erlang G=1 control + Harmonized 5-model coarsening ---
        so_tr, so_te, _ = load_stackoverflow()
        so_h = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
        so_ev, so_edges, so_el, so_ftr, so_atr, so_fte, so_ate = _prep_dataset(so_tr, so_te, 22, so_h, 2.0, None, 5)
        so_ev_j = jnp.asarray(so_ev, dtype=jnp.int32)
        roll_g1, n_p_g1, _ = make_talbot_frailty_drn(22, so_edges, 4, so_h, n_classes=1, pure_frailty=True, absorbing_node=None, max_log_fast=5.3, n_talbot=16)
        th0_g1 = np.concatenate([np.full(22, -0.5), np.zeros(22), so_el])
        mf_g1, xf_g1 = fit_talbot_model(roll_g1, th0_g1, so_ev_j, 2.0, so_ftr, so_atr, so_fte, so_ate, joint=False, maxiter=120, reg=1e-4)
        mj_g1, _ = fit_talbot_model(roll_g1, xf_g1, so_ev_j, 2.0, so_ftr, so_atr, so_fte, so_ate, joint=True, maxiter=120, reg=1e-4)
        mf_g1["params"] = n_p_g1; mf_g1["learned_gammas"] = [1.0]
        mj_g1["params"] = n_p_g1
        out["stackoverflow_v22_r8"]["pure_mixed_erlang_g1_fresh_only"] = mf_g1
        out["stackoverflow_v22_r8"]["pure_mixed_erlang_g1_joint"] = mj_g1
        print(
            f"[StackOverflow_V22] pure_mixed_erlang_g1 ({n_p_g1}p): "
            f"0-shot fresh={mf_g1['fresh_kl']:.4f} aged={mf_g1['aged_kl']:.4f} | joint aged={mj_g1['aged_kl']:.4f}",
            flush=True,
        )
        so_th0_hyp = np.concatenate([np.full(22, 1.5), np.full(22, -2.0), np.zeros(22), np.full(22, 0.5), so_el])
        so_th0_uh = np.concatenate([np.full(22, 1.2), np.full(22, 1.5), np.zeros(22), np.full(22, 0.4), so_el])
        out["stackoverflow_v22_r8"]["coarsening_r8"] = run_harmonized_coarsening(
            "StackOverflow_V22", so_tr, so_te, 22, so_edges, so_ev, 4, so_h, 2.0, None,
            [0.5, 2.0, 4.0, 8.0], psi_univ, 16, 90, so_th0_hyp, so_th0_uh,
        )

        # --- 2. MOOC V=98: Pure Frailty G=3 + Talbot accuracy/solver audit + Harmonized coarsening + Deep 2L-Mamba ---
        mo_tr, mo_te, _, mo_abs = load_mooc()
        mo_h = np.array([0.25, 1.0, 4.0, 12.0, 24.0, 48.0])
        mo_ev, mo_edges, mo_el, mo_ftr, mo_atr, mo_fte, mo_ate = _prep_dataset(mo_tr, mo_te, 98, mo_h, 6.0, mo_abs, 10)
        mo_ev_j = jnp.asarray(mo_ev, dtype=jnp.int32)
        mo_th0_hyp = np.concatenate([np.full(98, 1.5), np.full(98, -2.0), np.zeros(98), np.full(98, 0.5), mo_el])
        mo_th0_uh = np.concatenate([np.full(98, 1.2), np.full(98, 1.5), np.zeros(98), np.full(98, 0.4), mo_el])

        # 2a. Pure Frailty G=3 on MOOC
        roll_g3, n_p_g3, base_p3 = make_talbot_frailty_drn(98, mo_edges, 4, mo_h, n_classes=3, pure_frailty=True, absorbing_node=mo_abs, max_log_fast=5.3, n_talbot=12)
        th0_g3 = np.concatenate([np.full(98, -0.5), np.zeros(98), mo_el, np.full(2, 1.2), np.zeros(98 * 3)])
        mf_g3, xf_g3 = fit_talbot_model(roll_g3, th0_g3, mo_ev_j, 6.0, mo_ftr, mo_atr, mo_fte, mo_ate, joint=False, maxiter=80, reg=1e-4)
        mj_g3, _ = fit_talbot_model(roll_g3, xf_g3, mo_ev_j, 6.0, mo_ftr, mo_atr, mo_fte, mo_ate, joint=True, maxiter=80, reg=1e-4)
        d_lg3 = np.log1p(np.exp(xf_g3[base_p3 : base_p3 + 2])) + 0.15
        lg3 = np.concatenate([[0.0], np.cumsum(d_lg3)])
        gam3 = np.exp(np.clip(lg3 - lg3.mean(), -5, 5)).tolist()
        mf_g3["params"] = n_p_g3; mf_g3["learned_gammas"] = [round(g, 3) for g in gam3]
        mj_g3["params"] = n_p_g3
        out["mooc_v98_r8"]["pure_frailty_g3_fresh_only"] = mf_g3
        out["mooc_v98_r8"]["pure_frailty_g3_joint"] = mj_g3
        print(
            f"[MOOC_V98] pure_frailty_g3 ({n_p_g3}p, gammas={mf_g3['learned_gammas']}): "
            f"0-shot fresh={mf_g3['fresh_kl']:.4f} aged={mf_g3['aged_kl']:.4f} | joint aged={mj_g3['aged_kl']:.4f}",
            flush=True,
        )

        # 2b. Re-obtain fitted Talbot-DRN x_fr and audit accuracy vs exact dense 490x490 expm & SDIRK2
        roll_hyp_m, _ = make_talbot_hyper_drn(98, mo_edges, 4, mo_h, absorbing_node=mo_abs, max_log_fast=5.3, n_talbot=12)
        _, x_fr_m = fit_talbot_model(roll_hyp_m, mo_th0_hyp, mo_ev_j, 6.0, mo_ftr, mo_atr, mo_fte, mo_ate, joint=False, maxiter=80, reg=1e-4)
        out["mooc_v98_r8"]["talbot_accuracy_and_solver_benchmark"] = verify_mooc_talbot_accuracy_and_solvers(
            98, mo_edges, 4, mo_h, 6.0, mo_ev_j, x_fr_m, mo_ftr, mo_ate, mo_abs
        )

        # 2c. Harmonized 5-model coarsening on MOOC
        out["mooc_v98_r8"]["coarsening_r8"] = run_harmonized_coarsening(
            "MOOC_V98", mo_tr, mo_te, 98, mo_edges, mo_ev, 4, mo_h, 6.0, mo_abs,
            [1.0, 6.0, 12.0, 24.0], psi_univ, 12, 65, mo_th0_hyp, mo_th0_uh,
        )

        # 2d. Converged Deep 2L-Mamba AdamW on MOOC (unified eval_nodes!)
        out["mooc_v98_r8"]["deep_2layer_mamba_adamw"] = run_deep_adamw_mamba_mooc(
            mo_tr, 98, mo_edges, mo_h, 6.0, mo_ev_j, mo_ftr, mo_atr, mo_fte, mo_ate, absorbing_node=mo_abs
        )

        OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[R9] Updated {OUT_PATH} in {perf_counter() - t0:.1f}s.", flush=True)
        return

    if "--only-deep-mamba" in sys.argv:
        mo_tr, mo_te, _, mo_abs = load_mooc()
        horizons = np.array([0.25, 1.0, 4.0, 12.0, 24.0, 48.0])
        aged_thresh = 6.0
        n_nodes = 98
        eval_nodes_np, edges, _, fr_tr, ag_tr, fr_te, ag_te = _prep_dataset(mo_tr, mo_te, n_nodes, horizons, aged_thresh, mo_abs, 10)
        eval_nodes = jnp.asarray(eval_nodes_np, dtype=jnp.int32)
        m2_res = run_deep_adamw_mamba_mooc(
            mo_tr, n_nodes, edges, horizons, aged_thresh, eval_nodes,
            fr_tr, ag_tr, fr_te, ag_te,
            absorbing_node=mo_abs,
        )
        out = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        out["mooc_v98_r8"]["deep_2layer_mamba_adamw"] = m2_res
        OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Updated MOOC deep_2layer_mamba_adamw in {OUT_PATH}", flush=True)
        return

    t0 = perf_counter()
    # Calibrate data-free Universal 1D Pos-Sens clock ONCE (certified on k in [1.4, 7.8])
    psi_univ, cert = calibrate_universal_pos_clock(stages_red=8, stages_full=16, maxiter=200, k_min=1.4, k_max=7.8)
    print(f"Calibrated data-free Universal Pos-Sens clock (sup_surv={cert['sup_survival_abs_error']:.5f})", flush=True)

    # 1. StackOverflow V=22
    so_tr, so_te, _ = load_stackoverflow()
    so_res = run_benchmark_for_dataset(
        "StackOverflow_V22", so_tr, so_te, 22,
        horizons=np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]),
        aged_thresh=2.0, absorbing_node=None, dt_vals=[0.5, 2.0, 4.0, 8.0],
        psi_univ=psi_univ, min_edge_count=5, run_deep_mamba=False,
    )

    # 2. MOOC V=98
    mo_tr, mo_te, _, mo_abs = load_mooc()
    mo_res = run_benchmark_for_dataset(
        "MOOC_V98", mo_tr, mo_te, 98,
        horizons=np.array([0.25, 1.0, 4.0, 12.0, 24.0, 48.0]),
        aged_thresh=6.0, absorbing_node=mo_abs, dt_vals=[1.0, 6.0, 12.0, 24.0],
        psi_univ=psi_univ, min_edge_count=10, run_deep_mamba=True,
    )

    out = {
        "universal_clock_cert": cert,
        "stackoverflow_v22_r8": so_res,
        "mooc_v98_r8": mo_res,
        "total_seconds": round(perf_counter() - t0, 2),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_PATH} in {out['total_seconds']}s.", flush=True)


if __name__ == "__main__":
    main()
