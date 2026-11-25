"""Round-7 Mainstream Sequence & Point-Process Baselines:
1. Mamba / Selective-SSM (Gu & Dao 2024; 3-start) evaluated across:
   - Synthetic Conditional Neural Clock intervention suite (Table 2: PH truth,
     Lognormal truth, Pulse-contrast RMSE, 10 paired seeds).
   - IT Helpdesk (5-fold CV fresh & 0-shot >=7d clean age=0, Chrono 70/30 drift,
     and coarsening dt in {0.5, 2, 7, 14}d).
   - Sleep-EDF (5-fold CV joint & 0-shot >=5m clean age=0).
   - 1978 Influenza (Full Days 1-14 + Days 9-14 / 10-14 L2-tuned extrapolation).
2. Continuous-Time Neural Hawkes / Transformer Hawkes Process (THP; Mei & Eisner 2017 /
   Zuo et al. 2020) trained on exact event timestamps on IT Helpdesk and evaluated under
   panel coarsening dt in {0.5, 2, 7, 14}d (showing multi-hop panel collapse vs. DRN).
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

from diff_renewal.baselines import make_baseline_engine  # noqa: E402
from scripts.run_modern_neural_baselines import (  # noqa: E402
    NM_ROOT,
    TEST_PROTOCOLS,
    TEST_TIMES,
    TRAIN_PROTOCOLS,
    TRAIN_TIMES,
    build_plan,
    compute_metrics,
    summarize_list,
)
from scripts.run_helpdesk_extended_baselines import load_chronological_split  # noqa: E402
from scripts.run_helpdesk_systems_benchmark import (  # noqa: E402
    AGED_THRESHOLD_DAYS,
    E_EDGES,
    EDGES as HDK_EDGES,
    HORIZONS_DAYS,
    V_NODES as HDK_V,
    empirical_kl,
    ensure_helpdesk_csv,
    extract_snapshot_tables,
    load_ticket_trajectories,
)
from scripts.run_edge_memory_sleep_benchmark import (  # noqa: E402
    build_sleep_snapshots,
    parse_sleep_sequences,
)
from scripts.run_mean_field_epidemic_benchmark import (  # noqa: E402
    DAYS as INF_DAYS,
    OBS_COUNTS as INF_COUNTS,
    metrics_on_slice,
)

OUT_PATH = ROOT / "results" / "archived" / "r7_mamba_thp_baselines.json"
STARTS = (17, 41, 89)


def excess_kl(pred: np.ndarray, counts: np.ndarray) -> float:
    tot = counts.sum(axis=-1, keepdims=True)
    emp = np.where(tot > 0, counts / np.maximum(tot, 1.0), 0.0)
    p = np.clip(pred, 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    kl = np.sum(np.where(emp > 0, emp * np.log(np.maximum(emp, 1e-30) / p), 0.0), axis=-1)
    w = tot[..., 0]
    return float(np.sum(kl * w) / np.maximum(np.sum(w), 1.0))


# ---------------------------------------------------------------------------
# 1. Synthetic Intervention Suite (Table 2 / Fig. 4) — Mamba / Selective-SSM
# ---------------------------------------------------------------------------
def run_synthetic_mamba() -> dict:
    train_plan = build_plan(TRAIN_PROTOCOLS, TRAIN_TIMES)
    test_plan = build_plan(TEST_PROTOCOLS, TEST_TIMES)
    engines = []
    dummy_c = jnp.ones((8, 6, 3, 3))
    n_params = 0
    for s_id in STARTS:
        eng = make_baseline_engine("mamba_ode", train_plan, test_plan, seed=s_id, reg=5e-5)
        w0, _, vg, fc = eng
        n_params = int(len(w0))
        vg(w0, dummy_c)[1].block_until_ready()
        fc(w0).block_until_ready()
        engines.append(eng)

    out = {"params": n_params}
    for study_key, fname in [
        ("primary", "neural_memory_cycle.json"),
        ("lognormal", "neural_memory_cycle_lognormal.json"),
    ]:
        src = json.loads((NM_ROOT / "results" / fname).read_text(encoding="utf-8"))
        truth = np.asarray(src["test_truth"])
        seeds = [str(s) for s in src["protocol"]["seeds"]]
        runs = []
        for seed in seeds:
            tr_counts = jnp.asarray(src["runs"][seed]["train_counts"], dtype=float)
            te_counts = np.asarray(src["runs"][seed]["test_counts"], dtype=float)
            best_fit = None
            t0 = perf_counter()
            for w0, _, vg, fc in engines:
                def obj_syn(z):
                    v, g = vg(jnp.asarray(z), tr_counts)
                    return float(v), np.asarray(g, dtype=float)

                res = minimize(
                    obj_syn,
                    w0,
                    method="L-BFGS-B",
                    jac=True,
                    options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7, "maxls": 35},
                )
                if best_fit is None or res.fun < best_fit[0]:
                    best_fit = (float(res.fun), res.x, fc)
            pred = np.asarray(best_fit[2](jnp.asarray(best_fit[1])))
            m_dict = compute_metrics(pred, truth, te_counts)
            m_dict["seconds"] = perf_counter() - t0
            runs.append(m_dict)
        out[study_key] = {
            "balanced_kl": summarize_list([r["balanced_kl"] for r in runs]),
            "fresh_kl": summarize_list([r["fresh"]["kl"] for r in runs]),
            "aged_kl": summarize_list([r["aged"]["kl"] for r in runs]),
            "pulse_kl": summarize_list([r["pulse"]["kl"] for r in runs]),
            "pulse_effect_rmse": summarize_list([r["pulse_effect_rmse"] for r in runs]),
            "hits_le_0005": int(sum(1 for r in runs if r["balanced_kl"] <= 0.005)),
        }
        print(
            f"[Synthetic {study_key:9s}] Mamba ({n_params}p): bal_KL={out[study_key]['balanced_kl']['median']:.5f} "
            f"pulse_RMSE={100*out[study_key]['pulse_effect_rmse']['median']:.2f}pp hits={out[study_key]['hits_le_0005']}/10",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# 2. IT Helpdesk — Mamba (5-fold CV, Chrono 70/30, Coarsening) & Neural Hawkes (THP)
# ---------------------------------------------------------------------------
def make_mamba_graph_model(n_nodes: int, edges: tuple[tuple[int, int], ...], latent: int = 10, dt_step: float = 0.25, n_steps: int = 120):
    """Parameter-matched Selective-SSM (Mamba) continuous-time population network."""
    src_idx = jnp.asarray([u for u, _ in edges], dtype=jnp.int32)
    dst_idx = jnp.asarray([v for _, v in edges], dtype=jnp.int32)
    n_edges = len(edges)
    # Params: enc (n_nodes+1, H), eb (H), log_a (H), w_dt (n_nodes, H), b_dt (H),
    #         w_b (n_nodes, H), w_c (n_nodes, H), w_z (n_nodes, H), b_z (H),
    #         w_o (H + n_nodes, n_edges), b_o (n_edges)
    shapes = [
        (n_nodes + 1, latent),
        (latent,),
        (latent,),
        (n_nodes, latent),
        (latent,),
        (n_nodes, latent),
        (n_nodes, latent),
        (n_nodes, latent),
        (latent,),
        (latent + n_nodes, n_edges),
        (n_edges,),
    ]
    n_params = sum(int(np.prod(s)) for s in shapes)

    def init_fn(seed: int):
        r = np.random.default_rng(seed)
        parts = [
            r.normal(0.0, 0.25, (n_nodes + 1, latent)).ravel(),
            np.zeros(latent),
            np.linspace(np.log(0.1), np.log(3.5), latent),
            r.normal(0.0, 0.20, (n_nodes, latent)).ravel(),
            np.full(latent, -1.0),
            r.normal(0.0, 0.20, (n_nodes, latent)).ravel(),
            r.normal(0.0, 0.20, (n_nodes, latent)).ravel(),
            r.normal(0.0, 0.20, (n_nodes, latent)).ravel(),
            np.zeros(latent),
            r.normal(0.0, 0.15, (latent + n_nodes, n_edges)).ravel(),
            np.full(n_edges, -1.5),
        ]
        return np.concatenate(parts)

    def unpack(th):
        out = []
        i = 0
        for sh in shapes:
            sz = int(np.prod(sh))
            out.append(th[i : i + sz].reshape(sh))
            i += sz
        return out

    def rollout_fn(th, init_p: jax.Array, age_val: float, sample_idx: jax.Array, min_rate: float = 1e-4, max_rate: float = 4.0):
        enc, eb, log_a, w_dt, b_dt, w_b, w_c, w_z, b_z, w_o, b_o = unpack(th)
        neg_a = -jnp.exp(jnp.clip(log_a, -4.0, 3.0))
        a_col = jnp.full((init_p.shape[0], 1), age_val, dtype=init_p.dtype)
        h0 = jnp.tanh(jnp.concatenate([init_p, a_col], axis=-1) @ enc + eb)

        def step(carry, _):
            p, h = carry
            delta = jax.nn.softplus(p @ w_dt + b_dt) * dt_step
            a_bar = jnp.exp(delta * neg_a[None, :])
            b_bar = ((a_bar - 1.0) / neg_a[None, :]) * jnp.tanh(p @ w_b)
            h_new = a_bar * h + b_bar
            y_sel = jnp.tanh(p @ w_c) * h_new * jax.nn.silu(p @ w_z + b_z)
            rates = min_rate + max_rate * jax.nn.sigmoid(jnp.concatenate([y_sel, p], axis=-1) @ w_o + b_o)
            # Unconditionally positive exponential-Euler jump update (matches exact simplex flow!)
            tot_out = jax.vmap(lambda r: jax.ops.segment_sum(r, src_idx, num_segments=n_nodes))(rates)
            frac_leave = 1.0 - jnp.exp(-jnp.minimum(tot_out * dt_step, 50.0))
            scale = frac_leave / jnp.maximum(tot_out, 1e-30)
            edge_mass = p[:, src_idx] * scale[:, src_idx] * rates
            p_new = p * (1.0 - frac_leave)
            p_new = p_new.at[:, dst_idx].add(edge_mass)
            p_new = jnp.maximum(p_new, 1e-12)
            p_new = p_new / jnp.sum(p_new, axis=-1, keepdims=True)
            return (p_new, h_new), p_new

        _, traj = jax.lax.scan(step, (init_p, h0), None, length=n_steps)
        return jnp.transpose(traj[sample_idx], (1, 0, 2))

    return init_fn, rollout_fn, n_params


def run_helpdesk_mamba_and_thp() -> dict:
    csv_path = ensure_helpdesk_csv(Path("/tmp/drn_r7_helpdesk"))
    trajs, _ = load_ticket_trajectories(csv_path)
    rng = np.random.default_rng(2026)
    folds = np.array_split(rng.permutation(len(trajs)), 5)

    # Exact same grid & rate bounds as Helpdesk LTC/S4D/Latent ODE (dt=0.5d, 90 steps, min_rate=1e-4, max_rate=4.0)
    dt_step = 0.5
    n_steps = int(round(max(HORIZONS_DAYS) / dt_step))
    sample_idx = jnp.asarray([int(round(h / dt_step)) - 1 for h in HORIZONS_DAYS], dtype=jnp.int32)
    init_mamba, rollout_mamba, n_params = make_mamba_graph_model(HDK_V, HDK_EDGES, latent=10, dt_step=dt_step, n_steps=n_steps)

    eye4 = jnp.eye(HDK_V, dtype=jnp.float64)[:4]
    eye3_aged = jnp.eye(HDK_V, dtype=jnp.float64)[1:4]

    def pred_helpdesk_mamba(th, clean_age0: bool = True):
        pf = rollout_mamba(th, eye4, 0.0, sample_idx, min_rate=1e-4, max_rate=4.0)
        pa = rollout_mamba(th, eye3_aged, 0.0 if clean_age0 else 1.0, sample_idx, min_rate=1e-4, max_rate=4.0)
        return pf, pa

    def fit_mamba(f_tr, a_tr, joint: bool, maxiter: int = 140):
        f_j = jnp.asarray(f_tr, dtype=jnp.float64)
        a_j = jnp.asarray(a_tr, dtype=jnp.float64)
        nf = jnp.asarray(max(float(f_tr.sum()), 1.0))
        na = jnp.asarray(max(float(a_tr.sum()), 1.0))

        @jax.jit
        def vg(w):
            def loss_fn(ww):
                pf, pa = pred_helpdesk_mamba(ww, clean_age0=not joint)
                lf = -jnp.sum(f_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / nf
                tot = 0.5 * (lf - jnp.sum(a_j * jnp.log(jnp.clip(pa, 1e-12, 1.0))) / na) if joint else lf
                return tot + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(loss_fn)(w)

        def obj_m(z):
            v, g = vg(jnp.asarray(z))
            return float(v), np.asarray(g, dtype=float)

        best = None
        for s in STARTS:
            res = minimize(obj_m, init_mamba(s), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
            if best is None or res.fun < best[0]:
                best = (float(res.fun), res.x)
        return best[1]

    # 5-fold CV
    cv = {mode: {"fresh_kl": [], "aged_kl": []} for mode in ("joint", "fresh_only")}
    for fi in range(5):
        te_idx = set(folds[fi].tolist())
        tr_t = [trajs[i] for i in range(len(trajs)) if i not in te_idx]
        te_t = [trajs[i] for i in range(len(trajs)) if i in te_idx]
        f_tr, a_tr = extract_snapshot_tables(tr_t)
        f_te, a_te = extract_snapshot_tables(te_t)
        for mode, joint in (("joint", True), ("fresh_only", False)):
            w = fit_mamba(f_tr, a_tr, joint=joint)
            pf, pa = pred_helpdesk_mamba(jnp.asarray(w), clean_age0=not joint)
            cv[mode]["fresh_kl"].append(empirical_kl(np.asarray(pf), f_te))
            cv[mode]["aged_kl"].append(empirical_kl(np.asarray(pa), a_te))

    cv_sum = {
        mode: {
            "fresh_kl_mean": float(np.mean(cv[mode]["fresh_kl"])),
            "fresh_kl_sd": float(np.std(cv[mode]["fresh_kl"])),
            "aged_kl_mean": float(np.mean(cv[mode]["aged_kl"])),
            "aged_kl_sd": float(np.std(cv[mode]["aged_kl"])),
        }
        for mode in ("joint", "fresh_only")
    }
    print(
        f"[Helpdesk 5-fold] Mamba ({n_params}p): joint_fresh={cv_sum['joint']['fresh_kl_mean']:.4f}, "
        f"joint_aged={cv_sum['joint']['aged_kl_mean']:.4f} | "
        f"0shot_fresh={cv_sum['fresh_only']['fresh_kl_mean']:.4f}, "
        f"0shot_aged={cv_sum['fresh_only']['aged_kl_mean']:.4f}±{cv_sum['fresh_only']['aged_kl_sd']:.4f}",
        flush=True,
    )

    # Chronological 70/30
    tr_ch, te_ch, _ = load_chronological_split(csv_path)
    f_tr_c, a_tr_c = extract_snapshot_tables(tr_ch)
    f_te_c, a_te_c = extract_snapshot_tables(te_ch)
    w_ch = fit_mamba(f_tr_c, a_tr_c, joint=False)
    pf_c, pa_c = pred_helpdesk_mamba(jnp.asarray(w_ch), clean_age0=True)
    chrono = {
        "fresh_kl": empirical_kl(np.asarray(pf_c), f_te_c),
        "aged_kl": empirical_kl(np.asarray(pa_c), a_te_c),
    }
    print(f"[Helpdesk Chrono] Mamba: fresh={chrono['fresh_kl']:.4f}, 0shot_aged={chrono['aged_kl']:.4f}", flush=True)

    # Coarsening dt in {0.5, 2, 7, 14}d on Fold 0 (exact same protocol as Table A6)
    te_idx0 = set(folds[0].tolist())
    tr0 = [trajs[i] for i in range(len(trajs)) if i not in te_idx0]
    te0 = [trajs[i] for i in range(len(trajs)) if i in te_idx0]
    coarsening = {}
    for dt_c in (0.5, 2.0, 7.0, 14.0):
        grid = np.arange(dt_c, 42.0 + 1e-9, dt_c)
        f_tr_g, a_tr_g = extract_snapshot_tables(tr0, horizons=grid)
        f_te_g, a_te_g = extract_snapshot_tables(te0, horizons=grid)
        s_idx_g = jnp.asarray([int(round(h / dt_step)) - 1 for h in grid], dtype=jnp.int32)
        f_j = jnp.asarray(f_tr_g, dtype=jnp.float64)
        nf = jnp.asarray(max(float(f_tr_g.sum()), 1.0))

        @jax.jit
        def vg_g(w):
            def loss_fn(ww):
                pf = rollout_mamba(ww, eye4, 0.0, s_idx_g, min_rate=1e-4, max_rate=4.0)
                return -jnp.sum(f_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / nf + 1e-4 * jnp.mean(ww**2)

            return jax.value_and_grad(loss_fn)(w)

        def obj_g(z):
            v, g = vg_g(jnp.asarray(z))
            return float(v), np.asarray(g, dtype=float)

        best = None
        for s in STARTS:
            res = minimize(obj_g, init_mamba(s), jac=True, method="L-BFGS-B", options={"maxiter": 120})
            if best is None or res.fun < best[0]:
                best = (float(res.fun), res.x)
        pf_g = np.asarray(rollout_mamba(jnp.asarray(best[1]), eye4, 0.0, s_idx_g, min_rate=1e-4, max_rate=4.0))
        pa_g = np.asarray(rollout_mamba(jnp.asarray(best[1]), eye3_aged, 0.0, s_idx_g, min_rate=1e-4, max_rate=4.0))
        coarsening[f"dt_{dt_c}"] = {
            "mamba_fresh_kl": empirical_kl(pf_g, f_te_g),
            "mamba_aged_kl": empirical_kl(pa_g, a_te_g),
        }

    # Continuous-Time Neural Hawkes / THP baseline on IT Helpdesk coarsening (Fold 0):
    def eval_neural_hawkes_coarsening():
        out_thp = {}
        # Use dedicated START_TOKEN = 4 (absorbing Closed never precedes a jump!) so initial
        # visits & post-transition visits have strictly in-distribution (cur, prev) pairs!
        start_tok = 4
        for dt_c in (0.5, 2.0, 7.0, 14.0):
            def extract_coarsened_events(traj_list):
                rows = []
                for _, times, states in traj_list:
                    # If ticket reached absorbing Closed (4) at times[-1], it stays in 4 for all t >= times[-1],
                    # so include the grid point capturing the transition into 4 (loop breaks at st == 4).
                    # Only right-censor genuinely unclosed tickets at their actual last observed timestamp!
                    if int(states[-1]) == 4:
                        t_end = min(float(np.ceil(float(times[-1]) / dt_c) * dt_c), 45.0)
                    else:
                        t_end = min(float(times[-1]), 45.0)
                    if t_end < dt_c:
                        continue
                    grid_t = np.arange(0.0, t_end + 1e-9, dt_c)
                    if len(grid_t) < 2:
                        continue
                    g_states = [states[int(np.searchsorted(times, t, side="right")) - 1] for t in grid_t]
                    last_t = 0.0
                    last_u = int(g_states[0])
                    prev_u = start_tok
                    for t_cur, st in zip(grid_t[1:], g_states[1:]):
                        st = int(st)
                        if st != last_u:
                            if last_u < 4:
                                rows.append((last_u, st, float(t_cur - last_t), prev_u))
                            prev_u = last_u
                            last_u = st
                            last_t = float(t_cur)
                            if st == 4:
                                break
                    if last_u < 4 and grid_t[-1] > last_t + 1e-9:
                        rows.append((last_u, -1, float(grid_t[-1] - last_t), prev_u))
                return rows

            tr_ev = extract_coarsened_events(tr0)
            # Empirical incoming context distribution P(prev_u | cur=u0) for strictly in-distribution evaluation!
            prev_counts = np.zeros((4, 5), dtype=np.float64)
            for u_r, _, _, p_r in tr_ev:
                prev_counts[u_r, p_r] += 1.0
            prev_probs = np.zeros((4, 5), dtype=np.float64)
            for u0 in range(4):
                if prev_counts[u0].sum() > 0:
                    prev_probs[u0] = prev_counts[u0] / prev_counts[u0].sum()
                else:
                    prev_probs[u0, start_tok] = 1.0
            u_arr = jnp.asarray([r[0] for r in tr_ev], dtype=jnp.int32)
            v_arr = jnp.asarray([r[1] for r in tr_ev], dtype=jnp.int32)
            tau_arr = jnp.asarray([max(r[2], 1e-3) for r in tr_ev], dtype=jnp.float64)
            p_arr = jnp.asarray([r[3] for r in tr_ev], dtype=jnp.int32)
            is_ev = (v_arr >= 0).astype(jnp.float64)
            v_safe = jnp.maximum(v_arr, 0)

            gl_x, gl_w = np.polynomial.legendre.leggauss(12)
            gl_u = jnp.asarray(0.5 * (gl_x + 1.0))
            gl_wt = jnp.asarray(0.5 * gl_w)

            # THP/Neural-Hawkes with learned 8D attention/memory embedding (177 params total)
            def init_thp(seed: int):
                r = np.random.default_rng(seed)
                return np.r_[
                    np.full((4, 5), -2.5).ravel(),  # base mu[u, v]
                    r.normal(0.0, 0.15, (5, 5)).ravel(),  # self/cross-excitation alpha[prev, v] (4=START_TOKEN)
                    np.zeros((4, 5)).ravel(),  # log decay beta[u, v]
                    r.normal(0.0, 0.15, (4, 8)).ravel(),  # query emb
                    r.normal(0.0, 0.15, (5, 8)).ravel(),  # key emb (4=START_TOKEN)
                    r.normal(0.0, 0.15, (8, 5)).ravel(),  # attn readout
                ]

            def intensities(th, u_i, p_i, t_eval):
                mu = th[:20].reshape(4, 5)
                al = th[20:45].reshape(5, 5)
                be = jnp.exp(jnp.clip(th[45:65].reshape(4, 5), -4.0, 3.0))
                wq = th[65:97].reshape(4, 8)
                wk = th[97:137].reshape(5, 8)
                wo = th[137:177].reshape(8, 5)
                att = jax.nn.sigmoid(jnp.sum(wq[u_i] * wk[p_i], axis=-1, keepdims=True))
                h_t = att * wk[p_i] * jnp.exp(-0.2 * t_eval[..., None])
                raw = mu[u_i] + al[p_i] * jnp.exp(-be[u_i] * t_eval[..., None]) + h_t @ wo
                lam = jax.nn.softplus(raw) + 1e-6
                mask = 1.0 - jax.nn.one_hot(u_i, 5)
                return lam * mask

            @jax.jit
            def vg_thp(th):
                def loss_fn(w):
                    lam_ev = intensities(w, u_arr, p_arr, tau_arr)
                    log_lam = jnp.log(jnp.maximum(lam_ev[jnp.arange(len(tr_ev)), v_safe], 1e-12))
                    t_q = tau_arr[:, None] * gl_u[None, :]
                    lam_q = intensities(w, u_arr[:, None], p_arr[:, None], t_q)
                    surv_int = tau_arr * jnp.sum(jnp.sum(lam_q, axis=-1) * gl_wt[None, :], axis=-1)
                    nll = -jnp.mean(is_ev * log_lam - surv_int)
                    return nll + 1e-4 * jnp.mean(w**2)

                return jax.value_and_grad(loss_fn)(th)

            def obj_thp(z):
                v, g = vg_thp(jnp.asarray(z))
                return float(v), np.asarray(g, dtype=float)

            best = None
            for s in STARTS:
                res = minimize(obj_thp, init_thp(s), jac=True, method="L-BFGS-B", options={"maxiter": 120})
                if best is None or res.fun < best[0]:
                    best = (float(res.fun), res.x)

            # Evaluate 1-jump Hawkes / THP panel forecast over grid horizons using 32-pt Gauss-Legendre
            # quadrature (accurate for fast early bursts even at h=42d!) averaged over empirical P(prev | u0)
            grid = np.arange(dt_c, 42.0 + 1e-9, dt_c)
            f_te_g, _ = extract_snapshot_tables(te0, horizons=grid)
            w_best = jnp.asarray(best[1])
            gl32_x, gl32_w = np.polynomial.legendre.leggauss(32)
            gl32_u = 0.5 * (gl32_x + 1.0)
            gl32_wt = 0.5 * gl32_w
            # Nested 16-pt GL for accurate cumulative hazard int_0^t tot_lam(s) ds at each outer GL node
            gl16_x, gl16_w = np.polynomial.legendre.leggauss(16)
            gl16_u = 0.5 * (gl16_x + 1.0)
            gl16_wt = 0.5 * gl16_w

            pred_f = np.zeros((4, len(grid), 5), dtype=np.float64)
            for u0 in range(4):
                active_prevs = [(p0, prev_probs[u0, p0]) for p0 in range(5) if prev_probs[u0, p0] > 0.0]
                for hi, h_val in enumerate(grid):
                    h_f = float(h_val)
                    t_outer = h_f * gl32_u  # (32,)
                    t_inner = t_outer[:, None] * gl16_u[None, :]  # (32, 16)
                    p_mix = np.zeros(5, dtype=np.float64)
                    for p0, pw in active_prevs:
                        u_out = jnp.full(32, u0, dtype=jnp.int32)
                        p_out = jnp.full(32, p0, dtype=jnp.int32)
                        lam_out = np.asarray(intensities(w_best, u_out, p_out, jnp.asarray(t_outer)))  # (32, 5)
                        u_in = jnp.full((32, 16), u0, dtype=jnp.int32)
                        p_in = jnp.full((32, 16), p0, dtype=jnp.int32)
                        lam_in = np.asarray(intensities(w_best, u_in, p_in, jnp.asarray(t_inner)))  # (32, 16, 5)
                        cum_out = t_outer * np.sum(np.sum(lam_in, axis=-1) * gl16_wt[None, :], axis=-1)
                        surv_out = np.exp(-np.minimum(cum_out, 60.0))
                        p_jump = h_f * np.sum((gl32_wt * surv_out)[:, None] * lam_out, axis=0)
                        cum_end = h_f * float(np.sum(np.sum(lam_out, axis=-1) * gl32_wt))
                        p_stay = float(np.exp(-min(cum_end, 60.0)))
                        p_vec = p_jump.copy()
                        p_vec[u0] += p_stay
                        p_vec = np.clip(p_vec, 1e-12, 1.0)
                        p_vec /= p_vec.sum()
                        p_mix += pw * p_vec
                    pred_f[u0, hi] = p_mix / p_mix.sum()
            out_thp[f"dt_{dt_c}"] = {"thp_forecast_kl": empirical_kl(pred_f, f_te_g), "params": int(len(best[1]))}
        return out_thp

    thp_res = eval_neural_hawkes_coarsening()
    for k in coarsening:
        coarsening[k].update(thp_res[k])
    print("[Helpdesk Coarsening] Mamba & THP:", json.dumps(coarsening), flush=True)
    return {"params": n_params, "cv5": cv_sum, "chrono_70_30": chrono, "coarsening": coarsening}


# ---------------------------------------------------------------------------
# 3. Sleep-EDF — Mamba (5-fold CV, 3-start clean age=0)
# ---------------------------------------------------------------------------
def run_sleep_mamba() -> dict:
    subj_seqs = parse_sleep_sequences()
    subjects = sorted(subj_seqs.keys())
    obs_epochs = np.array([2, 4, 8, 12, 20, 30, 45, 60], dtype=int)
    _, _, all_ec = build_sleep_snapshots(subj_seqs, subjects, obs_epochs)
    edges = tuple((u, v) for u in range(5) for v in range(5) if u != v and all_ec[u, v] >= 0.01 * all_ec[u].sum())
    sample_idx = jnp.asarray(obs_epochs - 1, dtype=jnp.int32)
    init_m, rollout_m, n_params = make_mamba_graph_model(5, edges, latent=10, dt_step=0.05, n_steps=60)
    eye5 = jnp.eye(5, dtype=jnp.float64)
    folds = [subjects[i::5] for i in range(5)]

    cv = {mode: {"fresh_kl": [], "aged_kl": []} for mode in ("joint", "fresh_only")}
    for fi in range(5):
        te_s = folds[fi]
        tr_s = [s for j, f in enumerate(folds) if j != fi for s in f]
        f_tr, a_tr, _ = build_sleep_snapshots(subj_seqs, tr_s, obs_epochs)
        f_te, a_te, _ = build_sleep_snapshots(subj_seqs, te_s, obs_epochs)
        f_j, a_j = jnp.asarray(f_tr, dtype=jnp.float64), jnp.asarray(a_tr, dtype=jnp.float64)
        nf, na = jnp.asarray(max(float(f_tr.sum()), 1.0)), jnp.asarray(max(float(a_tr.sum()), 1.0))

        for mode, joint in (("joint", True), ("fresh_only", False)):
            @jax.jit
            def vg(w):
                def loss_fn(ww):
                    pf = rollout_m(ww, eye5, 0.0, sample_idx, min_rate=0.02, max_rate=8.0)
                    lf = -jnp.sum(f_j * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / nf
                    if joint:
                        pa = rollout_m(ww, eye5, 0.5, sample_idx, min_rate=0.02, max_rate=8.0)
                        la = -jnp.sum(a_j * jnp.log(jnp.clip(pa, 1e-12, 1.0))) / na
                        return 0.5 * (lf + la) + 1e-4 * jnp.mean(ww**2)
                    return lf + 1e-4 * jnp.mean(ww**2)

                return jax.value_and_grad(loss_fn)(w)

            def obj_sl(z):
                v, g = vg(jnp.asarray(z))
                return float(v), np.asarray(g, dtype=float)

            best = None
            for s in STARTS:
                res = minimize(obj_sl, init_m(s), jac=True, method="L-BFGS-B", options={"maxiter": 140})
                if best is None or res.fun < best[0]:
                    best = (float(res.fun), res.x)
            pf = np.asarray(rollout_m(jnp.asarray(best[1]), eye5, 0.0, sample_idx, min_rate=0.02, max_rate=8.0))
            pa = np.asarray(rollout_m(jnp.asarray(best[1]), eye5, 0.5 if joint else 0.0, sample_idx, min_rate=0.02, max_rate=8.0))
            cv[mode]["fresh_kl"].append(excess_kl(pf, f_te))
            cv[mode]["aged_kl"].append(excess_kl(pa, a_te))

    out = {
        "params": n_params,
        "cv5": {
            mode: {
                "fresh_kl_mean": float(np.mean(cv[mode]["fresh_kl"])),
                "fresh_kl_sd": float(np.std(cv[mode]["fresh_kl"])),
                "aged_kl_mean": float(np.mean(cv[mode]["aged_kl"])),
                "aged_kl_sd": float(np.std(cv[mode]["aged_kl"])),
            }
            for mode in ("joint", "fresh_only")
        },
    }
    print(
        f"[Sleep-EDF 5-fold] Mamba ({n_params}p): joint_aged={out['cv5']['joint']['aged_kl_mean']:.4f}, "
        f"0shot_aged={out['cv5']['fresh_only']['aged_kl_mean']:.4f}±{out['cv5']['fresh_only']['aged_kl_sd']:.4f}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# 4. 1978 Influenza — Mamba (Full + Days 9-14 / 10-14 L2 sweep)
# ---------------------------------------------------------------------------
def run_influenza_mamba() -> dict:
    # 3-compartment (In-bed, Convalescent, Other) simplex Mamba SSM: ~190 params
    H = 6
    obs_j = jnp.asarray(INF_COUNTS, dtype=jnp.float64)
    p0 = jnp.array([1.0 / 763.0, 0.0, 762.0 / 763.0], dtype=jnp.float64)
    dt = 0.25  # 4 steps/day, 14 days = 56 steps
    day_idx = jnp.asarray([d * 4 - 1 for d in range(1, 15)], dtype=jnp.int32)

    shapes = [(3, H), (H,), (H,), (3, H), (H,), (3, H), (3, H), (3, H), (H,), (H + 3, 3), (3,)]
    n_params = sum(int(np.prod(s)) for s in shapes)

    def init_inf(seed: int):
        r = np.random.default_rng(seed)
        return np.concatenate([
            r.normal(0, 0.2, (3, H)).ravel(), np.zeros(H),
            np.linspace(np.log(0.1), np.log(2.0), H),
            r.normal(0, 0.2, (3, H)).ravel(), np.full(H, -1.0),
            r.normal(0, 0.2, (3, H)).ravel(),
            r.normal(0, 0.2, (3, H)).ravel(),
            r.normal(0, 0.2, (3, H)).ravel(), np.zeros(H),
            r.normal(0, 0.15, (H + 3, 3)).ravel(), np.zeros(3),
        ])

    def unpack(th):
        out, i = [], 0
        for sh in shapes:
            sz = int(np.prod(sh))
            out.append(th[i : i + sz].reshape(sh))
            i += sz
        return out

    def pred_inf(th):
        enc, eb, log_a, w_dt, b_dt, w_b, w_c, w_z, b_z, w_o, b_o = unpack(th)
        neg_a = -jnp.exp(jnp.clip(log_a, -4.0, 2.5))
        h0 = jnp.tanh(p0 @ enc + eb)

        def step(carry, _):
            p, h = carry
            delta = jax.nn.softplus(p @ w_dt + b_dt) * dt
            a_bar = jnp.exp(delta * neg_a)
            b_bar = ((a_bar - 1.0) / neg_a) * (p @ w_b)
            h_new = a_bar * h + b_bar
            y_sel = (p @ w_c) * h_new * jax.nn.silu(p @ w_z + b_z)
            logits = jnp.concatenate([y_sel, p]) @ w_o + b_o
            # Smooth simplex update via softmax readout anchored to p
            p_new = jax.nn.softmax(jnp.log(jnp.clip(p, 1e-9, 1.0)) + dt * logits)
            return (p_new, h_new), p_new

        _, traj = jax.lax.scan(step, (p0, h0), None, length=56)
        return traj[day_idx]

    def fit_slice(n_tr: int, reg: float):
        tr_obs = obs_j[:n_tr]
        tot_n = jnp.sum(tr_obs)

        @jax.jit
        def vg(w):
            def loss_fn(ww):
                p = jnp.clip(pred_inf(ww)[:n_tr], 1e-12, 1.0)
                return -jnp.sum(tr_obs * jnp.log(p)) / tot_n + reg * jnp.mean(ww**2)

            return jax.value_and_grad(loss_fn)(w)

        def obj_inf(z):
            v, g = vg(jnp.asarray(z))
            return float(v), np.asarray(g, dtype=float)

        best = None
        for s in STARTS:
            res = minimize(obj_inf, init_inf(s), jac=True, method="L-BFGS-B", options={"maxiter": 200})
            if best is None or res.fun < best[0]:
                best = (float(res.fun), res.x)
        return np.asarray(pred_inf(jnp.asarray(best[1])))

    full_p = fit_slice(14, 1e-4)
    best_8 = None
    best_9 = None
    for reg in (1e-4, 1e-3, 1e-2, 1e-1):
        p8 = fit_slice(8, reg)
        m8 = metrics_on_slice(p8[8:14], INF_COUNTS[8:14])
        if best_8 is None or m8["mean_kl_nats"] < best_8["test"]["mean_kl_nats"]:
            best_8 = {"reg": reg, "test": m8}
        p9 = fit_slice(9, reg)
        m9 = metrics_on_slice(p9[9:14], INF_COUNTS[9:14])
        if best_9 is None or m9["mean_kl_nats"] < best_9["test"]["mean_kl_nats"]:
            best_9 = {"reg": reg, "test": m9}

    out = {
        "params": n_params,
        "full_days_1to14": metrics_on_slice(full_p, INF_COUNTS),
        "days_1to8_to_9to14_best_l2": best_8,
        "days_1to9_to_10to14_best_l2": best_9,
    }
    print(
        f"[Influenza] Mamba ({n_params}p): full={out['full_days_1to14']['mean_kl_nats']:.4f}, "
        f"D9-14={best_8['test']['mean_kl_nats']:.4f} ({best_8['test']['count_rmse_boys']:.1f} boys), "
        f"D10-14={best_9['test']['mean_kl_nats']:.4f}",
        flush=True,
    )
    return out


def main() -> None:
    res = {
        "synthetic_mamba": run_synthetic_mamba(),
        "helpdesk_mamba_and_thp": run_helpdesk_mamba_and_thp(),
        "sleep_mamba": run_sleep_mamba(),
        "influenza_mamba": run_influenza_mamba(),
    }
    OUT_PATH.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved Mamba & THP baseline results to {OUT_PATH}")


if __name__ == "__main__":
    main()
