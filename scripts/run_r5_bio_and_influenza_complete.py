"""Round-5 Complete Biological & Influenza Baselines:
1. Sleep-EDF: Equalized 3-start clean age=0 zero-shot neural ODEs (LTC, S4D, Neural-CME,
   Latent ODE) + 2-branch Universal Hyper-Pos-Sens (K_r=8 -> 45 states, 36 params).
2. Human Hematopoiesis (7-node tree, 10 donors): 3-start LTC & S4D on Donor 5-fold CV,
   pooled Days 1-5 -> Day 7, and strict 5-fold donor-held-out Days 1-5 -> unseen-donor Day 7,
   plus full-cone certified [1.4, 7.8] Universal Pos-Sens verification.
3. 1978 Influenza (763 boys): 3-start LTC & S4D on full Days 1-14 and unseen decline
   extrapolation (Days 1-8 -> 9-14 & Days 1-9 -> 10-14) across L2 in {5e-4, 5e-3, 2e-2, 1e-1}.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    build_universal_hyper_pos_network_generator,
    build_universal_pos_network_generator,
    calibrate_universal_pos_clock,
    multinomial_nll,
)
from scripts.run_edge_memory_sleep_benchmark import (  # noqa: E402
    build_sleep_snapshots,
    parse_sleep_sequences,
)
from scripts.run_hematopoiesis_branching_benchmark import (  # noqa: E402
    EDGES as HEMA_EDGES,
    aggregate_counts_by_day,
    load_hematopoiesis_facs,
    score_predictions,
)
from scripts.run_mean_field_epidemic_benchmark import (  # noqa: E402
    DAYS as INF_DAYS,
    OBS_COUNTS as INF_COUNTS,
    metrics_on_slice,
)


def excess_kl(pred: np.ndarray, counts: np.ndarray) -> float:
    tot = counts.sum(axis=-1, keepdims=True)
    emp = np.where(tot > 0, counts / np.maximum(tot, 1.0), 0.0)
    p = np.clip(pred, 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    kl = np.sum(np.where(emp > 0, emp * np.log(np.maximum(emp, 1e-30) / p), 0.0), axis=-1)
    w = tot[..., 0]
    return float(np.sum(kl * w) / np.maximum(np.sum(w), 1.0))


def run_hematopoiesis_ltc_s4d(psi_univ):
    df = load_hematopoiesis_facs()
    donors = sorted(df["donor_id"].unique())
    all_days = [1, 2, 3, 4, 5, 6, 7]
    ts_all = jnp.asarray(all_days, dtype=jnp.float64)
    init_hsc = jnp.zeros(7).at[0].set(1.0)
    n_edges = len(HEMA_EDGES)
    src_i = jnp.array([u for u, _ in HEMA_EDGES])
    dst_i = jnp.array([v for _, v in HEMA_EDGES])
    dt = 0.05
    n_steps = int(round(7.0 / dt))
    s_idx = jnp.rint(ts_all / dt).astype(int) - 1

    # LTC ODE (z=5, w=10 -> wh:12*10+10=130, wa:55, wt:55, wf:12*8+8=104 -> 344 params)
    z_l, w_l = 5, 10

    def init_ltc(seed):
        rng = np.random.default_rng(seed)
        return np.r_[
            rng.normal(0, 0.2, 12 * w_l), np.zeros(w_l),
            rng.normal(0, 0.15, w_l * z_l), np.zeros(z_l),
            rng.normal(0, 0.15, w_l * z_l), np.zeros(z_l),
            rng.normal(0, 0.15, 12 * n_edges), np.full(n_edges, -2.0),
        ]

    def pred_ltc(th):
        i = 0
        wh = th[i:i+12*w_l].reshape(12, w_l); i += 12*w_l
        bh = th[i:i+w_l]; i += w_l
        wa = th[i:i+w_l*z_l].reshape(w_l, z_l); i += w_l*z_l
        ba = th[i:i+z_l]; i += z_l
        wt = th[i:i+w_l*z_l].reshape(w_l, z_l); i += w_l*z_l
        bt = th[i:i+z_l]; i += z_l
        wf = th[i:i+12*n_edges].reshape(12, n_edges); i += 12*n_edges
        bf = th[i:i+n_edges]
        y0 = jnp.concatenate([init_hsc, jnp.zeros(z_l)])

        def fld(y):
            p, z = y[:7], y[7:]
            h = jnp.tanh(y @ wh + bh)
            dz = -(0.05 + 2.5 * jax.nn.sigmoid(h @ wt + bt)) * (z - jnp.tanh(h @ wa + ba))
            rates = 0.002 + 2.5 * jax.nn.sigmoid(y @ wf + bf)
            ef = p[src_i] * rates
            dp = jnp.zeros_like(p).at[dst_i].add(ef).at[src_i].add(-ef)
            return jnp.concatenate([dp, dz])

        def step(y, _):
            k1 = fld(y); k2 = fld(y + 0.5*dt*k1); k3 = fld(y + 0.5*dt*k2); k4 = fld(y + dt*k3)
            yn = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return yn, yn[:7]

        return jax.lax.scan(step, y0, None, length=n_steps)[1][s_idx]

    # S4D ODE (z=8, m=10 -> log_r:8, om:4, wb:7*8+8=64, wm:15*10+10=160, wo:10*8+8=88 -> 324 params)
    z_s, m_s = 8, 10

    def init_s4d(seed):
        rng = np.random.default_rng(seed)
        return np.r_[
            np.linspace(np.log(0.1), np.log(3.0), z_s), rng.normal(0, 0.2, z_s // 2),
            rng.normal(0, 0.2, 7 * z_s), np.zeros(z_s),
            rng.normal(0, 0.2, (7 + z_s) * m_s), np.zeros(m_s),
            rng.normal(0, 0.15, m_s * n_edges), np.full(n_edges, -2.0),
        ]

    def pred_s4d(th):
        half = z_s // 2
        i = 0
        lr = th[i:i+z_s]; i += z_s
        om = th[i:i+half]; i += half
        wb = th[i:i+7*z_s].reshape(7, z_s); i += 7*z_s
        bb = th[i:i+z_s]; i += z_s
        wm = th[i:i+(7+z_s)*m_s].reshape(7+z_s, m_s); i += (7+z_s)*m_s
        bm = th[i:i+m_s]; i += m_s
        wo = th[i:i+m_s*n_edges].reshape(m_s, n_edges); i += m_s*n_edges
        bo = th[i:i+n_edges]
        dec = jnp.exp(jnp.clip(lr, -4.0, 3.0))
        y0 = jnp.concatenate([init_hsc, jnp.zeros(z_s)])

        def fld(y):
            p, z = y[:7], y[7:]
            rot = jnp.concatenate([-om * z[half:], om * z[:half]])
            dz = -dec * z + rot + (p @ wb + bb)
            rates = 0.002 + 2.5 * jax.nn.sigmoid(jax.nn.gelu(y @ wm + bm) @ wo + bo)
            ef = p[src_i] * rates
            dp = jnp.zeros_like(p).at[dst_i].add(ef).at[src_i].add(-ef)
            return jnp.concatenate([dp, dz])

        def step(y, _):
            k1 = fld(y); k2 = fld(y + 0.5*dt*k1); k3 = fld(y + 0.5*dt*k2); k4 = fld(y + dt*k3)
            yn = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return yn, yn[:7]

        return jax.lax.scan(step, y0, None, length=n_steps)[1][s_idx]

    # Full-cone certified Universal Pos-Sens (k in [1.4, 7.8])
    def pred_univ(th):
        nm = jnp.concatenate([jnp.exp(th[:6]), jnp.array([1.0])])
        ns = jnp.concatenate([1.4 + 6.4 * jax.nn.sigmoid(th[6:12]), jnp.array([4.0])])
        s1, s2 = jax.nn.sigmoid(th[12]), jax.nn.sigmoid(th[13])
        rw = jnp.array([1.0, s1, 1.0 - s1, s2, 1.0 - s2, 1.0, 1.0, 1.0])
        gen, fr, rd = build_universal_pos_network_generator(nm, ns, rw, HEMA_EDGES, 7, psi_univ, stages_red=8)
        return jax.vmap(lambda t: rd @ jl.expm(t * gen) @ fr[:, 0])(ts_all)

    def make_vg(pfn, reg):
        @jax.jit
        def vg(w, c):
            def lf(ww):
                p = jnp.clip(pfn(ww), 1e-15, 1.0)
                p = p / jnp.sum(p, axis=1, keepdims=True)
                return -jnp.sum(c * jnp.log(p)) / jnp.maximum(jnp.sum(c), 1.0) + reg * jnp.sum(ww**2)
            return jax.value_and_grad(lf)(w)
        return vg

    def fit3(vg, ifn, counts, base_seed):
        cj = jnp.asarray(counts, dtype=jnp.float64)
        best = None
        for s in range(3):
            r = minimize(
                lambda z: tuple(map(lambda v: float(v) if np.ndim(v) == 0 else np.asarray(v, float), vg(jnp.asarray(z), cj))),
                ifn(base_seed + s), jac=True, method="L-BFGS-B", options={"maxiter": 220, "ftol": 1e-12}
            )
            if best is None or r.fun < best.fun:
                best = r
        return best.x

    vg_ltc = make_vg(pred_ltc, 2e-4)
    vg_s4d = make_vg(pred_s4d, 2e-4)
    vg_uni = make_vg(pred_univ, 1e-5)

    folds = [donors[i::5] for i in range(5)]
    specs = [
        ("ltc_ode", vg_ltc, init_ltc, pred_ltc, int(init_ltc(0).size), 12),
        ("s4d_ode", vg_s4d, init_s4d, pred_s4d, int(init_s4d(0).size), 15),
        ("universal_pos_sens_fullcone", vg_uni, lambda s: np.r_[np.full(6, 1.4 + 0.25 * (s % 3)), np.zeros(8)], pred_univ, 14, 49),
    ]
    out = {"donor_5fold_cv": {}, "pooled_day7": {}, "donor_heldout_day7": {}}
    _, c_all = aggregate_counts_by_day(df, all_days)
    c_p15 = c_all.copy(); c_p15[5:] = 0.0
    c_pd7 = np.zeros_like(c_all); c_pd7[6] = c_all[6]

    for mname, vg, ifn, pfn, npar, nst in specs:
        cv_s, d7_s = [], []
        for fi, te_d in enumerate(folds):
            tr_df = df[~df["donor_id"].isin(te_d)]
            te_df = df[df["donor_id"].isin(te_d)]
            _, c_tr = aggregate_counts_by_day(tr_df, all_days)
            _, c_te = aggregate_counts_by_day(te_df, all_days)
            w_cv = fit3(vg, ifn, c_tr, 100 + 10*fi)
            cv_s.append(score_predictions(np.asarray(pfn(jnp.asarray(w_cv))), c_te))
            c_tr15 = c_tr.copy(); c_tr15[5:] = 0.0
            c_ted7 = np.zeros_like(c_te); c_ted7[6] = c_te[6]
            w_ex = fit3(vg, ifn, c_tr15, 200 + 10*fi)
            d7_s.append(score_predictions(np.asarray(pfn(jnp.asarray(w_ex))), c_ted7))
        w_pool = fit3(vg, ifn, c_p15, 300)
        sc_pool = score_predictions(np.asarray(pfn(jnp.asarray(w_pool))), c_pd7)
        out["donor_5fold_cv"][mname] = {
            "states": nst, "params": npar,
            "kl": float(np.mean([r["kl"] for r in cv_s])),
            "prob_rmse": float(np.mean([r["prob_rmse"] for r in cv_s])),
        }
        out["pooled_day7"][mname] = {"day7_kl": sc_pool["kl"], "day7_rmse": sc_pool["prob_rmse"]}
        out["donor_heldout_day7"][mname] = {
            "kl_mean": float(np.mean([r["kl"] for r in d7_s])),
            "kl_sd": float(np.std([r["kl"] for r in d7_s], ddof=1)),
            "rmse_mean": float(np.mean([r["prob_rmse"] for r in d7_s])),
        }
        print(
            f"Hematopoiesis R5 {mname:27s} | CV={out['donor_5fold_cv'][mname]['kl']:.4f} "
            f"| PoolD7={sc_pool['kl']:.4f} | DonorD7={out['donor_heldout_day7'][mname]['kl_mean']:.4f}+/-{out['donor_heldout_day7'][mname]['kl_sd']:.3f}",
            flush=True,
        )
    return out


def run_influenza_ltc_s4d():
    ts_all = jnp.asarray(INF_DAYS, dtype=jnp.float64)
    dt = 0.05
    n_steps = int(round(14.0 / dt))
    s_idx = jnp.rint(ts_all / dt).astype(int) - 1
    z_l, w_l = 4, 8
    # LTC on 3 observed categories (in_bed, conval, other) + 4 latent = 7 states
    # init_logits: 2, z0: 4, wh: 7*8+8=64, wa: 8*4+4=36, wt: 8*4+4=36, wf: 7*3+3=24 -> 166 params

    def init_ltc(seed):
        rng = np.random.default_rng(seed)
        return np.r_[
            np.array([-5.5, -7.0]), np.zeros(z_l),
            rng.normal(0, 0.25, 7*w_l), np.zeros(w_l),
            rng.normal(0, 0.2, w_l*z_l), np.zeros(z_l),
            rng.normal(0, 0.2, w_l*z_l), np.zeros(z_l),
            rng.normal(0, 0.2, 7*3), np.zeros(3),
        ]

    def pred_ltc(th):
        i = 0
        il = th[i:i+2]; i += 2
        z0 = th[i:i+z_l]; i += z_l
        wh = th[i:i+7*w_l].reshape(7, w_l); i += 7*w_l
        bh = th[i:i+w_l]; i += w_l
        wa = th[i:i+w_l*z_l].reshape(w_l, z_l); i += w_l*z_l
        ba = th[i:i+z_l]; i += z_l
        wt = th[i:i+w_l*z_l].reshape(w_l, z_l); i += w_l*z_l
        bt = th[i:i+z_l]; i += z_l
        wf = th[i:i+7*3].reshape(7, 3); i += 7*3
        bf = th[i:i+3]
        p0 = jax.nn.softmax(jnp.array([il[0], il[1], 0.0]))
        y0 = jnp.concatenate([p0, z0])

        def fld(y):
            p, z = y[:3], y[3:]
            h = jnp.tanh(y @ wh + bh)
            dz = -(0.05 + 2.0 * jax.nn.sigmoid(h @ wt + bt)) * (z - jnp.tanh(h @ wa + ba))
            raw = y @ wf + bf
            dp = p * (raw - jnp.dot(p, raw))  # exact replicator simplex dynamics!
            return jnp.concatenate([dp, dz])

        def step(y, _):
            k1 = fld(y); k2 = fld(y + 0.5*dt*k1); k3 = fld(y + 0.5*dt*k2); k4 = fld(y + dt*k3)
            yn = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            pc = jnp.maximum(yn[:3], 1e-12); pc = pc / jnp.sum(pc)
            return jnp.concatenate([pc, yn[3:]]), pc

        return jax.lax.scan(step, y0, None, length=n_steps)[1][s_idx]

    # S4D on 3 categories + 6 latent = 9 states
    z_s, m_s = 6, 8

    def init_s4d(seed):
        rng = np.random.default_rng(seed)
        return np.r_[
            np.array([-5.5, -7.0]), np.zeros(z_s),
            np.linspace(np.log(0.1), np.log(2.5), z_s), rng.normal(0, 0.2, z_s // 2),
            rng.normal(0, 0.2, 3*z_s), np.zeros(z_s),
            rng.normal(0, 0.2, 9*m_s), np.zeros(m_s),
            rng.normal(0, 0.2, m_s*3), np.zeros(3),
        ]

    def pred_s4d(th):
        half = z_s // 2
        i = 0
        il = th[i:i+2]; i += 2
        z0 = th[i:i+z_s]; i += z_s
        lr = th[i:i+z_s]; i += z_s
        om = th[i:i+half]; i += half
        wb = th[i:i+3*z_s].reshape(3, z_s); i += 3*z_s
        bb = th[i:i+z_s]; i += z_s
        wm = th[i:i+9*m_s].reshape(9, m_s); i += 9*m_s
        bm = th[i:i+m_s]; i += m_s
        wo = th[i:i+m_s*3].reshape(m_s, 3); i += m_s*3
        bo = th[i:i+3]
        dec = jnp.exp(jnp.clip(lr, -4.0, 3.0))
        p0 = jax.nn.softmax(jnp.array([il[0], il[1], 0.0]))
        y0 = jnp.concatenate([p0, z0])

        def fld(y):
            p, z = y[:3], y[3:]
            rot = jnp.concatenate([-om * z[half:], om * z[:half]])
            dz = -dec * z + rot + (p @ wb + bb)
            raw = jax.nn.gelu(y @ wm + bm) @ wo + bo
            dp = p * (raw - jnp.dot(p, raw))
            return jnp.concatenate([dp, dz])

        def step(y, _):
            k1 = fld(y); k2 = fld(y + 0.5*dt*k1); k3 = fld(y + 0.5*dt*k2); k4 = fld(y + dt*k3)
            yn = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            pc = jnp.maximum(yn[:3], 1e-12); pc = pc / jnp.sum(pc)
            return jnp.concatenate([pc, yn[3:]]), pc

        return jax.lax.scan(step, y0, None, length=n_steps)[1][s_idx]

    out = {}
    for mname, ifn, pfn, nst in [("ltc_ode", init_ltc, pred_ltc, 7), ("s4d_ode", init_s4d, pred_s4d, 9)]:
        def fit_slice(n_tr, reg_val):
            obs_tr = jnp.asarray(INF_COUNTS[:n_tr], dtype=jnp.float64)
            n_tot = float(INF_COUNTS[:n_tr].sum())

            @jax.jit
            def vg(w):
                def lf(ww):
                    p = jnp.clip(pfn(ww)[:n_tr], 1e-12, 1.0)
                    return -jnp.sum(obs_tr * jnp.log(p)) / n_tot + reg_val * jnp.mean(ww**2)
                return jax.value_and_grad(lf)(w)

            best = None
            for s in range(3):
                r = minimize(
                    lambda z: tuple(map(lambda v: float(v) if np.ndim(v) == 0 else np.asarray(v, float), vg(jnp.asarray(z)))),
                    ifn(10 + s), jac=True, method="L-BFGS-B", options={"maxiter": 220}
                )
                if best is None or r.fun < best.fun:
                    best = r
            return np.asarray(pfn(jnp.asarray(best.x)))

        p_full = fit_slice(14, 5e-4)
        entry = {
            "states": nst, "params": int(ifn(0).size),
            "full_fit": metrics_on_slice(p_full, INF_COUNTS),
        }
        for n_tr in (8, 9):
            best_m, best_l2 = None, None
            for reg_val in (5e-4, 5e-3, 2e-2, 1e-1):
                p_ex = fit_slice(n_tr, reg_val)
                mt = metrics_on_slice(p_ex[n_tr:], INF_COUNTS[n_tr:])
                if best_m is None or mt["mean_kl_nats"] < best_m["mean_kl_nats"]:
                    best_m, best_l2 = mt, reg_val
            entry[f"days_1to{n_tr}_to_{n_tr+1}to14_best_l2"] = {"best_l2": best_l2, "test": best_m}
        print(
            f"Influenza R5 {mname:8s} | Full={entry['full_fit']['mean_kl_nats']:.4f} "
            f"| D9-14={entry['days_1to8_to_9to14_best_l2']['test']['mean_kl_nats']:.4f} ({entry['days_1to8_to_9to14_best_l2']['test']['count_rmse_boys']:.1f} boys) "
            f"| D10-14={entry['days_1to9_to_10to14_best_l2']['test']['mean_kl_nats']:.4f}",
            flush=True,
        )
        out[mname] = entry
    return out


def run_sleep_r5(psi_univ):
    subj_seqs = parse_sleep_sequences()
    subjects = sorted(subj_seqs.keys())
    obs_epochs = np.array([2, 4, 8, 12, 20, 30, 45, 60], dtype=int)
    aged_ten_min = 0.5
    _, _, all_ec = build_sleep_snapshots(subj_seqs, subjects, obs_epochs)
    edges = tuple(
        (u, v) for u in range(5) for v in range(5) if u != v and all_ec[u, v] >= 0.01 * all_ec[u].sum()
    )
    n_edges = len(edges)
    out_edges = [[] for _ in range(5)]
    for ei, (u, v) in enumerate(edges):
        out_edges[u].append(ei)
    dt = 0.05
    max_ep = int(obs_epochs[-1])
    sample_idx = jnp.asarray(obs_epochs - 1, dtype=jnp.int32)
    blk_sz = 9  # K_r=8 + 1 fast branch

    # 2-branch Universal Hyper-Pos-Sens (45 states, 5*4 + 16 = 36 params!)
    def pred_hpos(theta):
        fm = jnp.exp(jnp.clip(theta[:5], -4.5, 1.5))
        sm = jnp.exp(jnp.clip(theta[5:10], -1.5, 3.0))
        sk = 1.4 + 6.4 * jax.nn.sigmoid(theta[10:15])
        fw = jax.nn.sigmoid(theta[15:20])
        rl = theta[20 : 20 + n_edges]
        rw = jnp.zeros(n_edges, dtype=jnp.float64)
        for u in range(5):
            eids = jnp.asarray(out_edges[u])
            rw = rw.at[eids].set(jax.nn.softmax(rl[eids]))
        gen, fr, rd = build_universal_hyper_pos_network_generator(
            fm, sm, sk, fw, rw, edges, 5, psi_univ, stages_red=8
        )
        ag = jnp.zeros_like(fr)
        for u in range(5):
            s0, s1 = u * blk_sz, (u + 1) * blk_sz
            sa = jl.expm(aged_ten_min * gen[s0:s1, s0:s1]) @ fr[s0:s1, u]
            sa = jnp.maximum(sa, 1e-30)
            ag = ag.at[s0:s1, u].set(sa / jnp.sum(sa))
        tr = jl.expm(dt * gen)
        both = jnp.concatenate([fr, ag], axis=1)
        traj = jax.lax.scan(lambda st, _: (tr @ st, (rd @ (tr @ st)).T), both, None, length=max_ep)[1]
        sel = traj[sample_idx]
        return jnp.transpose(sel[:, :5, :], (1, 0, 2)), jnp.transpose(sel[:, 5:, :], (1, 0, 2))

    @jax.jit
    def vg_j(w, fc, ac):
        def lf(ww):
            pf, pa = pred_hpos(ww)
            return multinomial_nll(pf, fc) + multinomial_nll(pa, ac) + 2e-5 * jnp.sum(ww**2)
        return jax.value_and_grad(lf)(w)

    @jax.jit
    def vg_f(w, fc):
        def lf(ww):
            pf, _ = pred_hpos(ww)
            return multinomial_nll(pf, fc) + 2e-5 * jnp.sum(ww**2)
        return jax.value_and_grad(lf)(w)

    folds = [subjects[i::5] for i in range(5)]
    res = {m: {"fresh_kl": [], "aged_kl": []} for m in ("hyper_pos_sens_joint", "hyper_pos_sens_fresh_only")}
    for fi, te_s in enumerate(folds):
        tr_s = [s for s in subjects if s not in set(te_s)]
        f_tr, a_tr, _ = build_sleep_snapshots(subj_seqs, tr_s, obs_epochs)
        f_te, a_te, _ = build_sleep_snapshots(subj_seqs, te_s, obs_epochs)
        fc_j, ac_j = jnp.asarray(f_tr), jnp.asarray(a_tr)
        for mode in ("fresh_only", "joint"):
            bw, bv = None, float("inf")
            for st in range(2):
                rng = np.random.default_rng(900 + 10*fi + st)
                x0 = np.r_[np.full(5, -2.0), np.full(5, 0.5), rng.normal(0, 0.2, 5), np.zeros(5 + n_edges)]
                fn = (
                    (lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_j(jnp.asarray(x), fc_j, ac_j))))
                    if mode == "joint"
                    else (lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_f(jnp.asarray(x), fc_j))))
                )
                r = minimize(fn, x0, jac=True, method="L-BFGS-B", options={"maxiter": 160})
                if r.fun < bv:
                    bv, bw = float(r.fun), r.x.copy()
            pf, pa = pred_hpos(jnp.asarray(bw))
            res[f"hyper_pos_sens_{mode}"]["fresh_kl"].append(excess_kl(np.asarray(pf), f_te))
            res[f"hyper_pos_sens_{mode}"]["aged_kl"].append(excess_kl(np.asarray(pa), a_te))
        print(
            f"Sleep R5 Fold {fi+1}/5 Hyper-Pos-Sens | joint_aged={res['hyper_pos_sens_joint']['aged_kl'][-1]:.4f} "
            f"| 0-shot_aged={res['hyper_pos_sens_fresh_only']['aged_kl'][-1]:.4f}",
            flush=True,
        )

    return {
        "hyper_pos_sens_states": 45,
        "hyper_pos_sens_params": 20 + n_edges,
        "models_5fold_mean": {
            k: {
                "fresh_kl_mean": float(np.mean(v["fresh_kl"])),
                "fresh_kl_sd": float(np.std(v["fresh_kl"], ddof=1)),
                "aged_kl_mean": float(np.mean(v["aged_kl"])),
                "aged_kl_sd": float(np.std(v["aged_kl"], ddof=1)),
            }
            for k, v in res.items()
        },
    }


def main():
    psi_univ, univ_cert = calibrate_universal_pos_clock(
        stages_red=8, stages_full=16, maxiter=250, k_min=1.4, k_max=7.8
    )
    print(f"Full-cone Universal Pos-Sens [1.4, 7.8]: sup_surv={univ_cert['sup_survival_abs_error']:.5f}", flush=True)
    hema = run_hematopoiesis_ltc_s4d(psi_univ)
    inf = run_influenza_ltc_s4d()
    slp = run_sleep_r5(psi_univ)
    out = {
        "universal_clock_certificate_full_cone": univ_cert,
        "hematopoiesis_r5": hema,
        "influenza_r5": inf,
        "sleep_hyper_pos_sens_r5": slp,
    }
    out_path = ROOT / "results" / "archived" / "bio_influenza_r5_complete.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
