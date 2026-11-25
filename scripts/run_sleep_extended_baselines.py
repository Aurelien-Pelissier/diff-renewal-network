"""Extended Real-World Sleep-EDF Baselines: LTC, S4D, and Universal 1D Pos-Sens.

Evaluates on the exact same 75-adult 5-fold participant split of Sleep-EDF Expanded
(5 stages, 16 directed edges) as run_edge_memory_sleep_benchmark.py:
1. Multi-start Liquid Time-Constant ODE (LTC) conserving simplex mass.
2. Multi-start Diagonal State-Space ODE (S4D) conserving simplex mass.
3. Zero-shot Universal 1D Pos-Sens (K_r=8, 40 states, 26 params) plugged directly
   into the cyclic 5-node human sleep graph.
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
    build_universal_pos_network_generator,
    calibrate_universal_pos_clock,
    multinomial_nll,
)
from scripts.run_edge_memory_sleep_benchmark import (  # noqa: E402
    build_sleep_snapshots,
    parse_sleep_sequences,
)


def excess_kl(pred: np.ndarray, counts: np.ndarray) -> float:
    tot = counts.sum(axis=-1, keepdims=True)
    emp = np.where(tot > 0, counts / np.maximum(tot, 1.0), 0.0)
    p = np.clip(pred, 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    kl = np.sum(np.where(emp > 0, emp * np.log(np.maximum(emp, 1e-30) / p), 0.0), axis=-1)
    w = tot[..., 0]
    return float(np.sum(kl * w) / np.maximum(np.sum(w), 1.0))


def main():
    subj_seqs = parse_sleep_sequences()
    subjects = sorted(subj_seqs.keys())
    assert len(subjects) == 75
    obs_epochs = np.array([2, 4, 8, 12, 20, 30, 45, 60], dtype=int)
    aged_ten_min = 0.5

    _, _, all_edge_counts = build_sleep_snapshots(subj_seqs, subjects, obs_epochs)
    edges = tuple(
        (u, v)
        for u in range(5)
        for v in range(5)
        if u != v and all_edge_counts[u, v] >= 0.01 * all_edge_counts[u].sum()
    )
    n_edges = len(edges)
    src_idx = jnp.array([u for u, _ in edges])
    dst_idx = jnp.array([v for _, v in edges])

    dt = 0.05  # 30s epoch step in 10-min units (60 steps = 30 min)
    max_ep = int(obs_epochs[-1])
    sample_idx = jnp.asarray(obs_epochs - 1, dtype=jnp.int32)

    # 1. LTC ODE on 5-node, 16-edge Sleep graph
    z_ltc, w_ltc = 6, 12
    # enc: 6*6+6=42; wh: 11*12+12=144; wa: 78; wt: 78; wf: 11*16+16=192 -> 534 params
    def init_ltc(seed: int):
        rng = np.random.default_rng(seed)
        return np.concatenate(
            [
                rng.normal(0.0, 0.25, size=6 * z_ltc),
                np.zeros(z_ltc),
                rng.normal(0.0, 0.20, size=(5 + z_ltc) * w_ltc),
                np.zeros(w_ltc),
                rng.normal(0.0, 0.15, size=w_ltc * z_ltc),
                np.zeros(z_ltc),
                rng.normal(0.0, 0.15, size=w_ltc * z_ltc),
                np.zeros(z_ltc),
                rng.normal(0.0, 0.15, size=(5 + z_ltc) * n_edges),
                np.full(n_edges, -1.5),
            ]
        )

    def predict_ltc(theta):
        i = 0
        enc = theta[i : i + 6 * z_ltc].reshape(6, z_ltc); i += 6 * z_ltc
        eb = theta[i : i + z_ltc]; i += z_ltc
        wh = theta[i : i + (5 + z_ltc) * w_ltc].reshape(5 + z_ltc, w_ltc); i += (5 + z_ltc) * w_ltc
        bh = theta[i : i + w_ltc]; i += w_ltc
        wa = theta[i : i + w_ltc * z_ltc].reshape(w_ltc, z_ltc); i += w_ltc * z_ltc
        ba = theta[i : i + z_ltc]; i += z_ltc
        wt = theta[i : i + w_ltc * z_ltc].reshape(w_ltc, z_ltc); i += w_ltc * z_ltc
        bt = theta[i : i + z_ltc]; i += z_ltc
        wf = theta[i : i + (5 + z_ltc) * n_edges].reshape(5 + z_ltc, n_edges); i += (5 + z_ltc) * n_edges
        bf = theta[i : i + n_edges]

        def rollout(age_val: float):
            p0 = jnp.eye(5)
            z0 = jnp.tanh(jnp.concatenate([p0, jnp.full((5, 1), age_val)], axis=-1) @ enc + eb)
            y0 = jnp.concatenate([p0, z0], axis=-1)

            def rhs(y):
                p, z = y[:, :5], y[:, 5:]
                h = jnp.tanh(y @ wh + bh)
                dz = -(0.1 + 4.0 * jax.nn.sigmoid(h @ wt + bt)) * (z - jnp.tanh(h @ wa + ba))
                rates = 0.02 + 8.0 * jax.nn.sigmoid(y @ wf + bf)
                ef = p[:, src_idx] * rates
                dp = jnp.zeros_like(p).at[:, dst_idx].add(ef).at[:, src_idx].add(-ef)
                return jnp.concatenate([dp, dz], axis=-1)

            def step(y, _):
                k1 = rhs(y); k2 = rhs(y + 0.5 * dt * k1); k3 = rhs(y + 0.5 * dt * k2); k4 = rhs(y + dt * k3)
                yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                pc = jnp.maximum(yn[:, :5], 1e-15)
                pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
                return jnp.concatenate([pc, yn[:, 5:]], axis=-1), pc

            traj = jax.lax.scan(step, y0, None, length=max_ep)[1]
            return jnp.transpose(traj[sample_idx], (1, 0, 2))

        return rollout(0.0), rollout(aged_ten_min)

    # 2. S4D ODE on 5-node, 16-edge Sleep graph
    z_s4, m_s4 = 10, 12
    # enc: 70; log_r: 10; omega: 5; wb: 60; wm: 192; wo: 12*16+16=208 -> 545 params
    def init_s4d(seed: int):
        rng = np.random.default_rng(seed)
        return np.concatenate(
            [
                rng.normal(0.0, 0.25, size=6 * z_s4),
                np.zeros(z_s4),
                np.linspace(np.log(0.1), np.log(5.0), z_s4),
                rng.normal(0.0, 0.2, size=z_s4 // 2),
                rng.normal(0.0, 0.2, size=5 * z_s4),
                np.zeros(z_s4),
                rng.normal(0.0, 0.2, size=(5 + z_s4) * m_s4),
                np.zeros(m_s4),
                rng.normal(0.0, 0.15, size=m_s4 * n_edges),
                np.full(n_edges, -1.5),
            ]
        )

    def predict_s4d(theta):
        half = z_s4 // 2
        i = 0
        enc = theta[i : i + 6 * z_s4].reshape(6, z_s4); i += 6 * z_s4
        eb = theta[i : i + z_s4]; i += z_s4
        log_r = theta[i : i + z_s4]; i += z_s4
        omega = theta[i : i + half]; i += half
        wb = theta[i : i + 5 * z_s4].reshape(5, z_s4); i += 5 * z_s4
        bb = theta[i : i + z_s4]; i += z_s4
        wm = theta[i : i + (5 + z_s4) * m_s4].reshape(5 + z_s4, m_s4); i += (5 + z_s4) * m_s4
        bm = theta[i : i + m_s4]; i += m_s4
        wo = theta[i : i + m_s4 * n_edges].reshape(m_s4, n_edges); i += m_s4 * n_edges
        bo = theta[i : i + n_edges]
        decay = jnp.exp(jnp.clip(log_r, -4.0, 3.0))

        def rollout(age_val: float):
            p0 = jnp.eye(5)
            z0 = jnp.tanh(jnp.concatenate([p0, jnp.full((5, 1), age_val)], axis=-1) @ enc + eb)
            y0 = jnp.concatenate([p0, z0], axis=-1)

            def rhs(y):
                p, z = y[:, :5], y[:, 5:]
                rot = jnp.concatenate([-omega * z[:, half:], omega * z[:, :half]], axis=-1)
                dz = -decay * z + rot + (p @ wb + bb)
                rates = 0.02 + 8.0 * jax.nn.sigmoid(jax.nn.gelu(y @ wm + bm) @ wo + bo)
                ef = p[:, src_idx] * rates
                dp = jnp.zeros_like(p).at[:, dst_idx].add(ef).at[:, src_idx].add(-ef)
                return jnp.concatenate([dp, dz], axis=-1)

            def step(y, _):
                k1 = rhs(y); k2 = rhs(y + 0.5 * dt * k1); k3 = rhs(y + 0.5 * dt * k2); k4 = rhs(y + dt * k3)
                yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                pc = jnp.maximum(yn[:, :5], 1e-15)
                pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
                return jnp.concatenate([pc, yn[:, 5:]], axis=-1), pc

            traj = jax.lax.scan(step, y0, None, length=max_ep)[1]
            return jnp.transpose(traj[sample_idx], (1, 0, 2))

        return rollout(0.0), rollout(aged_ten_min)

    # 3. Universal 1D Pos-Sens (K_r=8 -> 40 states, full-cone [1.4, 7.8])
    psi_univ, univ_cert = calibrate_universal_pos_clock(
        stages_red=8, stages_full=16, maxiter=250, k_min=1.4, k_max=7.8
    )
    out_edges = [[] for _ in range(5)]
    for e_idx, (u, v) in enumerate(edges):
        out_edges[u].append(e_idx)

    def predict_univ(theta):
        nm = jnp.exp(jnp.clip(theta[:5], -3.0, 3.0))
        ns = 1.4 + 6.1 * jax.nn.sigmoid(theta[5:10])
        rl = theta[10 : 10 + n_edges]
        rw = jnp.zeros(n_edges, dtype=jnp.float64)
        for u in range(5):
            eids = jnp.asarray(out_edges[u])
            rw = rw.at[eids].set(jax.nn.softmax(rl[eids]))
        gen, fr, rd = build_universal_pos_network_generator(nm, ns, rw, edges, 5, psi_univ, stages_red=8)
        ag = jnp.zeros_like(fr)
        for u in range(5):
            s0, s1 = u * 8, (u + 1) * 8
            st_a = jl.expm(aged_ten_min * gen[s0:s1, s0:s1]) @ fr[s0:s1, u]
            st_a = jnp.maximum(st_a, 1e-30)
            ag = ag.at[s0:s1, u].set(st_a / jnp.sum(st_a))
        trans = jl.expm(dt * gen)
        both = jnp.concatenate([fr, ag], axis=1)

        def step(st, _):
            sn = trans @ st
            return sn, (rd @ sn).T

        traj = jax.lax.scan(step, both, None, length=max_ep)[1]
        sel = traj[sample_idx]
        return jnp.transpose(sel[:, :5, :], (1, 0, 2)), jnp.transpose(sel[:, 5:, :], (1, 0, 2))

    def make_fitter(pred_fn, reg: float = 1e-4):
        @jax.jit
        def vg_j(w, fc, ac):
            def lf(ww):
                pf, pa = pred_fn(ww)
                return multinomial_nll(pf, fc) + multinomial_nll(pa, ac) + reg * jnp.sum(ww**2)

            return jax.value_and_grad(lf)(w)

        @jax.jit
        def vg_f(w, fc):
            def lf(ww):
                pf, _ = pred_fn(ww)
                return multinomial_nll(pf, fc) + reg * jnp.sum(ww**2)

            return jax.value_and_grad(lf)(w)

        return vg_j, vg_f

    folds = [subjects[i::5] for i in range(5)]
    specs = [
        ("universal_pos_sens", lambda s: np.zeros(10 + n_edges), predict_univ, 1, 2e-5),
        ("ltc_ode", init_ltc, predict_ltc, 2, 1e-4),
        ("s4d_ode", init_s4d, predict_s4d, 2, 1e-4),
    ]
    res_all = {f"{m}_{mode}": {"fresh_kl": [], "aged_kl": []} for m, *_ in specs for mode in ("joint", "fresh_only")}

    for mname, init_fn, pred_fn, n_st, reg in specs:
        vg_j, vg_f = make_fitter(pred_fn, reg=reg)
        for fi, te_subjs in enumerate(folds):
            tr_subjs = [s for s in subjects if s not in set(te_subjs)]
            f_tr, a_tr, _ = build_sleep_snapshots(subj_seqs, tr_subjs, obs_epochs)
            f_te, a_te, _ = build_sleep_snapshots(subj_seqs, te_subjs, obs_epochs)
            fc_j, ac_j = jnp.asarray(f_tr), jnp.asarray(a_tr)

            for mode in ("joint", "fresh_only"):
                best_w, best_v = None, float("inf")
                for st in range(n_st):
                    x0 = init_fn(300 + 20 * fi + st)
                    if mode == "joint":
                        fun = lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_j(jnp.asarray(x), fc_j, ac_j)))
                    else:
                        fun = lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_f(jnp.asarray(x), fc_j)))
                    r = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter": 140})
                    if r.fun < best_v:
                        best_v, best_w = float(r.fun), r.x.copy()
                pf, pa = pred_fn(jnp.asarray(best_w))
                if mode == "fresh_only" and mname in ("ltc_ode", "s4d_ode"):
                    pa = pf
                fk = excess_kl(np.asarray(pf), f_te)
                ak = excess_kl(np.asarray(pa), a_te)
                res_all[f"{mname}_{mode}"]["fresh_kl"].append(fk)
                res_all[f"{mname}_{mode}"]["aged_kl"].append(ak)
                print(f"Sleep Fold {fi+1}/5 {mname}_{mode:10s}: fresh={fk:.5f} | aged={ak:.5f}", flush=True)

    out = {
        "universal_pos_sens_states": 40,
        "universal_pos_sens_params": 10 + n_edges,
        "ltc_ode_params": int(init_ltc(0).size),
        "s4d_ode_params": int(init_s4d(0).size),
        "models_5fold_mean": {
            k: {
                "fresh_kl_mean": float(np.mean(v["fresh_kl"])),
                "fresh_kl_sd": float(np.std(v["fresh_kl"], ddof=1)),
                "aged_kl_mean": float(np.mean(v["aged_kl"])),
                "aged_kl_sd": float(np.std(v["aged_kl"], ddof=1)),
                "folds": v,
            }
            for k, v in res_all.items()
        },
    }
    out_path = ROOT / "results" / "archived" / "sleep_extended_baselines.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
