"""Strict Donor-Held-Out Days 1-5 -> Unseen-Donor Day-7 Hematopoiesis Extrapolation.

Addresses pooled-donor evaluation by training each model strictly on Days 1-5
of 8 training donors and evaluating strictly on Day 7 (and Days 6-7) of the 2
completely unseen held-out donors across all 5 donor folds (mean +/- SD).
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
    solve_compact_renewal_network,
)
from scripts.run_hematopoiesis_branching_benchmark import (  # noqa: E402
    EDGES,
    aggregate_counts_by_day,
    load_hematopoiesis_facs,
    score_predictions,
)


def main():
    df = load_hematopoiesis_facs()
    donors = sorted(df["donor_id"].unique())
    assert len(donors) == 10
    all_days = [1, 2, 3, 4, 5, 6, 7]
    ts_all = jnp.asarray(all_days, dtype=jnp.float64)
    init_hsc = jnp.zeros(7).at[0].set(1.0)
    n_edges = len(EDGES)
    k_stages = 16
    src_idx = jnp.array([u for u, _ in EDGES])
    dst_idx = jnp.array([v for _, v in EDGES])

    psi_univ, _ = calibrate_universal_pos_clock(
        stages_red=8, stages_full=16, maxiter=250, k_min=1.4, k_max=7.8
    )

    def predict_drn(theta):
        means = jnp.exp(theta[:n_edges])
        shapes = 1.2 + 10.8 * jax.nn.sigmoid(theta[n_edges:])
        logits = jax.vmap(lambda kv: -0.5 * ((jnp.arange(1, k_stages + 1) - kv) / 0.85) ** 2)(shapes)
        return solve_compact_renewal_network(means, logits, EDGES, init_hsc, ts_all).ys

    def predict_univ(theta):
        nm = jnp.concatenate([jnp.exp(theta[:6]), jnp.array([1.0])])
        ns = jnp.concatenate([1.4 + 6.4 * jax.nn.sigmoid(theta[6:12]), jnp.array([4.0])])
        s_mpp, s_cmp = jax.nn.sigmoid(theta[12]), jax.nn.sigmoid(theta[13])
        rw = jnp.array([1.0, s_mpp, 1.0 - s_mpp, s_cmp, 1.0 - s_cmp, 1.0, 1.0, 1.0])
        gen, fr, rd = build_universal_pos_network_generator(nm, ns, rw, EDGES, 7, psi_univ, stages_red=8)
        return jax.vmap(lambda t: rd @ jl.expm(t * gen) @ fr[:, 0])(ts_all)

    def predict_ctmc(theta):
        rates = jnp.exp(theta)
        gen = jnp.zeros((7, 7)).at[dst_idx, src_idx].add(rates).at[src_idx, src_idx].add(-rates)
        return jax.vmap(lambda t: jl.expm(t * gen) @ init_hsc)(ts_all)

    def predict_cme(theta):
        w1, b1 = theta[:30].reshape(3, 10), theta[30:40]
        w2, b2 = theta[40:120].reshape(10, 8), theta[120:128]
        dt = 0.05
        n_steps = int(round(7.0 / dt))
        s_idx = jnp.rint(ts_all / dt).astype(int) - 1

        def step(p, i_s):
            tn = (i_s + 0.5) * dt / 7.0
            f_t = jnp.array([tn, tn**2, jnp.sin(jnp.pi * tn)])
            rates = 0.002 + 2.5 * jax.nn.sigmoid(jax.nn.gelu(f_t @ w1 + b1) @ w2 + b2 - 2.0)

            def fld(pc):
                ef = pc[src_idx] * rates
                return jnp.zeros_like(pc).at[dst_idx].add(ef).at[src_idx].add(-ef)

            k1 = fld(p); k2 = fld(p + 0.5 * dt * k1); k3 = fld(p + 0.5 * dt * k2); k4 = fld(p + dt * k3)
            pn = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return pn, pn

        return jax.lax.scan(step, init_hsc, jnp.arange(n_steps))[1][s_idx]

    def predict_lode(theta):
        w1, b1 = theta[:120].reshape(12, 10), theta[120:130]
        w2, b2 = theta[130:260].reshape(10, 13), theta[260:273]
        dt = 0.05
        n_steps = int(round(7.0 / dt))
        s_idx = jnp.rint(ts_all / dt).astype(int) - 1
        y0 = jnp.concatenate([init_hsc, jnp.zeros(5)])

        def step(y, _):
            def fld(yc):
                p = yc[:7]
                out = jnp.tanh(yc @ w1 + b1) @ w2 + b2
                rates = 0.002 + 2.5 * jax.nn.sigmoid(out[:8] - 2.0)
                ef = p[src_idx] * rates
                dp = jnp.zeros_like(p).at[dst_idx].add(ef).at[src_idx].add(-ef)
                return jnp.concatenate([dp, jnp.tanh(out[8:])])

            k1 = fld(y); k2 = fld(y + 0.5 * dt * k1); k3 = fld(y + 0.5 * dt * k2); k4 = fld(y + dt * k3)
            yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return yn, yn[:7]

        return jax.lax.scan(step, y0, None, length=n_steps)[1][s_idx]

    def make_vg(pred_fn, reg: float = 1e-5):
        @jax.jit
        def vg(w, counts):
            def lf(ww):
                p = jnp.clip(pred_fn(ww), 1e-15, 1.0)
                p = p / jnp.sum(p, axis=1, keepdims=True)
                return -jnp.sum(counts * jnp.log(p)) / jnp.maximum(jnp.sum(counts), 1.0) + reg * jnp.sum(ww**2)

            return jax.value_and_grad(lf)(w)

        return vg

    def fit(vg, inits, counts):
        c_j = jnp.asarray(counts, dtype=jnp.float64)
        best = None
        for x0 in inits:
            r = minimize(
                lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg(jnp.asarray(x), c_j))),
                np.asarray(x0, dtype=float),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 220, "ftol": 1e-12},
            )
            if best is None or r.fun < best.fun:
                best = r
        return best.x

    vg_drn = make_vg(predict_drn, 1e-5)
    vg_univ = make_vg(predict_univ, 1e-5)
    vg_ctmc = make_vg(predict_ctmc, 1e-5)
    vg_cme = make_vg(predict_cme, 2e-4)
    vg_lode = make_vg(predict_lode, 2e-4)

    folds = [donors[i::5] for i in range(5)]
    models = ("compiled_drn", "universal_pos_sens", "ctmc", "neural_cme", "latent_ode")
    day7_scores = {m: [] for m in models}
    day67_scores = {m: [] for m in models}

    for fi, te_donors in enumerate(folds):
        tr_df = df[~df["donor_id"].isin(te_donors)]
        te_df = df[df["donor_id"].isin(te_donors)]
        _, c_tr = aggregate_counts_by_day(tr_df, all_days)
        _, c_te = aggregate_counts_by_day(te_df, all_days)
        c_tr_1to5 = c_tr.copy(); c_tr_1to5[5:] = 0.0
        c_te_d7 = np.zeros_like(c_te); c_te_d7[6] = c_te[6]
        c_te_d67 = np.zeros_like(c_te); c_te_d67[5:] = c_te[5:]

        rng = np.random.default_rng(500 + fi)
        th = {
            "compiled_drn": fit(vg_drn, [np.r_[np.full(8, 1.5), np.zeros(8)], np.r_[np.full(8, 2.2), np.ones(8)]], c_tr_1to5),
            "universal_pos_sens": fit(vg_univ, [np.r_[np.full(6, 1.6), np.zeros(8)], np.r_[np.full(6, 2.2), np.ones(6), 0.0, 0.0]], c_tr_1to5),
            "ctmc": fit(vg_ctmc, [np.full(8, -1.8), np.full(8, -2.5)], c_tr_1to5),
            "neural_cme": fit(vg_cme, [rng.normal(0, 0.18, 128), rng.normal(0, 0.18, 128)], c_tr_1to5),
            "latent_ode": fit(vg_lode, [rng.normal(0, 0.18, 273), rng.normal(0, 0.18, 273)], c_tr_1to5),
        }
        preds = {
            "compiled_drn": np.asarray(predict_drn(jnp.asarray(th["compiled_drn"]))),
            "universal_pos_sens": np.asarray(predict_univ(jnp.asarray(th["universal_pos_sens"]))),
            "ctmc": np.asarray(predict_ctmc(jnp.asarray(th["ctmc"]))),
            "neural_cme": np.asarray(predict_cme(jnp.asarray(th["neural_cme"]))),
            "latent_ode": np.asarray(predict_lode(jnp.asarray(th["latent_ode"]))),
        }
        for m in models:
            if c_te_d7.sum() > 0:
                day7_scores[m].append(score_predictions(preds[m], c_te_d7))
            if c_te_d67.sum() > 0:
                day67_scores[m].append(score_predictions(preds[m], c_te_d67))
        if c_te_d67.sum() > 0:
            print(
                f"Donor-Held-Out Extrap Fold {fi+1}/5 (Days 6-7 unseen donors) | "
                + " | ".join(f"{m}={day67_scores[m][-1]['kl']:.4f}" for m in models),
                flush=True,
            )

    def agg(d):
        return {
            m: {
                "n_folds_with_target": len(rows),
                "kl_mean": float(np.mean([r["kl"] for r in rows])),
                "kl_sd": float(np.std([r["kl"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "rmse_mean": float(np.mean([r["prob_rmse"] for r in rows])),
                "rmse_sd": float(np.std([r["prob_rmse"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
            for m, rows in d.items()
        }

    out = {
        "protocol": "Strict 5-fold donor-held-out temporal extrapolation: train on Days 1-5 of 8 donors, evaluate on unseen Days 6-7 and Day 7 of 2 held-out donors",
        "donors_total": len(donors),
        "unseen_donor_day7": agg(day7_scores),
        "unseen_donor_days6to7": agg(day67_scores),
    }
    out_path = ROOT / "results" / "archived" / "hematopoiesis_donor_extrap.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
