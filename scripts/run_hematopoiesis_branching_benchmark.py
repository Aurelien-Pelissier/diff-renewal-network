"""Multi-Branching Primary Human Hematopoiesis Benchmark (Bast et al., iScience, 2021).

Evaluates Differentiable Renewal Networks, Zero-Shot Universal Topology-Agnostic Pos-Sens,
Memoryless CTMC, Time-Inhomogeneous Neural-CME(t), and Population Latent ODE on 78 destructive
FACS snapshots across 10 healthy human bone-marrow donor cultures (Days 1-7) over the 7-node
branching lineage hierarchy:
    HSC -> MPP -> {MLP, CMP},  CMP -> {GMP, MEP},  {MLP, GMP, MEP} -> Mature.

Tests:
1. 5-fold Donor-Grouped Cross-Validation (Days 1-7) on held-out human donor cultures.
2. Temporal Horizon Extrapolation (Train Days 1-5 -> Forecast Unseen Day 7 Lineage Emergence).
3. Biological Fate-Bifurcation Recovery (MPP lymphoid/myeloid & CMP granulocyte/erythroid splits + shapes).
"""

from hashlib import sha256
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
    build_universal_pos_network_generator,
    calibrate_universal_pos_clock,
    compact_network_state_size,
    moment_matched_logits,
    network_state_size,
    solve_compact_renewal_network,
)

HEMA_URL = "https://raw.githubusercontent.com/marrlab/HematopoiesisModelComparison/master/MATLAB/model_comparison_analysis/data/hematopoiesis_FACS_2019_07_data.txt"
HEMA_SHA256 = "6cc247fa32e41a021d07c20ea75e5ecc59fde2de5a4a0fc506713162e02dd17e"
CACHE_FILE = Path("/tmp/drn_hema_cache/hematopoiesis_FACS_2019_07_data.txt")

COMPARTMENTS = ["HSC", "MPP", "MLP", "CMP", "GMP", "MEP", "reifeZellen"]
COMP_LABELS = ["HSC", "MPP", "MLP", "CMP", "GMP", "MEP", "Mature"]
# Directed branching tree: 0:HSC -> 1:MPP -> {2:MLP, 3:CMP}, 3:CMP -> {4:GMP, 5:MEP}, {2,4,5} -> 6:Mature
EDGES = ((0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (2, 6), (4, 6), (5, 6))


def load_hematopoiesis_facs() -> pd.DataFrame:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists() or sha256(CACHE_FILE.read_bytes()).hexdigest() != HEMA_SHA256:
        raw = urllib.request.urlopen(HEMA_URL, timeout=30).read()
        assert sha256(raw).hexdigest() == HEMA_SHA256
        CACHE_FILE.write_bytes(raw)
    df = pd.read_csv(CACHE_FILE, sep=" ")
    assert len(df) == 78
    # Extract donor ID prefix before '_' in sample_name (e.g. '312_1a' -> '312')
    df["donor_id"] = df["sample_name"].astype(str).map(lambda s: s.split("_")[0])
    return df


def aggregate_counts_by_day(df: pd.DataFrame, days: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (days_arr, mean_composition_matrix [n_days, 7]) normalizing each FACS sample to simplex."""
    mat = np.zeros((len(days), 7), dtype=float)
    for i, d in enumerate(days):
        sub = df[df["Tag"] == d]
        if len(sub) > 0:
            raw = sub[COMPARTMENTS].to_numpy(dtype=float)
            row_sums = np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
            # Weight each biological replicate sample equally on the composition simplex
            mat[i] = (raw / row_sums).sum(axis=0)
    return np.asarray(days, dtype=float), mat


def score_predictions(pred_probs: np.ndarray, obs_counts: np.ndarray) -> dict:
    mask = obs_counts.sum(axis=1) > 0
    p = np.clip(pred_probs[mask], 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    c = obs_counts[mask]
    q = c / c.sum(axis=1, keepdims=True)
    w = c.sum(axis=1) / c.sum()
    kl = float(
        np.sum(
            w * np.sum(np.where(q > 0, q * (np.log(np.maximum(q, 1e-30)) - np.log(p)), 0.0), axis=1)
        )
    )
    rmse = float(np.sqrt(np.mean((p - q) ** 2)))
    return {"kl": kl, "prob_rmse": rmse}


def main() -> None:
    df = load_hematopoiesis_facs()
    donors = sorted(df["donor_id"].unique())
    all_days = [1, 2, 3, 4, 5, 6, 7]
    ts_all = jnp.asarray(all_days, dtype=float)
    init_hsc = jnp.zeros(7).at[0].set(1.0)
    n_edges = len(EDGES)

    # Calibrate 1D Universal Pos-Sens clock ONCE across full cone k in [1.4, 7.8] (completely data-free!)
    psi_univ, univ_cert = calibrate_universal_pos_clock(
        stages_red=8, stages_full=16, maxiter=250, k_min=1.4, k_max=7.8
    )

    k_stages = 16
    naive_states = network_state_size(EDGES, 7, k_stages)
    compact_states = compact_network_state_size(EDGES, 7, k_stages)

    # 1. Compiled DRN (K=16 marked-race): 8 edge means + 8 edge shapes (16 params)
    def predict_drn(theta):
        means = jnp.exp(theta[:n_edges])
        shapes = 1.2 + 10.8 * jax.nn.sigmoid(theta[n_edges:])
        # For shapes in [1.2, 3.0] smoothly blend stage-1..16 Gaussian logits
        logits = jax.vmap(lambda k_v: -0.5 * ((jnp.arange(1, k_stages + 1) - k_v) / 0.85) ** 2)(
            shapes
        )
        return solve_compact_renewal_network(means, logits, EDGES, init_hsc, ts_all).ys

    # 2. Zero-Shot Universal Pos-Sens Network (K_r=8 -> 49 states! 6 node means + 6 shapes + 2 split logits = 14 params)
    def predict_univ_pos(theta):
        # Nodes 0..5 active, node 6 sink
        node_means = jnp.concatenate([jnp.exp(theta[:6]), jnp.array([1.0])])
        node_shapes = jnp.concatenate([1.4 + 6.4 * jax.nn.sigmoid(theta[6:12]), jnp.array([4.0])])
        split_mpp = jax.nn.sigmoid(theta[12])  # MLP vs CMP
        split_cmp = jax.nn.sigmoid(theta[13])  # GMP vs MEP
        rw = jnp.array([1.0, split_mpp, 1.0 - split_mpp, split_cmp, 1.0 - split_cmp, 1.0, 1.0, 1.0])
        gen, fresh, read = build_universal_pos_network_generator(
            node_means, node_shapes, rw, EDGES, 7, psi_univ, stages_red=8
        )
        p_t = jax.vmap(lambda t: read @ jl.expm(t * gen) @ fresh[:, 0])(ts_all)
        return p_t

    univ_states = int(
        build_universal_pos_network_generator(
            np.ones(7), np.full(7, 5.0), np.ones(8), EDGES, 7, psi_univ, stages_red=8
        )[0].shape[0]
    )

    # 3. Memoryless CTMC (K=1, 8 edge rates)
    def predict_ctmc(theta):
        rates = jnp.exp(theta)
        gen = jnp.zeros((7, 7))
        for e_id, (u, v) in enumerate(EDGES):
            gen = gen.at[v, u].add(rates[e_id]).at[u, u].add(-rates[e_id])
        return jax.vmap(lambda t: jl.expm(t * gen) @ init_hsc)(ts_all)

    # 4. Time-Inhomogeneous Neural-CME(t) (width=10 -> 3*10 + 10 + 10*8 + 8 = 128 params)
    src_i = jnp.array([u for u, _ in EDGES])
    dst_i = jnp.array([v for _, v in EDGES])

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
                ef = pc[src_i] * rates
                return jnp.zeros_like(pc).at[dst_i].add(ef).at[src_i].add(-ef)

            k1 = fld(p)
            k2 = fld(p + 0.5 * dt * k1)
            k3 = fld(p + 0.5 * dt * k2)
            k4 = fld(p + dt * k3)
            pn = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return pn, pn

        return jax.lax.scan(step, init_hsc, jnp.arange(n_steps))[1][s_idx]

    # 5. Population Latent ODE (latent=5, width=10 -> 12*10 + 10 + 10*13 + 13 = 273 params)
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
                ef = p[src_i] * rates
                dp = jnp.zeros_like(p).at[dst_i].add(ef).at[src_i].add(-ef)
                return jnp.concatenate([dp, jnp.tanh(out[8:])])

            k1 = fld(y)
            k2 = fld(y + 0.5 * dt * k1)
            k3 = fld(y + 0.5 * dt * k2)
            k4 = fld(y + dt * k3)
            yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return yn, yn[:7]

        return jax.lax.scan(step, y0, None, length=n_steps)[1][s_idx]

    def make_fitter(pred_fn, reg: float):
        @jax.jit
        def loss(th, counts_m):
            p = jnp.clip(pred_fn(th), 1e-15, 1.0)
            p = p / jnp.sum(p, axis=1, keepdims=True)
            nll = -jnp.sum(counts_m * jnp.log(p)) / jnp.maximum(jnp.sum(counts_m), 1.0)
            return nll + reg * jnp.sum(th**2)

        return jax.jit(jax.value_and_grad(loss))

    vg_drn = make_fitter(predict_drn, 1e-5)
    vg_univ = make_fitter(predict_univ_pos, 1e-5)
    vg_ctmc = make_fitter(predict_ctmc, 1e-5)
    vg_cme = make_fitter(predict_cme, 2e-4)
    vg_lode = make_fitter(predict_lode, 2e-4)

    def fit_model(vg_fn, x0_list, counts_np, maxiter=220):
        c_j = jnp.asarray(counts_np)
        def _obj_h(z):
            v, g = vg_fn(jnp.asarray(z), c_j)
            return float(v), np.asarray(g, float)
        best = None
        for x0 in x0_list:
            res = minimize(
                _obj_h,
                x0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": 1e-12},
            )
            if best is None or res.fun < best.fun:
                best = res
        return best.x

    # Task 1: 5-fold Leave-Donor-Out Cross-Validation
    folds = [donors[i::5] for i in range(5)]
    cv_scores = {k: [] for k in ("compiled_drn", "universal_pos_sens", "ctmc", "neural_cme", "latent_ode")}

    for f_i, te_donors in enumerate(folds):
        tr_df = df[~df["donor_id"].isin(te_donors)]
        te_df = df[df["donor_id"].isin(te_donors)]
        _, c_tr = aggregate_counts_by_day(tr_df, all_days)
        _, c_te = aggregate_counts_by_day(te_df, all_days)

        rng = np.random.default_rng(400 + f_i)
        th_drn = fit_model(
            vg_drn,
            [np.r_[np.full(8, 1.5), np.zeros(8)], np.r_[np.full(8, 2.2), np.ones(8)]],
            c_tr,
        )
        th_univ = fit_model(
            vg_univ,
            [np.r_[np.full(6, 1.6), np.zeros(8)], np.r_[np.full(6, 2.2), np.ones(6), 0.0, 0.0]],
            c_tr,
        )
        th_ctmc = fit_model(vg_ctmc, [np.full(8, -1.8), np.full(8, -2.5)], c_tr)
        th_cme = fit_model(vg_cme, [rng.normal(0, 0.18, 128), rng.normal(0, 0.18, 128)], c_tr)
        th_lode = fit_model(vg_lode, [rng.normal(0, 0.18, 273), rng.normal(0, 0.18, 273)], c_tr)

        cv_scores["compiled_drn"].append(score_predictions(np.asarray(predict_drn(jnp.asarray(th_drn))), c_te))
        cv_scores["universal_pos_sens"].append(
            score_predictions(np.asarray(predict_univ_pos(jnp.asarray(th_univ))), c_te)
        )
        cv_scores["ctmc"].append(score_predictions(np.asarray(predict_ctmc(jnp.asarray(th_ctmc))), c_te))
        cv_scores["neural_cme"].append(score_predictions(np.asarray(predict_cme(jnp.asarray(th_cme))), c_te))
        cv_scores["latent_ode"].append(score_predictions(np.asarray(predict_lode(jnp.asarray(th_lode))), c_te))
        print(
            f"Hema Donor Fold {f_i} KL | DRN={cv_scores['compiled_drn'][-1]['kl']:.5f} | UnivPos={cv_scores['universal_pos_sens'][-1]['kl']:.5f} | CTMC={cv_scores['ctmc'][-1]['kl']:.5f} | CME={cv_scores['neural_cme'][-1]['kl']:.5f} | LODE={cv_scores['latent_ode'][-1]['kl']:.5f}",
            flush=True,
        )

    # Task 2: Temporal Horizon Extrapolation (Train Days 1-5 -> Predict Unseen Day 7 across all 12 donors!)
    _, c_full = aggregate_counts_by_day(df, all_days)
    c_train_1to5 = c_full.copy()
    c_train_1to5[5:] = 0.0  # mask out Day 6 and Day 7 completely during training!
    c_test_day7 = np.zeros_like(c_full)
    c_test_day7[6] = c_full[6]  # evaluate strictly on unseen Day 7!

    rng_ex = np.random.default_rng(909)
    th_drn_ex = fit_model(
        vg_drn, [np.r_[np.full(8, 1.5), np.zeros(8)], np.r_[np.full(8, 2.2), np.ones(8)]], c_train_1to5
    )
    th_univ_ex = fit_model(
        vg_univ,
        [np.r_[np.full(6, 1.6), np.zeros(8)], np.r_[np.full(6, 2.2), np.ones(6), 0.0, 0.0]],
        c_train_1to5,
    )
    th_ctmc_ex = fit_model(vg_ctmc, [np.full(8, -1.8), np.full(8, -2.5)], c_train_1to5)
    th_cme_ex = fit_model(vg_cme, [rng_ex.normal(0, 0.18, 128), rng_ex.normal(0, 0.18, 128)], c_train_1to5)
    th_lode_ex = fit_model(vg_lode, [rng_ex.normal(0, 0.18, 273), rng_ex.normal(0, 0.18, 273)], c_train_1to5)

    extrap_day7 = {
        "compiled_drn": score_predictions(np.asarray(predict_drn(jnp.asarray(th_drn_ex))), c_test_day7),
        "universal_pos_sens": score_predictions(
            np.asarray(predict_univ_pos(jnp.asarray(th_univ_ex))), c_test_day7
        ),
        "ctmc": score_predictions(np.asarray(predict_ctmc(jnp.asarray(th_ctmc_ex))), c_test_day7),
        "neural_cme": score_predictions(np.asarray(predict_cme(jnp.asarray(th_cme_ex))), c_test_day7),
        "latent_ode": score_predictions(np.asarray(predict_lode(jnp.asarray(th_lode_ex))), c_test_day7),
    }

    # Fit full-cohort DRN & Universal Pos-Sens for trajectory visualization & biological bifurcation readout
    th_univ_full = fit_model(
        vg_univ, [np.r_[np.full(6, 1.6), np.zeros(8)], np.r_[np.full(6, 2.2), np.ones(6), 0.0, 0.0]], c_full
    )
    th_drn_full = fit_model(
        vg_drn, [np.r_[np.full(8, 1.5), np.zeros(8)], np.r_[np.full(8, 2.2), np.ones(8)]], c_full
    )
    emp_props = (c_full / np.maximum(c_full.sum(axis=1, keepdims=True), 1.0)).tolist()

    out = {
        "dataset": "Bast et al. (iScience, 2021) Primary Human Hematopoiesis FACS",
        "doi": "10.1016/j.isci.2021.102120",
        "samples": int(len(df)),
        "donors": int(len(donors)),
        "compartments": COMP_LABELS,
        "edges": [list(e) for e in EDGES],
        "state_dimensions": {
            "naive_tensor_k16": naive_states,
            "compiled_race_k16": compact_states,
            "universal_pos_sens_kr8": univ_states,
        },
        "universal_clock_1d_certificate": univ_cert,
        "donor_5fold_cv_mean": {
            m: {
                "kl": round(float(np.mean([r["kl"] for r in rows])), 5),
                "kl_sd": round(float(np.std([r["kl"] for r in rows], ddof=1)), 5),
                "prob_rmse": round(float(np.mean([r["prob_rmse"] for r in rows])), 5),
            }
            for m, rows in cv_scores.items()
        },
        "day1to5_to_day7_extrapolation": {
            m: {"day7_kl": round(v["kl"], 5), "day7_rmse": round(v["prob_rmse"], 5)}
            for m, v in extrap_day7.items()
        },
        "inferred_bifurcations_univ_pos": {
            "mpp_to_mlp_fraction": round(float(jax.nn.sigmoid(th_univ_full[12])), 3),
            "mpp_to_cmp_fraction": round(float(1.0 - jax.nn.sigmoid(th_univ_full[12])), 3),
            "cmp_to_gmp_fraction": round(float(jax.nn.sigmoid(th_univ_full[13])), 3),
            "cmp_to_mep_fraction": round(float(1.0 - jax.nn.sigmoid(th_univ_full[13])), 3),
        },
        "trajectories_days_1to7": {
            "days": all_days,
            "empirical_fractions": emp_props,
            "compiled_drn_fractions": np.asarray(predict_drn(jnp.asarray(th_drn_full))).tolist(),
            "universal_pos_fractions": np.asarray(predict_univ_pos(jnp.asarray(th_univ_full))).tolist(),
            "ctmc_fractions": np.asarray(
                predict_ctmc(jnp.asarray(fit_model(vg_ctmc, [np.full(8, -1.8)], c_full)))
            ).tolist(),
        },
    }
    out_path = ROOT / "results" / "archived" / "hematopoiesis_branching_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps({k: out[k] for k in ("state_dimensions", "donor_5fold_cv_mean", "day1to5_to_day7_extrapolation", "inferred_bifurcations_univ_pos")}, indent=2))


if __name__ == "__main__":
    main()
