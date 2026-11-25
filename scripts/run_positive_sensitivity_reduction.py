"""Benchmark Positivity-Preserving Fisher-Sensitivity Reduction (Pos-Sens 33) on all 400 cyclic datasets.

Evaluates Pos-Sens 33 alongside Full 48, Symmetry 33, Interp. 33 (Baur et al., 2011),
and Interp. 18 across all 280 primary and 120 later-time 10,000-cell datasets using
the identical two-start SLSQP likelihood optimizer and paired bootstrap protocol.
"""

import json
import os
from pathlib import Path
import sys
from time import perf_counter
import warnings

import jax

jax.config.update("jax_enable_x64", True)
import jax.scipy.linalg as jl
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal.reduction import (  # noqa: E402
    READ_POS33,
    build_interpolatory_basis,
    calibrate_positive_33_surrogate,
    make_cycle_engine,
    make_interpolatory_engine,
    make_positive_33_engine,
    pos33_cycle_matrices,
)

NM_ROOT = Path(os.environ.get("DRN_DATA_ROOT", str(ROOT.parent / "non-markovian-diffrax")))
STARTS = [np.log([m] * 3 + [k] * 3) for m, k in [(1.8, 5.3), (2.3, 7.7)]]
BOUNDS = [(np.log(0.95), np.log(3.8))] * 3 + [(np.log(3.0), np.log(12.0))] * 3
FLOOR = 1e-8
SAMPLE = np.arange(2, 14)
LATE_STRATA = np.array([4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13])
LATE_ROWS = np.r_[0, 1, LATE_STRATA]

ORIGIN = np.log([2.0, 2.0, 2.0, 6.0, 6.0, 6.0])
MEAN_DIR = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0])
SHAPE_DIR = np.array([0.0, 0.0, 0.0, -1.0, 1.0, 0.0])
MIXED_DIR = MEAN_DIR + SHAPE_DIR
REGIMES = {
    name: ORIGIN + eps * axis
    for name, eps, axis in [
        ("symmetric", 0.0, MIXED_DIR),
        ("mixed_025", 0.025, MIXED_DIR),
        ("mixed_075", 0.075, MIXED_DIR),
        ("mixed_150", 0.15, MIXED_DIR),
        ("mixed_250", 0.25, MIXED_DIR),
        ("mean_only_150", 0.15, MEAN_DIR),
        ("shape_only_150", 0.15, SHAPE_DIR),
    ]
}


def wrap_late(engine):
    def evaluate(theta):
        p, j = engine(theta)
        return p[LATE_ROWS], j[LATE_ROWS]

    return evaluate


def fit_slsqp(engine, counts: np.ndarray, truth: np.ndarray) -> dict:
    n = counts.ravel() / counts.sum()
    cached_theta, cached_res = None, None

    def evaluate(theta):
        nonlocal cached_theta, cached_res
        if cached_theta is None or not np.array_equal(theta, cached_theta):
            p, jac = engine(theta)
            cached_res = p[SAMPLE].ravel(), jac[SAMPLE].reshape(-1, 6)
            cached_theta = theta.copy()
        return cached_res

    def objective(theta):
        p, jac = evaluate(theta)
        interior = p >= FLOOR
        logp = np.log(np.maximum(p, FLOOR))
        logp[~interior] += (p[~interior] - FLOOR) / FLOOR
        return float(-n @ logp), -(n / np.maximum(p, FLOOR)) @ jac

    constraints = dict(
        type="ineq",
        fun=lambda x: evaluate(x)[0] - FLOOR,
        jac=lambda x: evaluate(x)[1],
    )
    records = []
    t0 = perf_counter()
    for initial in STARTS:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Values in x were outside bounds", category=RuntimeWarning
            )
            res = minimize(
                objective,
                initial,
                method="SLSQP",
                jac=True,
                bounds=BOUNDS,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 150},
            )
        p, _ = evaluate(res.x)
        feasible = bool(np.isfinite(res.fun) and p.min() >= FLOOR - 1e-12)
        records.append(
            dict(
                theta=res.x.tolist(),
                loss=float(res.fun),
                feasible=feasible,
                converged=bool(res.success),
            )
        )
    elapsed = perf_counter() - t0
    candidates = [r for r in records if r["feasible"] and r["converged"]]
    if not candidates:
        return dict(valid=False, seconds=elapsed)
    best = min(candidates, key=lambda r: r["loss"])
    theta_hat = np.asarray(best["theta"])
    p_full_grid, _ = engine(theta_hat)
    err = (np.exp(theta_hat - truth) - 1.0).reshape(2, 3)
    rms = np.sqrt(np.mean(err**2, axis=1))
    return dict(
        valid=True,
        theta=theta_hat.tolist(),
        mean_rms=float(rms[0]),
        shape_rms=float(rms[1]),
        joint=bool(rms[0] <= 0.05 and rms[1] <= 0.10),
        loss=float(best["loss"]),
        min_fitted_probability=float(p_full_grid[SAMPLE].min()),
        min_early_extrapolation=float(p_full_grid[: SAMPLE[0]].min()),
        seconds=elapsed,
    )


def paired_bootstrap_ci(diffs: np.ndarray, seed: int = 99, n_boot: int = 10_000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    meds = np.median(diffs[idx], axis=1)
    return float(np.median(diffs)), [
        float(np.percentile(meds, 2.5)),
        float(np.percentile(meds, 97.5)),
    ]


def main() -> None:
    psi_star, cal_meta = calibrate_positive_33_surrogate(seed=0, maxiter=250)
    pos33_eval = make_positive_33_engine(psi_star)
    full48_eval = make_cycle_engine(reduced=False)
    q33, _ = build_interpolatory_basis(33)
    sp33_eval = make_interpolatory_engine(q33)

    # Warmup JIT
    for eng in (pos33_eval, full48_eval, sp33_eval):
        eng(STARTS[0])

    # Check diagnostics across all 7 regimes including fine early grid t in [0.02, 1.5]
    early_ts = np.linspace(0.02, 1.48, 30)
    regime_diagnostics = {}
    for rname, th in REGIMES.items():
        pf, jf = full48_eval(th)
        pr, jr = pos33_eval(th)
        ps, js = sp33_eval(th)
        gen_pos, fresh_pos = pos33_cycle_matrices(th, psi_star)
        min_dense_early = min(
            float(np.asarray(READ_POS33 @ jl.expm(float(t) * gen_pos) @ fresh_pos).min())
            for t in early_ts
        )
        regime_diagnostics[rname] = {
            "pos33_prob_max_abs": float(np.max(np.abs(pf[SAMPLE] - pr[SAMPLE]))),
            "pos33_jac_max_abs": float(np.max(np.abs(jf[SAMPLE] - jr[SAMPLE]))),
            "sp33_prob_max_abs": float(np.max(np.abs(pf[SAMPLE] - ps[SAMPLE]))),
            "sp33_jac_max_abs": float(np.max(np.abs(jf[SAMPLE] - js[SAMPLE]))),
            "pos33_min_early_prob": min_dense_early,
        }

    # Load archived 400 datasets and existing sensitivity_evidence.json
    sens_prev = json.loads(
        (ROOT / "results" / "archived" / "sensitivity_evidence.json").read_text(encoding="utf-8")
    )
    primary_raw = json.loads(
        (NM_ROOT / "results" / "cycle_mode_recovery.json").read_text(encoding="utf-8")
    )
    late_raw = json.loads(
        (NM_ROOT / "results" / "cycle_mode_late.json").read_text(encoding="utf-8")
    )
    archived_sr = json.loads(
        (NM_ROOT / "results" / "sensitivity_reduction.json").read_text(encoding="utf-8")
    )

    cases_summary = {}
    all_speedups = {"primary": [], "late": []}
    overall_counts = {
        "primary": {"joint": 0, "shape_errs": [], "mean_errs": [], "early_neg": 0, "near_floor": 0, "warm_ms": []},
        "late": {"joint": 0, "shape_errs": [], "mean_errs": [], "early_neg": 0, "near_floor": 0, "warm_ms": []},
    }

    for design, src_data, engine_pos, engine_full in [
        ("primary", primary_raw, pos33_eval, full48_eval),
        ("late", late_raw, wrap_late(pos33_eval), wrap_late(full48_eval)),
    ]:
        for case_name, group in src_data["cases"].items():
            key = f"{design}/{case_name}"
            theta_true = np.asarray(group["theta"])
            sr_runs = {r["seed"]: r["fits"] for r in archived_sr["cases"][key]["runs"]}

            pos_shapes, pos_means, full_shapes = [], [], []
            joint_hits = 0
            for idx, row in enumerate(group["runs"]):
                counts = np.asarray(row["counts"])
                # Timed paired comparison with Full 48 in alternating order
                if idx % 2 == 0:
                    f_pos = fit_slsqp(engine_pos, counts, theta_true)
                    f_full_t = fit_slsqp(engine_full, counts, theta_true)
                else:
                    f_full_t = fit_slsqp(engine_full, counts, theta_true)
                    f_pos = fit_slsqp(engine_pos, counts, theta_true)
                assert f_pos["valid"]
                # Verify continuous-time Markov positivity at t=0.05 for every fitted parameter
                gen_hat, fresh_hat = pos33_cycle_matrices(np.asarray(f_pos["theta"]), psi_star)
                p_t005 = np.asarray(READ_POS33 @ jl.expm(0.05 * gen_hat) @ fresh_hat)
                is_neg = bool(f_pos["min_early_extrapolation"] < 0.0 or p_t005.min() < 0.0)
                assert not is_neg, "Pos-Sens 33 must never produce negative probabilities!"

                pos_shapes.append(100.0 * f_pos["shape_rms"])
                pos_means.append(100.0 * f_pos["mean_rms"])
                full_shapes.append(100.0 * sr_runs[row["seed"]]["full48"]["shape_rms"])
                joint_hits += int(f_pos["joint"])
                overall_counts[design]["joint"] += int(f_pos["joint"])
                overall_counts[design]["shape_errs"].append(100.0 * f_pos["shape_rms"])
                overall_counts[design]["mean_errs"].append(100.0 * f_pos["mean_rms"])
                overall_counts[design]["early_neg"] += int(is_neg)
                overall_counts[design]["near_floor"] += int(f_pos["min_fitted_probability"] < 1e-6)
                overall_counts[design]["warm_ms"].append(1000.0 * f_pos["seconds"])
                all_speedups[design].append(f_full_t["seconds"] / f_pos["seconds"])

            diffs = np.asarray(pos_shapes) - np.asarray(full_shapes)
            # Deterministic cross-process seed (independent of PYTHONHASHSEED)
            det_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(key)) % 10000
            med_diff, ci_diff = paired_bootstrap_ci(diffs, seed=det_seed)
            cases_summary[key] = {
                "median_shape_percent": float(np.median(pos_shapes)),
                "median_mean_percent": float(np.median(pos_means)),
                "joint": joint_hits,
                "paired_shape_difference_vs_full_pp": {"median": med_diff, "ci95": ci_diff},
                "prev_models": {
                    m: sens_prev["cases"][key][m]["median_shape_percent"]
                    for m in ("full48", "sym33", "sp33", "sp18")
                },
            }
            print(
                f"{key:26s} | Pos33={np.median(pos_shapes):5.2f}% | Full48={cases_summary[key]['prev_models']['full48']:5.2f}% | Sp33={cases_summary[key]['prev_models']['sp33']:5.2f}% | Joint={joint_hits}/20",
                flush=True,
            )

    summary_out = {
        "calibration": cal_meta,
        "psi_star": psi_star.tolist(),
        "regime_diagnostics": regime_diagnostics,
        "overall": {
            d: {
                "pos33": {
                    "joint": overall_counts[d]["joint"],
                    "total": 280 if d == "primary" else 120,
                    "median_shape_percent": float(np.median(overall_counts[d]["shape_errs"])),
                    "median_mean_percent": float(np.median(overall_counts[d]["mean_errs"])),
                    "early_negative_fits": overall_counts[d]["early_neg"],
                    "near_floor_fits": overall_counts[d]["near_floor"],
                    "median_warm_ms": float(np.median(overall_counts[d]["warm_ms"])),
                    "median_paired_speedup_vs_full48": float(np.median(all_speedups[d])),
                },
                **{
                    m: sens_prev["overall"][d][m]
                    for m in ("full48", "sym33", "sp33", "sp18")
                },
            }
            for d in ("primary", "late")
        },
        "cases": cases_summary,
    }
    out_path = ROOT / "results" / "archived" / "positive_sensitivity_summary.json"
    out_path.write_text(json.dumps(summary_out, indent=2) + "\n", encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps(summary_out["overall"], indent=2))


if __name__ == "__main__":
    main()
