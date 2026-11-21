"""Evaluate modern multi-start neural ODE/SSM baselines against Conditional Neural Clocks.

Tests Multi-Start Latent ODE (387 params), Liquid Time-Constant ODE (LTC, 363 params),
and Diagonal Structured State-Space ODE (S4D, 363 params) across all 10 paired seeds
of both the Primary PH and Out-of-Class Lognormal-Mixture cyclic benchmarks, selecting
the best of 3 random starts solely by training penalized likelihood (zero test leakage).
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal.baselines import make_baseline_engine  # noqa: E402

NM_ROOT = Path("/usr/local/google/home/apelissier/non-markovian-diffrax")
HORIZON = 4.5
PERIOD = 0.25
TRAIN_TIMES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
TEST_TIMES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 4.5])
TRAIN_PROTOCOLS = ((0.0, 0.0, 9.0, 10.0), (1.0, 0.0, 9.0, 10.0), (0.0, 0.75, 9.0, 10.0), (0.0, 0.0, 1.0, 2.0))
TEST_PROTOCOLS = ((0.0, 0.0, 9.0, 10.0), (1.0, 0.0, 9.0, 10.0), (0.0, 1.5, 9.0, 10.0), (0.0, 1.5, 0.5, 1.5))
GROUPS = {"fresh": [0, 1, 4, 5], "aged": [2, 6], "pulse": [3, 7]}


@dataclass(frozen=True)
class Plan:
    traits: np.ndarray
    ages: np.ndarray
    baseline: np.ndarray
    exposure: np.ndarray
    sample: np.ndarray


def build_plan(protocols, times) -> Plan:
    initial, age, on, off = np.tile(np.array(protocols), (2, 1)).T
    grid = np.arange(round(HORIZON / PERIOD)) * PERIOD
    active = (grid[:, None] >= on) & (grid[:, None] < off)
    exposure = np.where(active, 1.0, initial)
    return Plan(
        np.repeat([-1.0, 1.0], len(protocols)),
        age,
        initial,
        exposure,
        np.rint(times / PERIOD).astype(int) - 1,
    )


def compute_metrics(p: np.ndarray, truth: np.ndarray, counts: np.ndarray) -> dict:
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    lp = np.log(np.maximum(p, 1e-30))
    res = {}
    for name, rows in GROUPS.items():
        q = truth[rows]
        n = counts[rows]
        res[name] = {
            "kl": float(np.sum(q * (np.log(np.maximum(q, 1e-30)) - lp[rows]), axis=-1).mean()),
            "test_nll": float(-(n * lp[rows]).sum() / n.sum()),
            "prob_rmse": float(np.sqrt(np.mean((p[rows] - q) ** 2))),
        }
    res["balanced_kl"] = float(np.mean([res[g]["kl"] for g in GROUPS]))
    after = TEST_TIMES > 0.5

    def contrast(arr):
        return (arr[GROUPS["pulse"]] - arr[GROUPS["aged"]])[:, after]

    d = contrast(p) - contrast(truth)
    res["pulse_effect_rmse"] = float(np.sqrt(np.mean(d**2)))
    return res


def summarize_list(vals: list[float]) -> dict:
    arr = np.asarray(vals)
    return {
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def main() -> None:
    train_plan = build_plan(TRAIN_PROTOCOLS, TRAIN_TIMES)
    test_plan = build_plan(TEST_PROTOCOLS, TEST_TIMES)
    baseline_kinds = ("latent_ode", "ltc_ode", "s4d_ode")
    starts = (17, 41, 89)

    # Pre-compile all engines across starts
    compiled_engines = {}
    dummy_c = jnp.ones((8, 6, 3, 3))
    for kind in baseline_kinds:
        compiled_engines[kind] = []
        for s_id in starts:
            eng = make_baseline_engine(kind, train_plan, test_plan, seed=s_id, reg=5e-5)
            w0, _, vg, fc = eng
            vg(w0, dummy_c)[1].block_until_ready()
            fc(w0).block_until_ready()
            compiled_engines[kind].append(eng)

    nm_prev = json.loads(
        (ROOT / "results" / "archived" / "neural_memory_summary.json").read_text(encoding="utf-8")
    )

    out_studies = {}
    for study_key, fname in [
        ("primary", "neural_memory_cycle.json"),
        ("lognormal", "neural_memory_cycle_lognormal.json"),
    ]:
        src = json.loads((NM_ROOT / "results" / fname).read_text(encoding="utf-8"))
        truth = np.asarray(src["test_truth"])
        seeds = [str(s) for s in src["protocol"]["seeds"]]

        per_kind_runs = {k: [] for k in baseline_kinds}
        for seed in seeds:
            tr_counts = jnp.asarray(src["runs"][seed]["train_counts"], dtype=float)
            te_counts = np.asarray(src["runs"][seed]["test_counts"], dtype=float)

            for kind in baseline_kinds:
                best_fit = None
                t0 = perf_counter()
                for w0, _, vg, fc in compiled_engines[kind]:
                    res = minimize(
                        lambda z: (
                            float(vg(jnp.asarray(z), tr_counts)[0]),
                            np.asarray(vg(jnp.asarray(z), tr_counts)[1], dtype=float),
                        ),
                        w0,
                        method="L-BFGS-B",
                        jac=True,
                        options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7, "maxls": 35},
                    )
                    if best_fit is None or res.fun < best_fit[0]:
                        best_fit = (float(res.fun), res.x, fc)
                elapsed = perf_counter() - t0
                pred = np.asarray(best_fit[2](jnp.asarray(best_fit[1])))
                m_dict = compute_metrics(pred, truth, te_counts)
                m_dict["seconds"] = elapsed
                per_kind_runs[kind].append(m_dict)
            print(
                f"{study_key:9s} seed {seed} | "
                + " | ".join(f"{k}: {per_kind_runs[k][-1]['balanced_kl']:.5f}" for k in baseline_kinds),
                flush=True,
            )

        models_summary = {
            "neural_phase": nm_prev[study_key]["models"]["neural_phase"],
            "mean_shape": nm_prev[study_key]["models"]["mean_shape"],
            "latent_ode_1start": nm_prev[study_key]["models"]["latent_ode"],
        }
        for kind in baseline_kinds:
            n_params = len(compiled_engines[kind][0][0])
            runs_k = per_kind_runs[kind]
            models_summary[f"{kind}_3start"] = {
                "n_parameters": n_params,
                "starts": len(starts),
                "warm_seconds_median": float(np.median([r["seconds"] for r in runs_k])),
                "hits_005": int(sum(r["balanced_kl"] <= 0.005 for r in runs_k)),
                "metrics": {
                    "fresh": {"kl": summarize_list([r["fresh"]["kl"] for r in runs_k])},
                    "aged": {"kl": summarize_list([r["aged"]["kl"] for r in runs_k])},
                    "pulse": {"kl": summarize_list([r["pulse"]["kl"] for r in runs_k])},
                    "balanced_kl": summarize_list([r["balanced_kl"] for r in runs_k]),
                    "pulse_effect_rmse": summarize_list([r["pulse_effect_rmse"] for r in runs_k]),
                    "test_nll_balanced": summarize_list(
                        [float(np.mean([r[g]["test_nll"] for g in GROUPS])) for r in runs_k]
                    ),
                },
            }
        out_studies[study_key] = models_summary

    out_path = ROOT / "results" / "archived" / "modern_neural_baselines.json"
    out_path.write_text(json.dumps(out_studies, indent=2) + "\n", encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
