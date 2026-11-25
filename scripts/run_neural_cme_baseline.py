"""Evaluate 3-start Time-Inhomogeneous Neural Master Equation (Neural-CME) baseline.

Tests whether arbitrary clock-time rate modulation A_phi(t, x, u) (342 parameters,
3 random starts) can substitute for residence-age renewal memory across all 10 seeds
of the Primary PH and Out-of-Class Lognormal-Mixture cyclic benchmarks.
"""

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
for p_str in (str(ROOT), str(ROOT / "src")):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from diff_renewal.baselines import make_baseline_engine  # noqa: E402
from scripts.run_modern_neural_baselines import (  # noqa: E402
    GROUPS,
    NM_ROOT,
    TEST_PROTOCOLS,
    TEST_TIMES,
    TRAIN_PROTOCOLS,
    TRAIN_TIMES,
    build_plan,
    compute_metrics,
    summarize_list,
)


def main() -> None:
    train_plan = build_plan(TRAIN_PROTOCOLS, TRAIN_TIMES)
    test_plan = build_plan(TEST_PROTOCOLS, TEST_TIMES)
    starts = (17, 41, 89)
    dummy_c = jnp.ones((8, 6, 3, 3))

    engines = []
    for s_id in starts:
        eng = make_baseline_engine("neural_cme", train_plan, test_plan, seed=s_id, reg=5e-5)
        w0, _, vg, fc = eng
        vg(w0, dummy_c)[1].block_until_ready()
        fc(w0).block_until_ready()
        engines.append(eng)

    mnb_path = ROOT / "results" / "archived" / "modern_neural_baselines.json"
    mnb = json.loads(mnb_path.read_text(encoding="utf-8"))

    for study_key, fname in [
        ("primary", "neural_memory_cycle.json"),
        ("lognormal", "neural_memory_cycle_lognormal.json"),
    ]:
        src = json.loads((NM_ROOT / "results" / fname).read_text(encoding="utf-8"))
        truth = np.asarray(src["test_truth"])
        seeds = [str(s) for s in src["protocol"]["seeds"]]
        runs_cme = []

        for seed in seeds:
            tr_counts = jnp.asarray(src["runs"][seed]["train_counts"], dtype=float)
            te_counts = np.asarray(src["runs"][seed]["test_counts"], dtype=float)
            best_fit = None
            t0 = perf_counter()
            for w0, _, vg, fc in engines:
                def _obj_cme(z):
                    v, g = vg(jnp.asarray(z), tr_counts)
                    return float(v), np.asarray(g, dtype=float)

                res = minimize(
                    _obj_cme,
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
            runs_cme.append(m_dict)
            print(f"{study_key:9s} seed {seed} | neural_cme_3start: {m_dict['balanced_kl']:.5f}", flush=True)

        mnb[study_key]["neural_cme_3start"] = {
            "n_parameters": len(engines[0][0]),
            "starts": len(starts),
            "warm_seconds_median": float(np.median([r["seconds"] for r in runs_cme])),
            "hits_005": int(sum(r["balanced_kl"] <= 0.005 for r in runs_cme)),
            "metrics": {
                "fresh": {"kl": summarize_list([r["fresh"]["kl"] for r in runs_cme])},
                "aged": {"kl": summarize_list([r["aged"]["kl"] for r in runs_cme])},
                "pulse": {"kl": summarize_list([r["pulse"]["kl"] for r in runs_cme])},
                "balanced_kl": summarize_list([r["balanced_kl"] for r in runs_cme]),
                "pulse_effect_rmse": summarize_list([r["pulse_effect_rmse"] for r in runs_cme]),
                "test_nll_balanced": summarize_list(
                    [float(np.mean([r[g]["test_nll"] for g in GROUPS])) for r in runs_cme]
                ),
            },
        }

    mnb_path.write_text(json.dumps(mnb, indent=2) + "\n", encoding="utf-8")
    print("Updated:", mnb_path)


if __name__ == "__main__":
    main()
