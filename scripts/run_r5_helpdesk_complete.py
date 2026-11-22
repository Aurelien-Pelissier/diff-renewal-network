"""Round-5 Complete IT Helpdesk Baselines: Equalized 3-Start Clean age=0 Neural ODEs,
LTC/S4D Panel Coarsening, and Two-Branch Universal Hyper-Pos-Sens (37 states, 25 params).

Closes all remaining IT Helpdesk loopholes:
1. Equalized 3-start clean zero-masked (age=0) zero-shot evaluation of ALL FOUR
   continuous-time neural ODE baselines (LTC, S4D, Latent ODE, Neural-CME) on
   5-fold ticket CV AND chronological 70/30 future-ticket split (zero random-weight noise).
2. LTC and S4D on all four panel-coarsening grids dt in {0.5, 2, 7, 14} days
   (clean age=0 zero-shot aged KL + fresh KL), filling the remaining table holes.
3. Full-cone certified (k in [1.4, 7.8]) 2-branch Universal Hyper-Pos-Sens
   (K_r=8 -> 9 states/active node = 37 states, 25 params) on 5-fold CV & Chronological split.
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
    calibrate_universal_pos_clock,
)
from scripts.run_helpdesk_extended_baselines import (  # noqa: E402
    init_helpdesk_ltc,
    init_helpdesk_s4d,
    load_chronological_split,
    predict_helpdesk_ltc,
    predict_helpdesk_s4d,
)
from scripts.run_helpdesk_systems_benchmark import (  # noqa: E402
    AGED_THRESHOLD_DAYS,
    E_EDGES,
    EDGES,
    HORIZONS_DAYS,
    V_NODES,
    empirical_kl,
    ensure_helpdesk_csv,
    extract_snapshot_tables,
    fit_model,
    init_helpdesk_latent_ode,
    init_helpdesk_neural_cme,
    load_ticket_trajectories,
    predict_helpdesk_latent_ode,
    predict_helpdesk_neural_cme,
)


def make_universal_hyper_pos_helpdesk(psi_univ, stages_red: int = 8):
    """37-state, 25-param 2-branch Universal Hyper-Pos-Sens on cyclic 5-node IT Helpdesk."""
    # Active nodes 0..3: 4 log fast means + 4 log slow means + 4 shape logits + 4 fast weight logits + 9 routing = 25 params
    out_edges = [[] for _ in range(V_NODES)]
    for e_idx, (u, v) in enumerate(EDGES):
        out_edges[u].append(e_idx)

    def predict(theta: jax.Array, horizons: jax.Array):
        fast_m = jnp.concatenate([jnp.exp(jnp.clip(theta[:4], -4.5, 2.5)), jnp.array([1.0])])
        slow_m = jnp.concatenate([jnp.exp(jnp.clip(theta[4:8], -1.5, 4.5)), jnp.array([1.0])])
        slow_k = jnp.concatenate([1.4 + 6.4 * jax.nn.sigmoid(theta[8:12]), jnp.array([4.0])])
        fast_w = jnp.concatenate([jax.nn.sigmoid(theta[12:16]), jnp.array([0.0])])
        rl = theta[16 : 16 + E_EDGES]
        rw = jnp.zeros(E_EDGES, dtype=jnp.float64)
        for u in range(4):
            eids = jnp.asarray(out_edges[u])
            rw = rw.at[eids].set(jax.nn.softmax(rl[eids]))

        gen, fresh, read = build_universal_hyper_pos_network_generator(
            fast_m, slow_m, slow_k, fast_w, rw, EDGES, V_NODES, psi_univ, stages_red=stages_red
        )
        blk_sz = stages_red + 1
        aged = jnp.zeros((gen.shape[0], 3), dtype=jnp.float64)
        for u in (1, 2, 3):
            s0, s1 = u * blk_sz, (u + 1) * blk_sz
            blk = gen[s0:s1, s0:s1]
            st_a = jl.expm(AGED_THRESHOLD_DAYS * blk) @ fresh[s0:s1, u]
            st_a = jnp.maximum(st_a, 1e-30)
            aged = aged.at[s0:s1, u - 1].set(st_a / jnp.sum(st_a))

        def eval_one(h):
            eh = jl.expm(h * gen)
            return (read @ (eh @ fresh[:, :4])).T, (read @ (eh @ aged)).T

        pf_all, pa_all = jax.vmap(eval_one)(horizons)
        return jnp.swapaxes(pf_all, 0, 1), jnp.swapaxes(pa_all, 0, 1)

    return predict


def main():
    csv_path = ensure_helpdesk_csv(Path("/tmp/drn_r5_helpdesk"))
    trajs, _ = load_ticket_trajectories(csv_path)
    h_jax = jnp.asarray(HORIZONS_DAYS, dtype=jnp.float64)

    # Full-cone certified Universal Pos-Sens clock k in [1.4, 7.8]
    psi_univ, univ_cert = calibrate_universal_pos_clock(
        stages_red=8, stages_full=16, maxiter=250, k_min=1.4, k_max=7.8
    )
    print(f"Certified Universal Pos-Sens [1.4, 7.8]: sup_surv={univ_cert['sup_survival_abs_error']:.5f}", flush=True)
    pred_hpos = make_universal_hyper_pos_helpdesk(psi_univ, stages_red=8)

    def init_hpos(seed: int):
        rng = np.random.default_rng(seed)
        return np.r_[
            np.log([0.6, 0.8, 1.5, 4.0]),
            np.log([6.0, 8.0, 15.0, 30.0]),
            rng.normal(0.0, 0.2, 4),
            np.zeros(4),
            np.zeros(E_EDGES),
        ]

    # Compiled reusable fitter per (predict_fn, joint) -> compiles ONCE per model instead of 150x!
    def make_fitter(predict_fn, *, joint: bool, horizons_arr: np.ndarray, reg: float = 1e-4):
        h_local = jnp.asarray(horizons_arr, dtype=jnp.float64)

        @jax.jit
        def vg(w, f_tr, a_tr, n_f, n_a):
            def loss_fn(ww):
                pf, pa = predict_fn(ww, h_local)
                lf = -jnp.sum(f_tr * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_f
                if joint:
                    la = -jnp.sum(a_tr * jnp.log(jnp.clip(pa, 1e-12, 1.0))) / n_a
                    tot = 0.5 * (lf + la)
                else:
                    tot = lf
                return tot + reg * jnp.mean(ww**2)

            return jax.value_and_grad(loss_fn)(w)

        def run_fit(init_w, f_tr_np, a_tr_np, maxiter: int = 150):
            f_j = jnp.asarray(f_tr_np, dtype=jnp.float64)
            a_j = jnp.asarray(a_tr_np, dtype=jnp.float64)
            nf = jnp.asarray(max(float(f_tr_np.sum()), 1.0), dtype=jnp.float64)
            na = jnp.asarray(max(float(a_tr_np.sum()), 1.0), dtype=jnp.float64)

            def obj(x):
                v, g = vg(jnp.asarray(x, dtype=jnp.float64), f_j, a_j, nf, na)
                return float(v), np.asarray(g, dtype=np.float64)

            r = minimize(obj, np.asarray(init_w, dtype=jnp.float64), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
            return r.x

        return run_fit

    # Exact same 5-fold split (seed=2026)
    rng = np.random.default_rng(2026)
    folds = np.array_split(rng.permutation(len(trajs)), 5)

    neural_specs = [
        ("ltc_ode", init_helpdesk_ltc, predict_helpdesk_ltc),
        ("s4d_ode", init_helpdesk_s4d, predict_helpdesk_s4d),
        ("neural_cme", init_helpdesk_neural_cme, predict_helpdesk_neural_cme),
        ("latent_ode", init_helpdesk_latent_ode, predict_helpdesk_latent_ode),
    ]

    fitters = {
        (m, j): make_fitter(pf, joint=j, horizons_arr=HORIZONS_DAYS)
        for m, _, pf in [("hyper_pos_sens", init_hpos, pred_hpos), *neural_specs]
        for j in (False, True)
    }

    # 1. Equalized 3-start clean age=0 5-fold CV for all 4 neural ODEs + 2-start Hyper-Pos-Sens
    cv_raw = {
        f"{m}_{mode}": {"fresh_kl": [], "aged_kl": []}
        for m in ("hyper_pos_sens", "ltc_ode", "s4d_ode", "neural_cme", "latent_ode")
        for mode in ("joint", "fresh_only")
    }

    for fi in range(5):
        te_idx = set(folds[fi].tolist())
        tr_t = [trajs[i] for i in range(len(trajs)) if i not in te_idx]
        te_t = [trajs[i] for i in range(len(trajs)) if i in te_idx]
        f_tr, a_tr = extract_snapshot_tables(tr_t)
        f_te, a_te = extract_snapshot_tables(te_t)

        # Hyper-Pos-Sens (2 starts)
        for mode, is_j in (("fresh_only", False), ("joint", True)):
            bw, bl = None, float("inf")
            for st in range(2):
                w = fitters[("hyper_pos_sens", is_j)](init_hpos(400 + 10 * fi + st), f_tr, a_tr, maxiter=160)
                pf_t, pa_t = pred_hpos(jnp.asarray(w), h_jax)
                lt = empirical_kl(np.asarray(pf_t), f_tr) + (empirical_kl(np.asarray(pa_t), a_tr) if is_j else 0.0)
                if lt < bl:
                    bl, bw = lt, w
            pf, pa = pred_hpos(jnp.asarray(bw), h_jax)
            cv_raw[f"hyper_pos_sens_{mode}"]["fresh_kl"].append(empirical_kl(np.asarray(pf), f_te))
            cv_raw[f"hyper_pos_sens_{mode}"]["aged_kl"].append(empirical_kl(np.asarray(pa), a_te))

        # All 4 neural ODEs: 3 starts, clean age=0 evaluation for fresh_only!
        for mname, init_fn, pred_fn in neural_specs:
            for mode, is_j in (("fresh_only", False), ("joint", True)):
                bw, bl = None, float("inf")
                for st in range(3):
                    w = fitters[(mname, is_j)](init_fn(500 + 20 * fi + st), f_tr, a_tr, maxiter=150)
                    pf_t, pa_t = pred_fn(jnp.asarray(w), h_jax)
                    lt = empirical_kl(np.asarray(pf_t), f_tr) + (empirical_kl(np.asarray(pa_t), a_tr) if is_j else 0.0)
                    if lt < bl:
                        bl, bw = lt, w
                pf, pa = pred_fn(jnp.asarray(bw), h_jax)
                pf_np = np.asarray(pf)
                # Clean age=0 zero-masked evaluation when fresh_only (eliminates untrained age-column noise)
                pa_eval = np.asarray(pa) if is_j else pf_np[1:4]
                cv_raw[f"{mname}_{mode}"]["fresh_kl"].append(empirical_kl(pf_np, f_te))
                cv_raw[f"{mname}_{mode}"]["aged_kl"].append(empirical_kl(pa_eval, a_te))
        print(
            f"Helpdesk R5 Fold {fi+1}/5 | HyperPos 0-sh={cv_raw['hyper_pos_sens_fresh_only']['aged_kl'][-1]:.4f} "
            f"| LTC 0-sh={cv_raw['ltc_ode_fresh_only']['aged_kl'][-1]:.4f} "
            f"| S4D 0-sh={cv_raw['s4d_ode_fresh_only']['aged_kl'][-1]:.4f} "
            f"| CME 0-sh={cv_raw['neural_cme_fresh_only']['aged_kl'][-1]:.4f} "
            f"| LODE 0-sh={cv_raw['latent_ode_fresh_only']['aged_kl'][-1]:.4f}",
            flush=True,
        )

    cv_summary = {
        k: {
            "fresh_kl_mean": float(np.mean(d["fresh_kl"])),
            "fresh_kl_sd": float(np.std(d["fresh_kl"], ddof=1)),
            "aged_kl_mean": float(np.mean(d["aged_kl"])),
            "aged_kl_sd": float(np.std(d["aged_kl"], ddof=1)),
            "folds": d,
        }
        for k, d in cv_raw.items()
    }

    # 2. Equalized 3-start clean age=0 Chronological 70/30 Split
    tr_ch, te_ch, cutoff_date = load_chronological_split(csv_path, train_frac=0.70)
    f_tr_c, a_tr_c = extract_snapshot_tables(tr_ch)
    f_te_c, a_te_c = extract_snapshot_tables(te_ch)
    chrono = {}

    for mname, init_fn, pred_fn, is_neural in [
        ("hyper_pos_sens", init_hpos, pred_hpos, False),
        *[(m, inf, prf, True) for m, inf, prf in neural_specs],
    ]:
        bw, bl = None, float("inf")
        for st in range(3):
            w = fitters[(mname, False)](init_fn(700 + st), f_tr_c, a_tr_c, maxiter=150)
            pf_t, _ = pred_fn(jnp.asarray(w), h_jax)
            lt = empirical_kl(np.asarray(pf_t), f_tr_c)
            if lt < bl:
                bl, bw = lt, w
        pf, pa = pred_fn(jnp.asarray(bw), h_jax)
        pf_np = np.asarray(pf)
        pa_eval = pf_np[1:4] if is_neural else np.asarray(pa)
        chrono[f"{mname}_fresh_only"] = {
            "fresh_kl": empirical_kl(pf_np, f_te_c),
            "aged_kl": empirical_kl(pa_eval, a_te_c),
        }
        print(f"Helpdesk R5 Chrono {mname:15s}: fresh={chrono[f'{mname}_fresh_only']['fresh_kl']:.4f} | aged={chrono[f'{mname}_fresh_only']['aged_kl']:.4f}", flush=True)

    # 3. Coarsening Panel Sweep dt in {0.5, 2, 7, 14}d for LTC, S4D, Neural-CME, Latent ODE (clean age=0, 2 starts)
    te_idx0 = set(folds[0].tolist())
    tr0 = [trajs[i] for i in range(len(trajs)) if i not in te_idx0]
    te0 = [trajs[i] for i in range(len(trajs)) if i in te_idx0]
    eval_horizons = np.array([1.0, 3.0, 7.0, 15.0, 30.0], dtype=float)
    h_eval_jax = jnp.asarray(eval_horizons, dtype=jnp.float64)
    fresh_te0, aged_te0 = extract_snapshot_tables(te0, horizons=eval_horizons)
    coarsening = []
    for dt in (0.5, 2.0, 7.0, 14.0):
        grid_full = np.arange(dt, 45.0 + 1e-9, dt)
        if len(grid_full) > 12:
            pick = np.unique(np.round(np.geomspace(1, len(grid_full), 12)).astype(int) - 1)
            grid_obs = grid_full[pick]
        else:
            grid_obs = grid_full
        grid_jax = jnp.asarray(grid_obs, dtype=jnp.float64)
        fresh_grid, _ = extract_snapshot_tables(tr0, horizons=grid_obs)
        f_grid_jax = jnp.asarray(fresh_grid, dtype=jnp.float64)
        n_grid_obs = float(np.maximum(fresh_grid.sum(), 1.0))
        row = {"delta_t_days": dt}
        for mname, init_fn, pred_fn in neural_specs:
            @jax.jit
            def n_vg(w):
                def loss_fn(ww):
                    pf, _ = pred_fn(ww, grid_jax)
                    nll = -jnp.sum(f_grid_jax * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_grid_obs
                    return nll + 1e-4 * jnp.mean(ww**2)

                return jax.value_and_grad(loss_fn)(w)

            bw, bl = None, float("inf")
            for st in range(2):
                r = minimize(
                    lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, np.float64), n_vg(jnp.asarray(x)))),
                    init_fn(800 + st),
                    jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": 150},
                )
                if r.fun < bl:
                    bl, bw = float(r.fun), r.x.copy()
            pf, _ = pred_fn(jnp.asarray(bw), h_eval_jax)
            pf_np = np.asarray(pf)
            row[f"{mname}_fresh_kl"] = empirical_kl(pf_np, fresh_te0)
            row[f"{mname}_aged_kl_clean_age0"] = empirical_kl(pf_np[1:4], aged_te0)
        print(
            f"Helpdesk R5 Coarsen dt={dt:4.1f}d | LTC={row['ltc_ode_aged_kl_clean_age0']:.4f} "
            f"| S4D={row['s4d_ode_aged_kl_clean_age0']:.4f} "
            f"| CME={row['neural_cme_aged_kl_clean_age0']:.4f} "
            f"| LODE={row['latent_ode_aged_kl_clean_age0']:.4f}",
            flush=True,
        )
        coarsening.append(row)

    out = {
        "protocol": "Equalized 3-start clean zero-masked (age=0) neural ODEs, LTC/S4D panel coarsening, and full-cone certified [1.4, 7.8] 2-branch Universal Hyper-Pos-Sens (37 states, 25 params)",
        "universal_clock_certificate_full_cone": univ_cert,
        "hyper_pos_sens_states": 37,
        "hyper_pos_sens_params": 25,
        "cv_5fold_clean_age0": cv_summary,
        "chronological_split_3start_clean_age0": {
            "cutoff_date": cutoff_date,
            "models": chrono,
        },
        "coarsening_neural_clean_age0": coarsening,
    }
    out_path = ROOT / "results" / "archived" / "helpdesk_r5_complete.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
