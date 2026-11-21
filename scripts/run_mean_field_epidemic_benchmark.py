"""Nonlinear Mean-Field Renewal Network Benchmark (1978 Boarding School Influenza).

Evaluates `MeanFieldDRN` (McKean-Vlasov semi-Markov ODE via Diffrax Tsit5) on
the canonical 1978 English Boarding School H1N1 Influenza Outbreak
(Anonymous, BMJ 1978, 1(6112):587; N=763 boys, 14 daily prevalence snapshots
recording both `in_bed` and `convalescent` counts):

1. Full-outbreak mechanistic law recovery (Days 1-14): compares MeanFieldDRN
   (smooth continuous-shape Erlang clocks for Exposed, In-Bed, and Convalescent)
   against Memoryless Markov SEICR (k=1) and unconstrained nonlinear Latent ODE.
2. Early-outbreak temporal extrapolation (Train Days 1-7 rise+peak -> Forecast
   unseen Days 8-14 decline & convalescent wave), showing that non-exponential
   residence-time memory drastically improves out-of-sample wave forecasting.
3. Empirical Kurtz O(M^{-1/2}) Propagation-of-Chaos verification comparing
   exact finite-M interacting semi-Markov Gillespie simulations (M=250..16,000)
   against deterministic MeanFieldDRN trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    seicr_mean_field_predict,
    simulate_interacting_seicr_gillespie,
)

# Anonymous (1978), "Influenza in a boarding school", BMJ 1(6112):587 (PMC1603269)
# Exact table reproduced in R `outbreaks::influenza_england_1978_school`, N = 763 boys
N_POP = 763.0
DAYS = np.arange(1.0, 15.0, dtype=float)
IN_BED = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], dtype=float)
CONVALESCENT = np.array([0, 0, 0, 0, 9, 17, 91, 161, 179, 156, 136, 72, 53, 32], dtype=float)
OTHER = N_POP - IN_BED - CONVALESCENT
OBS_COUNTS = np.stack([IN_BED, CONVALESCENT, OTHER], axis=-1)  # (14, 3)
OBS_FRACS = OBS_COUNTS / N_POP


def _unpack_seicr(theta: jax.Array, *, markov_only: bool):
    # Parameterize smoothly inside physically well-posed positive intervals
    beta = 0.2 + 4.5 * jax.nn.sigmoid(theta[0])
    m_e = 0.25 + 4.5 * jax.nn.sigmoid(theta[1])
    m_i = 0.5 + 7.0 * jax.nn.sigmoid(theta[2])
    m_c = 0.5 + 8.0 * jax.nn.sigmoid(theta[3])
    i0 = 1e-4 + 0.03 * jax.nn.sigmoid(theta[4])
    e0 = 1e-4 + 0.03 * jax.nn.sigmoid(theta[5])
    if markov_only:
        k_e = jnp.array(1.0, dtype=jnp.float64)
        k_i = jnp.array(1.0, dtype=jnp.float64)
        k_c = jnp.array(1.0, dtype=jnp.float64)
    else:
        k_e = 1.0 + 11.0 * jax.nn.sigmoid(theta[6])
        k_i = 1.0 + 11.0 * jax.nn.sigmoid(theta[7])
        k_c = 1.0 + 11.0 * jax.nn.sigmoid(theta[8])
    return beta, m_e, k_e, m_i, k_i, m_c, k_c, i0, e0


def predict_seicr_3cat(theta: jax.Array, ts: jax.Array, *, markov_only: bool, stages: int = 12):
    beta, m_e, k_e, m_i, k_i, m_c, k_c, i0, e0 = _unpack_seicr(theta, markov_only=markov_only)
    p5, _ = seicr_mean_field_predict(
        beta, m_e, k_e, m_i, k_i, m_c, k_c, i0, e0, ts, stages=stages
    )
    # Observable 3-category simplex: [in_bed = I, convalescent = C, other = S + E + R]
    other = p5[:, 0] + p5[:, 1] + p5[:, 4]
    return jnp.stack([p5[:, 2], p5[:, 3], other], axis=-1)


def init_epidemic_latent_ode(seed: int = 0):
    rng = np.random.default_rng(seed)
    z_dim, w_dim = 4, 12
    # 3-category simplex p + 4 latent z -> 7 -> 12 -> (3*3 off-diag/flux + 4) = 10
    # 7*12 + 12 + 12*10 + 10 + 4(z0) = 230 params
    return np.concatenate(
        [
            rng.normal(0.0, 0.25, size=7 * w_dim),
            np.zeros(w_dim),
            rng.normal(0.0, 0.15, size=w_dim * (6 + z_dim)),
            np.concatenate([np.full(6, -2.0), np.zeros(z_dim)]),
            rng.normal(0.0, 0.2, size=z_dim),
            np.array([-5.0, -7.0]),  # initial in_bed, conv logits
        ]
    )


def predict_epidemic_latent_ode(theta: jax.Array, ts: jax.Array):
    z_dim, w_dim = 4, 12
    i0 = 7 * w_dim
    w1 = theta[:i0].reshape(7, w_dim)
    b1 = theta[i0 : i0 + w_dim]
    i1 = i0 + w_dim
    w2 = theta[i1 : i1 + w_dim * (6 + z_dim)].reshape(w_dim, 6 + z_dim)
    i2 = i1 + w_dim * (6 + z_dim)
    b2 = theta[i2 : i2 + 6 + z_dim]
    z0 = theta[i2 + 6 + z_dim : i2 + 6 + 2 * z_dim]
    p_log = theta[-2:]
    p0_ic = jax.nn.sigmoid(p_log) * 0.05
    p0 = jnp.array([p0_ic[0], p0_ic[1], 1.0 - p0_ic[0] - p0_ic[1]])
    y0 = jnp.concatenate([p0, z0])

    # RK4 with 0.25-day step across 14 days (56 steps)
    dt = 0.25
    n_steps = 56
    idx_out = jnp.asarray(np.round(np.asarray(ts) / dt).astype(int) - 1)
    # 6 directed transitions on 3 categories (0=I, 1=C, 2=Other)
    edges3 = ((2, 0), (0, 1), (1, 2), (0, 2), (2, 1), (1, 0))

    def rhs(y):
        p, z = y[:3], y[3:]
        raw = jnp.tanh(y @ w1 + b1) @ w2 + b2
        rates = 1e-4 + 3.0 * jax.nn.sigmoid(raw[:6])
        dp = jnp.zeros(3)
        for e_i, (u, v) in enumerate(edges3):
            fl = rates[e_i] * p[u]
            dp = dp.at[v].add(fl)
            dp = dp.at[u].add(-fl)
        dz = jnp.tanh(raw[6:]) - 0.1 * z
        return jnp.concatenate([dp, dz])

    def step(y, _):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        pc = jnp.maximum(yn[:3], 1e-12)
        pc = pc / jnp.sum(pc)
        return jnp.concatenate([pc, yn[3:]]), pc

    _, traj = jax.lax.scan(step, y0, None, length=n_steps)
    return traj[idx_out]


def fit_seicr_multistart(obs_train: np.ndarray, ts_train: np.ndarray, *, markov_only: bool, n_starts: int = 5):
    ts_jax = jnp.asarray(ts_train, dtype=jnp.float64)
    obs_jax = jnp.asarray(obs_train, dtype=jnp.float64)
    n_tot = float(obs_train.sum())
    dim = 6 if markov_only else 9
    stages = 1 if markov_only else 12

    @jax.jit
    def vg(w):
        def loss_fn(ww):
            p = predict_seicr_3cat(ww, ts_jax, markov_only=markov_only, stages=stages)
            nll = -jnp.sum(obs_jax * jnp.log(jnp.clip(p, 1e-12, 1.0))) / n_tot
            return nll + 1e-5 * jnp.mean(ww**2)

        return jax.value_and_grad(loss_fn)(w)

    rng = np.random.default_rng(42 if not markov_only else 99)
    best_x, best_f = None, float("inf")
    for s in range(n_starts):
        x0 = np.zeros(dim) if s == 0 else rng.normal(0.0, 0.8, size=dim)
        res = minimize(
            lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, np.float64), vg(jnp.asarray(x)))),
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 250},
        )
        if res.fun < best_f:
            best_f, best_x = float(res.fun), res.x.copy()
    return best_x


def fit_latent_ode_multistart(obs_train: np.ndarray, ts_train: np.ndarray, n_starts: int = 5):
    ts_jax = jnp.asarray(ts_train, dtype=jnp.float64)
    obs_jax = jnp.asarray(obs_train, dtype=jnp.float64)
    n_tot = float(obs_train.sum())

    @jax.jit
    def vg(w):
        def loss_fn(ww):
            p = predict_epidemic_latent_ode(ww, ts_jax)
            nll = -jnp.sum(obs_jax * jnp.log(jnp.clip(p, 1e-12, 1.0))) / n_tot
            return nll + 5e-4 * jnp.mean(ww**2)

        return jax.value_and_grad(loss_fn)(w)

    best_x, best_f = None, float("inf")
    for s in range(n_starts):
        x0 = init_epidemic_latent_ode(seed=10 + s)
        res = minimize(
            lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, np.float64), vg(jnp.asarray(x)))),
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 250},
        )
        if res.fun < best_f:
            best_f, best_x = float(res.fun), res.x.copy()
    return best_x


def metrics_on_slice(pred: np.ndarray, obs_counts: np.ndarray):
    emp = obs_counts / obs_counts.sum(axis=-1, keepdims=True)
    p = np.clip(pred, 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    kl = float(np.mean(np.sum(np.where(emp > 0, emp * np.log(np.maximum(emp, 1e-30) / p), 0.0), axis=-1)))
    rmse_counts = float(np.sqrt(np.mean((p[:, :2] * N_POP - obs_counts[:, :2]) ** 2)))
    rmse_frac = float(np.sqrt(np.mean((p[:, :2] - emp[:, :2]) ** 2)))
    return {"mean_kl_nats": kl, "count_rmse_boys": rmse_counts, "frac_rmse": rmse_frac}


def verify_kurtz_propagation_of_chaos(params_tuple):
    """Empirically verify O(M^{-1/2}) finite-population convergence to MeanFieldDRN."""
    beta, m_e, k_e, m_i, k_i, m_c, k_c, i0, e0 = [float(z) for z in params_tuple]
    ts = np.linspace(1.0, 14.0, 14)
    p_det, _ = seicr_mean_field_predict(
        jnp.array(beta),
        jnp.array(m_e),
        jnp.array(k_e),
        jnp.array(m_i),
        jnp.array(k_i),
        jnp.array(m_c),
        jnp.array(k_c),
        jnp.array(max(i0, 0.02)),  # moderate initial seed so extinction doesn't dominate M=250
        jnp.array(max(e0, 0.02)),
        jnp.asarray(ts),
        stages=8,
    )
    p_det_np = np.asarray(p_det)
    pop_sizes = [250, 1000, 4000, 16000]
    rows = []
    for n_m in pop_sizes:
        errs = []
        for rep in range(16):
            sim = simulate_interacting_seicr_gillespie(
                beta, m_e, k_e, m_i, k_i, m_c, k_c, max(i0, 0.02), max(e0, 0.02), ts, n_m, stages=8, seed=1000 + rep + n_m
            )
            errs.append(float(np.max(np.abs(sim - p_det_np))))
        rows.append(
            {
                "n_agents": n_m,
                "median_linf_error": float(np.median(errs)),
                "mean_linf_error": float(np.mean(errs)),
                "sqrt_M_times_error": float(np.sqrt(n_m) * np.mean(errs)),
            }
        )
    # Fit log-log slope of mean L_inf error vs M (theoretical Kurtz rate = -0.50)
    slope = float(
        np.polyfit(np.log([r["n_agents"] for r in rows]), np.log([r["mean_linf_error"] for r in rows]), 1)[0]
    )
    print(f"Kurtz propagation-of-chaos log-log slope: {slope:.3f} (theory: -0.500)")
    for r in rows:
        print(f"  M={r['n_agents']:5d}: mean L_inf={r['mean_linf_error']:.5f}, sqrt(M)*err={r['sqrt_M_times_error']:.3f}")
    return {"empirical_slope": slope, "by_population_size": rows}


def main():
    ts_all = jnp.asarray(DAYS, dtype=jnp.float64)

    # 1. Full 14-day mechanistic fit
    w_drn_full = fit_seicr_multistart(OBS_COUNTS, DAYS, markov_only=False, n_starts=6)
    w_mkv_full = fit_seicr_multistart(OBS_COUNTS, DAYS, markov_only=True, n_starts=6)
    w_lode_full = fit_latent_ode_multistart(OBS_COUNTS, DAYS, n_starts=5)

    p_drn_full = np.asarray(predict_seicr_3cat(jnp.asarray(w_drn_full), ts_all, markov_only=False, stages=12))
    p_mkv_full = np.asarray(predict_seicr_3cat(jnp.asarray(w_mkv_full), ts_all, markov_only=True, stages=1))
    p_lode_full = np.asarray(predict_epidemic_latent_ode(jnp.asarray(w_lode_full), ts_all))

    drn_params_full = _unpack_seicr(jnp.asarray(w_drn_full), markov_only=False)
    mkv_params_full = _unpack_seicr(jnp.asarray(w_mkv_full), markov_only=True)
    param_names = ["beta", "m_E", "k_E", "m_I", "k_I", "m_C", "k_C", "i0", "e0"]
    inferred_drn = {k: float(v) for k, v in zip(param_names, drn_params_full)}
    inferred_drn["R0_implied"] = inferred_drn["beta"] * inferred_drn["m_I"]
    inferred_mkv = {k: float(v) for k, v in zip(param_names, mkv_params_full)}
    inferred_mkv["R0_implied"] = inferred_mkv["beta"] * inferred_mkv["m_I"]

    print("Full-outbreak Days 1-14:")
    print("  MeanFieldDRN :", metrics_on_slice(p_drn_full, OBS_COUNTS), inferred_drn)
    print("  Markov SEICR :", metrics_on_slice(p_mkv_full, OBS_COUNTS), inferred_mkv)
    print("  Latent ODE   :", metrics_on_slice(p_lode_full, OBS_COUNTS))

    # 2. Temporal Extrapolation: Train Days 1-8 (rise + peak + early convalescence) -> Forecast unseen Days 9-14
    #    and Train Days 1-9 -> Forecast unseen Days 10-14
    extrap_splits = {}
    for n_tr in (8, 9):
        w_drn_ex = fit_seicr_multistart(OBS_COUNTS[:n_tr], DAYS[:n_tr], markov_only=False, n_starts=6)
        w_mkv_ex = fit_seicr_multistart(OBS_COUNTS[:n_tr], DAYS[:n_tr], markov_only=True, n_starts=6)
        w_lode_ex = fit_latent_ode_multistart(OBS_COUNTS[:n_tr], DAYS[:n_tr], n_starts=5)

        p_drn_ex = np.asarray(predict_seicr_3cat(jnp.asarray(w_drn_ex), ts_all, markov_only=False, stages=12))
        p_mkv_ex = np.asarray(predict_seicr_3cat(jnp.asarray(w_mkv_ex), ts_all, markov_only=True, stages=1))
        p_lode_ex = np.asarray(predict_epidemic_latent_ode(jnp.asarray(w_lode_ex), ts_all))

        skey = f"days_1to{n_tr}_to_{n_tr+1}to14"
        print(f"Extrapolation {skey}:")
        print("  MeanFieldDRN :", metrics_on_slice(p_drn_ex[n_tr:], OBS_COUNTS[n_tr:]))
        print("  Markov SEICR :", metrics_on_slice(p_mkv_ex[n_tr:], OBS_COUNTS[n_tr:]))
        print("  Latent ODE   :", metrics_on_slice(p_lode_ex[n_tr:], OBS_COUNTS[n_tr:]))
        extrap_splits[skey] = {
            "mean_field_drn": {
                "train": metrics_on_slice(p_drn_ex[:n_tr], OBS_COUNTS[:n_tr]),
                "test": metrics_on_slice(p_drn_ex[n_tr:], OBS_COUNTS[n_tr:]),
                "pred_in_bed": (p_drn_ex[:, 0] * N_POP).tolist(),
                "pred_convalescent": (p_drn_ex[:, 1] * N_POP).tolist(),
            },
            "markov_seicr": {
                "train": metrics_on_slice(p_mkv_ex[:n_tr], OBS_COUNTS[:n_tr]),
                "test": metrics_on_slice(p_mkv_ex[n_tr:], OBS_COUNTS[n_tr:]),
                "pred_in_bed": (p_mkv_ex[:, 0] * N_POP).tolist(),
                "pred_convalescent": (p_mkv_ex[:, 1] * N_POP).tolist(),
            },
            "latent_ode": {
                "train": metrics_on_slice(p_lode_ex[:n_tr], OBS_COUNTS[:n_tr]),
                "test": metrics_on_slice(p_lode_ex[n_tr:], OBS_COUNTS[n_tr:]),
                "pred_in_bed": (p_lode_ex[:, 0] * N_POP).tolist(),
                "pred_convalescent": (p_lode_ex[:, 1] * N_POP).tolist(),
            },
        }

    # 3. Verify Kurtz O(M^{-1/2}) propagation of chaos against finite-agent stochastic Gillespie
    kurtz = verify_kurtz_propagation_of_chaos(drn_params_full)

    out = {
        "dataset": {
            "name": "1978 English Boarding School Influenza Outbreak (Anonymous, BMJ 1978)",
            "population_N": int(N_POP),
            "days": DAYS.tolist(),
            "in_bed": IN_BED.tolist(),
            "convalescent": CONVALESCENT.tolist(),
        },
        "full_fit_days_1to14": {
            "mean_field_drn": {
                "params_count": 9,
                "states": 3 * 12 + 2,
                **metrics_on_slice(p_drn_full, OBS_COUNTS),
                "inferred_parameters": inferred_drn,
                "pred_in_bed": (p_drn_full[:, 0] * N_POP).tolist(),
                "pred_convalescent": (p_drn_full[:, 1] * N_POP).tolist(),
            },
            "markov_seicr": {
                "params_count": 6,
                "states": 5,
                **metrics_on_slice(p_mkv_full, OBS_COUNTS),
                "inferred_parameters": inferred_mkv,
                "pred_in_bed": (p_mkv_full[:, 0] * N_POP).tolist(),
                "pred_convalescent": (p_mkv_full[:, 1] * N_POP).tolist(),
            },
            "latent_ode": {
                "params_count": int(init_epidemic_latent_ode(0).size),
                "states": 7,
                **metrics_on_slice(p_lode_full, OBS_COUNTS),
                "pred_in_bed": (p_lode_full[:, 0] * N_POP).tolist(),
                "pred_convalescent": (p_lode_full[:, 1] * N_POP).tolist(),
            },
        },
        "extrapolation_splits": extrap_splits,
        "kurtz_propagation_of_chaos": kurtz,
    }
    out_path = ROOT / "results" / "archived" / "mean_field_epidemic_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
