"""Round-6 Uniformity Closure: Complete E. coli Neural ODE Suite (LTC, S4D, Neural-CME,
clean age=0 Latent ODE) & IT Helpdesk Hyper-Pos-Sens Coarsening Sweep.

Closes the last two minor uniformity gaps across Appendix Tables A5 and A6:
1. Single-cell E. coli 3-phase ring (Tanouchi et al., 2017; 18,972 cycles, 49 clusters,
   exact 5-fold cluster CV): equalized 3-start clean age=0 zero-shot and supervised joint
   fits for LTC ODE, S4D ODE, Neural-CME(t), and Latent ODE.
2. IT Helpdesk coarsening sweep (dt in {0.5, 2, 7, 14} days, exact Fold 0 split):
   2-branch Universal Hyper-Pos-Sens (37 states, 25 params, full-cone [1.4, 7.8]).
"""

from __future__ import annotations

from hashlib import md5
import json
from pathlib import Path
import sys
import urllib.request

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

from diff_renewal import calibrate_universal_pos_clock, multinomial_nll  # noqa: E402
import scripts.run_real_continuous_sleep_network as rcs  # noqa: E402
from scripts.run_helpdesk_systems_benchmark import (  # noqa: E402
    E_EDGES,
    empirical_kl,
    ensure_helpdesk_csv,
    extract_snapshot_tables,
    load_ticket_trajectories,
)
from scripts.run_r5_helpdesk_complete import make_universal_hyper_pos_helpdesk  # noqa: E402

TANOUCHI_URLS = {
    25: ("https://ndownloader.figshare.com/files/6397359", "56db1c7c9e91e53f859c211e442413a9"),
    27: ("https://ndownloader.figshare.com/files/7235000", "0f45d32d6df0e833c56a363db9b33131"),
    37: ("https://ndownloader.figshare.com/files/6397368", "8db408b9826cd881f3771744ff503c51"),
}


def ensure_tanouchi(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for temp, (url, exp_md5) in TANOUCHI_URLS.items():
        zpath = cache_dir / f"tanouchi_{temp}.zip"
        if not zpath.is_file() or md5(zpath.read_bytes()).hexdigest() != exp_md5:
            urllib.request.urlretrieve(url, zpath)
            assert md5(zpath.read_bytes()).hexdigest() == exp_md5, f"MD5 mismatch for {temp}C"


def run_ecoli_neural_suite() -> dict:
    cache_dir = Path("/tmp/drn_r6_ecoli_cache")
    ensure_tanouchi(cache_dir)
    rcs.CACHE = cache_dir
    records = rcs.load_tanouchi_cycles()

    obs_min = np.array([6, 12, 18, 24, 36, 48, 60, 75, 90], dtype=int)
    ts_hr = obs_min / 60.0
    aged_hr = 12.0 / 60.0
    temps = (25, 27, 37)
    temp_feat = {25: -1.0, 27: -0.667, 37: 1.0}
    dt = 0.025
    n_steps = int(round(ts_hr[-1] / dt))
    sample_idx = jnp.rint(jnp.asarray(ts_hr) / dt).astype(int) - 1

    clusters = sorted({r["cluster"] for r in records})
    folds = [clusters[i::5] for i in range(5)]

    # 1. LTC ODE on 3-node E. coli ring (z_dim=6, w_dim=10 -> 283 params)
    def init_ecoli_ltc(seed: int):
        rng = np.random.default_rng(seed)
        z_d, w_d = 6, 10
        # enc: 5*6+6=36; wh: 10*10+10=110; wa: 10*6+6=66; wt: 10*6+6=66; wf: 10*3+3=33 -> 311 params
        return np.concatenate([
            rng.normal(0.0, 0.22, 5 * z_d), np.zeros(z_d),
            rng.normal(0.0, 0.20, 10 * w_d), np.zeros(w_d),
            rng.normal(0.0, 0.15, w_d * z_d), np.zeros(z_d),
            rng.normal(0.0, 0.15, w_d * z_d), np.zeros(z_d),
            rng.normal(0.0, 0.15, 10 * 3), np.zeros(3),
        ])

    def pred_ecoli_ltc(th, x_temp: float, is_aged: bool):
        z_d, w_d = 6, 10
        i = 0
        enc = th[i:i+5*z_d].reshape(5, z_d); i += 5*z_d
        eb = th[i:i+z_d]; i += z_d
        wh = th[i:i+10*w_d].reshape(10, w_d); i += 10*w_d
        bh = th[i:i+w_d]; i += w_d
        wa = th[i:i+w_d*z_d].reshape(w_d, z_d); i += w_d*z_d
        ba = th[i:i+z_d]; i += z_d
        wt = th[i:i+w_d*z_d].reshape(w_d, z_d); i += w_d*z_d
        bt = th[i:i+z_d]; i += z_d
        wf = th[i:i+10*3].reshape(10, 3); i += 10*3
        bf = th[i:i+3]
        age_v = aged_hr if is_aged else 0.0
        prep = jnp.eye(3)
        z0 = jnp.tanh(jnp.column_stack([prep, jnp.full((3, 1), x_temp), jnp.full((3, 1), age_v)]) @ enc + eb)
        y0 = jnp.column_stack([prep, z0])

        def field(y):
            p, z = y[:, :3], y[:, 3:]
            inp = jnp.column_stack([y, jnp.full((3, 1), x_temp)])
            h = jnp.tanh(inp @ wh + bh)
            dz = -(0.1 + 6.0 * jax.nn.sigmoid(h @ wt + bt)) * (z - jnp.tanh(h @ wa + ba))
            flux = p * (0.05 + 12.0 * jax.nn.sigmoid(inp @ wf + bf))
            dp = jnp.roll(flux, 1, axis=-1) - flux
            return jnp.column_stack([dp, dz])

        def step(y, _):
            k1 = field(y); k2 = field(y + 0.5*dt*k1); k3 = field(y + 0.5*dt*k2); k4 = field(y + dt*k3)
            yn = y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            pc = jnp.maximum(yn[:, :3], 1e-15); pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
            return jnp.column_stack([pc, yn[:, 3:]]), pc

        traj = jax.lax.scan(step, y0, None, length=n_steps)[1]
        return jnp.transpose(traj[sample_idx], (1, 0, 2))

    # 2. S4D ODE on 3-node E. coli ring (z_dim=8, m_dim=10 -> 287 params)
    def init_ecoli_s4d(seed: int):
        rng = np.random.default_rng(seed)
        z_d, m_d = 8, 10
        # enc: 5*8+8=48; log_r: 8; om: 4; wb: 4*8+8=40; wm: 12*10+10=130; wo: 10*3+3=33 -> 263 params
        return np.concatenate([
            rng.normal(0.0, 0.22, 5 * z_d), np.zeros(z_d),
            np.linspace(np.log(0.2), np.log(8.0), z_d),
            rng.normal(0.0, 0.3, z_d // 2),
            rng.normal(0.0, 0.2, 4 * z_d), np.zeros(z_d),
            rng.normal(0.0, 0.2, 12 * m_d), np.zeros(m_d),
            rng.normal(0.0, 0.15, m_d * 3), np.zeros(3),
        ])

    def pred_ecoli_s4d(th, x_temp: float, is_aged: bool):
        z_d, m_d = 8, 10
        half = z_d // 2
        i = 0
        enc = th[i:i+5*z_d].reshape(5, z_d); i += 5*z_d
        eb = th[i:i+z_d]; i += z_d
        decay = jnp.exp(jnp.clip(th[i:i+z_d], -4.0, 4.0)); i += z_d
        om = th[i:i+half]; i += half
        wb = th[i:i+4*z_d].reshape(4, z_d); i += 4*z_d
        bb = th[i:i+z_d]; i += z_d
        wm = th[i:i+12*m_d].reshape(12, m_d); i += 12*m_d
        bm = th[i:i+m_d]; i += m_d
        wo = th[i:i+m_d*3].reshape(m_d, 3); i += m_d*3
        bo = th[i:i+3]
        age_v = aged_hr if is_aged else 0.0
        prep = jnp.eye(3)
        z0 = jnp.tanh(jnp.column_stack([prep, jnp.full((3, 1), x_temp), jnp.full((3, 1), age_v)]) @ enc + eb)
        y0 = jnp.column_stack([prep, z0])

        def field(y):
            p, z = y[:, :3], y[:, 3:]
            rot = jnp.concatenate([-om * z[:, half:], om * z[:, :half]], axis=-1)
            pt = jnp.column_stack([p, jnp.full((3, 1), x_temp)])
            dz = -decay * z + rot + (pt @ wb + bb)
            inp = jnp.column_stack([y, jnp.full((3, 1), x_temp)])
            flux = p * (0.05 + 12.0 * jax.nn.sigmoid(jax.nn.gelu(inp @ wm + bm) @ wo + bo))
            dp = jnp.roll(flux, 1, axis=-1) - flux
            return jnp.column_stack([dp, dz])

        def step(y, _):
            k1 = field(y); k2 = field(y + 0.5*dt*k1); k3 = field(y + 0.5*dt*k2); k4 = field(y + dt*k3)
            yn = y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            pc = jnp.maximum(yn[:, :3], 1e-15); pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
            return jnp.column_stack([pc, yn[:, 3:]]), pc

        traj = jax.lax.scan(step, y0, None, length=n_steps)[1]
        return jnp.transpose(traj[sample_idx], (1, 0, 2))

    # 3. Time-Inhomogeneous Neural-CME(t) on 3-node E. coli ring (w_dim=20 -> 183 params)
    def init_ecoli_cme(seed: int):
        rng = np.random.default_rng(seed)
        w_d = 20
        # feats: [t_n, sin(2pi t_n), cos(2pi t_n), x_temp, age_flag] -> 5*20+20+20*3+3 = 183 params
        return np.concatenate([
            rng.normal(0.0, 0.25, 5 * w_d), np.zeros(w_d),
            rng.normal(0.0, 0.15, w_d * 3), np.zeros(3),
        ])

    def pred_ecoli_cme(th, x_temp: float, is_aged: bool):
        w_d = 20
        w1 = th[:5*w_d].reshape(5, w_d)
        b1 = th[5*w_d:6*w_d]
        w2 = th[6*w_d:6*w_d+3*w_d].reshape(w_d, 3)
        b2 = th[6*w_d+3*w_d:]
        age_f = 1.0 if is_aged else 0.0

        def step(carry, _):
            p, tc = carry
            tn = (tc + 0.5 * dt) / 1.5
            feat = jnp.array([tn, jnp.sin(2.0*jnp.pi*tn), jnp.cos(2.0*jnp.pi*tn), x_temp, age_f])
            rates = 0.05 + 12.0 * jax.nn.sigmoid(jax.nn.gelu(feat @ w1 + b1) @ w2 + b2)

            def rhs(ps):
                fl = ps * rates[None, :]
                return jnp.roll(fl, 1, axis=-1) - fl

            k1 = rhs(p); k2 = rhs(p + 0.5*dt*k1); k3 = rhs(p + 0.5*dt*k2); k4 = rhs(p + dt*k3)
            pn = jnp.maximum(p + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4), 1e-15)
            pn = pn / jnp.sum(pn, axis=-1, keepdims=True)
            return (pn, tc + dt), pn

        traj = jax.lax.scan(step, (jnp.eye(3), 0.0), None, length=n_steps)[1]
        return jnp.transpose(traj[sample_idx], (1, 0, 2))

    # 4. Population Latent ODE (exact same 285-param architecture as real_continuous_benchmarks.py)
    def init_ecoli_lode(seed: int):
        return np.random.default_rng(seed).normal(0.0, 0.22, size=285)

    def pred_ecoli_lode(th, x_temp: float, is_aged: bool):
        enc = th[:30].reshape(5, 6); eb = th[30:36]
        w1 = th[36:156].reshape(10, 12); b1 = th[156:168]
        w2 = th[168:276].reshape(12, 9); b2 = th[276:285]
        age_v = aged_hr if is_aged else 0.0
        prep = jnp.eye(3)
        z0 = jnp.tanh(jnp.column_stack([prep, jnp.full((3, 1), x_temp), jnp.full((3, 1), age_v)]) @ enc + eb)
        y0 = jnp.column_stack([prep, z0])

        def field(y):
            p = y[:, :3]
            inp = jnp.column_stack([y, jnp.full((3, 1), x_temp)])
            out = jnp.tanh(inp @ w1 + b1) @ w2 + b2
            flux = p * (0.05 + 12.0 * jax.nn.sigmoid(out[:, :3]))
            dp = jnp.roll(flux, 1, axis=-1) - flux
            return jnp.column_stack([dp, jnp.tanh(out[:, 3:])])

        def step(y, _):
            k1 = field(y); k2 = field(y + 0.5*dt*k1); k3 = field(y + 0.5*dt*k2); k4 = field(y + dt*k3)
            yn = y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            pc = jnp.maximum(yn[:, :3], 1e-15); pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
            return jnp.column_stack([pc, yn[:, 3:]]), pc

        traj = jax.lax.scan(step, y0, None, length=n_steps)[1]
        return jnp.transpose(traj[sample_idx], (1, 0, 2))

    specs = [
        ("ltc_ode", 9, 311, init_ecoli_ltc, pred_ecoli_ltc),
        ("s4d_ode", 11, 263, init_ecoli_s4d, pred_ecoli_s4d),
        ("neural_cme", 3, 183, init_ecoli_cme, pred_ecoli_cme),
        ("latent_ode", 9, 285, init_ecoli_lode, pred_ecoli_lode),
    ]

    def make_ecoli_fitters(pred_fn):
        @jax.jit
        def vg_joint(th, f_c, a_c):
            def lf(w):
                tot = 0.0
                for idx_t, t_k in enumerate(temps):
                    xf = temp_feat[t_k]
                    tot = tot + multinomial_nll(pred_fn(w, xf, False), f_c[idx_t]) + 0.5 * multinomial_nll(pred_fn(w, xf, True), a_c[idx_t])
                return tot / 3.0 + 1e-4 * jnp.sum(w**2)
            return jax.value_and_grad(lf)(th)

        @jax.jit
        def vg_fo(th, f_c):
            def lf(w):
                tot = 0.0
                for idx_t, t_k in enumerate(temps):
                    tot = tot + multinomial_nll(pred_fn(w, temp_feat[t_k], False), f_c[idx_t])
                return tot / 3.0 + 1e-4 * jnp.sum(w**2)
            return jax.value_and_grad(lf)(th)

        @jax.jit
        def eval_all(th):
            pf = jnp.stack([pred_fn(th, temp_feat[t_k], False) for t_k in temps], axis=0)
            pa = jnp.stack([pred_fn(th, temp_feat[t_k], True) for t_k in temps], axis=0)
            return pf, pa

        return vg_joint, vg_fo, eval_all

    compiled = {m: make_ecoli_fitters(pf) for m, _, _, _, pf in specs}
    raw = {f"{m}_{mode}": {"fresh_kl": [], "aged_kl": []} for m, *_ in specs for mode in ("joint", "fresh_only")}

    def score_batch(p_arr, c_arr):
        kls = []
        for idx_t in range(3):
            ph = np.clip(p_arr[idx_t], 1e-12, 1.0)
            ph /= ph.sum(axis=-1, keepdims=True)
            ca = c_arr[idx_t]
            qe = ca / np.maximum(ca.sum(axis=-1, keepdims=True), 1.0)
            we = ca.sum(axis=-1) / np.maximum(ca.sum(), 1.0)
            kls.append(float(np.sum(we * np.sum(np.where(qe > 0, qe * (np.log(np.maximum(qe, 1e-30)) - np.log(ph)), 0.0), axis=-1))))
        return float(np.mean(kls))

    for fi, te_cl in enumerate(folds):
        tr_recs = [r for r in records if r["cluster"] not in set(te_cl)]
        te_recs = [r for r in records if r["cluster"] in set(te_cl)]
        tr_snap = rcs.build_ecoli_cohort_snapshots(tr_recs, obs_min)
        te_snap = rcs.build_ecoli_cohort_snapshots(te_recs, obs_min)
        f_tr = jnp.asarray([tr_snap[t]["fresh_counts"] for t in temps])
        a_tr = jnp.asarray([tr_snap[t]["aged_counts"] for t in temps])
        f_te = np.asarray([te_snap[t]["fresh_counts"] for t in temps])
        a_te = np.asarray([te_snap[t]["aged_counts"] for t in temps])

        for mname, _, _, init_fn, _ in specs:
            vg_j, vg_fo, ev_fn = compiled[mname]
            bj, bfo = None, None
            for st in range(3):
                w0 = init_fn(600 + 30 * fi + st)
                rj = minimize(lambda z: (float(vg_j(z, f_tr, a_tr)[0]), np.asarray(vg_j(z, f_tr, a_tr)[1], float)), w0, jac=True, method="L-BFGS-B", options={"maxiter": 160})
                if bj is None or rj.fun < bj.fun:
                    bj = rj
                rfo = minimize(lambda z: (float(vg_fo(z, f_tr)[0]), np.asarray(vg_fo(z, f_tr)[1], float)), w0, jac=True, method="L-BFGS-B", options={"maxiter": 160})
                if bfo is None or rfo.fun < bfo.fun:
                    bfo = rfo
            pf_j, pa_j = map(np.asarray, ev_fn(jnp.asarray(bj.x)))
            pf_fo, _ = map(np.asarray, ev_fn(jnp.asarray(bfo.x)))
            raw[f"{mname}_joint"]["fresh_kl"].append(score_batch(pf_j, f_te))
            raw[f"{mname}_joint"]["aged_kl"].append(score_batch(pa_j, a_te))
            raw[f"{mname}_fresh_only"]["fresh_kl"].append(score_batch(pf_fo, f_te))
            # Clean age=0 zero-masked evaluation for fresh_only!
            raw[f"{mname}_fresh_only"]["aged_kl"].append(score_batch(pf_fo, a_te))
        print(
            f"E. coli R6 Fold {fi+1}/5 | 0-shot aged (clean age=0): "
            f"LTC={raw['ltc_ode_fresh_only']['aged_kl'][-1]:.4f} "
            f"S4D={raw['s4d_ode_fresh_only']['aged_kl'][-1]:.4f} "
            f"CME={raw['neural_cme_fresh_only']['aged_kl'][-1]:.4f} "
            f"LODE={raw['latent_ode_fresh_only']['aged_kl'][-1]:.4f}",
            flush=True,
        )

    out = {"states_params": {m: {"states": st, "params": pr} for m, st, pr, *_ in specs}, "models_5fold_clean_age0": {}}
    for k, d in raw.items():
        out["models_5fold_clean_age0"][k] = {
            "fresh_kl": round(float(np.mean(d["fresh_kl"])), 5),
            "fresh_kl_sd": round(float(np.std(d["fresh_kl"], ddof=1)), 5),
            "aged_kl": round(float(np.mean(d["aged_kl"])), 5),
            "aged_kl_sd": round(float(np.std(d["aged_kl"], ddof=1)), 5),
        }
    return out


def run_helpdesk_hyper_pos_coarsening() -> list[dict]:
    csv_path = ensure_helpdesk_csv(Path("/tmp/drn_r6_helpdesk"))
    trajs, _ = load_ticket_trajectories(csv_path)
    rng = np.random.default_rng(2026)
    folds = np.array_split(rng.permutation(len(trajs)), 5)
    te_idx0 = set(folds[0].tolist())
    tr0 = [trajs[i] for i in range(len(trajs)) if i not in te_idx0]
    te0 = [trajs[i] for i in range(len(trajs)) if i in te_idx0]

    psi_univ, _ = calibrate_universal_pos_clock(stages_red=8, stages_full=16, maxiter=250, k_min=1.4, k_max=7.8)
    pred_hpos = make_universal_hyper_pos_helpdesk(psi_univ, stages_red=8)

    def init_hpos(seed: int):
        r = np.random.default_rng(seed)
        return np.r_[np.log([0.6, 0.8, 1.5, 4.0]), np.log([6.0, 8.0, 15.0, 30.0]), r.normal(0.0, 0.2, 4), np.zeros(4), np.zeros(E_EDGES)]

    eval_horizons = np.array([1.0, 3.0, 7.0, 15.0, 30.0], dtype=float)
    h_eval_jax = jnp.asarray(eval_horizons, dtype=jnp.float64)
    fresh_te0, aged_te0 = extract_snapshot_tables(te0, horizons=eval_horizons)

    rows = []
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
        n_obs = float(np.maximum(fresh_grid.sum(), 1.0))

        @jax.jit
        def vg(w):
            def lf(ww):
                pf, _ = pred_hpos(ww, grid_jax)
                return -jnp.sum(f_grid_jax * jnp.log(jnp.clip(pf, 1e-12, 1.0))) / n_obs + 1e-4 * jnp.mean(ww**2)
            return jax.value_and_grad(lf)(w)

        bw, bl = None, float("inf")
        for st in range(2):
            res = minimize(lambda x: (float(vg(jnp.asarray(x))[0]), np.asarray(vg(jnp.asarray(x))[1], float)), init_hpos(900 + st), jac=True, method="L-BFGS-B", options={"maxiter": 150})
            if res.fun < bl:
                bl, bw = float(res.fun), res.x.copy()
        pf, pa = pred_hpos(jnp.asarray(bw), h_eval_jax)
        row = {
            "delta_t_days": dt,
            "hyper_pos_sens_fresh_kl": empirical_kl(np.asarray(pf), fresh_te0),
            "hyper_pos_sens_aged_kl": empirical_kl(np.asarray(pa), aged_te0),
        }
        print(f"Helpdesk R6 Hyper-Pos-Sens Coarsen dt={dt:4.1f}d | fresh={row['hyper_pos_sens_fresh_kl']:.4f} | aged={row['hyper_pos_sens_aged_kl']:.4f}", flush=True)
        rows.append(row)
    return rows


def main():
    ecoli = run_ecoli_neural_suite()
    hdk_coarsen = run_helpdesk_hyper_pos_coarsening()
    out = {
        "ecoli_r6_complete_neural_odes": ecoli,
        "helpdesk_r6_hyper_pos_coarsening": hdk_coarsen,
    }
    out_path = ROOT / "results" / "archived" / "r6_uniformity_closure.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
