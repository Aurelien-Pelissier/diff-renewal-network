"""Round-6b Integrity & Protocol-Closure Audit Script.

Fixes every concrete protocol-to-code discrepancy identified in blind empirical & forensic review:
1. Sleep-EDF:
   - Strict fresh_only CTMC (eliminating joint-data leakage into the 0-shot column).
   - Equalized 3-start clean zero-masked (age=0) 5-fold CV for ALL FOUR neural ODEs
     (LTC 534p, S4D 545p, Neural-CME 394p, Latent ODE 540p), PLUS clock-shifted
     Neural-CME(t + a) check.
2. 1978 Influenza:
   - Adds 3-start Time-Inhomogeneous Neural-CME(t) (full fit + Days 9-14 / 10-14 L2<=0.1 sweep)
     so all four neural ODE baselines are present on all five real-world domains.
3. Human Hematopoiesis:
   - Logs exact GMP / MEP bifurcation split directly from the full-cone [1.4, 7.8]
     Universal Pos-Sens fit (eliminating cross-archive attribution).
4. Cyclic Reduction Plain-ME 33 (K_r=11) Ablation & Same-Machine Warm Timing Audit:
   - Evaluates plain K_r=11 mixed-Erlang vs Pos-Sens 33 vs Interp. 33 vs Full 48 on
     identical hardware across eps in {0, 0.15, 0.25} and boundary shapes k in {8.5, 9.5, 10.5}.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

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
    mixed_erlang_components,
    multinomial_nll,
    smooth_erlang_logits,
)
from diff_renewal.traits import _shape_center  # noqa: E402
from diff_renewal.reduction import (  # noqa: E402
    ROUTE,
    STEP,
    STEPS,
    build_interpolatory_basis,
    make_cycle_engine,
    make_interpolatory_engine,
    make_positive_33_engine,
)
from scripts.run_edge_memory_sleep_benchmark import (  # noqa: E402
    build_sleep_snapshots,
    parse_sleep_sequences,
)
from scripts.run_hematopoiesis_branching_benchmark import (  # noqa: E402
    EDGES as HEMA_EDGES,
    aggregate_counts_by_day,
    load_hematopoiesis_facs,
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


def run_sleep_integrity() -> dict:
    subj_seqs = parse_sleep_sequences()
    subjects = sorted(subj_seqs.keys())
    assert len(subjects) == 75
    obs_epochs = np.array([2, 4, 8, 12, 20, 30, 45, 60], dtype=int)
    aged_ten_min = 0.5
    _, _, all_ec = build_sleep_snapshots(subj_seqs, subjects, obs_epochs)
    edges = tuple(
        (u, v) for u in range(5) for v in range(5) if u != v and all_ec[u, v] >= 0.01 * all_ec[u].sum()
    )
    n_edges = len(edges)
    src_idx = jnp.array([u for u, _ in edges])
    dst_idx = jnp.array([v for _, v in edges])
    dt = 0.05
    max_ep = int(obs_epochs[-1])
    sample_idx = jnp.asarray(obs_epochs - 1, dtype=jnp.int32)
    ts_ten_min = jnp.asarray(obs_epochs * 0.5 / 10.0, dtype=jnp.float64)
    folds = [subjects[i::5] for i in range(5)]

    # 1. Strict fresh_only & joint CTMC (16 params)
    def pred_ctmc(th):
        rates = jnp.exp(jnp.clip(th, -6.0, 4.0))
        gen = jnp.zeros((5, 5), dtype=jnp.float64)
        for ei, (u, v) in enumerate(edges):
            gen = gen.at[v, u].add(rates[ei]).at[u, u].add(-rates[ei])
        probs = jax.vmap(lambda t: jl.expm(t * gen).T)(ts_ten_min)
        p_out = jnp.transpose(probs, (1, 0, 2))
        return p_out, p_out

    # 2. LTC ODE (534 params)
    z_ltc, w_ltc = 6, 12
    def init_ltc(seed: int):
        r = np.random.default_rng(seed)
        return np.r_[
            r.normal(0, 0.25, 6*z_ltc), np.zeros(z_ltc),
            r.normal(0, 0.20, 11*w_ltc), np.zeros(w_ltc),
            r.normal(0, 0.15, w_ltc*z_ltc), np.zeros(z_ltc),
            r.normal(0, 0.15, w_ltc*z_ltc), np.zeros(z_ltc),
            r.normal(0, 0.15, 11*n_edges), np.full(n_edges, -1.5),
        ]

    def pred_ltc(th):
        i = 0
        enc = th[i:i+6*z_ltc].reshape(6, z_ltc); i += 6*z_ltc
        eb = th[i:i+z_ltc]; i += z_ltc
        wh = th[i:i+11*w_ltc].reshape(11, w_ltc); i += 11*w_ltc
        bh = th[i:i+w_ltc]; i += w_ltc
        wa = th[i:i+w_ltc*z_ltc].reshape(w_ltc, z_ltc); i += w_ltc*z_ltc
        ba = th[i:i+z_ltc]; i += z_ltc
        wt = th[i:i+w_ltc*z_ltc].reshape(w_ltc, z_ltc); i += w_ltc*z_ltc
        bt = th[i:i+z_ltc]; i += z_ltc
        wf = th[i:i+11*n_edges].reshape(11, n_edges); i += 11*n_edges
        bf = th[i:i+n_edges]

        def rollout(age_v: float):
            p0 = jnp.eye(5)
            z0 = jnp.tanh(jnp.concatenate([p0, jnp.full((5, 1), age_v)], axis=-1) @ enc + eb)
            y0 = jnp.concatenate([p0, z0], axis=-1)
            def rhs(y):
                p, z = y[:, :5], y[:, 5:]
                h = jnp.tanh(y @ wh + bh)
                dz = -(0.1 + 4.0 * jax.nn.sigmoid(h @ wt + bt)) * (z - jnp.tanh(h @ wa + ba))
                ef = p[:, src_idx] * (0.02 + 8.0 * jax.nn.sigmoid(y @ wf + bf))
                dp = jnp.zeros_like(p).at[:, dst_idx].add(ef).at[:, src_idx].add(-ef)
                return jnp.concatenate([dp, dz], axis=-1)
            def step(y, _):
                k1 = rhs(y); k2 = rhs(y + 0.5*dt*k1); k3 = rhs(y + 0.5*dt*k2); k4 = rhs(y + dt*k3)
                yn = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                pc = jnp.maximum(yn[:, :5], 1e-15); pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
                return jnp.concatenate([pc, yn[:, 5:]], axis=-1), pc
            return jnp.transpose(jax.lax.scan(step, y0, None, length=max_ep)[1][sample_idx], (1, 0, 2))
        return rollout(0.0), rollout(aged_ten_min)

    # 3. S4D ODE (545 params)
    z_s4, m_s4 = 10, 12
    def init_s4d(seed: int):
        r = np.random.default_rng(seed)
        return np.r_[
            r.normal(0, 0.25, 6*z_s4), np.zeros(z_s4),
            np.linspace(np.log(0.1), np.log(5.0), z_s4), r.normal(0, 0.2, z_s4//2),
            r.normal(0, 0.2, 5*z_s4), np.zeros(z_s4),
            r.normal(0, 0.2, 15*m_s4), np.zeros(m_s4),
            r.normal(0, 0.15, m_s4*n_edges), np.full(n_edges, -1.5),
        ]

    def pred_s4d(th):
        half = z_s4 // 2
        i = 0
        enc = th[i:i+6*z_s4].reshape(6, z_s4); i += 6*z_s4
        eb = th[i:i+z_s4]; i += z_s4
        dec = jnp.exp(jnp.clip(th[i:i+z_s4], -4.0, 3.0)); i += z_s4
        om = th[i:i+half]; i += half
        wb = th[i:i+5*z_s4].reshape(5, z_s4); i += 5*z_s4
        bb = th[i:i+z_s4]; i += z_s4
        wm = th[i:i+15*m_s4].reshape(15, m_s4); i += 15*m_s4
        bm = th[i:i+m_s4]; i += m_s4
        wo = th[i:i+m_s4*n_edges].reshape(m_s4, n_edges); i += m_s4*n_edges
        bo = th[i:i+n_edges]

        def rollout(age_v: float):
            p0 = jnp.eye(5)
            z0 = jnp.tanh(jnp.concatenate([p0, jnp.full((5, 1), age_v)], axis=-1) @ enc + eb)
            y0 = jnp.concatenate([p0, z0], axis=-1)
            def rhs(y):
                p, z = y[:, :5], y[:, 5:]
                rot = jnp.concatenate([-om * z[:, half:], om * z[:, :half]], axis=-1)
                dz = -dec * z + rot + (p @ wb + bb)
                ef = p[:, src_idx] * (0.02 + 8.0 * jax.nn.sigmoid(jax.nn.gelu(y @ wm + bm) @ wo + bo))
                dp = jnp.zeros_like(p).at[:, dst_idx].add(ef).at[:, src_idx].add(-ef)
                return jnp.concatenate([dp, dz], axis=-1)
            def step(y, _):
                k1 = rhs(y); k2 = rhs(y + 0.5*dt*k1); k3 = rhs(y + 0.5*dt*k2); k4 = rhs(y + dt*k3)
                yn = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                pc = jnp.maximum(yn[:, :5], 1e-15); pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
                return jnp.concatenate([pc, yn[:, 5:]], axis=-1), pc
            return jnp.transpose(jax.lax.scan(step, y0, None, length=max_ep)[1][sample_idx], (1, 0, 2))
        return rollout(0.0), rollout(aged_ten_min)

    # 4. Neural-CME(t) (394 params, plus clock-shifted t -> t + a rollout!)
    cme_w = 18
    def init_cme(seed: int):
        r = np.random.default_rng(seed)
        return r.normal(0.0, 0.18, size=4*cme_w + cme_w + cme_w*n_edges + n_edges)

    def pred_cme_mode(th, age_v: float, t_shift: float = 0.0):
        w1 = th[:4*cme_w].reshape(4, cme_w)
        b1 = th[4*cme_w:5*cme_w]
        w2 = th[5*cme_w:5*cme_w+cme_w*n_edges].reshape(cme_w, n_edges)
        b2 = th[5*cme_w+cme_w*n_edges:]
        def step(p, i_st):
            t_norm = ((i_st + 0.5) * dt + t_shift) / 3.0
            feat = jnp.array([t_norm, jnp.sin(2.0*jnp.pi*t_norm), jnp.cos(2.0*jnp.pi*t_norm), age_v])
            rates = 0.02 + 8.0 * jax.nn.sigmoid(jax.nn.gelu(feat @ w1 + b1) @ w2 + b2 - 1.5)
            def fld(pc):
                ef = pc[:, src_idx] * rates[None, :]
                return jnp.zeros_like(pc).at[:, dst_idx].add(ef).at[:, src_idx].add(-ef)
            k1 = fld(p); k2 = fld(p + 0.5*dt*k1); k3 = fld(p + 0.5*dt*k2); k4 = fld(p + dt*k3)
            pn = jnp.maximum(p + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 1e-15)
            pn = pn / jnp.sum(pn, axis=-1, keepdims=True)
            return pn, pn
        return jnp.transpose(jax.lax.scan(step, jnp.eye(5), jnp.arange(max_ep))[1][sample_idx], (1, 0, 2))

    def pred_cme(th):
        return pred_cme_mode(th, 0.0, 0.0), pred_cme_mode(th, aged_ten_min, 0.0)

    pred_cme_shifted_jit = jax.jit(lambda th: pred_cme_mode(th, 0.0, aged_ten_min))

    # 5. Latent ODE (540 params: enc 6*6+6=42; w1 11*14+14=168; w2 14*22+22=330 -> 540 params)
    z_lo, w_lo = 6, 14
    def init_lode(seed: int):
        return np.random.default_rng(seed).normal(0.0, 0.18, size=540)

    def pred_lode(th):
        i = 0
        enc = th[i:i+36].reshape(6, z_lo); i += 36
        eb = th[i:i+z_lo]; i += z_lo
        w1 = th[i:i+11*w_lo].reshape(11, w_lo); i += 11*w_lo
        b1 = th[i:i+w_lo]; i += w_lo
        w2 = th[i:i+w_lo*(n_edges+z_lo)].reshape(w_lo, n_edges+z_lo); i += w_lo*(n_edges+z_lo)
        b2 = th[i:i+n_edges+z_lo]
        def rollout(age_v: float):
            p0 = jnp.eye(5)
            z0 = jnp.tanh(jnp.concatenate([p0, jnp.full((5, 1), age_v)], axis=-1) @ enc + eb)
            y0 = jnp.concatenate([p0, z0], axis=-1)
            def rhs(y):
                p = y[:, :5]
                raw = jnp.tanh(y @ w1 + b1) @ w2 + b2
                ef = p[:, src_idx] * (0.02 + 8.0 * jax.nn.sigmoid(raw[:, :n_edges] - 1.5))
                dp = jnp.zeros_like(p).at[:, dst_idx].add(ef).at[:, src_idx].add(-ef)
                return jnp.concatenate([dp, jnp.tanh(raw[:, n_edges:])], axis=-1)
            def step(y, _):
                k1 = rhs(y); k2 = rhs(y + 0.5*dt*k1); k3 = rhs(y + 0.5*dt*k2); k4 = rhs(y + dt*k3)
                yn = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                pc = jnp.maximum(yn[:, :5], 1e-15); pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
                return jnp.concatenate([pc, yn[:, 5:]], axis=-1), pc
            return jnp.transpose(jax.lax.scan(step, y0, None, length=max_ep)[1][sample_idx], (1, 0, 2))
        return rollout(0.0), rollout(aged_ten_min)

    def make_fitters(pfn, reg=1e-4):
        @jax.jit
        def vg_j(w, fc, ac):
            def lf(ww):
                pf, pa = pfn(ww)
                return multinomial_nll(pf, fc) + multinomial_nll(pa, ac) + reg * jnp.sum(ww**2)
            return jax.value_and_grad(lf)(w)
        @jax.jit
        def vg_f(w, fc):
            def lf(ww):
                pf, _ = pfn(ww)
                return multinomial_nll(pf, fc) + reg * jnp.sum(ww**2)
            return jax.value_and_grad(lf)(w)
        p_jit = jax.jit(pfn)
        return vg_j, vg_f, p_jit

    specs = [
        ("ctmc", 1, lambda s: np.full(n_edges, -1.0), pred_ctmc, 1e-5),
        ("ltc_ode", 3, init_ltc, pred_ltc, 1e-4),
        ("s4d_ode", 3, init_s4d, pred_s4d, 1e-4),
        ("neural_cme", 3, init_cme, pred_cme, 1e-4),
        ("latent_ode", 3, init_lode, pred_lode, 1e-4),
    ]
    comp = {m: make_fitters(pf, reg) for m, _, _, pf, reg in specs}
    raw = {f"{m}_{md}": {"fresh_kl": [], "aged_kl": []} for m, *_ in specs for md in ("joint", "fresh_only")}
    raw["neural_cme_shifted_t_plus_a"] = {"aged_kl": []}

    for fi, te_subjs in enumerate(folds):
        tr_subjs = [s for s in subjects if s not in set(te_subjs)]
        f_tr, a_tr, _ = build_sleep_snapshots(subj_seqs, tr_subjs, obs_epochs)
        f_te, a_te, _ = build_sleep_snapshots(subj_seqs, te_subjs, obs_epochs)
        fc_j, ac_j = jnp.asarray(f_tr), jnp.asarray(a_tr)

        for mname, n_st, ifn, _, _ in specs:
            vg_j, vg_f, p_jit = comp[mname]
            bj, bf = None, None
            for st in range(n_st):
                w0 = ifn(800 + 25 * fi + st)
                rj = minimize(lambda z: (float(vg_j(z, fc_j, ac_j)[0]), np.asarray(vg_j(z, fc_j, ac_j)[1], float)), w0, jac=True, method="L-BFGS-B", options={"maxiter": 180})
                if bj is None or rj.fun < bj.fun:
                    bj = rj
                rf = minimize(lambda z: (float(vg_f(z, fc_j)[0]), np.asarray(vg_f(z, fc_j)[1], float)), w0, jac=True, method="L-BFGS-B", options={"maxiter": 180})
                if bf is None or rf.fun < bf.fun:
                    bf = rf
            pf_j, pa_j = map(np.asarray, p_jit(jnp.asarray(bj.x)))
            pf_f, _ = map(np.asarray, p_jit(jnp.asarray(bf.x)))
            raw[f"{mname}_joint"]["fresh_kl"].append(excess_kl(pf_j, f_te))
            raw[f"{mname}_joint"]["aged_kl"].append(excess_kl(pa_j, a_te))
            raw[f"{mname}_fresh_only"]["fresh_kl"].append(excess_kl(pf_f, f_te))
            # Clean age=0 zero-masked evaluation (and exact memoryless CTMC fresh-only!)
            raw[f"{mname}_fresh_only"]["aged_kl"].append(excess_kl(pf_f, a_te))
            if mname == "neural_cme":
                pa_sh = np.asarray(pred_cme_shifted_jit(jnp.asarray(bf.x)))
                raw["neural_cme_shifted_t_plus_a"]["aged_kl"].append(excess_kl(pa_sh, a_te))
        print(
            f"Sleep R6b Fold {fi+1}/5 | CTMC(fo)={raw['ctmc_fresh_only']['aged_kl'][-1]:.4f} "
            f"| LTC={raw['ltc_ode_fresh_only']['aged_kl'][-1]:.4f} "
            f"| S4D={raw['s4d_ode_fresh_only']['aged_kl'][-1]:.4f} "
            f"| CME={raw['neural_cme_fresh_only']['aged_kl'][-1]:.4f} (sh={raw['neural_cme_shifted_t_plus_a']['aged_kl'][-1]:.4f}) "
            f"| LODE={raw['latent_ode_fresh_only']['aged_kl'][-1]:.4f}",
            flush=True,
        )

    out = {}
    for k, d in raw.items():
        out[k] = {
            f"{mk}_mean": round(float(np.mean(mv)), 5) for mk, mv in d.items()
        } | {
            f"{mk}_sd": round(float(np.std(mv, ddof=1)), 5) for mk, mv in d.items()
        }
    return out


def run_influenza_neural_cme() -> dict:
    """3-start Time-Inhomogeneous Neural-CME(t) on 1978 Influenza (3 observed simplex categories)."""
    ts_all = jnp.asarray(INF_DAYS, dtype=jnp.float64)
    dt = 0.05
    n_steps = int(round(14.0 / dt))
    s_idx = jnp.rint(ts_all / dt).astype(int) - 1
    w_d = 16
    # init_logits(2) + w1(3*16=48) + b1(16) + w2(16*6=96 directed edges among 3 cats) + b2(6) = 168 params
    edges3 = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    s3 = jnp.array([u for u, _ in edges3])
    d3 = jnp.array([v for _, v in edges3])

    def init_cme(seed: int):
        r = np.random.default_rng(seed)
        return np.r_[np.array([-5.5, -7.0]), r.normal(0, 0.25, 3*w_d), np.zeros(w_d), r.normal(0, 0.15, w_d*6), np.full(6, -2.0)]

    def pred_cme(th):
        il = th[:2]
        w1 = th[2:2+3*w_d].reshape(3, w_d)
        b1 = th[2+3*w_d:2+4*w_d]
        w2 = th[2+4*w_d:2+4*w_d+6*w_d].reshape(w_d, 6)
        b2 = th[2+4*w_d+6*w_d:]
        p0 = jax.nn.softmax(jnp.array([il[0], il[1], 0.0]))
        def step(p, i_st):
            tn = (i_st + 0.5) * dt / 14.0
            feat = jnp.array([tn, jnp.sin(2.0*jnp.pi*tn), jnp.cos(2.0*jnp.pi*tn)])
            rates = 1e-4 + 4.0 * jax.nn.sigmoid(jax.nn.gelu(feat @ w1 + b1) @ w2 + b2)
            def fld(pc):
                ef = pc[s3] * rates
                return jnp.zeros_like(pc).at[d3].add(ef).at[s3].add(-ef)
            k1 = fld(p); k2 = fld(p + 0.5*dt*k1); k3 = fld(p + 0.5*dt*k2); k4 = fld(p + dt*k3)
            pn = jnp.maximum(p + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 1e-12)
            pn = pn / jnp.sum(pn)
            return pn, pn
        return jax.lax.scan(step, p0, jnp.arange(n_steps))[1][s_idx]

    p_jit = jax.jit(pred_cme)
    def fit_slice(n_tr, reg_v):
        obs_tr = jnp.asarray(INF_COUNTS[:n_tr], dtype=jnp.float64)
        n_tot = float(INF_COUNTS[:n_tr].sum())
        @jax.jit
        def vg(w):
            def lf(ww):
                p = jnp.clip(pred_cme(ww)[:n_tr], 1e-12, 1.0)
                return -jnp.sum(obs_tr * jnp.log(p)) / n_tot + reg_v * jnp.mean(ww**2)
            return jax.value_and_grad(lf)(w)
        best = None
        for s in range(3):
            r = minimize(lambda z: (float(vg(z)[0]), np.asarray(vg(z)[1], float)), init_cme(20 + s), jac=True, method="L-BFGS-B", options={"maxiter": 220})
            if best is None or r.fun < best.fun:
                best = r
        return np.asarray(p_jit(jnp.asarray(best.x)))

    out = {"states": 3, "params": int(init_cme(0).size), "full_fit": metrics_on_slice(fit_slice(14, 5e-4), INF_COUNTS)}
    for n_tr in (8, 9):
        bm, bl2 = None, None
        for rv in (5e-4, 5e-3, 2e-2, 1e-1):
            mt = metrics_on_slice(fit_slice(n_tr, rv)[n_tr:], INF_COUNTS[n_tr:])
            if bm is None or mt["mean_kl_nats"] < bm["mean_kl_nats"]:
                bm, bl2 = mt, rv
        out[f"days_1to{n_tr}_to_{n_tr+1}to14_best_l2"] = {"best_l2": bl2, "test": bm}
    print(
        f"Influenza R6b Neural-CME | Full={out['full_fit']['mean_kl_nats']:.4f} "
        f"| D9-14={out['days_1to8_to_9to14_best_l2']['test']['mean_kl_nats']:.4f} ({out['days_1to8_to_9to14_best_l2']['test']['count_rmse_boys']:.1f} boys) "
        f"| D10-14={out['days_1to9_to_10to14_best_l2']['test']['mean_kl_nats']:.4f}",
        flush=True,
    )
    return out


def run_hematopoiesis_fullcone_bifurcation() -> dict:
    """Log exact GMP/MEP bifurcation split directly from the full-cone [1.4, 7.8] Universal Pos-Sens fit."""
    df = load_hematopoiesis_facs()
    psi_u, cert = calibrate_universal_pos_clock(stages_red=8, stages_full=16, maxiter=250, k_min=1.4, k_max=7.8)
    days_list = sorted(int(d) for d in df["Tag"].unique())
    days_arr, counts_mat = aggregate_counts_by_day(df, days_list)
    day_means = counts_mat / np.maximum(counts_mat.sum(axis=1, keepdims=True), 1e-12)
    ts_j = jnp.asarray(days_arr - 1.0, dtype=jnp.float64)
    p0_j = jnp.asarray(day_means[0], dtype=jnp.float64)
    emp_j = jnp.asarray(counts_mat[1:], dtype=jnp.float64)
    n_tot = float(counts_mat[1:].sum())

    def pred(th):
        nm = jnp.concatenate([jnp.exp(jnp.clip(th[:6], -2.0, 3.0)), jnp.array([1.0])])
        ns = jnp.concatenate([1.4 + 6.4 * jax.nn.sigmoid(th[6:12]), jnp.array([4.0])])
        p_mpp_cmp = jax.nn.sigmoid(th[12])
        p_cmp_gmp = jax.nn.sigmoid(th[13])
        rw = jnp.array([1.0, 1.0 - p_mpp_cmp, p_mpp_cmp, p_cmp_gmp, 1.0 - p_cmp_gmp, 1.0, 1.0, 1.0])
        gen, fr, rd = build_universal_pos_network_generator(nm, ns, rw, HEMA_EDGES, 7, psi_u, stages_red=8)
        x0 = fr @ p0_j
        return jax.vmap(lambda t: rd @ (jl.expm(t * gen) @ x0))(ts_j[1:])

    @jax.jit
    def vg(w):
        def lf(ww):
            p = jnp.clip(pred(ww), 1e-12, 1.0)
            return -jnp.sum(emp_j * jnp.log(p)) / n_tot + 1e-5 * jnp.mean(ww**2)
        return jax.value_and_grad(lf)(w)

    res = minimize(lambda z: (float(vg(z)[0]), np.asarray(vg(z)[1], float)), np.r_[np.log([2.0]*6), np.zeros(8)], jac=True, method="L-BFGS-B", options={"maxiter": 250})
    p_cmp_gmp = float(1.0 / (1.0 + np.exp(-res.x[13])))
    p_mpp_cmp = float(1.0 / (1.0 + np.exp(-res.x[12])))
    out = {
        "cone": [1.4, 7.8],
        "sup_survival_abs_error": round(cert["sup_survival_abs_error"], 5),
        "cmp_to_gmp_pct": round(100.0 * p_cmp_gmp, 1),
        "cmp_to_mep_pct": round(100.0 * (1.0 - p_cmp_gmp), 1),
        "mpp_to_cmp_pct": round(100.0 * p_mpp_cmp, 1),
    }
    print("Hematopoiesis R6b full-cone [1.4, 7.8] bifurcations:", out, flush=True)
    return out


def run_plain_me33_ablation() -> dict:
    """Compare Plain-ME 33 (K_r=11 plain mixed-Erlang, zero offline calibration) vs Pos-Sens 33
    vs Interp. 33 vs Full 48 on identical hardware across interior (k~6) and boundary (k in [8.5, 10.5]) regimes."""
    pos_arch = json.loads((ROOT / "results" / "archived" / "positive_sensitivity_summary.json").read_text())
    psi_star = jnp.asarray(pos_arch["psi_star"], dtype=jnp.float64)

    full48 = make_cycle_engine(reduced=False)
    pos33 = make_positive_33_engine(psi_star)
    q33, _ = build_interpolatory_basis(dimension=33)
    sp33 = make_interpolatory_engine(q33)

    # Plain-ME 33: standard K_r=11 smooth mixed-Erlang truncation (zero offline calibration!)
    local11 = jnp.eye(11, k=-1) - jnp.eye(11)
    exit11 = jnp.kron(jnp.eye(3), jnp.eye(1, 11, 10))
    read11 = jnp.kron(jnp.eye(3), jnp.ones((1, 11)))
    def _aux_me33(theta):
        mean, shape = jnp.exp(jnp.asarray(theta)).reshape(2, 3)
        centers = jax.vmap(lambda k: _shape_center(k, 11, 0.5))(shape)
        logits11 = jax.vmap(lambda c: smooth_erlang_logits(c, stages=11, width=0.5))(centers)
        rates, entry = jax.vmap(mixed_erlang_components)(mean, logits11)
        fresh = jl.block_diag(*(e[:, None] for e in entry))
        gen = jnp.kron(jnp.diag(rates), local11) + fresh @ ROUTE @ (rates[:, None] * exit11)
        tr = jl.expm(STEP * gen)
        p = jax.lax.scan(lambda st, _: (tr @ st, (read11 @ (tr @ st)).T), fresh, None, length=STEPS)[1]
        return p, p
    me33_jit = jax.jit(jax.jacfwd(_aux_me33, has_aux=True))
    def plain_me33(th):
        j, p = me33_jit(jnp.asarray(th))
        return np.asarray(p), np.asarray(j)

    # Warm all 4 engines and measure same-machine warm eval+Jacobian latency (median of 30 runs)
    origin = np.log([2.0, 2.0, 2.0, 6.0, 6.0, 6.0])
    for fn in (full48, pos33, sp33, plain_me33):
        fn(origin)
    timings = {}
    for name, fn in [("full48", full48), ("pos33", pos33), ("interp33", sp33), ("plain_me33", plain_me33)]:
        ts = []
        for _ in range(30):
            t0 = perf_counter()
            fn(origin)
            ts.append(1000.0 * (perf_counter() - t0))
        timings[name] = round(float(np.median(ts)), 3)

    # Compare probability and Jacobian defects across k=6 (eps=0, 0.15, 0.25) AND near-boundary k in {8.5, 9.5, 10.5}
    sample = np.arange(2, 14)
    mix_d = np.array([1.0, 0.0, -1.0, -1.0, 1.0, 0.0])
    regimes = {
        "k6_sym_eps0": origin,
        "k6_mixed_eps015": origin + 0.15 * mix_d,
        "k6_mixed_eps025": origin + 0.25 * mix_d,
        "boundary_k8p5": np.log([2.0, 2.0, 2.0, 8.5, 8.5, 8.5]),
        "boundary_k9p5": np.log([2.0, 2.0, 2.0, 9.5, 9.5, 9.5]),
        "boundary_k10p5": np.log([2.0, 2.0, 2.0, 10.5, 10.5, 10.5]),
    }
    defects = {}
    for rname, th in regimes.items():
        pf, jf = full48(th)
        defects[rname] = {}
        for mname, fn in [("pos33", pos33), ("plain_me33", plain_me33), ("interp33", sp33)]:
            pr, jr = fn(th)
            defects[rname][mname] = {
                "prob_max_abs": round(float(np.max(np.abs(pf[sample] - pr[sample]))), 6),
                "jac_max_abs": round(float(np.max(np.abs(jf[sample] - jr[sample]))), 6),
            }
    print("Same-machine warm timings (ms):", timings, flush=True)
    print("Plain-ME 33 vs Pos-Sens 33 vs Interp. 33 defects:", json.dumps(defects, indent=2), flush=True)
    return {"same_machine_warm_ms": timings, "defects": defects}


def main():
    plain_ab = run_plain_me33_ablation()
    inf_cme = run_influenza_neural_cme()
    hema_bif = run_hematopoiesis_fullcone_bifurcation()
    sleep_int = run_sleep_integrity()
    out = {
        "sleep_r6b_3start_clean_age0_and_fresh_ctmc": sleep_int,
        "influenza_r6b_neural_cme": inf_cme,
        "hematopoiesis_r6b_fullcone_bifurcation": hema_bif,
        "cyclic_r6b_plain_me33_ablation_and_timings": plain_ab,
    }
    out_path = ROOT / "results" / "archived" / "r6_integrity_fixes.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
