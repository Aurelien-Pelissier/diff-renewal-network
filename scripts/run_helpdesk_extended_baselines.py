"""Extended Real-World IT Helpdesk Baselines: LTC, S4D, Universal 1D Pos-Sens, & Chronological Split.

Evaluates on the exact same 4,576-ticket Polato (2017) IT Helpdesk workflow
(MD5: a91d74377c55be5cbdfda05dae69db30) and identical 5-fold ticket split:
1. Real-world Liquid Time-Constant ODE (LTC) & Diagonal State-Space ODE (S4D)
   (both fresh-only zero-shot aged transfer and supervised joint).
2. Zero-shot Universal 1D Pos-Sens (K_r=8, 33 states) plugged directly into
   the cyclic 5-node IT Helpdesk graph.
3. Strict Chronological Split (first 70% tickets -> last 30% future tickets)
   testing robustness under real-world operational drift across all 8 architectures.
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    build_universal_pos_network_generator,
    calibrate_universal_pos_clock,
)
from scripts.run_helpdesk_systems_benchmark import (  # noqa: E402
    AGED_THRESHOLD_DAYS,
    CORE_STATES,
    E_EDGES,
    EDGES,
    HORIZONS_DAYS,
    K_RENEWAL,
    STATE_MAP,
    V_NODES,
    empirical_kl,
    ensure_helpdesk_csv,
    extract_snapshot_tables,
    fit_model,
    init_edge_memory_drn,
    init_helpdesk_latent_ode,
    init_helpdesk_neural_cme,
    load_ticket_trajectories,
    predict_ctmc,
    predict_edge_memory_drn,
    predict_helpdesk_latent_ode,
    predict_helpdesk_neural_cme,
    predict_renewal_drn,
)


# ---------------------------------------------------------------------------
# Real-world LTC ODE on 5-node, 9-edge Helpdesk graph
# ---------------------------------------------------------------------------
def init_helpdesk_ltc(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z_dim, w_dim = 6, 12
    # enc: 6*6+6=42; wh: 11*12+12=144; wa: 12*6+6=78; wt: 12*6+6=78; wf: 11*9+9=108 -> 450 params
    return np.concatenate(
        [
            rng.normal(0.0, 0.25, size=6 * z_dim),
            np.zeros(z_dim),
            rng.normal(0.0, 0.20, size=(V_NODES + z_dim) * w_dim),
            np.zeros(w_dim),
            rng.normal(0.0, 0.15, size=w_dim * z_dim),
            np.zeros(z_dim),
            rng.normal(0.0, 0.15, size=w_dim * z_dim),
            np.zeros(z_dim),
            rng.normal(0.0, 0.15, size=(V_NODES + z_dim) * E_EDGES),
            np.full(E_EDGES, -2.0),
        ]
    )


def predict_helpdesk_ltc(theta: jax.Array, horizons: jax.Array):
    z_dim, w_dim = 6, 12
    i = 0
    enc = theta[i : i + 6 * z_dim].reshape(6, z_dim); i += 6 * z_dim
    eb = theta[i : i + z_dim]; i += z_dim
    wh = theta[i : i + (V_NODES + z_dim) * w_dim].reshape(V_NODES + z_dim, w_dim); i += (V_NODES + z_dim) * w_dim
    bh = theta[i : i + w_dim]; i += w_dim
    wa = theta[i : i + w_dim * z_dim].reshape(w_dim, z_dim); i += w_dim * z_dim
    ba = theta[i : i + z_dim]; i += z_dim
    wt = theta[i : i + w_dim * z_dim].reshape(w_dim, z_dim); i += w_dim * z_dim
    bt = theta[i : i + z_dim]; i += z_dim
    wf = theta[i : i + (V_NODES + z_dim) * E_EDGES].reshape(V_NODES + z_dim, E_EDGES); i += (V_NODES + z_dim) * E_EDGES
    bf = theta[i : i + E_EDGES]

    n_steps, dt = 90, 0.5
    h_idx = jnp.asarray(np.round(np.asarray(horizons) / dt).astype(int) - 1)

    def run_cohort(is_aged_flag, init_p):
        age_col = jnp.full((init_p.shape[0], 1), is_aged_flag)
        z0 = jnp.tanh(jnp.concatenate([init_p, age_col], axis=-1) @ enc + eb)
        y0 = jnp.concatenate([init_p, z0], axis=-1)

        def rhs(y):
            p, z = y[:, :V_NODES], y[:, V_NODES:]
            h = jnp.tanh(y @ wh + bh)
            target = jnp.tanh(h @ wa + ba)
            tau_inv = 0.05 + 3.0 * jax.nn.sigmoid(h @ wt + bt)
            dz = -tau_inv * (z - target)
            rates = 1e-4 + 4.0 * jax.nn.sigmoid(y @ wf + bf)
            dp = jnp.zeros_like(p)
            for e_i, (u, v) in enumerate(EDGES):
                fl = rates[:, e_i] * p[:, u]
                dp = dp.at[:, v].add(fl)
                dp = dp.at[:, u].add(-fl)
            return jnp.concatenate([dp, dz], axis=-1)

        def step(y, _):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            pc = jnp.maximum(yn[:, :V_NODES], 1e-15)
            pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
            return jnp.concatenate([pc, yn[:, V_NODES:]], axis=-1), pc

        _, traj = jax.lax.scan(step, y0, None, length=n_steps)
        return jnp.swapaxes(traj[h_idx], 0, 1)

    return run_cohort(0.0, jnp.eye(V_NODES)[:4]), run_cohort(1.0, jnp.eye(V_NODES)[1:4])


# ---------------------------------------------------------------------------
# Real-world S4D ODE on 5-node, 9-edge Helpdesk graph
# ---------------------------------------------------------------------------
def init_helpdesk_s4d(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z_dim, m_dim = 10, 12
    # enc: 6*10+10=70; log_r: 10; omega: 5; wb: 5*10+10=60; wm: 15*12+12=192; wo: 12*9+9=117 -> 454 params
    return np.concatenate(
        [
            rng.normal(0.0, 0.25, size=6 * z_dim),
            np.zeros(z_dim),
            np.linspace(np.log(0.04), np.log(2.5), z_dim),
            rng.normal(0.0, 0.2, size=z_dim // 2),
            rng.normal(0.0, 0.2, size=V_NODES * z_dim),
            np.zeros(z_dim),
            rng.normal(0.0, 0.2, size=(V_NODES + z_dim) * m_dim),
            np.zeros(m_dim),
            rng.normal(0.0, 0.15, size=m_dim * E_EDGES),
            np.full(E_EDGES, -2.0),
        ]
    )


def predict_helpdesk_s4d(theta: jax.Array, horizons: jax.Array):
    z_dim, m_dim = 10, 12
    half = z_dim // 2
    i = 0
    enc = theta[i : i + 6 * z_dim].reshape(6, z_dim); i += 6 * z_dim
    eb = theta[i : i + z_dim]; i += z_dim
    log_r = theta[i : i + z_dim]; i += z_dim
    omega = theta[i : i + half]; i += half
    wb = theta[i : i + V_NODES * z_dim].reshape(V_NODES, z_dim); i += V_NODES * z_dim
    bb = theta[i : i + z_dim]; i += z_dim
    wm = theta[i : i + (V_NODES + z_dim) * m_dim].reshape(V_NODES + z_dim, m_dim); i += (V_NODES + z_dim) * m_dim
    bm = theta[i : i + m_dim]; i += m_dim
    wo = theta[i : i + m_dim * E_EDGES].reshape(m_dim, E_EDGES); i += m_dim * E_EDGES
    bo = theta[i : i + E_EDGES]

    decay = jnp.exp(jnp.clip(log_r, -5.0, 3.0))
    n_steps, dt = 90, 0.5
    h_idx = jnp.asarray(np.round(np.asarray(horizons) / dt).astype(int) - 1)

    def run_cohort(is_aged_flag, init_p):
        age_col = jnp.full((init_p.shape[0], 1), is_aged_flag)
        z0 = jnp.tanh(jnp.concatenate([init_p, age_col], axis=-1) @ enc + eb)
        y0 = jnp.concatenate([init_p, z0], axis=-1)

        def rhs(y):
            p, z = y[:, :V_NODES], y[:, V_NODES:]
            z1, z2 = z[:, :half], z[:, half:]
            rot = jnp.concatenate([-omega * z2, omega * z1], axis=-1)
            dz = -decay * z + rot + (p @ wb + bb)
            hid = jax.nn.gelu(y @ wm + bm)
            rates = 1e-4 + 4.0 * jax.nn.sigmoid(hid @ wo + bo)
            dp = jnp.zeros_like(p)
            for e_i, (u, v) in enumerate(EDGES):
                fl = rates[:, e_i] * p[:, u]
                dp = dp.at[:, v].add(fl)
                dp = dp.at[:, u].add(-fl)
            return jnp.concatenate([dp, dz], axis=-1)

        def step(y, _):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            pc = jnp.maximum(yn[:, :V_NODES], 1e-15)
            pc = pc / jnp.sum(pc, axis=-1, keepdims=True)
            return jnp.concatenate([pc, yn[:, V_NODES:]], axis=-1), pc

        _, traj = jax.lax.scan(step, y0, None, length=n_steps)
        return jnp.swapaxes(traj[h_idx], 0, 1)

    return run_cohort(0.0, jnp.eye(V_NODES)[:4]), run_cohort(1.0, jnp.eye(V_NODES)[1:4])


# ---------------------------------------------------------------------------
# Universal 1D Pos-Sens plugged zero-shot into cyclic 5-node IT Helpdesk graph
# ---------------------------------------------------------------------------
def make_universal_pos_helpdesk(psi_univ, stages_red: int = 8):
    # Parameters: 4 active node log-means + 4 shape logits + 9 routing logits = 17 params
    out_edges = [[] for _ in range(V_NODES)]
    for e_idx, (u, v) in enumerate(EDGES):
        out_edges[u].append(e_idx)

    def predict(theta: jax.Array, horizons: jax.Array):
        node_means = jnp.concatenate([jnp.exp(jnp.clip(theta[:4], -3.0, 4.5)), jnp.array([1.0])])
        node_shapes = jnp.concatenate([1.4 + 6.1 * jax.nn.sigmoid(theta[4:8]), jnp.array([4.0])])
        r_logits = theta[8 : 8 + E_EDGES]
        rw = jnp.zeros(E_EDGES, dtype=jnp.float64)
        for u in range(4):
            eids = jnp.asarray(out_edges[u])
            rw = rw.at[eids].set(jax.nn.softmax(r_logits[eids]))

        gen, fresh, read = build_universal_pos_network_generator(
            node_means, node_shapes, rw, EDGES, V_NODES, psi_univ, stages_red=stages_red
        )
        # Analytical survivor initialization at AGED_THRESHOLD_DAYS on each transient node block
        aged = jnp.zeros((gen.shape[0], 3), dtype=jnp.float64)
        for u in (1, 2, 3):
            s0, s1 = u * stages_red, (u + 1) * stages_red
            blk = gen[s0:s1, s0:s1]
            st_a = jl.expm(AGED_THRESHOLD_DAYS * blk) @ fresh[s0:s1, u]
            st_a = jnp.maximum(st_a, 1e-30)
            st_a = st_a / jnp.sum(st_a)
            aged = aged.at[s0:s1, u - 1].set(st_a)

        def eval_one(h):
            eh = jl.expm(h * gen)
            return (read @ (eh @ fresh[:, :4])).T, (read @ (eh @ aged)).T

        pf_all, pa_all = jax.vmap(eval_one)(horizons)
        return jnp.swapaxes(pf_all, 0, 1), jnp.swapaxes(pa_all, 0, 1)

    return predict


def load_chronological_split(csv_path: Path, train_frac: float = 0.70):
    """Split 4,576 tickets strictly chronologically by opening timestamp."""
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["Complete Timestamp"], format="%Y/%m/%d %H:%M:%S.%f")
    df = df.sort_values(["Case ID", "ts"])
    sub = df[df["Activity"].isin(CORE_STATES)].copy()
    sub = sub[
        ~((sub["Case ID"] == sub["Case ID"].shift(1)) & (sub["Activity"] == sub["Activity"].shift(1)))
    ].copy()
    sub["state"] = sub["Activity"].map(STATE_MAP).astype(int)

    items = []
    for case_id, grp in sub.groupby("Case ID", sort=False):
        sts = grp["state"].to_numpy(dtype=np.int32)
        t0 = grp["ts"].iloc[0]
        times = (grp["ts"] - t0).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        if len(sts) < 2 or np.any(np.diff(times) < 0):
            continue
        items.append((t0, (str(case_id), times, sts)))
    items.sort(key=lambda x: x[0])
    cut = int(round(train_frac * len(items)))
    tr_trajs = [x[1] for x in items[:cut]]
    te_trajs = [x[1] for x in items[cut:]]
    cutoff_date = str(items[cut][0].date())
    return tr_trajs, te_trajs, cutoff_date


def main():
    csv_path = ensure_helpdesk_csv(Path("/tmp/drn_r4_helpdesk"))
    trajs, _ = load_ticket_trajectories(csv_path)
    h_jax = jnp.asarray(HORIZONS_DAYS, dtype=jnp.float64)

    # Calibrate 1D Universal Pos-Sens clock ONCE (completely data-free!)
    psi_univ, univ_cert = calibrate_universal_pos_clock(stages_red=8, stages_full=16, maxiter=250)
    pred_univ = make_universal_pos_helpdesk(psi_univ, stages_red=8)

    # Exact same 5-fold ticket split as run_helpdesk_systems_benchmark.py (seed=2026)
    rng = np.random.default_rng(2026)
    folds = np.array_split(rng.permutation(len(trajs)), 5)

    ext_specs = [
        ("universal_pos_sens_fresh_only", lambda s: np.r_[np.log([4.0, 4.0, 10.0, 28.0]), np.zeros(4 + E_EDGES)], pred_univ, False),
        ("universal_pos_sens_joint", lambda s: np.r_[np.log([4.0, 4.0, 10.0, 28.0]), np.zeros(4 + E_EDGES)], pred_univ, True),
        ("ltc_ode_fresh_only", init_helpdesk_ltc, predict_helpdesk_ltc, False),
        ("ltc_ode_joint", init_helpdesk_ltc, predict_helpdesk_ltc, True),
        ("s4d_ode_fresh_only", init_helpdesk_s4d, predict_helpdesk_s4d, False),
        ("s4d_ode_joint", init_helpdesk_s4d, predict_helpdesk_s4d, True),
    ]

    fold_res = {m: {"fresh_kl": [], "aged_kl": []} for m, *_ in ext_specs}
    for fi in range(5):
        te_idx = set(folds[fi].tolist())
        tr_t = [trajs[i] for i in range(len(trajs)) if i not in te_idx]
        te_t = [trajs[i] for i in range(len(trajs)) if i in te_idx]
        f_tr, a_tr = extract_snapshot_tables(tr_t)
        f_te, a_te = extract_snapshot_tables(te_t)
        print(f"Helpdesk Extended Fold {fi+1}/5:", flush=True)
        for mname, init_fn, pred_fn, is_joint in ext_specs:
            # 2 starts for neural ODEs, 1 for deterministic Universal Pos-Sens
            n_st = 2 if ("ltc" in mname or "s4d" in mname) else 1
            best_w, best_loss = None, float("inf")
            for st in range(n_st):
                w_hat = fit_model(init_fn(200 + 10 * fi + st), pred_fn, f_tr, a_tr, joint=is_joint, maxiter=150)
                pf_tr, pa_tr = pred_fn(jnp.asarray(w_hat), h_jax)
                l_tr = empirical_kl(np.asarray(pf_tr), f_tr) + (empirical_kl(np.asarray(pa_tr), a_tr) if is_joint else 0.0)
                if l_tr < best_loss:
                    best_loss, best_w = l_tr, w_hat
            pf, pa = pred_fn(jnp.asarray(best_w), h_jax)
            fk = empirical_kl(np.asarray(pf), f_te)
            ak = empirical_kl(np.asarray(pa), a_te)
            fold_res[mname]["fresh_kl"].append(fk)
            fold_res[mname]["aged_kl"].append(ak)
            print(f"  {mname:30s}: fresh={fk:.5f} | aged={ak:.5f}", flush=True)

    cv_summary = {
        m: {
            "fresh_kl_mean": float(np.mean(d["fresh_kl"])),
            "fresh_kl_sd": float(np.std(d["fresh_kl"], ddof=1)),
            "aged_kl_mean": float(np.mean(d["aged_kl"])),
            "aged_kl_sd": float(np.std(d["aged_kl"], ddof=1)),
            "folds": d,
        }
        for m, d in fold_res.items()
    }

    # Chronological 70/30 future-ticket split across ALL 8 architectures!
    tr_ch, te_ch, cutoff_date = load_chronological_split(csv_path, train_frac=0.70)
    f_tr_c, a_tr_c = extract_snapshot_tables(tr_ch)
    f_te_c, a_te_c = extract_snapshot_tables(te_ch)
    print(f"\nChronological Split (cutoff {cutoff_date}: {len(tr_ch)} train -> {len(te_ch)} future test):", flush=True)

    all_chrono = [
        ("edge_memory_drn_fresh_only", init_edge_memory_drn, predict_edge_memory_drn, False),
        ("edge_memory_drn_joint", init_edge_memory_drn, predict_edge_memory_drn, True),
        ("renewal_drn_fresh_only", lambda s: np.concatenate([np.log(np.full(E_EDGES, 8.0)), np.zeros(E_EDGES * K_RENEWAL)]), predict_renewal_drn, False),
        ("universal_pos_sens_fresh_only", lambda s: np.r_[np.log([4.0, 4.0, 10.0, 28.0]), np.zeros(4 + E_EDGES)], pred_univ, False),
        ("ctmc", lambda s: np.full(E_EDGES, -2.0), predict_ctmc, False),
        ("ltc_ode_fresh_only", init_helpdesk_ltc, predict_helpdesk_ltc, False),
        ("s4d_ode_fresh_only", init_helpdesk_s4d, predict_helpdesk_s4d, False),
        ("neural_cme_fresh_only", init_helpdesk_neural_cme, predict_helpdesk_neural_cme, False),
        ("latent_ode_fresh_only", init_helpdesk_latent_ode, predict_helpdesk_latent_ode, False),
    ]
    chrono_res = {}
    for mname, init_fn, pred_fn, is_joint in all_chrono:
        w_hat = fit_model(init_fn(77), pred_fn, f_tr_c, a_tr_c, joint=is_joint, maxiter=150)
        pf, pa = pred_fn(jnp.asarray(w_hat), h_jax)
        fk = empirical_kl(np.asarray(pf), f_te_c)
        ak = empirical_kl(np.asarray(pa), a_te_c)
        chrono_res[mname] = {"fresh_kl": fk, "aged_kl": ak}
        print(f"  Chrono {mname:30s}: fresh={fk:.5f} | aged={ak:.5f}", flush=True)

    out = {
        "universal_pos_sens_states": 4 * 8 + 1,
        "universal_pos_sens_params": 4 + 4 + E_EDGES,
        "ltc_ode_params": int(init_helpdesk_ltc(0).size),
        "s4d_ode_params": int(init_helpdesk_s4d(0).size),
        "universal_clock_certificate": univ_cert,
        "cv_5fold_extended": cv_summary,
        "chronological_split": {
            "cutoff_date": cutoff_date,
            "train_tickets": len(tr_ch),
            "test_tickets": len(te_ch),
            "models": chrono_res,
        },
    }
    out_path = ROOT / "results" / "archived" / "helpdesk_extended_baselines.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
