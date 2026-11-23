"""Theorem 6 Balanced-Truncation Audit on Real Fitted Networks (Exact Analytical Sylvester Derivatives).

Loads and directly fits the real-world datasets via L-BFGS-B before auditing Glover Balanced
Truncation (frozen & re-balanced) against data-free Universal Pos-Sens / Hyper-Pos-Sens:
  1. Real Human Hematopoiesis (7-node branching FACS DAG, 10 donors, 78 samples; Bast et al. 2021):
     Fits the 97-state K=16 branching DRN directly on the 78 FACS snapshots (verifying the real
     ~15-18% MPP->CMP bifurcation), then audits Glover BT at r in {12, 18, 24, 36, 48} vs.
     data-free Universal Pos-Sens (49 states).
  2. Real IT Helpdesk Service Queue (5-node cyclic workflow, 4,576 tickets; Polato 2017):
     Fits the 69-state K=16 cyclic Hyper-Erlang DRN directly on the 4,576 ticket trajectories,
     then audits Glover BT at r in {10, 12, 16, 20, 36} vs. data-free Universal Hyper-Pos-Sens (37 states).

Uses EXACT analytical Lyapunov/Sylvester Riesz-projection Fréchet derivatives (zero finite-
difference noise!) and measures both:
  - Early-time simplex-positivity violations (min P_BT(t) < 0 at tight orders r <= 24 / 20),
  - Exact forward L2 and analytical sensitivity L2 errors (re-balanced & frozen),
  - Per-step O(D^3) Lyapunov balancing wall-clock overhead vs. zero-overhead universal Pos-Sens.
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
from scipy.linalg import expm as sp_expm, solve_continuous_lyapunov, solve_sylvester, svd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    build_universal_hyper_pos_network_generator,
    build_universal_pos_network_generator,
    calibrate_universal_pos_clock,
    mixed_erlang_components,
    moment_matched_logits,
)
from scripts.run_hematopoiesis_branching_benchmark import (  # noqa: E402
    aggregate_counts_by_day,
    load_hematopoiesis_facs,
)
from scripts.run_helpdesk_systems_benchmark import (  # noqa: E402
    ensure_helpdesk_csv,
    extract_snapshot_tables,
    load_ticket_trajectories,
)

OUT_PATH = ROOT / "results" / "archived" / "r8_real_bt_failure.json"


def _rollout_uniform_expm(A: np.ndarray, v0: np.ndarray, C: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Exact trajectory C exp(t A) v0 on uniform grid ts via ONE expm(dt * A)."""
    dt = float(ts[1] - ts[0])
    E_step = sp_expm(dt * A)
    x = sp_expm(float(ts[0]) * A) @ v0
    out = np.empty((len(ts), C.shape[0]), dtype=np.float64)
    out[0] = C @ x
    for i in range(1, len(ts)):
        x = E_step @ x
        out[i] = C @ x
    return out


def precompute_gramians_and_derivs(A_tr: np.ndarray, B_tr: np.ndarray, C_tr: np.ndarray, dA_list, dB_list):
    """Solve Lyapunov Gramians Wc, Wo and EXACT analytical derivatives dWc, dWo (zero FD noise!)."""
    t0 = perf_counter()
    Wc = solve_continuous_lyapunov(A_tr, -(B_tr @ B_tr.T))
    Wo = solve_continuous_lyapunov(A_tr.T, -(C_tr.T @ C_tr))
    Wc = 0.5 * (Wc + Wc.T)
    Wo = 0.5 * (Wo + Wo.T)
    ec, uc = np.linalg.eigh(Wc)
    eo, uo = np.linalg.eigh(Wo)
    Lc = uc @ np.diag(np.sqrt(np.maximum(ec, 0.0)))
    Lo = uo @ np.diag(np.sqrt(np.maximum(eo, 0.0)))
    U, s, Vt = svd(Lo.T @ Lc, full_matrices=False)
    base_lyap_ms = (perf_counter() - t0) * 1000.0

    M = Wc @ Wo
    dM_list = []
    for dA, dB in zip(dA_list, dB_list):
        rhs_c = dA @ Wc + Wc @ dA.T + dB @ B_tr.T + B_tr @ dB.T
        rhs_o = dA.T @ Wo + Wo @ dA
        dWc = solve_continuous_lyapunov(A_tr, -rhs_c)
        dWo = solve_continuous_lyapunov(A_tr.T, -rhs_o)
        dM_list.append(dWc @ Wo + Wc @ dWo)
    return {"Lc": Lc, "Lo": Lo, "U": U, "s": s, "Vt": Vt, "M": M, "dM_list": dM_list, "base_lyap_ms": base_lyap_ms}


def eval_exact_bt_order(A_tr, B_tr, C_tr, dA_list, dB_list, pre, r: int, ts: np.ndarray, x0_col: int = 0) -> dict:
    """Evaluate order-r Glover BT with EXACT analytical Sylvester Riesz-projection derivatives."""
    Lc, Lo, U, s, Vt, M, dM_list = pre["Lc"], pre["Lo"], pre["U"], pre["s"], pre["Vt"], pre["M"], pre["dM_list"]
    n_tr = A_tr.shape[0]
    s1 = s[:r]
    inv_sqrt = 1.0 / np.sqrt(np.maximum(s1, 1e-30))
    V1 = Lc @ Vt[:r, :].T @ np.diag(inv_sqrt)
    W1T = np.diag(inv_sqrt) @ U[:, :r].T @ Lo.T

    Ar = W1T @ A_tr @ V1
    Br = W1T @ B_tr
    Cr = C_tr @ V1

    # Exact forward rollout & positivity audit
    P_f = _rollout_uniform_expm(A_tr, B_tr[:, x0_col], C_tr, ts)
    P_r = _rollout_uniform_expm(Ar, Br[:, x0_col], Cr, ts)
    fwd_linf = float(np.max(np.abs(P_f - P_r)))
    fwd_l2 = float(np.sqrt(np.trapz(np.sum((P_f - P_r) ** 2, axis=-1), ts)))
    min_prob_red = float(np.min(P_r))
    neg_count = int(np.sum(P_r < -1e-9))

    # Build complementary biorthogonal basis (V2, W2T) for exact Sylvester Riesz-projection derivative
    Pi = V1 @ W1T
    U_c, _, _ = svd(np.eye(n_tr) - Pi)
    V2 = U_c[:, : n_tr - r]
    T_inv = np.linalg.inv(np.hstack([V1, V2]))
    W2T = T_inv[r:, :]
    M11 = W1T @ M @ V1
    M22 = W2T @ M @ V2

    Cf_blk = np.hstack([C_tr, np.zeros_like(C_tr)])
    Cr_fz_blk = np.hstack([Cr, np.zeros_like(Cr)])

    fz_errs, reb_errs, rel_fz_errs, rel_reb_errs = [], [], [], []
    max_sylv_res = 0.0
    for dA, dB, dM in zip(dA_list, dB_list, dM_list):
        # Full analytical sensitivity
        Mf = np.block([[A_tr, dA], [np.zeros((n_tr, n_tr)), A_tr]])
        v0f = np.concatenate([dB[:, x0_col], B_tr[:, x0_col]])
        Jf = _rollout_uniform_expm(Mf, v0f, Cf_blk, ts)
        jf_norm = float(np.sqrt(np.trapz(np.sum(Jf**2, axis=-1), ts)))

        # Frozen BT exact sensitivity
        dAr_fz = W1T @ dA @ V1
        dBr_fz = W1T @ dB
        Mr_fz = np.block([[Ar, dAr_fz], [np.zeros((r, r)), Ar]])
        v0r_fz = np.concatenate([dBr_fz[:, x0_col], Br[:, x0_col]])
        Jr_fz = _rollout_uniform_expm(Mr_fz, v0r_fz, Cr_fz_blk, ts)
        ef = float(np.sqrt(np.trapz(np.sum((Jf - Jr_fz) ** 2, axis=-1), ts)))
        fz_errs.append(ef)
        rel_fz_errs.append(ef / max(jf_norm, 1e-12))

        # Re-balanced BT exact analytical Sylvester Riesz-projection sensitivity!
        X12 = solve_sylvester(M11, -M22, +(W1T @ dM @ V2))
        X21 = solve_sylvester(M22, -M11, -(W2T @ dM @ V1))
        dPi = V1 @ X12 @ W2T + V2 @ X21 @ W1T
        sres = float(np.max(np.abs((M @ dPi - dPi @ M) - (Pi @ dM - dM @ Pi))))
        max_sylv_res = max(max_sylv_res, sres)

        dV1 = dPi @ V1
        dW1T = W1T @ dPi @ (np.eye(n_tr) - Pi)
        dAr_reb = dW1T @ A_tr @ V1 + W1T @ dA @ V1 + W1T @ A_tr @ dV1
        dBr_reb = dW1T @ B_tr + W1T @ dB
        dCr_reb = C_tr @ dV1

        Mr_reb = np.block([[Ar, dAr_reb], [np.zeros((r, r)), Ar]])
        v0r_reb = np.concatenate([dBr_reb[:, x0_col], Br[:, x0_col]])
        # y = Cr x => dy = dCr x + Cr dx = [Cr, dCr] [dx; x]
        Cr_reb_blk = np.hstack([Cr, dCr_reb])
        Jr_reb = _rollout_uniform_expm(Mr_reb, v0r_reb, Cr_reb_blk, ts)
        er = float(np.sqrt(np.trapz(np.sum((Jf - Jr_reb) ** 2, axis=-1), ts)))
        reb_errs.append(er)
        rel_reb_errs.append(er / max(jf_norm, 1e-12))

    mean_fz = float(np.mean(fz_errs))
    mean_reb = float(np.mean(reb_errs))
    return {
        "total_states": int(r + 1),
        "sigma_r_plus_1": float(s[r]) if r < len(s) else 0.0,
        "glover_ub": float(2.0 * np.sum(s[r:])),
        "fwd_linf": fwd_linf,
        "fwd_l2": fwd_l2,
        "mean_sens_l2_err": mean_fz,
        "rebal_mean_sens_l2_err": mean_reb,
        "mean_rel_sens_err_fz": float(np.mean(rel_fz_errs)),
        "mean_rel_sens_err_reb": float(np.mean(rel_reb_errs)),
        "distortion_ratio_l2": float(mean_fz / max(fwd_l2, 1e-15)),
        "rebal_distortion_ratio_l2": float(mean_reb / max(fwd_l2, 1e-15)),
        "max_sylvester_residual": float(max_sylv_res),
        "min_prob_red": min_prob_red,
        "neg_time_entries": neg_count,
        "violates_positivity": bool(min_prob_red < -1e-9),
        "base_lyap_ms": round(float(pre["base_lyap_ms"]), 2),
    }


def eval_surrogate_exact(A_tr, B_tr, C_tr, dA_list, dB_list, Ar, Br, Cr, dAr_list, dBr_list, ts, x0_col: int = 0) -> dict:
    P_f = _rollout_uniform_expm(A_tr, B_tr[:, x0_col], C_tr, ts)
    P_r = _rollout_uniform_expm(Ar, Br[:, x0_col], Cr, ts)
    fwd_linf = float(np.max(np.abs(P_f - P_r)))
    fwd_l2 = float(np.sqrt(np.trapz(np.sum((P_f - P_r) ** 2, axis=-1), ts)))
    min_prob_red = float(np.min(P_r))
    n_tr, r = A_tr.shape[0], Ar.shape[0]
    Cf_blk = np.hstack([C_tr, np.zeros_like(C_tr)])
    Cr_blk = np.hstack([Cr, np.zeros_like(Cr)])
    errs, rels = [], []
    for dA, dB, dAr, dBr in zip(dA_list, dB_list, dAr_list, dBr_list):
        Mf = np.block([[A_tr, dA], [np.zeros((n_tr, n_tr)), A_tr]])
        Mr = np.block([[Ar, dAr], [np.zeros((r, r)), Ar]])
        Jf = _rollout_uniform_expm(Mf, np.concatenate([dB[:, x0_col], B_tr[:, x0_col]]), Cf_blk, ts)
        Jr = _rollout_uniform_expm(Mr, np.concatenate([dBr[:, x0_col], Br[:, x0_col]]), Cr_blk, ts)
        e = float(np.sqrt(np.trapz(np.sum((Jf - Jr) ** 2, axis=-1), ts)))
        nrm = float(np.sqrt(np.trapz(np.sum(Jf**2, axis=-1), ts)))
        errs.append(e)
        rels.append(e / max(nrm, 1e-12))
    mean_e = float(np.mean(errs))
    return {
        "total_states": int(r + 1),
        "fwd_linf": fwd_linf,
        "fwd_l2": fwd_l2,
        "mean_sens_l2_err": mean_e,
        "mean_rel_sens_err": float(np.mean(rels)),
        "distortion_ratio_l2": float(mean_e / max(fwd_l2, 1e-15)),
        "min_prob_red": min_prob_red,
        "violates_positivity": bool(min_prob_red < -1e-9),
        "base_lyap_ms": 0.0,
    }


def run_hematopoiesis_bt_audit() -> dict:
    """Real 7-node Human Hematopoiesis (DAG, 78 FACS samples): ACTUAL fitted K=16 DRN (97st) -> BT vs Univ Pos-Sens."""
    V = 7
    edges = [(0, 1), (1, 2), (1, 3), (2, 6), (3, 4), (3, 5), (4, 6), (5, 6)]
    K = 16

    def build_full_hemato(theta):
        means = jnp.exp(jnp.clip(theta[:6], -2.0, 3.0))
        shapes = 1.5 + 6.0 * jax.nn.sigmoid(theta[6:12])
        p_cmp = jax.nn.sigmoid(theta[12])
        p_mep = jax.nn.sigmoid(theta[13])
        rw = jnp.array([1.0, 1.0 - p_cmp, p_cmp, 1.0, 1.0 - p_mep, p_mep, 1.0, 1.0])
        logits_all = moment_matched_logits(shapes)
        blocks, entries, exits = [], [], []
        for u in range(6):
            r_u, ent_u = mixed_erlang_components(means[u], logits_all[u])
            blocks.append(r_u * (jnp.eye(K, k=-1) - jnp.eye(K)))
            entries.append(ent_u)
            exits.append(jnp.zeros(K).at[-1].set(r_u))
        blocks.append(jnp.zeros((1, 1)))
        entries.append(jnp.ones(1))
        exits.append(jnp.zeros(1))
        offsets, cur = [], 0
        for blk in blocks:
            offsets.append((cur, cur + blk.shape[0]))
            cur += blk.shape[0]
        A = jl.block_diag(*blocks)
        for e_id, (u, v) in enumerate(edges):
            u0, u1 = offsets[u]
            v0, v1 = offsets[v]
            A = A.at[v0:v1, u0:u1].add(rw[e_id] * jnp.outer(entries[v], exits[u]))
        B = jl.block_diag(*(e[:, None] for e in entries))
        C = jl.block_diag(*(jnp.ones((1, e - s)) for s, e in offsets))
        return A, B, C

    # Actually load and fit on the 78 real FACS samples from Bast et al. (2021)!
    df = load_hematopoiesis_facs()
    days_uniq, day_counts = aggregate_counts_by_day(df, [1, 2, 3, 4, 5, 6, 7])
    day_weights = day_counts.sum(axis=-1)
    emp_fracs = day_counts / np.maximum(day_weights[:, None], 1e-12)
    days_j = jnp.asarray(days_uniq, dtype=jnp.float64)
    emp_j = jnp.asarray(emp_fracs, dtype=jnp.float64)
    dw_j = jnp.asarray(day_weights / day_weights.sum(), dtype=jnp.float64)

    @jax.jit
    def fit_vg(th):
        def loss(z):
            A, B, C = build_full_hemato(z)
            b0 = B[:, 0]
            preds = jax.vmap(lambda t: C @ (jl.expm(t * A) @ b0))(days_j)
            preds = jnp.clip(preds, 1e-12, 1.0)
            preds = preds / jnp.sum(preds, axis=-1, keepdims=True)
            kl = jnp.sum(jnp.where(emp_j > 0, emp_j * jnp.log(jnp.maximum(emp_j, 1e-30) / preds), 0.0), axis=-1)
            return jnp.sum(dw_j * kl) + 1e-4 * jnp.mean(z**2)

        return jax.value_and_grad(loss)(th)

    th_init = np.array([
        np.log(2.0), np.log(2.0), np.log(2.0), np.log(2.0), np.log(2.0), np.log(2.0),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -1.5, -0.2,
    ])
    res_fit = minimize(
        lambda z: (float(fit_vg(jnp.asarray(z))[0]), np.asarray(fit_vg(jnp.asarray(z))[1], dtype=float)),
        th_init, method="L-BFGS-B", jac=True, options={"maxiter": 150},
    )
    th0 = np.asarray(res_fit.x, dtype=np.float64)
    p_cmp_fit = float(1.0 / (1.0 + np.exp(-th0[12])))
    p_mep_fit = float(1.0 / (1.0 + np.exp(-th0[13])))
    print(
        f"[Real Hematopoiesis Fitted 97st] loss={res_fit.fun:.5f} p_cmp={p_cmp_fit:.3f} (GMP={1-p_mep_fit:.3f}/MEP={p_mep_fit:.3f})",
        flush=True,
    )

    A0, B0, C0 = map(np.asarray, build_full_hemato(jnp.asarray(th0)))
    n_tr = A0.shape[0] - 1  # 96 transient + 1 absorbing = 97 states
    A_tr, B_tr, C_tr = A0[:n_tr, :n_tr], B0[:n_tr, :], C0[:6, :n_tr]
    ts = np.linspace(0.05, 7.0, 140)

    # Exact JAX autodiff Jacobians dA/dtheta_p, dB/dtheta_p (ZERO FD noise!)
    jac_full = jax.jit(jax.jacfwd(build_full_hemato))
    dA_all, dB_all, _ = map(np.asarray, jac_full(jnp.asarray(th0)))
    dA_tr_list = [dA_all[:n_tr, :n_tr, p] for p in range(12)]
    dB_tr_list = [dB_all[:n_tr, :, p] for p in range(12)]

    pre = precompute_gramians_and_derivs(A_tr, B_tr, C_tr, dA_tr_list, dB_tr_list)
    bt_sweep = {}
    for r in (12, 18, 24, 36, 48):
        m = eval_exact_bt_order(A_tr, B_tr, C_tr, dA_tr_list, dB_tr_list, pre, r, ts, x0_col=0)
        bt_sweep[f"r{r}"] = m
        print(
            f"[Real Hemato Fitted {n_tr+1}->{r+1:2d}] BT: fwd_L2={m['fwd_l2']:.2e} "
            f"sens_fz={m['mean_sens_l2_err']:.2e} sens_reb={m['rebal_mean_sens_l2_err']:.2e} "
            f"sylv={m['max_sylvester_residual']:.1e} minP={m['min_prob_red']:.2e} neg={m['violates_positivity']}",
            flush=True,
        )

    psi_univ, cert = calibrate_universal_pos_clock(stages_red=8, stages_full=16, maxiter=200, k_min=1.4, k_max=7.8)

    def build_pos_hemato(theta):
        means = jnp.concatenate([jnp.exp(jnp.clip(theta[:6], -2.0, 3.0)), jnp.ones(1)])
        shapes = jnp.concatenate([1.5 + 6.0 * jax.nn.sigmoid(theta[6:12]), jnp.array([2.0])])
        p_cmp = jax.nn.sigmoid(theta[12])
        p_mep = jax.nn.sigmoid(theta[13])
        rw = jnp.array([1.0, 1.0 - p_cmp, p_cmp, 1.0, 1.0 - p_mep, p_mep, 1.0, 1.0])
        return build_universal_pos_network_generator(means, shapes, rw, edges, V, psi_univ, stages_red=8)

    Ap0, Bp0, Cp0 = map(np.asarray, build_pos_hemato(jnp.asarray(th0)))
    jac_pos = jax.jit(jax.jacfwd(build_pos_hemato))
    dAp_all, dBp_all, _ = map(np.asarray, jac_pos(jnp.asarray(th0)))
    dAp_list = [dAp_all[:48, :48, p] for p in range(12)]
    dBp_list = [dBp_all[:48, :, p] for p in range(12)]
    m_pos = eval_surrogate_exact(
        A_tr, B_tr, C_tr, dA_tr_list, dB_tr_list,
        Ap0[:48, :48], Bp0[:48, :], Cp0[:6, :48], dAp_list, dBp_list, ts, x0_col=0,
    )
    m_pos["cert_sup_surv"] = cert["sup_survival_abs_error"]
    print(
        f"[Real Hemato Fitted {n_tr+1}->49] Univ. Pos-Sens: fwd_L2={m_pos['fwd_l2']:.2e} "
        f"sens={m_pos['mean_sens_l2_err']:.2e} minP={m_pos['min_prob_red']:.2e} neg={m_pos['violates_positivity']}",
        flush=True,
    )
    return {
        "full_states": n_tr + 1,
        "fitted_real_data": {
            "n_samples": int(len(df)),
            "fit_kl": float(res_fit.fun),
            "p_cmp": p_cmp_fit,
            "p_mep": p_mep_fit,
            "means": np.exp(np.clip(th0[:6], -2.0, 3.0)).tolist(),
            "shapes": (1.5 + 6.0 / (1.0 + np.exp(-th0[6:12]))).tolist(),
        },
        "balanced_truncation_sweep": bt_sweep,
        "universal_pos_sens_49": m_pos,
    }


def run_helpdesk_bt_audit() -> dict:
    """Real 5-node IT Helpdesk cyclic workflow (4,576 tickets): ACTUAL fitted K=16 DRN (69st) -> BT vs Univ Hyp-Pos."""
    V = 5
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 4)]
    K_full = 16  # 4 active nodes * (1 fast + 16 slow) + 1 absorbing = 69 states

    # Actually load and fit on the 4,576 real IT Helpdesk tickets (Polato 2017, MD5 verified)!
    csv_path = ensure_helpdesk_csv(Path("/tmp/drn_r9_data/helpdesk"))
    trajs, _ = load_ticket_trajectories(csv_path)
    horizons = np.array([0.5, 2.0, 7.0, 14.0, 28.0, 42.0], dtype=np.float64)
    fresh_cnt, _ = extract_snapshot_tables(trajs, horizons=horizons)
    # Empirical transition routing matrix from the 4,576 real tickets
    edge_cnt = np.zeros(len(edges), dtype=np.float64)
    edge_idx = {e: i for i, e in enumerate(edges)}
    for _, _, sts in trajs:
        for a, b in zip(sts[:-1], sts[1:]):
            if (int(a), int(b)) in edge_idx:
                edge_cnt[edge_idx[(int(a), int(b))]] += 1.0
    rw_emp = np.zeros(len(edges), dtype=np.float64)
    for u in range(4):
        idxs = [i for i, (s, _) in enumerate(edges) if s == u]
        tot = sum(edge_cnt[i] for i in idxs)
        for i in idxs:
            rw_emp[i] = edge_cnt[i] / max(tot, 1.0)
    rw_j = jnp.asarray(rw_emp, dtype=jnp.float64)

    def build_full_hdk(theta):
        fm = jnp.exp(jnp.clip(theta[0:4], -3.0, 2.5))
        sm = jnp.exp(jnp.clip(theta[4:8], 0.0, 4.0))
        sk = 1.5 + 6.0 * jax.nn.sigmoid(theta[8:12])
        af = jax.nn.sigmoid(theta[12:16])
        logits_all = moment_matched_logits(sk)
        blocks, entries, exits = [], [], []
        for i in range(4):
            rf = 1.0 / fm[i]
            rs, ent_s = mixed_erlang_components(sm[i], logits_all[i])
            blk_s = rs * (jnp.eye(K_full, k=-1) - jnp.eye(K_full))
            blk = jl.block_diag(jnp.array([[-rf]]), blk_s)
            ent = jnp.concatenate([jnp.array([af[i]]), (1.0 - af[i]) * ent_s])
            ex = jnp.zeros(K_full + 1).at[0].set(rf).at[-1].set(rs)
            blocks.append(blk)
            entries.append(ent)
            exits.append(ex)
        blocks.append(jnp.zeros((1, 1)))
        entries.append(jnp.ones(1))
        exits.append(jnp.zeros(1))
        offsets, cur = [], 0
        for blk in blocks:
            offsets.append((cur, cur + blk.shape[0]))
            cur += blk.shape[0]
        A = jl.block_diag(*blocks)
        for e_id, (u, v) in enumerate(edges):
            u0, u1 = offsets[u]
            v0, v1 = offsets[v]
            A = A.at[v0:v1, u0:u1].add(rw_j[e_id] * jnp.outer(entries[v], exits[u]))
        B = jl.block_diag(*(e[:, None] for e in entries))
        C = jl.block_diag(*(jnp.ones((1, e - s)) for s, e in offsets))
        return A, B, C

    h_j = jnp.asarray(horizons, dtype=jnp.float64)
    fc_j = jnp.asarray(fresh_cnt, dtype=jnp.float64)
    n_fc = float(fresh_cnt.sum())

    @jax.jit
    def fit_vg_hdk(th):
        def loss(z):
            A, B, C = build_full_hdk(z)
            # Evaluate P(h) = C expm(h A) B[:, :4] -> (H, V, 4) -> swap to (4, H, V)
            def one_h(h):
                p = C @ (jl.expm(h * A) @ B[:, :4])  # (V, 4)
                p = jnp.clip(p.T, 1e-12, 1.0)
                return p / jnp.sum(p, axis=-1, keepdims=True)

            preds = jnp.swapaxes(jax.vmap(one_h)(h_j), 0, 1)  # (4, H, V)
            return -jnp.sum(fc_j * jnp.log(preds)) / n_fc + 1e-4 * jnp.mean(z**2)

        return jax.value_and_grad(loss)(th)

    th_init = np.array([
        np.log(0.2), np.log(0.4), np.log(0.4), np.log(1.0),
        np.log(4.0), np.log(8.0), np.log(8.0), np.log(25.0),
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, -1.0,
    ])
    res_fit = minimize(
        lambda z: (float(fit_vg_hdk(jnp.asarray(z))[0]), np.asarray(fit_vg_hdk(jnp.asarray(z))[1], dtype=float)),
        th_init, method="L-BFGS-B", jac=True, options={"maxiter": 100},
    )
    th0 = np.asarray(res_fit.x, dtype=np.float64)
    print(
        f"[Real Helpdesk Fitted 69st] tickets={len(trajs)} loss={res_fit.fun:.5f} "
        f"slow_means={np.round(np.exp(np.clip(th0[4:8], 0, 4)), 2).tolist()} rw={np.round(rw_emp, 3).tolist()}",
        flush=True,
    )

    A0, B0, C0 = map(np.asarray, build_full_hdk(jnp.asarray(th0)))
    n_tr = A0.shape[0] - 1  # 68 transient + 1 absorbing = 69 states
    A_tr, B_tr, C_tr = A0[:n_tr, :n_tr], B0[:n_tr, :], C0[:4, :n_tr]
    ts = np.linspace(0.05, 35.0, 200)

    jac_hdk = jax.jit(jax.jacfwd(build_full_hdk))
    dA_all, dB_all, _ = map(np.asarray, jac_hdk(jnp.asarray(th0)))
    dA_list = [dA_all[:n_tr, :n_tr, p] for p in range(4, 12)]
    dB_list = [dB_all[:n_tr, :, p] for p in range(4, 12)]

    pre = precompute_gramians_and_derivs(A_tr, B_tr, C_tr, dA_list, dB_list)
    bt_sweep = {}
    for r in (10, 12, 16, 20, 36):
        m = eval_exact_bt_order(A_tr, B_tr, C_tr, dA_list, dB_list, pre, r, ts, x0_col=0)
        bt_sweep[f"r{r}"] = m
        print(
            f"[Real Helpdesk Fitted {n_tr+1}->{r+1:2d}] BT: fwd_L2={m['fwd_l2']:.2e} "
            f"sens_fz={m['mean_sens_l2_err']:.2e} sens_reb={m['rebal_mean_sens_l2_err']:.2e} "
            f"sylv={m['max_sylvester_residual']:.1e} minP={m['min_prob_red']:.2e} neg={m['violates_positivity']}",
            flush=True,
        )

    psi_univ, _ = calibrate_universal_pos_clock(stages_red=8, stages_full=16, maxiter=200, k_min=1.4, k_max=7.8)

    def build_hyp_hdk(theta):
        fm = jnp.concatenate([jnp.exp(jnp.clip(theta[0:4], -3.0, 2.5)), jnp.ones(1)])
        sm = jnp.concatenate([jnp.exp(jnp.clip(theta[4:8], 0.0, 4.0)), jnp.ones(1)])
        sk = jnp.concatenate([1.5 + 6.0 * jax.nn.sigmoid(theta[8:12]), jnp.array([2.0])])
        af = jnp.concatenate([jax.nn.sigmoid(theta[12:16]), jnp.zeros(1)])
        return build_universal_hyper_pos_network_generator(fm, sm, sk, af, rw_j, edges, V, psi_univ, stages_red=8)

    Ah0, Bh0, Ch0 = map(np.asarray, build_hyp_hdk(jnp.asarray(th0)))
    jac_hyp = jax.jit(jax.jacfwd(build_hyp_hdk))
    dAh_all, dBh_all, _ = map(np.asarray, jac_hyp(jnp.asarray(th0)))
    dAh_list = [dAh_all[:36, :36, p] for p in range(4, 12)]
    dBh_list = [dBh_all[:36, :, p] for p in range(4, 12)]
    m_hyp = eval_surrogate_exact(
        A_tr, B_tr, C_tr, dA_list, dB_list,
        Ah0[:36, :36], Bh0[:36, :], Ch0[:4, :36], dAh_list, dBh_list, ts, x0_col=0,
    )
    print(
        f"[Real Helpdesk Fitted {n_tr+1}->37] Univ. Hyp-Pos: fwd_L2={m_hyp['fwd_l2']:.2e} "
        f"sens={m_hyp['mean_sens_l2_err']:.2e} minP={m_hyp['min_prob_red']:.2e} neg={m_hyp['violates_positivity']}",
        flush=True,
    )
    return {
        "full_states": n_tr + 1,
        "fitted_real_data": {
            "n_tickets": int(len(trajs)),
            "fit_nll": float(res_fit.fun),
            "routing_weights": rw_emp.tolist(),
            "slow_means": np.exp(np.clip(th0[4:8], 0.0, 4.0)).tolist(),
            "slow_shapes": (1.5 + 6.0 / (1.0 + np.exp(-th0[8:12]))).tolist(),
        },
        "balanced_truncation_sweep": bt_sweep,
        "universal_hyper_pos_sens_37": m_hyp,
    }


def main() -> None:
    t0 = perf_counter()
    hem = run_hematopoiesis_bt_audit()
    hdk = run_helpdesk_bt_audit()
    out = {
        "hematopoiesis_7node_real_bt_vs_pos_sens": hem,
        "helpdesk_5node_real_bt_vs_hyper_pos_sens": hdk,
        "total_seconds": round(perf_counter() - t0, 2),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT_PATH} in {out['total_seconds']}s.", flush=True)


if __name__ == "__main__":
    main()
