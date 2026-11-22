"""Unit tests for stiffness- and stage-count-independent Laplace-Schur solvers."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from diff_renewal import (
    build_universal_hyper_pos_network_generator,
    calibrate_universal_pos_clock,
    hyper_erlang_aged_phases_and_log_surv,
    laplace_schur_implicit_solve,
    solve_coxian_hyper_talbot,
    solve_frailty_renewal_talbot,
    solve_renewal_talbot,
    universal_pos_clock,
)


def _dense_hyper_generator(lam_f, lam_s, alpha_f, b_slow, R_mat):
    V, K = b_slow.shape
    D = V * (K + 1)
    A = np.zeros((D, D), dtype=np.float64)
    for u in range(V):
        o = u * (K + 1)
        lf, ls = float(lam_f[u]), float(lam_s[u])
        A[o, o] = -lf
        for j in range(K):
            A[o + 1 + j, o + 1 + j] = -ls
            if j < K - 1:
                A[o + 2 + j, o + 1 + j] = ls
        for v in range(V):
            rvu = float(R_mat[v, u])
            if rvu > 0:
                ov = v * (K + 1)
                av = float(alpha_f[v])
                bsv = np.asarray(b_slow[v])
                A[ov, o] += rvu * lf * av
                A[ov + 1 : ov + 1 + K, o] += rvu * lf * (1.0 - av) * bsv
                A[ov, o + K] += rvu * ls * av
                A[ov + 1 : ov + 1 + K, o + K] += rvu * ls * (1.0 - av) * bsv
    return A


def test_laplace_schur_implicit_and_talbot_exact_stiff():
    rng = np.random.default_rng(7)
    V, K = 8, 5
    lam_f = np.array([150.0, 80.0, 20.0, 120.0, 40.0, 10.0, 60.0, 0.0])  # stiffness > 5,000!
    lam_s = np.array([0.05, 0.2, 0.03, 0.15, 0.08, 0.4, 0.1, 0.0])
    alpha_f = np.array([0.7, 0.3, 0.5, 0.6, 0.4, 0.8, 0.2, 0.0])
    b_slow = rng.dirichlet(np.ones(K), size=V)
    R_mat = rng.uniform(0.0, 1.0, size=(V, V))
    np.fill_diagonal(R_mat, 0.0)
    R_mat[:, -1] = 0.0
    for u in range(V - 1):
        R_mat[:, u] /= R_mat[:, u].sum()
    active = np.array([1.0] * (V - 1) + [0.0])

    A_dense = _dense_hyper_generator(lam_f, lam_s, alpha_f, b_slow, R_mat)
    assert np.max(np.abs(A_dense.sum(axis=0))) < 1e-13

    # 1. Exact block-bidiagonal Schur implicit solve to <1e-13
    Y = rng.uniform(0.0, 1.0, size=(V - 1, V, K + 1))
    gh = 0.37
    X_schur = np.asarray(
        laplace_schur_implicit_solve(
            jnp.asarray(Y),
            gh,
            jnp.asarray(lam_f),
            jnp.asarray(lam_s),
            jnp.asarray(alpha_f),
            jnp.asarray(b_slow),
            jnp.asarray(R_mat),
        )
    )
    D = V * (K + 1)
    X_ref = np.linalg.solve(np.eye(D) - gh * A_dense, Y.reshape(V - 1, D).T).T.reshape(V - 1, V, K + 1)
    assert np.max(np.abs(X_schur - X_ref)) < 1e-13

    # 2. Exact fresh AND aged Laplace-Schur-Talbot vs dense expm at stiffness 5,000!
    horizons = np.array([0.25, 1.0, 4.0, 16.0])
    age = 5.0
    p_fr, p_ag = solve_renewal_talbot(
        jnp.asarray(lam_f),
        jnp.asarray(lam_s),
        jnp.asarray(alpha_f),
        jnp.asarray(b_slow),
        jnp.asarray(R_mat),
        horizons,
        jnp.asarray(active),
        eval_nodes=jnp.arange(V - 1, dtype=jnp.int32),
        age=age,
        n_talbot=20,
        return_both_fresh_and_aged=True,
    )
    p_fr = np.asarray(p_fr)
    p_ag = np.asarray(p_ag)

    pf_a, ps_a, _ = hyper_erlang_aged_phases_and_log_surv(
        jnp.asarray(lam_f), jnp.asarray(lam_s), jnp.asarray(alpha_f), jnp.asarray(b_slow), age, jnp.asarray(active)
    )
    X0_fr = np.zeros((V - 1, V, K + 1))
    X0_ag = np.zeros((V - 1, V, K + 1))
    for u in range(V - 1):
        X0_fr[u, u, 0] = alpha_f[u]
        X0_fr[u, u, 1:] = (1.0 - alpha_f[u]) * b_slow[u]
        X0_ag[u, u, 0] = float(pf_a[u])
        X0_ag[u, u, 1:] = np.asarray(ps_a[u])
    for hi, h in enumerate(horizons):
        E_h = np.asarray(jl.expm(float(h) * A_dense))
        ref_fr = (E_h @ X0_fr.reshape(V - 1, D).T).T.reshape(V - 1, V, K + 1).sum(axis=-1)
        ref_ag = (E_h @ X0_ag.reshape(V - 1, D).T).T.reshape(V - 1, V, K + 1).sum(axis=-1)
        assert np.max(np.abs(p_fr[:, hi, :] - ref_fr)) < 1e-8
        assert np.max(np.abs(p_ag[:, hi, :] - ref_ag)) < 1e-8


def test_frailty_and_coxian_talbot_exact():
    rng = np.random.default_rng(11)
    V, K = 5, 4
    lam_f = np.array([15.0, 8.0, 25.0, 12.0, 0.0])
    lam_s = np.array([0.4, 0.25, 0.5, 0.3, 0.0])
    alpha_f = np.array([0.4, 0.5, 0.6, 0.3, 0.0])
    b_slow = rng.dirichlet(np.ones(K), size=V)
    R_mat = np.array(
        [
            [0.0, 0.5, 0.2, 0.1, 0.0],
            [0.4, 0.0, 0.3, 0.2, 0.0],
            [0.3, 0.3, 0.0, 0.4, 0.0],
            [0.2, 0.1, 0.3, 0.0, 0.0],
            [0.1, 0.1, 0.2, 0.3, 0.0],
        ]
    )
    active = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    horizons = np.array([0.5, 2.0, 8.0])
    age = 3.0

    # Frailty-DRN (G=2 persistent classes) exact Bayesian posterior check
    gammas = np.array([0.25, 2.5])
    pi_w = np.tile(np.array([0.4, 0.6]), (V, 1))
    fr_f, ag_f = solve_frailty_renewal_talbot(
        jnp.asarray(lam_f),
        jnp.asarray(lam_s),
        jnp.asarray(alpha_f),
        jnp.asarray(b_slow),
        jnp.asarray(R_mat),
        jnp.asarray(gammas),
        jnp.asarray(pi_w),
        horizons,
        jnp.asarray(active),
        eval_nodes=jnp.arange(V - 1, dtype=jnp.int32),
        age=age,
        n_talbot=20,
        return_both_fresh_and_aged=True,
    )
    # Direct reference: mixture of class-wise expm with exact Bayes posterior
    ref_fr = np.zeros_like(np.asarray(fr_f))
    ref_ag = np.zeros_like(np.asarray(ag_f))
    log_s_list = []
    class_ag_list = []
    D = V * (K + 1)
    for g_idx, gam in enumerate(gammas):
        A_g = _dense_hyper_generator(lam_f * gam, lam_s * gam, alpha_f, b_slow, R_mat)
        pf_a, ps_a, ls_g = hyper_erlang_aged_phases_and_log_surv(
            jnp.asarray(lam_f * gam),
            jnp.asarray(lam_s * gam),
            jnp.asarray(alpha_f),
            jnp.asarray(b_slow),
            age,
            jnp.asarray(active),
        )
        log_s_list.append(np.asarray(ls_g[: V - 1]))
        X0_fr = np.zeros((V - 1, V, K + 1))
        X0_ag = np.zeros((V - 1, V, K + 1))
        for u in range(V - 1):
            X0_fr[u, u, 0] = alpha_f[u]
            X0_fr[u, u, 1:] = (1.0 - alpha_f[u]) * b_slow[u]
            X0_ag[u, u, 0] = float(pf_a[u])
            X0_ag[u, u, 1:] = np.asarray(ps_a[u])
        c_ag = np.zeros_like(ref_ag)
        for hi, h in enumerate(horizons):
            E_h = np.asarray(jl.expm(float(h) * A_g))
            ref_fr[:, hi, :] += pi_w[: V - 1, g_idx : g_idx + 1] * (
                (E_h @ X0_fr.reshape(V - 1, D).T).T.reshape(V - 1, V, K + 1).sum(axis=-1)
            )
            c_ag[:, hi, :] = (E_h @ X0_ag.reshape(V - 1, D).T).T.reshape(V - 1, V, K + 1).sum(axis=-1)
        class_ag_list.append(c_ag)
    log_s_arr = np.stack(log_s_list, axis=-1)
    lw = np.log(pi_w[: V - 1]) + log_s_arr
    lw -= lw.max(axis=-1, keepdims=True)
    post = np.exp(lw) / np.exp(lw).sum(axis=-1, keepdims=True)
    for g_idx in range(2):
        ref_ag += post[:, g_idx, None, None] * class_ag_list[g_idx]
    assert np.max(np.abs(np.asarray(fr_f) - ref_fr)) < 1e-8
    assert np.max(np.abs(np.asarray(ag_f) - ref_ag)) < 1e-8

    # Universal Hyper-Pos-Sens non-uniform Coxian Talbot check vs dense generator!
    psi_univ, _ = calibrate_universal_pos_clock(stages_red=6, stages_full=16, maxiter=40, k_min=1.5, k_max=5.5)
    edges = [(u, v) for u in range(V - 1) for v in range(V) if R_mat[v, u] > 0]
    rw = np.array([R_mat[v, u] for u, v in edges])
    fast_m = np.array([0.2, 0.3, 0.15, 0.25, 1.0])
    slow_m = np.array([3.0, 4.0, 2.5, 5.0, 1.0])
    slow_k = np.array([3.5, 4.2, 2.8, 5.0, 2.0])
    gen, fresh, read = build_universal_hyper_pos_network_generator(
        fast_m, slow_m, slow_k, alpha_f, rw, edges, V, psi_univ, stages_red=6
    )
    stage_rates, entries = jax.vmap(
        lambda m, k: universal_pos_clock(m, k, jnp.asarray(psi_univ), stages_red=6)
    )(jnp.asarray(slow_m), jnp.asarray(slow_k))
    p_cox_fr, p_cox_ag = solve_coxian_hyper_talbot(
        jnp.asarray(1.0 / fast_m) * jnp.asarray(active),
        stage_rates * jnp.asarray(active)[:, None],
        jnp.asarray(alpha_f) * jnp.asarray(active),
        entries,
        jnp.asarray(R_mat),
        horizons,
        jnp.asarray(active),
        eval_nodes=jnp.arange(V - 1, dtype=jnp.int32),
        age=age,
        n_talbot=24,
        return_both_fresh_and_aged=True,
    )
    gen_np, fresh_np, read_np = np.asarray(gen), np.asarray(fresh), np.asarray(read)
    for hi, h in enumerate(horizons):
        ref_h = (read_np @ np.asarray(jl.expm(float(h) * gen_np)) @ fresh_np).T[: V - 1]
        assert np.max(np.abs(np.asarray(p_cox_fr)[:, hi, :] - ref_h)) < 1e-8
    assert np.all(np.isfinite(np.asarray(p_cox_ag)))
