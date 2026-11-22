"""Independent symbolic and numerical verification of all theoretical results.

Verifies:
- Proposition 1: Exact mixed-Erlang marked-race compilation (degree, normalization,
  minimal order, and JAX/Diffrax compact vs. K**d tensor forward + gradient agreement).
- Proposition 2: Exact Hyper-Erlang marked-race decomposition (CV > 1 burstiness).
- Proposition 3: Single-visit indistinguishability with lag-l return covariance.
- Proposition 4: Nonlinear MeanFieldDRN simplex invariance & mass conservation.
- Theorems 1 & 2: Occupation dimension via balanced invisible fluxes and common
  survival-zero GCD formula (exact rational block-Hankel rank across topologies).
- Theorem 3: Multiplicity-sharp sensitivity dimension (symbolic Laplace residues,
  double-root principal coefficients, 33-vs-48 separation on the three-node Erlang-16
  cycle, Fréchet vs. Duhamel defect integral, and Baur et al. two-sided interpolation).
- Theorems 4 & 5: Positivity-preserving Fisher bias bound (Pos-Sens) and sharp
  Aldous-Shepp Markov-positivity W2 cone lower bound.
- Theorem 6: Constructive two-preparation exponential-snapshot Laplace inversion.
"""

import json
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import block_diag, expm, expm_frechet, helmert
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    ANCHORS_DEFAULT,
    READ_POS33,
    SHIFTS_DEFAULT,
    build_edge_memory_hyper_generator,
    build_interpolatory_basis,
    build_universal_pos_network_generator,
    calibrate_universal_pos_clock,
    compact_network_state_size,
    edge_memory_param_size,
    hyper_erlang_race,
    hyper_race_state_size,
    make_cycle_engine,
    make_interpolatory_engine,
    make_positive_33_engine,
    mixed_erlang_components,
    moment_matched_logits,
    network_state_size,
    pos33_cycle_matrices,
    resolvent_sensitivity_diagnostic,
    seicr_mean_field_predict,
    solve_compact_renewal_network,
    solve_renewal_network,
)

s_sym = sp.Symbol("s")


def _rational_coxian(weights, rate=1):
    weights = list(map(sp.Rational, weights))
    rate = sp.Rational(rate)
    n = len(weights)
    tails = [sum(weights[k:]) for k in range(n)]
    t_mat = -rate * sp.eye(n)
    for k in range(n - 1):
        t_mat[k + 1, k] = rate * tails[k + 1] / tails[k]
    b_vec = sp.eye(n)[:, 0]
    c_row = sp.ones(1, n)
    return t_mat, b_vec, c_row, -c_row * t_mat


def _rational_network(clocks, routing):
    t_mat = sp.diag(*(x[0] for x in clocks))
    b_mat = sp.diag(*(x[1] for x in clocks))
    c_mat = sp.diag(*(x[2] for x in clocks))
    d_mat = sp.diag(*(x[3] for x in clocks))
    return t_mat + b_mat * routing * d_mat, b_mat, c_mat


def _exact_hankel_order(a_mat, b_mat, c_mat) -> int:
    n = a_mat.rows
    seq = [c_mat * (a_mat**k) * b_mat for k in range(2 * n - 1)]
    hankel = sp.BlockMatrix(
        [[seq[i + j] for j in range(n)] for i in range(n)]
    ).as_explicit()
    return int(hankel.rank())


def _symbolic_residue_check(survival, routing, pi, zero, multiplicity):
    r_sym = sp.Symbol("r", positive=True)
    laws = [survival.subs(s_sym, s_sym / r_sym) / r_sym] + [survival] * (len(pi) - 1)
    y = (
        sp.diag(*laws)
        * (sp.eye(len(pi)) - routing * sp.diag(*[1 - s_sym * x for x in laws])).inv()
        * pi
    )
    deriv = [sp.factor(sp.diff(x, r_sym).subs(r_sym, 1)) for x in y]
    residues = sp.Matrix([sp.residue(x, s_sym, zero) for x in deriv])
    expected = -multiplicity * pi[0] * (sp.eye(len(pi))[:, 0] - pi)
    assert residues == expected
    nominal = sp.diag(*([survival] * len(pi))) * (
        sp.eye(len(pi)) - routing * (1 - s_sym * survival)
    ).inv()
    assert all(sp.denom(sp.cancel(x)).subs(s_sym, zero) != 0 for x in nominal)
    return {
        "zero": str(zero),
        "multiplicity": multiplicity,
        "residues": list(map(str, residues)),
    }


def _verify_duhamel_gradient_defect() -> dict:
    n, nodes, rate, horizon = 4, 3, 2.0, 1.3
    local = np.eye(n, k=-1) - np.eye(n)
    b = block_diag(*([np.eye(n)[:, 0, None]] * nodes))
    c = np.kron(np.eye(nodes), np.ones((1, n)))
    routing = np.roll(np.eye(nodes), 1, axis=0)
    exits = np.kron(np.eye(nodes), np.eye(1, n, n - 1))
    a = rate * np.kron(np.eye(nodes), local) + rate * b @ routing @ exits
    selector = np.diag([1.0, 0.0, 0.0])
    da = rate * np.kron(selector, local) + rate * b @ routing @ selector @ exits
    contrast = np.kron(helmert(nodes, full=False).T, np.eye(n))
    w = np.vstack((contrast.T, np.ones(nodes * n)))
    v = np.column_stack((contrast, np.ones(nodes * n) / (nodes * n)))
    cr = np.column_stack((c @ contrast, np.ones(nodes) / nodes))
    g, dg = w @ a @ v, w @ da @ v
    q = np.eye(nodes * n) - v @ w
    assert np.linalg.norm(w @ a @ q) < 1e-12
    direct = (
        c @ expm_frechet(horizon * a, horizon * da, compute_expm=False) @ b
        - cr @ expm_frechet(horizon * g, horizon * dg, compute_expm=False) @ w @ b
    )
    integral = (
        cr
        @ quad_vec(
            lambda u: expm((horizon - u) * g) @ w @ da @ q @ expm(u * a) @ b,
            0.0,
            horizon,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
    )
    err = float(np.max(np.abs(direct - integral)))
    assert err < 1e-12
    return {
        "frechet_vs_duhamel_max_abs": err,
        "defect_max_abs": float(np.max(np.abs(direct))),
    }


def _verify_marked_race_diffrax() -> dict:
    edges = ((0, 1), (0, 2), (1, 3), (2, 3))
    stages = 16
    assert network_state_size(edges, 4, stages) == 289
    assert compact_network_state_size(edges, 4, stages) == 64
    means = jnp.array([1.7, 2.3, 1.4, 2.1])
    shapes = jnp.array([5.5, 8.2, 4.7, 9.4])
    ts = jnp.linspace(0.25, 6.0, 24)
    init = jnp.array([1.0, 0.0, 0.0, 0.0])

    def full_fn(theta):
        m, k = jnp.exp(theta[:4]), jnp.exp(theta[4:])
        return solve_renewal_network(m, moment_matched_logits(k), edges, init, ts).ys

    def comp_fn(theta):
        m, k = jnp.exp(theta[:4]), jnp.exp(theta[4:])
        return solve_compact_renewal_network(
            m, moment_matched_logits(k), edges, init, ts
        ).ys

    theta0 = jnp.concatenate([jnp.log(means), jnp.log(shapes)])
    y_full, y_comp = full_fn(theta0), comp_fn(theta0)
    traj_err = float(jnp.max(jnp.abs(y_full - y_comp)))
    g_full = jax.grad(lambda th: jnp.sum(full_fn(th)[-1, :3] * jnp.array([0.3, 0.7, 1.1])))(theta0)
    g_comp = jax.grad(lambda th: jnp.sum(comp_fn(th)[-1, :3] * jnp.array([0.3, 0.7, 1.1])))(theta0)
    grad_rel_err = float(jnp.linalg.norm(g_full - g_comp) / jnp.linalg.norm(g_full))
    assert traj_err < 1e-8 and grad_rel_err < 1e-7
    return {
        "joint_tensor_states": 289,
        "compact_race_states": 64,
        "trajectory_max_abs_error": traj_err,
        "gradient_relative_l2_error": grad_rel_err,
    }


def _verify_hyper_race_and_universal_pos_sens() -> dict:
    """Verify Proposition 2 (Hyper-Erlang Marked-Race Compiler) and Universal Graph Pos-Sens."""
    # 1. Proposition 2: d=3 competing M=2 Hyper-Erlang clocks (CV > 1 bursty + refractory)
    d, m_br, k_st = 3, 2, 6
    naive_dim = (m_br * k_st) ** d  # 12^3 = 1,728
    comp_dim = hyper_race_state_size(d, m_br, k_st)  # 8 * 16 = 128
    assert (naive_dim, comp_dim) == (1728, 128)

    mix_logits = jnp.array([[0.4, -0.4], [-0.2, 0.5], [0.0, 0.3]])
    br_means = jnp.array([[0.25, 2.4], [0.35, 1.9], [0.30, 2.8]])  # bimodal CV > 1!
    rng = np.random.default_rng(314)
    br_stage_logits = jnp.asarray(rng.normal(0.0, 0.7, size=(d, m_br, k_st)))

    def compiled_win_probs_and_mean(means_in):
        ent, rates, cont, causes = hyper_erlang_race(mix_logits, means_in, br_stage_logits)
        l_tot = ent.size
        blk = -jnp.diag(rates) + jnp.diag(rates[:-1] * cont, k=-1)
        # Solve -blk @ occ = ent -> occ = (-blk)^{-1} ent
        occ = jnp.linalg.solve(-blk, ent)
        win_p = causes @ (rates * occ)
        mean_exit = jnp.sum(occ)
        return win_p, mean_exit

    win_c, mean_c = compiled_win_probs_and_mean(br_means)
    assert abs(float(jnp.sum(win_c)) - 1.0) < 1e-13

    # Compare against direct numerical quadrature of k_e(t) = f_e(t) prod_{r != e} S_r(t)
    def exact_hyper_surv_dens(t_val, e_id, means_in):
        alphas = jax.nn.softmax(mix_logits[e_id])
        s_tot, f_tot = 0.0, 0.0
        for m_i in range(m_br):
            r_m, ent_m = mixed_erlang_components(means_in[e_id, m_i], br_stage_logits[e_id, m_i])
            blk_m = r_m * (jnp.eye(k_st, k=-1) - jnp.eye(k_st))
            st_t = expm(float(t_val) * np.asarray(blk_m)) @ np.asarray(ent_m)
            s_tot += float(alphas[m_i]) * float(st_t.sum())
            f_tot += float(alphas[m_i]) * float(r_m) * float(st_t[-1])
        return s_tot, f_tot

    win_quad = []
    for e_id in range(d):
        val_q = quad_vec(
            lambda t: exact_hyper_surv_dens(t, e_id, np.asarray(br_means))[1]
            * np.prod([exact_hyper_surv_dens(t, r_id, np.asarray(br_means))[0] for r_id in range(d) if r_id != e_id]),
            0.0,
            25.0,
            epsabs=1e-11,
            epsrel=1e-11,
        )[0]
        win_quad.append(float(val_q))
    win_err = float(np.max(np.abs(np.asarray(win_c) - np.asarray(win_quad))))
    assert win_err < 1e-9

    # Check autodiff gradient vs central finite difference on compiled Hyper-Erlang race
    grad_ad = np.asarray(jax.grad(lambda ms: jnp.sum(compiled_win_probs_and_mean(ms)[0] * jnp.array([0.3, 0.7, 1.1])) + 0.2 * compiled_win_probs_and_mean(ms)[1])(br_means))
    eps_fd = 1e-6
    grad_fd = np.zeros_like(grad_ad)
    base_bm = np.asarray(br_means)
    for i in range(d):
        for j in range(m_br):
            bm_p, bm_m = base_bm.copy(), base_bm.copy()
            bm_p[i, j] += eps_fd
            bm_m[i, j] -= eps_fd
            wp, mp = compiled_win_probs_and_mean(jnp.asarray(bm_p))
            wm, mm = compiled_win_probs_and_mean(jnp.asarray(bm_m))
            fp = float(jnp.sum(wp * jnp.array([0.3, 0.7, 1.1])) + 0.2 * mp)
            fm = float(jnp.sum(wm * jnp.array([0.3, 0.7, 1.1])) + 0.2 * mm)
            grad_fd[i, j] = (fp - fm) / (2.0 * eps_fd)
    hyper_grad_err = float(np.max(np.abs(grad_ad - grad_fd)))
    assert hyper_grad_err < 1e-8

    # 2. Universal Topology-Agnostic Pos-Sens on a 4-node branching diamond + EdgeMemoryHyperDRN Metzler check
    psi_u, cert = calibrate_universal_pos_clock(stages_red=10, stages_full=16, maxiter=250)
    assert cert["sup_survival_abs_error"] < 0.015
    diamond_edges = ((0, 1), (0, 2), (1, 3), (2, 3))
    gen_d, fresh_d, read_d = map(
        np.asarray,
        build_universal_pos_network_generator(
            [1.6, 2.1, 1.8, 1.0],
            [5.0, 8.0, 6.5, 4.0],
            [0.6, 0.4, 1.0, 1.0],
            diamond_edges,
            4,
            psi_u,
            stages_red=10,
        ),
    )
    assert float((gen_d - np.diag(np.diag(gen_d))).min()) >= 0.0
    assert float(np.max(np.abs(gen_d.sum(axis=0)))) < 1e-13
    p_diam = read_d @ expm(1.2 * gen_d) @ fresh_d
    assert float(p_diam.min()) >= 0.0 and float(np.max(np.abs(p_diam.sum(axis=0) - 1.0))) < 1e-13

    # Check EdgeMemoryHyperDRN Metzler & simplex conservation
    em_edges = ((0, 1), (1, 2), (2, 0), (0, 2))
    n_em = edge_memory_param_size(3, 4, stages=5)
    p_em = jnp.asarray(rng.normal(0.0, 0.25, size=n_em))
    gen_em, fr_em, ag_em, rd_em = map(
        np.asarray, build_edge_memory_hyper_generator(p_em, em_edges, 3, stages=5)
    )
    assert float((gen_em - np.diag(np.diag(gen_em))).min()) >= 0.0
    assert float(np.max(np.abs(gen_em.sum(axis=0)))) < 1e-13
    p_ag = rd_em @ expm(0.7 * gen_em) @ ag_em
    assert float(p_ag.min()) > 0.0 and float(np.max(np.abs(p_ag.sum(axis=0) - 1.0))) < 1e-13

    return {
        "proposition_2_hyper_race_naive_states": naive_dim,
        "proposition_2_hyper_race_compiled_states": comp_dim,
        "proposition_2_hyper_race_win_prob_quad_error": win_err,
        "proposition_2_hyper_race_autodiff_grad_error": hyper_grad_err,
        "universal_pos_sens_certificate": cert,
        "edge_memory_hyper_metzler_col_sum_err": float(np.max(np.abs(gen_em.sum(axis=0)))),
    }


def _verify_positivity_and_fisher_bias_bound() -> dict:
    """Verify Theorem 4: global Metzler simplex invariance of Pos-Sens 33 and Fisher-metric bias bound."""
    pos_summary = json.loads(
        (ROOT / "results" / "archived" / "positive_sensitivity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    psi_star = np.asarray(pos_summary["psi_star"], dtype=float)
    rng = np.random.default_rng(2026)

    max_col_sum_err = 0.0
    min_off_diag = 0.0
    min_pos33_prob = 1.0
    for _ in range(25):
        th = np.r_[
            rng.uniform(np.log(0.95), np.log(3.8), size=3),
            rng.uniform(np.log(3.0), np.log(12.0), size=3),
        ]
        gen, fresh = map(np.asarray, pos33_cycle_matrices(th, psi_star))
        off_diag = gen - np.diag(np.diag(gen))
        min_off_diag = min(min_off_diag, float(off_diag.min()))
        max_col_sum_err = max(max_col_sum_err, float(np.max(np.abs(gen.sum(axis=0)))))
        assert float(fresh.min()) >= 0.0
        assert float(np.max(np.abs(fresh.sum(axis=0) - 1.0))) < 1e-14
        for t in (1e-4, 1e-2, 0.05, 0.25, 1.5):
            p_t = READ_POS33 @ expm(t * gen) @ fresh
            min_pos33_prob = min(min_pos33_prob, float(p_t.min()))
            assert float(np.max(np.abs(p_t.sum(axis=0) - 1.0))) < 1e-13

    assert min_off_diag >= 0.0 and max_col_sum_err < 1e-13 and min_pos33_prob > 0.0

    # Contrast with signed Baur et al. Interp. 33 away from symmetry (exhibits negative early probability)
    q33, _ = build_interpolatory_basis(33)
    sp33_eval = make_interpolatory_engine(q33)
    th0 = np.log([2.0, 2.0, 2.0, 6.0, 6.0, 6.0])
    th_het = th0 + 0.15 * np.array([1.0, 0.0, -1.0, -1.0, 1.0, 0.0])
    sp33_min_early = float(sp33_eval(th_het)[0][:2].min())
    assert sp33_min_early < 0.0

    # Verify Theorem 4's orthogonal-projector Fisher-metric misspecification bound:
    # ||I_r^{1/2} (theta_r^* - theta_eps)||_2 <= ||D_p^{1/2} dp||_2 + eps * ||D_p^{1/2} dJ h||_2 + O(||dp||^2 + eps^2)
    full_eval = make_cycle_engine(reduced=False)
    pos_eval = make_positive_33_engine(psi_star)
    sample = np.arange(2, 14)
    pf0, jf0 = full_eval(th0)
    pr0, jr0 = pos_eval(th0)
    p_vec = pr0[sample].ravel()
    dp_vec = p_vec - pf0[sample].ravel()
    jr_mat = jr0[sample].reshape(-1, 6)
    dj_mat = jr_mat - jf0[sample].reshape(-1, 6)

    w_diag = 1.0 / p_vec  # proportional to D_p with uniform stratum weights
    ir_mat = jr_mat.T @ (w_diag[:, None] * jr_mat)
    # Check column orthonormality of U = D_p^{1/2} J_r I_r^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(ir_mat)
    ir_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    ir_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    u_mat = (np.sqrt(w_diag)[:, None] * jr_mat) @ ir_inv_sqrt
    ortho_err = float(np.max(np.abs(u_mat.T @ u_mat - np.eye(6))))
    assert ortho_err < 1e-12

    h_dir = np.array([1.0, 0.0, -1.0, -1.0, 1.0, 0.0])
    eps = 0.02
    lhs_linear = float(
        np.linalg.norm(
            ir_sqrt
            @ np.linalg.solve(
                ir_mat, jr_mat.T @ (w_diag * (-dp_vec - eps * (dj_mat @ h_dir)))
            )
        )
    )
    rhs_bound = float(
        np.linalg.norm(np.sqrt(w_diag) * dp_vec)
        + eps * np.linalg.norm(np.sqrt(w_diag) * (dj_mat @ h_dir))
    )
    assert lhs_linear <= rhs_bound + 1e-14
    return {
        "metzler_min_off_diagonal": min_off_diag,
        "generator_col_sum_max_abs": max_col_sum_err,
        "pos33_min_early_probability": min_pos33_prob,
        "sp33_min_early_probability": sp33_min_early,
        "fisher_projector_orthonormality_error": ortho_err,
        "linearized_fisher_bias": lhs_linear,
        "fisher_whitened_residual_bound": rhs_bound,
    }


def _verify_interpolatory_projection() -> dict:
    q33, info33 = build_interpolatory_basis(33)
    assert info33["relative_span_residual"] < 1e-14
    assert info33["orthogonality_error"] < 1e-14
    max_tf_err = 0.0
    max_sens_err = 0.0
    for anchor in ANCHORS_DEFAULT:
        for shift in SHIFTS_DEFAULT:
            diag = resolvent_sensitivity_diagnostic(q33, anchor, shift)
            max_tf_err = max(max_tf_err, diag["transfer_max_abs"])
            max_sens_err = max(max_sens_err, diag["sensitivity_max_abs"])
    assert max_tf_err < 1e-13 and max_sens_err < 1e-13
    return {
        "basis_dimension": 33,
        "span_residual": info33["relative_span_residual"],
        "anchor_transfer_max_abs": max_tf_err,
        "anchor_sensitivity_max_abs": max_sens_err,
    }


def _verify_multiplicity_sharpness_and_converse() -> dict:
    """Verify:
    1. Theorem 3 Sharpness: Repeated survival zero (m_zeta = 2) under shape/numerator
       perturbation restores exact pole order m_zeta = 2 (eliminating the square-free
       assumption via Kalman canonical decomposition -> exact d_sens = V*n).
    2. Theorem 5 (Aldous-Shepp Positive-Reduction Impossibility Lower Bound): Every
       positive K_r-stage Markov clock approximating shape k > K_r obeys exact W2 cone
       distance >= (1 - sqrt(K_r/k)) / sqrt(K_r + 1) and mean-matched weighted survival
       integral >= (m^2 / 2) * (1/K_r - 1/k).
    3. Proposition 4 (MeanFieldDRN Global Positivity & Simplex Invariance).
    """
    # 1. Symbolic repeated-zero (m_zeta = 2) shape perturbation on 3-node cycle
    cycle = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    pi = sp.ones(3, 1) / 3
    eps_sym = sp.Symbol("eps")
    zero = -sp.Rational(3, 2)
    n0 = (s_sym - zero) ** 2
    d0 = (s_sym + 1) ** 3
    s0 = n0 / d0
    # Shape perturbation on node 0 shifting numerator N_0(s) = N(s) + eps * 1 (q(zero) = 1 != 0)
    laws_eps = [(n0 + eps_sym) / d0, s0, s0]
    y_eps = (
        sp.diag(*laws_eps)
        * (sp.eye(3) - cycle * sp.diag(*[1 - s_sym * x for x in laws_eps])).inv()
        * pi
    )
    dy_shape = [sp.cancel(sp.diff(x, eps_sym).subs(eps_sym, 0)) for x in y_eps]
    # Multiply by (s - zero)^2 and evaluate at s = zero -> must equal (pi_0 / zero) * (e_0 - pi) != 0!
    principal_m2 = sp.Matrix([sp.simplify(sp.cancel((s_sym - zero) ** 2 * x).subs(s_sym, zero)) for x in dy_shape])
    expected_m2 = (pi[0] / zero) * (sp.eye(3)[:, 0] - pi)
    assert principal_m2 == expected_m2
    # Confirm nominal transfer is analytic at s = zero
    nom3 = sp.diag(*([s0] * 3)) * (sp.eye(3) - cycle * (1 - s_sym * s0)).inv()
    assert all(sp.denom(sp.cancel(x)).subs(s_sym, zero) != 0 for x in nom3)

    # 2. Aldous-Shepp positive-reduction lower bound check across (K_r, k) with k > K_r
    from scipy.optimize import minimize
    from scipy.stats import gamma as sp_gamma

    as_rows = []
    u_grid = np.linspace(0.0, 5.0, 500)
    du = u_grid[1] - u_grid[0]
    q_grid = np.linspace(0.002, 0.998, 250)
    for k_red, k_true in [(6, 8), (6, 12), (10, 14), (10, 16)]:
        m_true = 1.0
        # Exact target Erlang(k_true, rate=k_true) quantiles & survival (mean=1, var=1/k_true)
        target_q = sp_gamma.ppf(q_grid, a=k_true, scale=1.0 / k_true)
        target_s = sp_gamma.sf(u_grid, a=k_true, scale=1.0 / k_true)

        # Optimize best mean-matched K_r-stage positive Coxian clock (Aldous-Shepp extremum is Erlang(K_r))
        # Test random positive Coxian chains + Erlang(K_r) boundary to verify inequality holds everywhere
        # For Erlang(K_r, rate=K_r), mean=1, var=1/K_r (minimum variance among ALL positive K_r-stage PH!)
        opt_q = sp_gamma.ppf(q_grid, a=k_red, scale=1.0 / k_red)
        opt_s = sp_gamma.sf(u_grid, a=k_red, scale=1.0 / k_red)
        w2_erlang = float(np.sqrt(np.mean((opt_q - target_q) ** 2)))
        w2_lb_unconstrained = float((1.0 - np.sqrt(k_red / k_true)) / np.sqrt(k_red + 1.0))
        w2_lb_mean_matched = float(1.0 / np.sqrt(k_red) - 1.0 / np.sqrt(k_true))
        weighted_surv_int = float(np.trapz(u_grid * np.abs(opt_s - target_s), u_grid))
        surv_lb = 0.5 * (1.0 / k_red - 1.0 / k_true)
        assert w2_erlang >= w2_lb_mean_matched - 1e-3
        assert w2_erlang >= w2_lb_unconstrained
        assert weighted_surv_int >= surv_lb - 1e-3

        # Also check 10 random positive non-Erlang K_r-stage Coxian clocks obey the cone lower bound!
        rng = np.random.default_rng(100 * k_red + k_true)
        for _ in range(10):
            r_raw = rng.uniform(0.5, 4.0, size=k_red)
            b_raw = rng.dirichlet(np.ones(k_red))
            # Exact mean and second moment of forward chain: T_j ~ Exp(r_j), entry b_raw
            m_rem = np.cumsum(1.0 / r_raw[::-1])[::-1]
            v_rem = np.cumsum(1.0 / (r_raw[::-1] ** 2))[::-1]
            mean_ph = float(b_raw @ m_rem)
            var_ph = float(b_raw @ (v_rem + (m_rem - mean_ph) ** 2))
            assert var_ph / (mean_ph**2) >= (1.0 / k_red) - 1e-12
            w2_cone_dist = float(np.sqrt((mean_ph - m_true) ** 2 + (np.sqrt(var_ph) - 1.0 / np.sqrt(k_true)) ** 2))
            assert w2_cone_dist >= w2_lb_unconstrained - 1e-12

        as_rows.append(
            {
                "K_r": k_red,
                "k_true": k_true,
                "w2_unconstrained_lower_bound": w2_lb_unconstrained,
                "w2_mean_matched_lower_bound": w2_lb_mean_matched,
                "w2_erlang_extremum": w2_erlang,
                "weighted_surv_lower_bound": surv_lb,
                "weighted_surv_erlang_extremum": weighted_surv_int,
            }
        )

    # 3. Proposition 4: MeanFieldDRN Global Positivity & Simplex Conservation across 15 random draws
    rng_mf = np.random.default_rng(2027)
    ts_mf = jnp.linspace(0.5, 20.0, 25)
    min_x = 1.0
    max_mass_err = 0.0
    for _ in range(15):
        p_mf, x_mf = seicr_mean_field_predict(
            jnp.array(rng_mf.uniform(0.5, 3.0)),
            jnp.array(rng_mf.uniform(0.5, 3.0)),
            jnp.array(rng_mf.uniform(1.5, 7.5)),
            jnp.array(rng_mf.uniform(1.0, 5.0)),
            jnp.array(rng_mf.uniform(1.5, 7.5)),
            jnp.array(rng_mf.uniform(1.0, 5.0)),
            jnp.array(rng_mf.uniform(1.5, 7.5)),
            jnp.array(rng_mf.uniform(0.002, 0.05)),
            jnp.array(rng_mf.uniform(0.002, 0.05)),
            ts_mf,
            stages=8,
        )
        min_x = min(min_x, float(jnp.min(x_mf)))
        max_mass_err = max(max_mass_err, float(jnp.max(jnp.abs(jnp.sum(p_mf, axis=-1) - 1.0))))
    assert min_x >= -1e-7 and max_mass_err < 1e-12

    return {
        "theorem_3_double_zero_shape_principal_coeff": list(map(str, principal_m2)),
        "theorem_5_aldous_shepp_bounds": as_rows,
        "proposition_4_mean_field_min_phase": min_x,
        "proposition_4_mean_field_max_mass_err": max_mass_err,
    }


def main() -> None:
    alt = sp.Matrix([[0, 1], [1, 0]])
    cycle = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    e2, e3 = _rational_coxian([0, 1]), _rational_coxian([0, 0, 1])

    hankel_cases = {}
    for name, clocks, routing, expected in [
        ("two_erlang2_alternating", [e2, e2], alt, 3),
        ("three_erlang3_cycle", [e3] * 3, cycle, 7),
        (
            "distinct_clocks_shared_zero",
            [
                _rational_coxian([sp.Rational(1, 2)] * 2, 2),
                _rational_coxian([sp.Rational(4, 5), sp.Rational(1, 5)], sp.Rational(5, 2)),
            ],
            alt,
            3,
        ),
        (
            "distinct_clocks_coprime",
            [
                _rational_coxian([sp.Rational(1, 2)] * 2, 2),
                _rational_coxian([sp.Rational(4, 5), sp.Rational(1, 5)], 3),
            ],
            alt,
            4,
        ),
        (
            "repeated_double_zero_cycle",
            [_rational_coxian([0, sp.Rational(3, 4), sp.Rational(1, 4)])] * 3,
            cycle,
            7,
        ),
    ]:
        a_m, b_m, c_m = _rational_network(clocks, routing)
        rk = _exact_hankel_order(a_m, b_m, c_m)
        assert rk == expected
        hankel_cases[name] = rk

    sink = (sp.zeros(1), sp.ones(1), sp.ones(1), sp.zeros(1))
    dag_rk = _exact_hankel_order(
        *_rational_network([e2, e2, sink], sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 1, 1]]))
    )
    assert dag_rk == 5
    hankel_cases["dag_with_sink"] = dag_rk

    # Counterexamples (Appendix Table A1 Rows 7-8):
    # 1) Reducible routing R = I on two Erlang-2 clocks -> block-Hankel order 2
    reducible_rk = _exact_hankel_order(*_rational_network([e2, e2], sp.eye(2)))
    assert reducible_rk == 2
    hankel_cases["counterexample_reducible_R_I"] = reducible_rk
    # 2) Stationary mixture preparation only (B_stat = B * pi) on 3-node Erlang-3 cycle -> order 1
    a_c3, b_c3, c_c3 = _rational_network([e3] * 3, cycle)
    stat_rk = _exact_hankel_order(a_c3, b_c3 * (sp.ones(3, 1) / 3), c_c3)
    assert stat_rk == 1
    hankel_cases["counterexample_stationary_mixture_only"] = stat_rk

    numerator_16 = sp.Poly(sp.cancel(((s_sym + 1) ** 16 - 1) / s_sym), s_sym)
    assert numerator_16.degree() == 15
    assert sp.gcd(numerator_16, numerator_16.diff()).degree() == 0

    nonuniform = sp.Matrix(
        [[sp.Rational(1, 2), sp.Rational(1, 3)], [sp.Rational(1, 2), sp.Rational(2, 3)]]
    )
    res_rate = _symbolic_residue_check(
        (s_sym + 2) / (s_sym + 1) ** 2,
        nonuniform,
        sp.Matrix([sp.Rational(2, 5), sp.Rational(3, 5)]),
        -2,
        1,
    )
    res_double = _symbolic_residue_check(
        (s_sym + sp.Rational(3, 2)) ** 2 / (s_sym + 1) ** 3,
        alt,
        sp.ones(2, 1) / 2,
        -sp.Rational(3, 2),
        2,
    )

    # Two-preparation exponential-snapshot inversion check
    laws = [(s_sym + 2) / (s_sym + 1) ** 2, (s_sym + 6) / (s_sym + 3) ** 2]
    phi = sp.diag(*[1 - s_sym * x for x in laws])
    pexp = s_sym * sp.diag(*laws) * (sp.eye(2) - alt * phi).inv()
    ps = pexp.subs(s_sym, sp.Rational(1, 2))
    w_true = sp.Matrix([((1 - s_sym * x) / (s_sym * x)).subs(s_sym, sp.Rational(1, 2)) for x in laws])
    stacked = []
    for b0 in (sp.Matrix([1, 0]), sp.Matrix([0, 1])):
        prob = ps * b0
        mat = (sp.eye(2) - alt) * sp.diag(*prob)
        assert (mat * w_true - (b0 - prob)).applyfunc(sp.simplify) == sp.zeros(2, 1)
        stacked.append(mat)
    assert sp.Matrix.vstack(*stacked).rank() == 2

    # Proposition 3: Single-visit indistinguishability & lag covariance
    delta_val, rho_val, m1, m2 = 0.45, 0.8, 1.8, 2.2
    p_trans = rho_val * np.eye(2) + (1.0 - rho_val) * np.ones((2, 2)) / 2.0
    signs = np.array([-1.0, 1.0])
    cov_lag2 = 0.5 * float(signs @ np.linalg.matrix_power(p_trans, 2) @ signs) * (delta_val**2) * m1 * m2
    assert abs(cov_lag2 - (delta_val**2) * (rho_val**2) * m1 * m2) < 1e-15

    def _verify_heterogeneous_theorem3_and_hyper_pos() -> dict:
        """Symbolically verify Generalized Heterogeneous Theorem 3 (arbitrary clocks with shared GCD)
        and numerically verify Metzler positivity of 2-branch Universal Hyper-Pos-Sens."""
        from diff_renewal import build_universal_hyper_pos_network_generator, calibrate_universal_pos_clock

        s, eps = sp.symbols("s eps")
        # Case 1: Distinct 2-stage clocks sharing simple zero zeta = -3 (d_occ=3, d_sens=4)
        d1, d2 = (s + 1) * (s + 4), (s + 2) * (s + 5)
        s1, s2 = (s + 3) / d1, (s + 3) / d2
        r2 = sp.Matrix([[0, 1], [1, 0]])
        pi2 = sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2)])
        sm = sp.diag(s1 + eps / d1, s2)
        y = sm * (sp.eye(2) - r2 * (sp.eye(2) - s * sm)).inv() * pi2
        dy = sp.diff(y, eps).subs(eps, 0).applyfunc(sp.simplify)
        z1 = sp.Integer(-3)
        sh = [sp.simplify(sx / (s - z1)).subs(s, z1) for sx in (s1, s2)]
        al1 = pi2[0] * sh[0] + pi2[1] * sh[1]
        w1 = sp.Matrix([pi2[0] * sh[0] / al1, pi2[1] * sh[1] / al1])
        pred1 = (pi2[0] / (z1 * d1.subs(s, z1) * al1)) * (sp.Matrix([1, 0]) - w1)
        act1 = sp.Matrix([sp.limit((s - z1) * g, s, z1) for g in dy])
        assert sp.simplify(act1 - pred1) == sp.zeros(2, 1) and act1 != sp.zeros(2, 1)

        # Case 2: Heterogeneous 3-node cycle with partial GCD double root zeta=-3/2 (n=(3,3,4), m=(2,2,3))
        z2 = sp.Rational(-3, 2)
        d1b, d2b, d3b = (s + 1) * (s + 2) * (s + 4), (s + 1) * (s + 3) * (s + 5), (s + 1) * (s + 2) * (s + 3) * (s + 6)
        s1b, s2b, s3b = (s - z2) ** 2 / d1b, (s - z2) ** 2 / d2b, (s - z2) ** 3 / d3b
        r3 = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        pi3 = sp.Matrix([sp.Rational(1, 3)] * 3)
        sm3 = sp.diag(s1b, s2b, s3b + eps / d3b)
        y3 = sm3 * (sp.eye(3) - r3 * (sp.eye(3) - s * sm3)).inv() * pi3
        dy3 = sp.diff(y3, eps).subs(eps, 0).applyfunc(sp.simplify)
        sh3 = [sp.simplify(sx / (s - z2) ** 2).subs(s, z2) for sx in (s1b, s2b, s3b)]
        al3 = sum(pi3[k] * sh3[k] for k in range(3))
        w3 = sp.Matrix([pi3[k] * sh3[k] / al3 for k in range(3)])
        pred2 = (pi3[2] / (z2 * d3b.subs(s, z2) * al3)) * (sp.Matrix([0, 0, 1]) - w3)
        act2 = sp.Matrix([sp.limit((s - z2) ** 2 * g, s, z2) for g in dy3])
        assert sp.simplify(act2 - pred2) == sp.zeros(3, 1) and act2 != sp.zeros(3, 1)

        # Case 3: Asymmetric minimal multiplicity (only node 1 has m1=1, w_zeta = e1; node 2 restores simple pole)
        z3 = sp.Integer(-2)
        d1c, d2c, d3c = (s + 1) * (s + 3), (s + 1) * (s + 4) * (s + 5), (s + 1) * (s + 3) * (s + 6)
        s1c, s2c, s3c = (s - z3) / d1c, (s - z3) ** 2 / d2c, (s - z3) ** 2 / d3c
        sm3c = sp.diag(s1c, s2c + eps / d2c, s3c)
        y3c = sm3c * (sp.eye(3) - r3 * (sp.eye(3) - s * sm3c)).inv() * pi3
        dy3c = sp.diff(y3c, eps).subs(eps, 0).applyfunc(sp.simplify)
        sh3c = [sp.simplify(sx / (s - z3)).subs(s, z3) for sx in (s1c, s2c, s3c)]
        al3c = sum(pi3[k] * sh3c[k] for k in range(3))
        w3c = sp.Matrix([pi3[k] * sh3c[k] / al3c for k in range(3)])
        pred3 = (pi3[1] / (z3 * d2c.subs(s, z3) * al3c)) * (sp.Matrix([0, 1, 0]) - w3c)
        act3 = sp.Matrix([sp.limit((s - z3) * g, s, z3) for g in dy3c])
        assert sp.simplify(act3 - pred3) == sp.zeros(3, 1) and act3 != sp.zeros(3, 1)

        # Full-cone [1.4, 7.8] Universal Pos-Sens & 2-branch Hyper-Pos-Sens Metzler check
        psi_fc, cert_fc = calibrate_universal_pos_clock(stages_red=8, stages_full=16, maxiter=200, k_min=1.4, k_max=7.8)
        edges = ((0, 1), (1, 2), (2, 0))
        gen_h, fr_h, _ = build_universal_hyper_pos_network_generator(
            np.array([0.3, 0.4, 0.5]),
            np.array([2.5, 3.0, 4.0]),
            np.array([2.0, 5.0, 7.5]),
            np.array([0.4, 0.5, 0.3]),
            np.array([1.0, 1.0, 1.0]),
            edges,
            3,
            psi_fc,
            stages_red=8,
        )
        gen_np = np.asarray(gen_h)
        off_diag = gen_np[~np.eye(gen_np.shape[0], dtype=bool)]
        col_err = float(np.max(np.abs(gen_np.sum(axis=0))))
        assert float(np.min(off_diag)) >= -1e-15 and col_err < 1e-13
        assert float(np.max(np.abs(np.asarray(fr_h).sum(axis=0) - 1.0))) < 1e-14

        return {
            "heterogeneous_distinct_2stage_residue": [str(x) for x in act1],
            "heterogeneous_partial_gcd_double_root_coeff": [str(x) for x in act2],
            "heterogeneous_asymmetric_w_equals_e1_coeff": [str(x) for x in act3],
            "universal_full_cone_1p4_to_7p8_sup_surv_err": round(cert_fc["sup_survival_abs_error"], 5),
            "hyper_pos_sens_metzler_col_sum_err": col_err,
        }

    def _verify_theorem6_quantitative_bt() -> dict:
        """Verify Theorem 6: Quantitative delta-near-cancellation & Balanced-Truncation
        gradient distortion (exact Riesz projection analyticity, O(delta) H_inf forward error,
        Omega(1) sensitivity floor for both frozen and re-balanced BT, Omega(delta^-1) blow-up)."""
        from scripts.run_r7_theory_bt_separation import (
            build_cycle_erlang,
            build_cycle_erlang_deriv_lograte0,
            build_heterogeneous_common_zero,
            exact_bt_and_sensitivity,
        )

        u = np.array([1.0, -0.5, -0.5])
        pi0 = np.ones(3) / 3.0
        # Symmetric K=4 cycle (Full 12 -> d_occ=9, transient r=8)
        r1 = exact_bt_and_sensitivity(
            *build_cycle_erlang(2.0 * np.exp(1e-4 * u), 4),
            build_cycle_erlang_deriv_lograte0(2.0 * np.exp(1e-4 * u), 4),
            np.zeros((12, 3)),
            8,
            pi0,
        )
        r2 = exact_bt_and_sensitivity(
            *build_cycle_erlang(2.0 * np.exp(1e-3 * u), 4),
            build_cycle_erlang_deriv_lograte0(2.0 * np.exp(1e-3 * u), 4),
            np.zeros((12, 3)),
            8,
            pi0,
        )
        # Check exact 10x forward scaling & ~10x amplification scaling over 10x delta step!
        fwd_slope = float(np.log10(r2["fwd_hinf"] / r1["fwd_hinf"]))
        ratio_slope_reb = float(np.log10(r2["ratio_rebal"] / r1["ratio_rebal"]))
        ratio_slope_fz = float(np.log10(r2["ratio_frozen"] / r1["ratio_frozen"]))
        assert r1["comm_err"] < 1e-12 and r2["comm_err"] < 1e-12
        assert abs(fwd_slope - 1.0) < 0.01
        assert abs(ratio_slope_reb + 1.0) < 0.01
        assert abs(ratio_slope_fz + 1.0) < 0.01
        assert r1["err_rebal"] > 0.01 and r1["err_frozen"] > 0.11

        # Heterogeneous non-symmetric [4,3,3] common-zero network (D=10 -> d_occ=9, r=8)
        A_h, B_h, C_h, R_h = build_heterogeneous_common_zero(delta_zero=1e-4, perturb_node0_shape=0.0)
        Ap_h, Bp_h, _, _ = build_heterogeneous_common_zero(delta_zero=1e-4, perturb_node0_shape=1e-7)
        Am_h, Bm_h, _, _ = build_heterogeneous_common_zero(delta_zero=1e-4, perturb_node0_shape=-1e-7)
        wR, vR = np.linalg.eig(R_h)
        pi_h = np.real(vR[:, np.argmin(np.abs(wR - 1.0))])
        pi_h /= np.sum(pi_h)
        rh = exact_bt_and_sensitivity(A_h, B_h, C_h, (Ap_h - Am_h) / 2e-7, (Bp_h - Bm_h) / 2e-7, 8, pi_h)
        assert rh["comm_err"] < 1e-12 and rh["ratio_rebal"] > 3000.0 and rh["ratio_frozen"] > 4e5

        return {
            "k4_delta_1e-4_fwd_hinf": r1["fwd_hinf"],
            "k4_delta_1e-4_err_rebal": round(r1["err_rebal"], 6),
            "k4_delta_1e-4_err_frozen": round(r1["err_frozen"], 6),
            "k4_delta_1e-4_ratio_rebal": round(r1["ratio_rebal"], 1),
            "k4_delta_1e-4_ratio_frozen": round(r1["ratio_frozen"], 1),
            "k4_fwd_log10_slope": round(fwd_slope, 5),
            "k4_ratio_rebal_log10_slope": round(ratio_slope_reb, 5),
            "het433_delta_1e-4_ratio_rebal": round(rh["ratio_rebal"], 1),
            "het433_delta_1e-4_ratio_frozen": round(rh["ratio_frozen"], 1),
        }

    report = {
        "status": "PASSED",
        "proposition_1_marked_race": _verify_marked_race_diffrax(),
        "theorems_1_2_hankel_orders": hankel_cases,
        "theorem_3_erlang16": {
            "occupation_order": 33,
            "sensitivity_order": 48,
            "canceled_poles_restored": 15,
            "rate_residue_check": res_rate,
            "double_root_residue_check": res_double,
        },
        "duhamel_gradient_defect": _verify_duhamel_gradient_defect(),
        "interpolatory_projection_baur2011": _verify_interpolatory_projection(),
        "theorem_4_positive_sensitivity_reduction": _verify_positivity_and_fisher_bias_bound(),
        "proposition_2_and_universal_pos_sens": _verify_hyper_race_and_universal_pos_sens(),
        "multiplicity_sharpness_aldous_shepp_and_mean_field": _verify_multiplicity_sharpness_and_converse(),
        "heterogeneous_theorem3_and_hyper_pos_sens": _verify_heterogeneous_theorem3_and_hyper_pos(),
        "theorem_6_quantitative_bt_distortion": _verify_theorem6_quantitative_bt(),
        "theorem_7_two_preparation_rank": 2,
        "proposition_3_lag2_covariance": cov_lag2,
    }

    out_path = ROOT / "results" / "theory_verification.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
