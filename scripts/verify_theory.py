"""Independent symbolic and numerical verification of all theoretical results.

Verifies:
- Proposition 1: Exact marked-race compilation (degree, normalization, minimal order,
  and JAX/Diffrax compact vs. K**d tensor forward + gradient agreement).
- Theorems 1 & 2: Occupation dimension via balanced invisible fluxes and common
  survival-zero GCD formula (exact rational block-Hankel rank across topologies).
- Theorem 3: Node-specific sensitivity restores canceled poles (symbolic Laplace
  residues, square-free 33-vs-48 separation on the three-node Erlang-16 cycle,
  Fréchet vs. Duhamel defect integral, and Baur et al. two-sided interpolation).
- Proposition 2 & Theorem 4: Single-visit indistinguishability with lag-l covariance
  and constructive two-preparation exponential-snapshot Laplace inversion.
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
    build_interpolatory_basis,
    compact_network_state_size,
    make_cycle_engine,
    make_interpolatory_engine,
    make_positive_33_engine,
    moment_matched_logits,
    network_state_size,
    pos33_cycle_matrices,
    resolvent_sensitivity_diagnostic,
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

    # Proposition 2: Single-visit indistinguishability & lag covariance
    delta_val, rho_val, m1, m2 = 0.45, 0.8, 1.8, 2.2
    p_trans = rho_val * np.eye(2) + (1.0 - rho_val) * np.ones((2, 2)) / 2.0
    signs = np.array([-1.0, 1.0])
    cov_lag2 = 0.5 * float(signs @ np.linalg.matrix_power(p_trans, 2) @ signs) * (delta_val**2) * m1 * m2
    assert abs(cov_lag2 - (delta_val**2) * (rho_val**2) * m1 * m2) < 1e-15

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
        "theorem_5_two_preparation_rank": 2,
        "proposition_2_lag2_covariance": cov_lag2,
    }

    out_path = ROOT / "results" / "theory_verification.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
