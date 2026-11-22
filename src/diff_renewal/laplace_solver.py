"""Stiffness- and stage-count-independent Laplace-Schur renewal solvers.

Exploits the exact renewal resolvent identity from Theorem 3,
    P(s) = diag(S_i(s)) * (I_V - R * diag(phi_i(s)))^{-1},
to collapse a D_st = sum_i n_i phase-type network (arbitrary stage count K and
arbitrary fast/slow rate stiffness ratio lambda_f / lambda_s) into V x V node-level
solves:
  1. `solve_renewal_talbot`: Weideman (2006) optimal fixed-Talbot contour quadrature
     computing exact fresh AND aged occupancies simultaneously from a single shared
     resolvent inverse Z(s) = (I_V - R * diag(phi(s)))^{-1}.
  2. `solve_frailty_renewal_talbot`: G-component persistent latent-speed Frailty-DRN
     operationalizing Proposition 3 with exact Bayesian frailty conditioning
     P(g | T > a) proportional to pi_g * S_g(a) (persisting across transitions!).
  3. `solve_coxian_hyper_talbot`: Exact Laplace-Schur-Talbot solver for arbitrary
     non-uniform Coxian + fast exponential clocks (e.g. Universal Hyper-Pos-Sens
     plugged zero-shot into large-V graphs).
  4. `laplace_schur_sdirk2_solve`: Exact block-bidiagonal Schur-complement implicit
     2-stage L-stable SDIRK2 time-stepper for (I - gamma*h*A) X = Y.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np


def weideman_talbot_contour(
    horizons: np.ndarray | jax.Array,
    *,
    n_nodes: int = 16,
) -> tuple[jax.Array, jax.Array]:
    """Build Weideman (2006) optimal fixed-Talbot upper-half-plane contour nodes & weights.

    Uses conjugate symmetry of real-valued probability trajectories: evaluating only the
    M = n_nodes // 2 upper-half-plane nodes and taking Re(sum(w * F(s))) recovers the
    exact Bromwich inverse Laplace integral to ~1e-8 (n_nodes=16) or ~1e-10 (n_nodes=20).
    """
    h_arr = np.asarray(horizons, dtype=np.float64)
    m_half = int(n_nodes) // 2
    step = 2.0 * np.pi / float(n_nodes)
    theta = (np.arange(m_half, dtype=np.float64) + 0.5) * step
    cot_t = 1.0 / np.tan(0.6407 * theta)
    csc2_t = 1.0 + cot_t**2
    z_unit = -0.6122 + 0.5017 * theta * cot_t + 0.2645j * theta
    dz_unit = 0.5017 * (cot_t - 0.6407 * theta * csc2_t) + 0.2645j

    s_all = []
    w_all = []
    for t_val in h_arr:
        scale = float(n_nodes) / float(t_val)
        s = scale * z_unit
        w = 2.0 * (step / (2.0j * np.pi)) * scale * dz_unit * np.exp(s * float(t_val))
        s_all.append(s)
        w_all.append(w)
    return (
        jnp.asarray(np.stack(s_all, axis=0), dtype=jnp.complex128),
        jnp.asarray(np.stack(w_all, axis=0), dtype=jnp.complex128),
    )


def hyper_erlang_aged_phases_and_log_surv(
    lam_f: jax.Array,
    lam_s: jax.Array,
    alpha_f: jax.Array,
    b_slow: jax.Array,
    age: float,
    active_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Exact overflow-free aged phase weights (p_fast, p_slow) and log survival log S_v(a).

    Assumes ordered rates lam_f >= lam_s >= 0 on active nodes and lam_f = lam_s = 0 on
    absorbing nodes (active_mask == 0).
    """
    k_slow = int(b_slow.shape[-1])
    if age <= 0.0:
        p_f = alpha_f * active_mask
        p_s = (1.0 - alpha_f)[:, None] * b_slow * active_mask[:, None]
        log_s = jnp.zeros_like(lam_f)
        return p_f, p_s, log_s

    js = jnp.arange(k_slow, dtype=lam_f.dtype)
    diff_idx = js[:, None] - js[None, :]
    valid = diff_idx >= 0
    d_safe = jnp.where(valid, diff_idx, 0.0)
    xa = jnp.maximum(lam_s * age, 1e-12)
    log_p = (
        d_safe[None, :, :] * jnp.log(xa)[:, None, None]
        - jax.scipy.special.gammaln(d_safe + 1.0)[None, :, :]
    )
    m_sh = jnp.maximum(
        jnp.max(jnp.where(valid[None, :, :], log_p, -1e9), axis=(1, 2)),
        0.0,
    )
    p_mat = jnp.where(
        valid[None, :, :],
        jnp.exp(jnp.clip(log_p - m_sh[:, None, None], -700.0, 0.0)),
        0.0,
    )
    # Factor out exp(-lam_s * age + m_sh):
    # S_v(a) = exp(-lam_s*a + m_sh) * [alpha*exp(-(lam_f-lam_s)*a - m_sh) + (1-alpha)*sum(P_mat @ b_slow)]
    slow_un = (1.0 - alpha_f)[:, None] * jnp.einsum("vji,vi->vj", p_mat, b_slow)
    df = jnp.maximum(lam_f - lam_s, 0.0) * age
    fast_un = alpha_f * jnp.exp(-jnp.clip(df + m_sh, 0.0, 700.0))
    total_un = jnp.maximum(fast_un + jnp.sum(slow_un, axis=-1), 1e-300)

    p_f = jnp.where(active_mask > 0.5, fast_un / total_un, 0.0)
    p_s = jnp.where(active_mask[:, None] > 0.5, slow_un / total_un[:, None], 0.0)
    log_surv = jnp.where(
        active_mask > 0.5,
        -lam_s * age + m_sh + jnp.log(total_un),
        0.0,
    )
    return p_f, p_s, log_surv


def _hyper_erlang_laplace(
    s_c: jax.Array,
    lam_f: jax.Array,
    lam_s: jax.Array,
    p_f: jax.Array,
    p_s: jax.Array,
    active_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact Laplace transform phi_v(s) and survival transform S_v(s) for 2-branch Hyper-Erlang."""
    k_slow = int(p_s.shape[-1])
    qf = lam_f / (s_c + lam_f)
    qs = lam_s / (s_c + lam_s)
    # Entry at stage j (0..K-1) needs (K - j) exponential transitions to exit!
    pows = jnp.stack([qs ** (k_slow - j) for j in range(k_slow)], axis=-1)
    phi = (p_f * qf + jnp.sum(p_s * pows, axis=-1)) * active_mask
    s_v = (1.0 - phi) / s_c
    return phi, s_v


def solve_renewal_talbot(
    lam_f: jax.Array,
    lam_s: jax.Array,
    alpha_f: jax.Array,
    b_slow: jax.Array,
    routing_mat: jax.Array,
    horizons: np.ndarray | jax.Array,
    active_mask: jax.Array,
    *,
    eval_nodes: jax.Array | None = None,
    age: float = 0.0,
    n_talbot: int = 16,
    return_both_fresh_and_aged: bool = False,
):
    """Exact stiffness- and K-independent Laplace-Schur-Talbot renewal network solver.

    Computes P(t)[u, v] = P(X(t) = v | X(0) = u) for all eval_nodes u and all horizons t
    simultaneously from a single shared V x V resolvent inverse Z(s) = (I - R diag(phi(s)))^{-1}.
    When `return_both_fresh_and_aged=True`, returns `(P_fresh, P_aged)` at ZERO extra matrix solves!
    """
    v_cnt = int(routing_mat.shape[0])
    ev_idx = jnp.arange(v_cnt, dtype=jnp.int32) if eval_nodes is None else eval_nodes
    s_grid, w_grid = weideman_talbot_contour(horizons, n_nodes=n_talbot)

    pf_fresh = alpha_f * active_mask
    ps_fresh = (1.0 - alpha_f)[:, None] * b_slow * active_mask[:, None]
    need_aged = return_both_fresh_and_aged or (age > 0.0)
    if need_aged:
        pf_aged, ps_aged, _ = hyper_erlang_aged_phases_and_log_surv(
            lam_f, lam_s, alpha_f, b_slow, age, active_mask
        )
    else:
        pf_aged, ps_aged = pf_fresh, ps_fresh

    eye_v = jnp.eye(v_cnt, dtype=jnp.complex128)
    r_c = routing_mat.astype(jnp.complex128)

    def one_s(s_c):
        phi, s_v = _hyper_erlang_laplace(s_c, lam_f, lam_s, pf_fresh, ps_fresh, active_mask)
        m_mat = eye_v - r_c * phi[None, :]
        z_inv = jnp.linalg.solve(m_mat, eye_v)  # (V, V), col u = start u
        p_fr = s_v[:, None] * z_inv  # (V, V) [v, u]
        if not need_aged:
            return p_fr, p_fr
        phi_a, s_va = _hyper_erlang_laplace(s_c, lam_f, lam_s, pf_aged, ps_aged, active_mask)
        # P^(a)(s) = diag(S^(a)) + diag(S) * Z * R * diag(phi^(a))
        zr = z_inv @ r_c
        p_ag = jnp.diag(s_va) + (s_v[:, None] * zr) * phi_a[None, :]
        return p_fr, p_ag

    def one_h(s_vec, w_vec):
        p_frs, p_ags = jax.vmap(one_s)(s_vec)  # (M, V, V)
        fr_h = jnp.real(jnp.einsum("m,mvu->uv", w_vec, p_frs))[ev_idx]
        fr_h = jnp.maximum(fr_h, 1e-12)
        fr_h = fr_h / jnp.sum(fr_h, axis=-1, keepdims=True)
        if not need_aged:
            return fr_h, fr_h
        ag_h = jnp.real(jnp.einsum("m,mvu->uv", w_vec, p_ags))[ev_idx]
        ag_h = jnp.maximum(ag_h, 1e-12)
        ag_h = ag_h / jnp.sum(ag_h, axis=-1, keepdims=True)
        return fr_h, ag_h

    fr_all, ag_all = jax.vmap(one_h)(s_grid, w_grid)  # (H, B_ev, V)
    fr_out = jnp.swapaxes(fr_all, 0, 1)  # (B_ev, H, V)
    ag_out = jnp.swapaxes(ag_all, 0, 1)
    if return_both_fresh_and_aged:
        return fr_out, ag_out
    return ag_out if age > 0.0 else fr_out


def solve_frailty_renewal_talbot(
    lam_f: jax.Array,
    lam_s: jax.Array,
    alpha_f: jax.Array,
    b_slow: jax.Array,
    routing_mat: jax.Array,
    frailty_multipliers: jax.Array,
    frailty_weights: jax.Array,
    horizons: np.ndarray | jax.Array,
    active_mask: jax.Array,
    *,
    eval_nodes: jax.Array | None = None,
    age: float = 0.0,
    n_talbot: int = 16,
    return_both_fresh_and_aged: bool = False,
):
    """Exact G-component persistent latent-speed Frailty-DRN (operationalizing Proposition 3).

    Parameters
    ----------
    frailty_multipliers : (G,) positive global speed multipliers gamma_g
    frailty_weights : (V, G) simplex prior weights pi_{v, g} per starting node v
    """
    v_cnt = int(routing_mat.shape[0])
    ev_idx = jnp.arange(v_cnt, dtype=jnp.int32) if eval_nodes is None else eval_nodes

    def one_class(gamma_g):
        lf_g = lam_f * gamma_g
        ls_g = lam_s * gamma_g
        p_fr, p_ag = solve_renewal_talbot(
            lf_g,
            ls_g,
            alpha_f,
            b_slow,
            routing_mat,
            horizons,
            active_mask,
            eval_nodes=ev_idx,
            age=age,
            n_talbot=n_talbot,
            return_both_fresh_and_aged=True,
        )
        _, _, log_s = hyper_erlang_aged_phases_and_log_surv(
            lf_g, ls_g, alpha_f, b_slow, age, active_mask
        )
        return p_fr, p_ag, log_s[ev_idx]

    # vmap across G persistent speed classes!
    p_fr_g, p_ag_g, log_s_g = jax.vmap(one_class)(frailty_multipliers)  # (G, B_ev, H, V), (G, B_ev)
    pi_ev = frailty_weights[ev_idx]  # (B_ev, G)

    # Fresh mixture: sum_g pi_{u, g} * P_g(t)
    p_fresh = jnp.einsum("bg,gbhv->bhv", pi_ev, p_fr_g)
    p_fresh = jnp.maximum(p_fresh, 1e-12)
    p_fresh = p_fresh / jnp.sum(p_fresh, axis=-1, keepdims=True)

    # Exact Bayesian frailty posterior at age a: w_{u, g}(a) proportional to pi_{u, g} * S_{u, g}(a)!
    log_post = jnp.log(jnp.maximum(pi_ev, 1e-30)) + log_s_g.T  # (B_ev, G)
    post_w = jax.nn.softmax(log_post, axis=-1)
    p_aged = jnp.einsum("bg,gbhv->bhv", post_w, p_ag_g)
    p_aged = jnp.maximum(p_aged, 1e-12)
    p_aged = p_aged / jnp.sum(p_aged, axis=-1, keepdims=True)

    if return_both_fresh_and_aged:
        return p_fresh, p_aged
    return p_aged if age > 0.0 else p_fresh


def solve_coxian_hyper_talbot(
    lam_f: jax.Array,
    stage_rates: jax.Array,
    alpha_f: jax.Array,
    b_slow: jax.Array,
    routing_mat: jax.Array,
    horizons: np.ndarray | jax.Array,
    active_mask: jax.Array,
    *,
    eval_nodes: jax.Array | None = None,
    age: float = 0.0,
    n_talbot: int = 16,
    return_both_fresh_and_aged: bool = False,
):
    """Exact Laplace-Schur-Talbot solver for arbitrary non-uniform Coxian + fast exp clocks.

    Enables zero-shot plugging of Universal Hyper-Pos-Sens (K_r non-uniform Coxian stages +
    1 fast exponential branch) into arbitrary large-V graphs without dense D_st x D_st matrices.
    """
    v_cnt = int(routing_mat.shape[0])
    k_red = int(stage_rates.shape[-1])
    ev_idx = jnp.arange(v_cnt, dtype=jnp.int32) if eval_nodes is None else eval_nodes
    s_grid, w_grid = weideman_talbot_contour(horizons, n_nodes=n_talbot)

    pf_fresh = alpha_f * active_mask
    ps_fresh = (1.0 - alpha_f)[:, None] * b_slow * active_mask[:, None]
    need_aged = return_both_fresh_and_aged or (age > 0.0)

    if need_aged:
        # Exact per-node K_r x K_r matrix exponential (K_r=8 -> tiny 8x8 per node, <0.2ms!)
        def aged_one_node(lf_i, r_i, af_i, bs_i, act_i):
            # Shift by min_rate to prevent underflow for arbitrary age
            min_r = jnp.minimum(lf_i, jnp.min(r_i))
            t_blk = -jnp.diag(r_i - min_r) + jnp.diag(r_i[:-1], k=-1)
            slow_surv = (1.0 - af_i) * (jl.expm(age * t_blk) @ bs_i)
            fast_surv = af_i * jnp.exp(-jnp.clip((lf_i - min_r) * age, 0.0, 700.0))
            tot = jnp.maximum(fast_surv + jnp.sum(slow_surv), 1e-300)
            pf_a = jnp.where(act_i > 0.5, fast_surv / tot, 0.0)
            ps_a = jnp.where(act_i > 0.5, slow_surv / tot, jnp.zeros(k_red, dtype=r_i.dtype))
            return pf_a, ps_a

        pf_aged, ps_aged = jax.vmap(aged_one_node)(
            lam_f, stage_rates, alpha_f, b_slow, active_mask
        )
    else:
        pf_aged, ps_aged = pf_fresh, ps_fresh

    eye_v = jnp.eye(v_cnt, dtype=jnp.complex128)
    r_c = routing_mat.astype(jnp.complex128)

    def cox_laplace(s_c, p_f, p_s):
        qf = lam_f / (s_c + lam_f)
        # Suffix cumulative product of r_m / (s + r_m) from m = j .. K_r - 1
        step_q = stage_rates / (s_c + stage_rates)  # (V, K_r)
        # Reverse cumprod along axis=-1
        suf_q = jnp.cumprod(step_q[:, ::-1], axis=-1)[:, ::-1]
        phi = (p_f * qf + jnp.sum(p_s * suf_q, axis=-1)) * active_mask
        s_v = (1.0 - phi) / s_c
        return phi, s_v

    def one_s(s_c):
        phi, s_v = cox_laplace(s_c, pf_fresh, ps_fresh)
        m_mat = eye_v - r_c * phi[None, :]
        z_inv = jnp.linalg.solve(m_mat, eye_v)
        p_fr = s_v[:, None] * z_inv
        if not need_aged:
            return p_fr, p_fr
        phi_a, s_va = cox_laplace(s_c, pf_aged, ps_aged)
        zr = z_inv @ r_c
        p_ag = jnp.diag(s_va) + (s_v[:, None] * zr) * phi_a[None, :]
        return p_fr, p_ag

    def one_h(s_vec, w_vec):
        p_frs, p_ags = jax.vmap(one_s)(s_vec)
        fr_h = jnp.real(jnp.einsum("m,mvu->uv", w_vec, p_frs))[ev_idx]
        fr_h = jnp.maximum(fr_h, 1e-12)
        fr_h = fr_h / jnp.sum(fr_h, axis=-1, keepdims=True)
        if not need_aged:
            return fr_h, fr_h
        ag_h = jnp.real(jnp.einsum("m,mvu->uv", w_vec, p_ags))[ev_idx]
        ag_h = jnp.maximum(ag_h, 1e-12)
        ag_h = ag_h / jnp.sum(ag_h, axis=-1, keepdims=True)
        return fr_h, ag_h

    fr_all, ag_all = jax.vmap(one_h)(s_grid, w_grid)
    fr_out = jnp.swapaxes(fr_all, 0, 1)
    ag_out = jnp.swapaxes(ag_all, 0, 1)
    if return_both_fresh_and_aged:
        return fr_out, ag_out
    return ag_out if age > 0.0 else fr_out


def laplace_schur_implicit_solve(
    rhs_y: jax.Array,
    gamma_h: float,
    lam_f: jax.Array,
    lam_s: jax.Array,
    alpha_f: jax.Array,
    b_slow: jax.Array,
    routing_mat: jax.Array,
) -> jax.Array:
    """Exact block-bidiagonal Laplace-Schur solve of (I - gamma_h * A) X = rhs_y.

    Reduces the D_st = V*(K+1) linear system exactly to a V x V M-matrix system
    (I_V - diag(phi_v(1 / gamma_h)) * R) f = f_0, sharing one LU across batch dimension B.
    """
    v_cnt = int(routing_mat.shape[0])
    k_slow = int(b_slow.shape[-1])

    def _inv_local(y_arr):
        zf = y_arr[:, :, 0] / (1.0 + gamma_h * lam_f[None, :])
        inv_d = 1.0 / (1.0 + gamma_h * lam_s[None, :])
        q_step = (gamma_h * lam_s[None, :]) * inv_d
        zs = [y_arr[:, :, 1] * inv_d]
        for j in range(1, k_slow):
            zs.append(y_arr[:, :, 1 + j] * inv_d + q_step * zs[-1])
        z_slow = jnp.stack(zs, axis=-1)
        z_all = jnp.concatenate([zf[:, :, None], z_slow], axis=-1)
        exit_flux = lam_f[None, :] * zf + lam_s[None, :] * zs[-1]
        return z_all, exit_flux

    qf = (gamma_h * lam_f) / (1.0 + gamma_h * lam_f)
    qs = (gamma_h * lam_s) / (1.0 + gamma_h * lam_s)
    rem_pows = jnp.stack([qs ** (k_slow - j) for j in range(k_slow)], axis=-1)
    phi = alpha_f * qf + (1.0 - alpha_f) * jnp.sum(b_slow * rem_pows, axis=-1)

    z0, f0 = _inv_local(rhs_y)
    m_mat = jnp.eye(v_cnt, dtype=rhs_y.dtype) - phi[:, None] * routing_mat
    f_sol = jnp.linalg.solve(m_mat, f0.T).T
    inflow = gamma_h * (f_sol @ routing_mat.T)
    b_in = jnp.concatenate(
        [
            (inflow * alpha_f[None, :])[:, :, None],
            (inflow * (1.0 - alpha_f)[None, :])[:, :, None] * b_slow[None, :, :],
        ],
        axis=-1,
    )
    z_in, _ = _inv_local(b_in)
    return z0 + z_in
