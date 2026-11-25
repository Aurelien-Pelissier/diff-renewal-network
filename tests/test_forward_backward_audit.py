"""Regression tests for forward, backward, finite-difference, and gradient-health audit.

Guards against:
1. Zero-gradient bilinear saddle in EdgeMemoryHyperDRN (`init_edge_memory_params`).
2. Dead parameter slices in `baselines.py` (`s4d_ode` `omega`), unclipped RK4 NaN gradients.
3. `laplace_solver.py` `lam_f < lam_s` rate ordering and zero-prior frailty posterior hijack.
4. `traits.py` `_shape_center` upper-boundary clamping (e.g., `K=11, k=10.5`) and boundary JVP.
5. `reduction.py` sink-node NaN gradient poisoning of `psi_univ` and `stages_full != 16`.
6. `memory.py` non-unit `clock` fast-mode `direct` scaling.
"""

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np

import diff_renewal as dr
from diff_renewal.traits import _shape_center


@dataclass(frozen=True)
class _AuditPlan:
    traits: np.ndarray
    baseline: np.ndarray
    ages: np.ndarray
    exposure: np.ndarray
    sample: np.ndarray


def _fd_rel_err(loss_fn, w0: np.ndarray, eps: float = 1e-6, seed: int = 0) -> tuple[np.ndarray, float]:
    _, g = jax.value_and_grad(loss_fn)(jnp.asarray(w0, dtype=jnp.float64))
    g_np = np.asarray(g, dtype=np.float64)
    v = np.random.default_rng(seed).normal(size=w0.shape)
    v /= max(np.linalg.norm(v), 1e-12)
    fd = (float(loss_fn(jnp.asarray(w0 + eps * v))) - float(loss_fn(jnp.asarray(w0 - eps * v)))) / (2.0 * eps)
    ad = float(np.dot(g_np, v))
    return g_np, abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)


def test_edge_memory_init_breaks_bilinear_saddle_and_preserves_rank():
    edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 4))
    n_nodes = 5
    stages, rank = 4, 2
    w0 = dr.init_edge_memory_params(n_nodes, len(edges), stages=stages, rank=rank, base_mean=5.0, seed=17)
    assert w0.size == dr.edge_memory_param_size(n_nodes, len(edges), stages=stages, rank=rank)

    def loss_fn(w):
        gen, fr, ag, rd = dr.build_edge_memory_hyper_generator(
            w, edges, n_nodes, stages=stages, rank=rank, aged_horizon=50.0
        )
        e_h = jax.scipy.linalg.expm(1.5 * gen)
        pf = rd @ (e_h @ fr[:, :4])
        pa = rd @ (e_h @ ag[:, 1:4])
        wt_f = jnp.linspace(0.2, 1.0, pf.size).reshape(pf.shape)
        wt_a = jnp.linspace(0.3, 1.1, pa.size).reshape(pa.shape)
        return jnp.sum(pf * wt_f) + jnp.sum(pa * wt_a)

    g, rel = _fd_rel_err(loss_fn, w0)
    assert np.all(np.isfinite(g))
    assert rel < 1e-5
    # All e_emb parameters (last rank * n_edges entries) must receive strictly nonzero gradients!
    e_emb_g = g[-rank * len(edges) :]
    assert np.all(np.abs(e_emb_g) > 0.0)


def test_baselines_zero_dead_params_and_s4d_stability():
    plan = _AuditPlan(
        traits=np.array([-0.5, 0.2, 0.8]),
        baseline=np.array([1.0, 0.5, -0.3]),
        ages=np.array([0.0, 0.75, 1.5]),
        exposure=np.array([[0.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 0.0, 0.2], [0.0, 0.5, 0.0]]),
        sample=np.array([0, 2, 3]),
    )
    counts = jnp.asarray(np.random.default_rng(7).dirichlet([2.0, 2.0, 2.0], size=(3, 3, 3)) * 50.0)

    for kind in ("latent_ode", "ltc_ode", "s4d_ode", "neural_cme", "mamba_ode"):
        params = dr.init_population_baseline(kind, seed=17)
        flat0, unpack = ravel_pytree(params)

        def loss_fn(w):
            p = dr.predict_population_baseline(kind, unpack(w), plan)
            p = jnp.maximum(p, 1e-12)
            p = p / jnp.sum(p, axis=-1, keepdims=True)
            return -jnp.sum(counts * jnp.log(p)) / jnp.sum(counts)

        g, rel = _fd_rel_err(loss_fn, np.asarray(flat0))
        assert np.all(np.isfinite(g))
        assert rel < 1e-5
        assert int(np.sum(np.abs(g) == 0.0)) == 0, f"{kind} has dead zero-gradient parameters"

    # Specifically test S4D with large log_rates=5.0: must remain finite (no RK4 grad NaN!)
    p_s4d = list(dr.init_population_baseline("s4d_ode", seed=17))
    p_s4d[2] = jnp.full_like(p_s4d[2], 5.0)
    flat_hi, unp_hi = ravel_pytree(tuple(p_s4d))
    _, g_hi = jax.value_and_grad(
        lambda w: jnp.sum(dr.predict_population_baseline("s4d_ode", unp_hi(w), plan) ** 2)
    )(flat_hi)
    assert np.all(np.isfinite(np.asarray(g_hi)))


def test_laplace_solver_arbitrary_rate_order_and_zero_prior_frailty():
    # 1. lam_f < lam_s exact aged phases, log_surv, and nonzero grad w.r.t. lam_f!
    lam_f = jnp.array([0.5, 3.0])
    lam_s = jnp.array([2.0, 1.0])
    alpha_f = jnp.array([0.4, 0.4])
    b_slow = jnp.array([[0.3, 0.7], [0.5, 0.5]])
    act = jnp.array([1.0, 1.0])
    age = 2.0

    def f_age(lf):
        pf, ps, ls = dr.hyper_erlang_aged_phases_and_log_surv(lf, lam_s, alpha_f, b_slow, age, act)
        return jnp.sum(pf) + jnp.sum(ls)

    pf, ps, log_s = dr.hyper_erlang_aged_phases_and_log_surv(lam_f, lam_s, alpha_f, b_slow, age, act)
    # Exact analytical check for node 0: S_fast = 0.4*exp(-1.0), S_slow = 0.6*exp(-4.0)*(0.3*(1+4) + 0.7) = 0.6*exp(-4)*2.2
    s_f0 = 0.4 * np.exp(-1.0)
    s_s0 = 0.6 * np.exp(-4.0) * (0.3 * 5.0 + 0.7)
    np.testing.assert_allclose(float(pf[0]), s_f0 / (s_f0 + s_s0), rtol=1e-12)
    np.testing.assert_allclose(float(log_s[0]), np.log(s_f0 + s_s0), rtol=1e-12)
    g_lf = np.asarray(jax.grad(f_age)(lam_f))
    assert np.all(np.isfinite(g_lf)) and abs(g_lf[0]) > 1e-4

    # 2. Zero-prior frailty class must NEVER hijack aged posterior at large age!
    r_mat = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    gammas = jnp.array([0.05, 1.0])
    pi_w = jnp.array([[0.0, 1.0], [0.0, 1.0]])
    _, p_ag = dr.solve_frailty_renewal_talbot(
        jnp.array([2.0, 2.0]), jnp.array([1.0, 1.0]), jnp.array([0.3, 0.3]),
        jnp.array([[0.5, 0.5], [0.5, 0.5]]), r_mat, gammas, pi_w,
        np.array([0.5, 2.0]), act, age=30.0, n_talbot=12, return_both_fresh_and_aged=True,
    )
    _, p_ag_ref = dr.solve_renewal_talbot(
        jnp.array([2.0, 2.0]), jnp.array([1.0, 1.0]), jnp.array([0.3, 0.3]),
        jnp.array([[0.5, 0.5], [0.5, 0.5]]), r_mat, np.array([0.5, 2.0]),
        act, age=30.0, n_talbot=12, return_both_fresh_and_aged=True,
    )
    np.testing.assert_allclose(np.asarray(p_ag), np.asarray(p_ag_ref), atol=1e-10)


def test_traits_shape_center_high_shape_and_boundary_jvp():
    # K=11 at k=10.5 must NOT clamp early and must match FD!
    k_val = jnp.array(10.5)
    c_val = float(_shape_center(k_val, 11, 0.5))
    assert c_val > 10.0
    g_ad = float(jax.grad(lambda x: _shape_center(x, 11, 0.5))(k_val))
    eps = 1e-6
    g_fd = (float(_shape_center(k_val + eps, 11, 0.5)) - float(_shape_center(k_val - eps, 11, 0.5))) / (2 * eps)
    np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5)

    # Strictly outside clamped bounds (e.g., shape=0.9 or shape=18.0 for K=16) must give exact 0.0 JVP!
    assert float(jax.grad(lambda x: _shape_center(x, 16, 0.5))(jnp.array(0.9))) == 0.0
    assert float(jax.grad(lambda x: _shape_center(x, 16, 0.5))(jnp.array(18.0))) == 0.0


def test_reduction_sink_nan_mask_and_stages_full_flexibility():
    # 1. calibrate_universal_pos_clock must support stages_full != 16
    psi_u, cert = dr.calibrate_universal_pos_clock(stages_red=6, stages_full=12, maxiter=10)
    assert np.all(np.isfinite(psi_u)) and cert["stages_full"] == 12

    # 2. Placeholder 0.0 mean/shape on sink node 2 must NEVER poison psi_univ gradients with NaN!
    edges = ((0, 1), (1, 2))
    g_pos = jax.grad(
        lambda psi: jnp.sum(
            dr.build_universal_pos_network_generator(
                jnp.array([1.5, 2.0, 0.0]), jnp.array([4.0, 5.0, 0.0]),
                jnp.array([1.0, 1.0]), edges, 3, psi, stages_red=6
            )[0] ** 2
        )
    )(jnp.asarray(psi_u))
    assert np.all(np.isfinite(np.asarray(g_pos)))

    g_hyp = jax.grad(
        lambda psi: jnp.sum(
            dr.build_universal_hyper_pos_network_generator(
                jnp.array([0.3, 0.4, 0.0]), jnp.array([2.0, 3.0, 0.0]),
                jnp.array([4.0, 5.0, 0.0]), jnp.array([0.4, 0.5, 0.0]),
                jnp.array([1.0, 1.0]), edges, 3, psi, stages_red=6
            )[0] ** 2
        )
    )(jnp.asarray(psi_u))
    assert np.all(np.isfinite(np.asarray(g_hyp)))


def test_memory_direct_term_scales_with_nonunit_clock():
    ts = jnp.linspace(0.0, 2.0, 11)
    # Fast mode lam=1000, w=1000 -> removed mass M = 1.0. Under clock c=2.0, z -> u / (c * lam) => mem -> M*u/c
    sol_fast = dr.solve_memory(
        lambda t, y, mem, a: -y + mem,
        lambda t, y, a: 1.0,
        y0=0.0,
        ts=ts,
        rates=jnp.array([1000.0]),
        weights=jnp.array([1000.0]),
        clock=lambda t, y, a: 2.0,
    )
    sol_direct = dr.solve_memory(
        lambda t, y, mem, a: -y + mem,
        lambda t, y, a: 1.0,
        y0=0.0,
        ts=ts,
        rates=jnp.array([1.0]),
        weights=jnp.array([0.0]),
        direct=1.0,
        clock=lambda t, y, a: 2.0,
    )
    np.testing.assert_allclose(np.asarray(sol_direct.ys[-1]), np.asarray(sol_fast.ys[-1]), rtol=1e-2)
