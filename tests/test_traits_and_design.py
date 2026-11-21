"""Unit tests for observed traits, Fisher allocation, neural clocks, and memory."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from diff_renewal import (
    cap_mixed_erlang,
    capped_network_tv_bound,
    coarsened_fisher,
    endpoint_coefficients,
    init_clock_network,
    multinomial_nll,
    neural_clock,
    optimize_fisher_allocation,
    smooth_erlang_logits,
    solve_memory,
    static_preserving_cutoff,
    survivor_phases,
    trait_network_probabilities,
)


def test_traits_and_multinomial_loss():
    endpoints = jnp.array(
        [[[0.4, 0.7], [0.5, 0.8], [0.6, 0.9]], [[1.4, 1.8], [1.5, 1.9], [1.6, 2.0]]]
    )
    coeffs = endpoint_coefficients(endpoints)
    traits = jnp.array([[-1.0], [0.0], [1.0]])
    edges = ((0, 1), (1, 2), (2, 0))
    initials = jnp.eye(3)
    ts = jnp.array([1.0, 2.5, 4.0])
    probs = trait_network_probabilities(coeffs, traits, edges, initials, ts)
    assert probs.shape == (3, 3, 3, 3)
    np.testing.assert_allclose(np.asarray(probs.sum(axis=-1)), 1.0, atol=1e-8)

    nll = multinomial_nll(probs, 100.0 * probs)
    assert float(nll) > 0.0


def test_fisher_allocation_and_diagnostics():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(2, 6, 4, 3))
    info = np.einsum("scmp,scmq->scpq", raw, raw)
    pilot = np.broadcast_to(0.5 * np.eye(3), (2, 3, 3)).copy()
    alloc = optimize_fisher_allocation(info, pilot, 7200, [1, 2])
    assert alloc["success"]
    assert int(alloc["counts"].sum()) == 7200
    assert alloc["criterion"] <= alloc["initial_criterion"] + 1e-12

    p = np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    dp = np.array(
        [
            [[0.1, -0.05], [-0.04, 0.02], [-0.06, 0.03]],
            [[-0.08, 0.04], [0.03, -0.01], [0.05, -0.03]],
        ]
    )
    cf = coarsened_fisher(p, dp, groups=[0, 0, 1], sample_sizes=[100, 100])
    np.testing.assert_allclose(cf["fine"], cf["coarse"] + cf["loss"], atol=1e-12)


def test_neural_clock_capping_and_memory_solver():
    net = init_clock_network(jax.random.PRNGKey(3), features=3, stages=16, width=8)
    rate, entry = neural_clock(net, jnp.array([[0.0, -1.0, 0.5], [1.0, 1.0, 0.0]]))
    assert rate.shape == (2,) and entry.shape == (2, 16)
    np.testing.assert_allclose(np.asarray(entry.sum(axis=-1)), 1.0, atol=1e-12)

    surv0 = survivor_phases(rate, entry, 0.0)
    np.testing.assert_allclose(np.asarray(surv0.sum(axis=-1)), 1.0, atol=1e-12)
    surv1 = survivor_phases(rate, entry, 1.2)
    assert np.all(np.asarray(surv1.sum(axis=-1)) < 1.0)

    c_mean, _, tail = cap_mixed_erlang(2.0, smooth_erlang_logits(8.0, stages=16), cap=12)
    assert float(c_mean) <= 2.0 + 1e-12 and 0.0 <= float(tail) <= 1.0
    tv = capped_network_tv_bound([float(tail)] * 3, [4.0, 4.0, 4.0], ((0, 1), (1, 2), (2, 0)), 3.0)
    assert 0.0 <= tv["path_tv_bound"] <= 1.0

    cutoff = static_preserving_cutoff([0.5, 2.0, 10.0], [0.3, 0.4, 0.5], cutoff=3.0)
    assert np.isclose(cutoff.direct, 0.05) and cutoff.screen_bound(1.0) > 0.0

    ts = jnp.linspace(0.0, 3.0, 31)
    g = jax.grad(
        lambda p: solve_memory(
            rhs=lambda t, y, mem, par: -par[0] * y + par[1] * mem,
            source=lambda t, y, par: jnp.tanh(y),
            y0=1.0,
            ts=ts,
            rates=jnp.array([0.4, 1.5]),
            weights=jnp.array([0.6, 0.4]),
            args=p,
        ).ys[-1]
    )(jnp.array([0.8, 0.25]))
    assert g.shape == (2,) and np.all(np.isfinite(np.asarray(g)))
