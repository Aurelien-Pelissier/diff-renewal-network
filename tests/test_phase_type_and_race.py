"""Unit tests for phase-type blocks, marked-race compilation, and compact networks."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from diff_renewal import (
    compact_network_state_size,
    coxian_generator,
    erlang_generator,
    mixed_erlang_components,
    mixed_erlang_moments,
    mixed_erlang_race,
    moment_matched_logits,
    network_state_size,
    phase_type_dynamics,
    smooth_erlang_logits,
    solve_compact_renewal_network,
    solve_erlang_chain,
    solve_renewal_network,
)


def test_phase_type_moments_and_conservation():
    q, alpha = erlang_generator(4, 2.0)
    dp, flux = phase_type_dynamics(q, alpha, jnp.array([0.4, 0.3, 0.2, 0.1]), 0.25)
    assert np.isclose(float(jnp.sum(dp) + flux), 0.25, atol=1e-12)

    logits = smooth_erlang_logits(7.2, stages=16, width=0.5)
    rate, entry = mixed_erlang_components(2.5, logits)
    mean, var, shape = mixed_erlang_moments(rate, entry)
    assert np.isclose(float(mean), 2.5, atol=1e-12)
    assert float(var) > 0.0 and 6.0 < float(shape) < 8.5

    q_cox, a_cox = coxian_generator(jnp.zeros(5), jnp.zeros(4))
    assert q_cox.shape == (5, 5) and float(jnp.sum(a_cox)) == 1.0


def test_marked_race_and_compact_network_exactness():
    edges = ((0, 1), (0, 2), (1, 3), (2, 3))
    assert network_state_size(edges, 4, 12) == 12**2 + 12 + 12 + 1
    assert compact_network_state_size(edges, 4, 12) == 4 + 11 * 4

    means = jnp.array([1.6, 2.2, 1.4, 1.9])
    logits = moment_matched_logits(jnp.array([4.5, 7.5, 5.5, 9.0]))
    rate, cont, causes = mixed_erlang_race(means[:2], logits[:2])
    assert float(rate) > 0.0
    row_sums = jnp.sum(causes, axis=0) + jnp.pad(cont, (0, 1))
    np.testing.assert_allclose(np.asarray(row_sums), 1.0, atol=1e-12)

    ts = jnp.linspace(0.5, 5.0, 10)
    init = jnp.array([1.0, 0.0, 0.0, 0.0])
    y_full = solve_renewal_network(means, logits, edges, init, ts).ys
    y_comp = solve_compact_renewal_network(means, logits, edges, init, ts).ys
    np.testing.assert_allclose(np.asarray(y_full), np.asarray(y_comp), atol=1e-8)

    g_full = jax.grad(lambda m: jnp.sum(solve_renewal_network(m, logits, edges, init, ts).ys[-1, :3] * jnp.array([0.2, 0.5, 0.9])))(means)
    g_comp = jax.grad(lambda m: jnp.sum(solve_compact_renewal_network(m, logits, edges, init, ts).ys[-1, :3] * jnp.array([0.2, 0.5, 0.9])))(means)
    np.testing.assert_allclose(np.asarray(g_full), np.asarray(g_comp), rtol=1e-6, atol=1e-8)


def test_sequential_erlang_chain_mass_conservation():
    means = jnp.array([1.5, 2.5])
    logits = jnp.stack([smooth_erlang_logits(5.0, stages=16), smooth_erlang_logits(9.0, stages=16)])
    ts = jnp.linspace(0.0, 6.0, 13)
    ys = np.asarray(solve_erlang_chain(means, logits, ts).ys)
    assert ys.shape == (13, 3)
    np.testing.assert_allclose(ys.sum(axis=1), 1.0, atol=1e-9)
