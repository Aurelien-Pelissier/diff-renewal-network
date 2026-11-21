"""Unit tests for MeanFieldDRN, matrix-free SparseDRN, and absorbing-sink EdgeMemoryHyperDRN."""

from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    build_edge_memory_hyper_generator,
    edge_memory_param_size,
    moment_matched_logits,
    seicr_mean_field_predict,
    simulate_interacting_seicr_gillespie,
    solve_compact_renewal_network,
    solve_sparse_renewal_network,
)


def test_sparse_renewal_network_matches_dense_and_conserves_mass():
    v_nodes, deg, stages = 8, 2, 16
    dst = np.array([[(u + 1) % v_nodes, (u + 2) % v_nodes] for u in range(v_nodes)], dtype=np.int32)
    edges = tuple((u, int(dst[u, r])) for u in range(v_nodes) for r in range(deg))
    rng = np.random.default_rng(12)
    means = jnp.asarray(rng.uniform(1.3, 2.7, size=(v_nodes, deg)))
    shapes = jnp.asarray(rng.uniform(4.5, 9.5, size=(v_nodes, deg)))
    init = jnp.zeros(v_nodes).at[0].set(1.0)
    ts = jnp.linspace(0.5, 3.5, 6)
    weights = jnp.asarray(rng.uniform(0.2, 1.2, size=(len(ts), v_nodes)))

    def f_dense(th):
        m = jnp.exp(th[: v_nodes * deg])
        k = jnp.exp(th[v_nodes * deg :])
        ys = solve_compact_renewal_network(m, moment_matched_logits(k), edges, init, ts).ys
        return jnp.sum(ys * weights)

    def f_sparse(th):
        m = jnp.exp(th[: v_nodes * deg]).reshape(v_nodes, deg)
        k = jnp.exp(th[v_nodes * deg :]).reshape(v_nodes, deg)
        ys = solve_sparse_renewal_network(m, moment_matched_logits(k), dst, init, ts, rtol=1e-8, atol=1e-10)
        return jnp.sum(ys * weights)

    th0 = jnp.concatenate([jnp.log(means.reshape(-1)), jnp.log(shapes.reshape(-1))])
    vd, gd = jax.value_and_grad(f_dense)(th0)
    vs, gs = jax.value_and_grad(f_sparse)(th0)
    assert abs(float(vd - vs)) < 1e-8
    assert float(jnp.linalg.norm(gd - gs) / jnp.linalg.norm(gd)) < 1e-7


def test_mean_field_seicr_positivity_grad_and_gillespie_agreement():
    ts = jnp.linspace(1.0, 10.0, 10)

    def obj(th):
        p5, x = seicr_mean_field_predict(
            th[0], th[1], th[2], th[3], th[4], th[5], th[6], th[7], th[8], ts, stages=6
        )
        return jnp.sum(p5[:, 2] ** 2), (p5, x)

    th0 = jnp.array([2.4, 1.2, 3.5, 2.2, 5.0, 1.8, 3.0, 0.03, 0.02])
    (val, (p5, x)), grad = jax.value_and_grad(obj, has_aux=True)(th0)
    assert np.isfinite(float(val)) and np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.min(x)) >= -1e-7
    assert float(jnp.max(jnp.abs(jnp.sum(p5, axis=-1) - 1.0))) < 1e-12

    # Stochastic Gillespie average at M=3000 stays close to deterministic MeanFieldDRN
    sims = [
        simulate_interacting_seicr_gillespie(
            2.4, 1.2, 3.5, 2.2, 5.0, 1.8, 3.0, 0.03, 0.02, np.asarray(ts), 3000, stages=6, seed=s
        )
        for s in range(4)
    ]
    sim_mean = np.mean(sims, axis=0)
    assert float(np.max(np.abs(sim_mean - np.asarray(p5)))) < 0.04


def test_edge_memory_hyper_with_absorbing_sink():
    edges = ((0, 1), (1, 2), (2, 1), (1, 3), (2, 3))  # node 3 is absorbing sink (out-degree 0)
    n_p = edge_memory_param_size(4, len(edges), stages=4)
    gen, fresh, aged, read = map(
        np.asarray, build_edge_memory_hyper_generator(jnp.zeros(n_p), edges, 4, stages=4)
    )
    off = gen - np.diag(np.diag(gen))
    assert float(off.min()) >= 0.0
    assert float(np.max(np.abs(gen.sum(axis=0)))) < 1e-13
    assert float(np.max(np.abs((read @ fresh) - np.eye(4)))) < 1e-13
    assert float(np.max(np.abs((read @ aged) - np.eye(4)))) < 1e-13
