"""Unit tests for Positivity-Preserving Sensitivity Reduction, Vectorized Race Solver, and Modern Neural Baselines."""

from dataclasses import dataclass
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np

from diff_renewal import (
    READ_POS33,
    build_edge_memory_hyper_generator,
    build_universal_pos_network_generator,
    calibrate_universal_pos_clock,
    compact_network_state_size,
    edge_memory_param_size,
    hyper_erlang_race,
    hyper_race_state_size,
    init_population_baseline,
    make_baseline_engine,
    make_positive_33_engine,
    moment_matched_logits,
    network_state_size,
    pos33_cycle_matrices,
    predict_population_baseline,
    solve_compact_renewal_network,
    solve_renewal_network,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _DummyPlan:
    traits: np.ndarray
    ages: np.ndarray
    baseline: np.ndarray
    exposure: np.ndarray
    sample: np.ndarray


def test_pos33_metzler_and_simplex_invariance():
    pos_summary = json.loads(
        (ROOT / "results" / "archived" / "positive_sensitivity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    psi_star = np.asarray(pos_summary["psi_star"], dtype=float)
    theta = np.log([1.6, 2.1, 2.7, 4.2, 6.8, 9.5])
    gen, fresh = pos33_cycle_matrices(theta, psi_star)
    gen_np = np.asarray(gen)
    off = gen_np - np.diag(np.diag(gen_np))
    assert off.min() >= 0.0
    assert np.max(np.abs(gen_np.sum(axis=0))) < 1e-13

    for t in (0.01, 0.1, 1.0, 4.0):
        p_t = np.asarray(READ_POS33 @ jl.expm(t * gen) @ fresh)
        assert p_t.min() > 0.0
        assert np.max(np.abs(p_t.sum(axis=0) - 1.0)) < 1e-13

    engine = make_positive_33_engine(psi_star)
    vals, jac = engine(theta)
    assert vals.shape == (14, 3, 3)
    assert jac.shape == (14, 3, 3, 6)
    assert vals.min() > 0.0
    assert np.all(np.isfinite(jac))


def test_vectorized_compact_race_degree3_parity():
    edges = tuple((i, (i + s) % 5) for i in range(5) for s in (1, 2, 3))
    stages = 8
    assert network_state_size(edges, 5, stages) == 5 * (8**3)
    assert compact_network_state_size(edges, 5, stages) == 5 + 7 * 15
    rng = np.random.default_rng(123)
    means = jnp.asarray(rng.uniform(1.3, 2.6, size=15))
    shapes = jnp.asarray(rng.uniform(3.8, 9.5, size=15))
    ts = jnp.linspace(0.4, 3.2, 8)
    init = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    w = jnp.array([0.2, 0.5, 0.9, 1.1, 0.4])
    theta0 = jnp.concatenate([jnp.log(means), jnp.log(shapes)])

    def full_loss(th):
        m, k = jnp.exp(th[:15]), jnp.exp(th[15:])
        ys = solve_renewal_network(m, moment_matched_logits(k), edges, init, ts).ys
        return jnp.sum(ys * w[None, :])

    def comp_loss(th):
        m, k = jnp.exp(th[:15]), jnp.exp(th[15:])
        ys = solve_compact_renewal_network(m, moment_matched_logits(k), edges, init, ts).ys
        return jnp.sum(ys * w[None, :])

    vf, gf = jax.value_and_grad(full_loss)(theta0)
    vc, gc = jax.value_and_grad(comp_loss)(theta0)
    assert abs(float(vf - vc)) < 1e-8
    assert float(jnp.linalg.norm(gf - gc) / jnp.linalg.norm(gf)) < 1e-6


def test_modern_population_baselines_mass_and_gradients():
    plan = _DummyPlan(
        traits=np.array([-1.0, 1.0]),
        ages=np.array([0.0, 0.75]),
        baseline=np.array([0.0, 1.0]),
        exposure=np.ones((8, 2)),
        sample=np.array([1, 3, 5, 7]),
    )
    counts = jnp.ones((2, 4, 3, 3)) * 10.0
    expected_params = {
        "latent_ode": 387,
        "ltc_ode": 378,
        "s4d_ode": 429,
        "neural_cme": 342,
        "mamba_ode": 414,
    }

    for kind, n_exp in expected_params.items():
        params = init_population_baseline(kind, seed=17)
        pred = np.asarray(predict_population_baseline(kind, params, plan))
        assert pred.shape == (2, 4, 3, 3)
        assert pred.min() > 0.0
        assert np.max(np.abs(pred.sum(axis=-1) - 1.0)) < 1e-12

        w0, _, vg, fc = make_baseline_engine(kind, plan, plan, seed=17)
        assert len(w0) == n_exp
        loss_val, grad_val = vg(jnp.asarray(w0), counts)
        assert np.isfinite(float(loss_val))
        assert np.all(np.isfinite(np.asarray(grad_val)))
        assert fc(jnp.asarray(w0)).shape == (2, 4, 3, 3)


def test_hyper_erlang_race_and_edge_memory_drn():
    assert hyper_race_state_size(4, 2, 16) == 976  # vs (32)^4 = 1,048,576!
    mix = jnp.zeros((2, 2))
    means = jnp.array([[0.3, 2.1], [0.4, 1.8]])
    logits = jnp.zeros((2, 2, 5))
    ent, rates, cont, causes = hyper_erlang_race(mix, means, logits)
    assert ent.size == hyper_race_state_size(2, 2, 5)
    assert abs(float(jnp.sum(ent)) - 1.0) < 1e-13
    assert float(jnp.min(causes)) >= 0.0

    psi_u, cert = calibrate_universal_pos_clock(stages_red=6, stages_full=16, maxiter=120)
    assert cert["sup_survival_abs_error"] < 0.03
    edges = ((0, 1), (0, 2), (1, 3), (2, 3))
    gen, fr, rd = build_universal_pos_network_generator(
        [1.5, 2.0, 1.7, 1.0], [4.5, 7.5, 6.0, 4.0], [0.5, 0.5, 1.0, 1.0], edges, 4, psi_u, stages_red=6
    )
    assert float(jnp.max(jnp.abs(jnp.sum(gen, axis=0)))) < 1e-13
    p_out = rd @ jl.expm(1.0 * gen) @ fr
    assert float(jnp.min(p_out)) >= 0.0

    em_edges = ((0, 1), (1, 2), (2, 0))
    n_p = edge_memory_param_size(3, 3, stages=4)
    gen_e, fr_e, ag_e, rd_e = build_edge_memory_hyper_generator(jnp.zeros(n_p), em_edges, 3, stages=4)
    assert float(jnp.max(jnp.abs(jnp.sum(gen_e, axis=0)))) < 1e-13
    p_ag = rd_e @ jl.expm(0.5 * gen_e) @ ag_e
    assert float(jnp.min(p_ag)) > 0.0
    assert float(jnp.max(jnp.abs(jnp.sum(p_ag, axis=0) - 1.0))) < 1e-13
