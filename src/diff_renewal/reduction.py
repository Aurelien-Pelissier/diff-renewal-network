"""Symmetry-based continuation, Baur et al. projection, and Positive Fisher-Sensitivity Reduction."""

import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from scipy.linalg import helmert, solve, svd
from scipy.optimize import minimize

from .phase_type import mixed_erlang_components
from .traits import moment_matched_logits

STAGES = 16
K_POS_RED = 11
STEP = 0.5
STEPS = 14
CONTRASTS = np.kron(helmert(3, full=False).T, np.eye(STAGES))
READ = np.kron(np.eye(3), np.ones((1, STAGES)))
W_SYM = np.vstack((CONTRASTS.T, np.ones(3 * STAGES)))
READ_SYM = np.column_stack((READ @ CONTRASTS, np.ones(3) / 3.0))
ROUTE = np.roll(np.eye(3), 1, axis=0)
EXIT = np.kron(np.eye(3), np.eye(1, STAGES, STAGES - 1))
LOCAL = np.eye(STAGES, k=-1) - np.eye(STAGES)

READ_POS33 = jnp.kron(jnp.eye(3), jnp.ones((1, K_POS_RED)))
ANCHORS_DEFAULT = [np.log([m] * 3 + [k] * 3) for m, k in [(1.8, 5.3), (2.3, 7.7)]]
SHIFTS_DEFAULT = [1.0 / 7.0, 0.5, 2.0]


def cycle_symmetry_matrices(theta, *, pure: bool = False):
    """Return (full_48_generator, fresh_entry_48x3, reduced_33_generator)."""
    mean, shape = jnp.exp(jnp.asarray(theta)).reshape(2, 3)
    if pure:
        rates = STAGES / mean
        entry = jnp.tile(jnp.eye(STAGES)[0], (3, 1))
    else:
        rates, entry = jax.vmap(mixed_erlang_components)(
            mean, moment_matched_logits(shape)
        )
    fresh = jl.block_diag(*(e[:, None] for e in entry))
    generator = jnp.kron(jnp.diag(rates), LOCAL) + fresh @ ROUTE @ (
        rates[:, None] * EXIT
    )
    residence = jnp.sum(jnp.cumsum(entry, axis=1) / rates[:, None], axis=0)
    profile = residence / residence.sum()
    lift = jnp.column_stack((CONTRASTS, jnp.tile(profile / 3.0, 3)))
    return generator, fresh, W_SYM @ generator @ lift


def make_cycle_engine(reduced: bool = False, *, pure: bool = False):
    """Return JIT-compiled evaluator (theta -> probabilities, Jacobian) on STEP*1..STEPS."""

    def auxiliary(theta):
        generator, fresh, small = cycle_symmetry_matrices(theta, pure=pure)
        if reduced:
            generator, fresh, read = small, W_SYM @ fresh, READ_SYM
        else:
            read = READ
        transition = jl.expm(STEP * generator)

        def advance(state, _):
            state = transition @ state
            return state, (read @ state).T

        p = jax.lax.scan(advance, fresh, None, length=STEPS)[1]
        return p, p

    compiled = jax.jit(jax.jacfwd(auxiliary, has_aux=True))

    def evaluate(theta):
        derivative, values = compiled(jnp.asarray(theta))
        return np.asarray(values), np.asarray(derivative)

    return evaluate


def build_interpolatory_basis(
    dimension: int = 33,
    *,
    anchors=None,
    shifts=None,
):
    """Build two-sided parametric interpolation basis Q (48 x dimension) from anchor points."""
    if dimension not in (18, 33):
        raise ValueError("Supported predeclared basis dimensions are 18 and 33")
    t0 = time.perf_counter()
    anchors_list = ANCHORS_DEFAULT if anchors is None else list(anchors)
    shifts_list = SHIFTS_DEFAULT if shifts is None else list(shifts)
    used_anchors = anchors_list[:1] if dimension == 18 else anchors_list

    mass = READ.T / np.sqrt(float(STAGES))
    columns = []
    for theta in used_anchors:
        a, b, _ = map(np.asarray, cycle_symmetry_matrices(theta))
        for shift in shifts_list:
            resolvent = shift * np.eye(3 * STAGES) - a
            for snapshots in (solve(resolvent, b), solve(resolvent.T, READ.T)):
                columns.append(snapshots / np.linalg.norm(snapshots, axis=0))
    snapshots = np.column_stack(columns)
    residual = snapshots - mass @ (mass.T @ snapshots)
    u, s, _ = svd(residual, full_matrices=False)
    q = np.column_stack((mass, u[:, : dimension - 3]))
    q = np.linalg.qr(q)[0]
    info = dict(
        dimension=dimension,
        anchors=[np.asarray(x).tolist() for x in used_anchors],
        shifts=list(shifts_list),
        residual_singular_values=s.tolist(),
        relative_span_residual=float(
            np.linalg.norm(snapshots - q @ (q.T @ snapshots))
            / np.linalg.norm(snapshots)
        ),
        orthogonality_error=float(np.linalg.norm(q.T @ q - np.eye(dimension))),
        output_span_residual=float(np.linalg.norm(READ - READ @ q @ q.T)),
        seconds=time.perf_counter() - t0,
    )
    return q, info


def make_interpolatory_engine(q):
    """Return JIT-compiled projected forward+Jacobian evaluator for basis Q."""
    q_jax = jnp.asarray(q)
    read_proj = jnp.asarray(READ) @ q_jax

    def auxiliary(theta):
        a, b, _ = cycle_symmetry_matrices(theta)
        transition = jl.expm(STEP * (q_jax.T @ a @ q_jax))

        def advance(state, _):
            state = transition @ state
            return state, (read_proj @ state).T

        p = jax.lax.scan(advance, q_jax.T @ b, None, length=STEPS)[1]
        return p, p

    compiled = jax.jit(jax.jacfwd(auxiliary, has_aux=True))

    def evaluate(theta):
        derivative, values = compiled(jnp.asarray(theta))
        return np.asarray(values), np.asarray(derivative)

    return evaluate


def resolvent_sensitivity_diagnostic(q, theta, shift: float):
    """Check transfer and sensitivity interpolation accuracy in Laplace domain."""
    a, b, _ = map(np.asarray, cycle_symmetry_matrices(theta))
    da, db, _ = map(np.asarray, jax.jacfwd(cycle_symmetry_matrices)(jnp.asarray(theta)))
    r = solve(shift * np.eye(3 * STAGES) - a, np.eye(3 * STAGES))
    ar, br, cr = q.T @ a @ q, q.T @ b, READ @ q
    rr = solve(shift * np.eye(q.shape[1]) - ar, np.eye(q.shape[1]))
    full, small = READ @ r @ b, cr @ rr @ br
    jf = np.stack(
        [READ @ r @ (da[:, :, i] @ r @ b + db[:, :, i]) for i in range(6)], axis=-1
    )
    jr = np.stack(
        [
            cr @ rr @ (q.T @ da[:, :, i] @ q @ rr @ br + q.T @ db[:, :, i])
            for i in range(6)
        ],
        axis=-1,
    )
    return dict(
        transfer_max_abs=float(np.max(np.abs(full - small))),
        sensitivity_max_abs=float(np.max(np.abs(jf - jr))),
    )


def _pos_node_chain(mean, shape, psi):
    """Map (mean, shape) to a valid K_POS_RED=11 positive Coxian chain with exact mean."""
    s_norm = (jnp.log(shape) - jnp.log(6.0)) / 0.7
    feats = jnp.array([1.0, s_norm, s_norm**2, s_norm**3])
    w_logits = psi[: 4 * K_POS_RED].reshape(4, K_POS_RED)
    w_rates = psi[4 * K_POS_RED : 8 * K_POS_RED].reshape(4, K_POS_RED)
    entry = jax.nn.softmax(feats @ w_logits)
    rel_rates = jnp.exp(0.6 * jnp.tanh(feats @ w_rates))
    rem_mean_unit = jnp.cumsum((1.0 / rel_rates)[::-1])[::-1]
    unit_mean = jnp.dot(entry, rem_mean_unit)
    stage_rates = (unit_mean / mean) * rel_rates
    return stage_rates, entry


def pos33_cycle_matrices(theta, psi):
    """Build 33x33 strictly Metzler mass-conserving generator and non-negative entry matrix."""
    means, shapes = jnp.exp(jnp.asarray(theta)).reshape(2, 3)
    stage_rates, entries = jax.vmap(lambda m, k: _pos_node_chain(m, k, psi))(
        means, shapes
    )
    blocks = [
        -jnp.diag(r_vec) + jnp.diag(r_vec[:-1], k=-1) for r_vec in stage_rates
    ]
    local_gen = jl.block_diag(*blocks)
    fresh = jl.block_diag(*(e[:, None] for e in entries))
    out_flux = jnp.kron(
        jnp.diag(stage_rates[:, -1]), jnp.eye(1, K_POS_RED, K_POS_RED - 1)
    )
    generator = local_gen + fresh @ jnp.asarray(ROUTE) @ out_flux
    return generator, fresh


def calibrate_positive_33_surrogate(*, seed: int = 0, maxiter: int = 250):
    """Offline data-free Fisher-sensitivity calibration of the 33-state positive PH family."""
    t0 = time.perf_counter()
    full_eval = make_cycle_engine(reduced=False)
    rng = np.random.default_rng(seed)
    anchors = list(ANCHORS_DEFAULT)
    for m_base, k_base in [(1.5, 4.0), (2.0, 6.0), (2.6, 9.0), (1.8, 10.5), (2.4, 3.6)]:
        anchors.append(np.log([m_base] * 3 + [k_base] * 3))
    for _ in range(12):
        ms = np.exp(rng.uniform(np.log(1.2), np.log(3.2), size=3))
        ks = np.exp(rng.uniform(np.log(3.3), np.log(11.2), size=3))
        anchors.append(np.log(np.r_[ms, ks]))
    anchor_arr = jnp.asarray(anchors)
    pf_full = jnp.asarray([full_eval(np.asarray(th))[0] for th in anchor_arr])
    jf_full = jnp.asarray([full_eval(np.asarray(th))[1] for th in anchor_arr])

    def _eval_one(th, psi):
        def aux(t_vec):
            gen, fresh = pos33_cycle_matrices(t_vec, psi)
            trans = jl.expm(STEP * gen)

            def step(st, _):
                st = trans @ st
                return st, (READ_POS33 @ st).T

            p = jax.lax.scan(step, fresh, None, length=STEPS)[1]
            return p, p

        j, p = jax.jacfwd(aux, has_aux=True)(th)
        return p, j

    @jax.jit
    def loss_fn(psi):
        def per_anchor(th, pf, jf):
            pr, jr = _eval_one(th, psi)
            w = 1.0 / jnp.maximum(pf, 1e-3)
            return jnp.mean(w * (pr - pf) ** 2) + 0.35 * jnp.mean(
                w[..., None] * (jr - jf) ** 2
            )

        return jnp.mean(jax.vmap(per_anchor)(anchor_arr, pf_full, jf_full)) + 1e-6 * jnp.mean(
            psi**2
        )

    vg = jax.jit(jax.value_and_grad(loss_fn))
    psi0 = np.zeros(8 * K_POS_RED)
    fit = minimize(
        lambda z: (float(vg(z)[0]), np.asarray(vg(z)[1], dtype=float)),
        psi0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    psi_star = np.asarray(fit.x, dtype=float)
    meta = dict(
        dimension=3 * K_POS_RED,
        stages_per_node=K_POS_RED,
        n_anchors=len(anchors),
        calibration_loss=float(fit.fun),
        iterations=int(fit.nit),
        seconds=time.perf_counter() - t0,
    )
    return psi_star, meta


def make_positive_33_engine(psi):
    """Return JIT-compiled evaluator (theta -> probabilities, Jacobian) for Pos-Sens 33."""
    psi_jax = jnp.asarray(psi)

    def auxiliary(theta):
        gen, fresh = pos33_cycle_matrices(theta, psi_jax)
        trans = jl.expm(STEP * gen)

        def step(st, _):
            st = trans @ st
            return st, (READ_POS33 @ st).T

        p = jax.lax.scan(step, fresh, None, length=STEPS)[1]
        return p, p

    compiled = jax.jit(jax.jacfwd(auxiliary, has_aux=True))

    def evaluate(theta):
        deriv, vals = compiled(jnp.asarray(theta))
        return np.asarray(vals), np.asarray(deriv)

    return evaluate
