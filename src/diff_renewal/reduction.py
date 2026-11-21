"""Symmetry-based continuation and sensitivity-interpolating reduction on renewal cycles.

Implements both the stationary-profile 33-state symmetry closure and two-sided
parametric resolvent interpolation (Baur et al., 2011) for the 48-state
three-node mixed-Erlang cycle.
"""

import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from scipy.linalg import helmert, solve, svd

from .phase_type import mixed_erlang_components
from .traits import moment_matched_logits

STAGES = 16
STEP = 0.5
STEPS = 14
CONTRASTS = np.kron(helmert(3, full=False).T, np.eye(STAGES))
READ = np.kron(np.eye(3), np.ones((1, STAGES)))
W_SYM = np.vstack((CONTRASTS.T, np.ones(3 * STAGES)))
READ_SYM = np.column_stack((READ @ CONTRASTS, np.ones(3) / 3.0))
ROUTE = np.roll(np.eye(3), 1, axis=0)
EXIT = np.kron(np.eye(3), np.eye(1, STAGES, STAGES - 1))
LOCAL = np.eye(STAGES, k=-1) - np.eye(STAGES)

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
    """Return a JIT-compiled evaluator (theta -> probabilities, Jacobian) on STEP*1..STEPS."""

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
    """Return a JIT-compiled projected forward+Jacobian evaluator for basis Q."""
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
