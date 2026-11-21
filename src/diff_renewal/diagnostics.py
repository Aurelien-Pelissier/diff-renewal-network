"""Fisher information geometry and adjoint sensitivity residual accounting."""

import numpy as np


def multinomial_score_design(probabilities, derivatives, sample_sizes=1):
    """Weighted score matrix W satisfying Fisher = W.T @ W."""
    p, dp = np.asarray(probabilities, float), np.asarray(derivatives, float)
    if p.ndim != 2 or dp.ndim != 3 or dp.shape[:2] != p.shape:
        raise ValueError("Expected probabilities (T,C), derivatives (T,C,P)")
    if not np.all(np.isfinite(p)) or not np.all(p > 0) or not np.all(np.isfinite(dp)):
        raise ValueError("A finite, strictly positive, fixed probability support is required")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-9, rtol=0):
        raise ValueError("Probabilities must sum to one")
    if not np.allclose(dp.sum(axis=1), 0.0, atol=1e-8, rtol=0):
        raise ValueError("Probability derivatives must sum to zero")
    n = np.broadcast_to(np.asarray(sample_sizes, float), (p.shape[0],))
    if not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("Sample sizes must be finite and nonnegative")
    return (np.sqrt(n[:, None] / p)[..., None] * dp).reshape(-1, dp.shape[-1])


def coarsened_fisher(probabilities, derivatives, groups, sample_sizes=1):
    """Exact conditional-score information lost under a fixed category partition."""
    p, dp = np.asarray(probabilities, float), np.asarray(derivatives, float)
    design = multinomial_score_design(p, dp, sample_sizes)
    groups = np.asarray(groups)
    if groups.shape != (p.shape[1],):
        raise ValueError("Supply one group label per category")
    labels, index = np.unique(groups, return_inverse=True)
    q = np.stack([p[:, index == i].sum(axis=1) for i in range(len(labels))], axis=1)
    dq = np.stack([dp[:, index == i].sum(axis=1) for i in range(len(labels))], axis=1)
    coarse = multinomial_score_design(q, dq, sample_sizes)
    n = np.broadcast_to(np.asarray(sample_sizes, float), (len(p),))
    residual = np.sqrt(n[:, None] * p)[..., None] * (
        dp / p[..., None] - (dq / q[..., None])[:, index]
    )
    residual = residual.reshape(-1, dp.shape[-1])
    return dict(
        fine=design.T @ design,
        coarse=coarse.T @ coarse,
        loss=residual.T @ residual,
        residual_design=residual,
        coarse_probabilities=q,
        coarse_derivatives=dq,
    )


def efficient_fisher(design, interest, nuisance, relative_cutoff: float = 1e-10):
    """Project out nuisance tangent scores using an explicit SVD rank cutoff."""
    w = np.asarray(design, float)
    if w.ndim != 2 or not np.all(np.isfinite(w)) or not 0 <= relative_cutoff < 1:
        raise ValueError("Expected finite 2D design and cutoff in [0,1)")
    interest, nuisance = list(interest), list(nuisance)
    if set(interest) & set(nuisance) or len(set(interest + nuisance)) != len(
        interest + nuisance
    ):
        raise ValueError("Interest and nuisance indices must be distinct")
    if any(i < 0 or i >= w.shape[1] for i in interest + nuisance):
        raise ValueError("Parameter index out of bounds")
    retained = w[:, interest]
    rank = 0
    if nuisance:
        u, s, _ = np.linalg.svd(w[:, nuisance], full_matrices=False)
        rank = int(np.count_nonzero(s > relative_cutoff * s[0])) if len(s) and s[0] else 0
        retained = retained - u[:, :rank] @ (u[:, :rank].T @ retained)
    return dict(
        information=retained.T @ retained,
        residual_design=retained,
        nuisance_rank=rank,
        relative_cutoff=relative_cutoff,
    )


def adjoint_residual_estimate(
    transitions,
    approximate_states,
    initial,
    terminal_observable,
    *,
    adjoints=None,
    adjoint_error_bounds=None,
):
    """Exact discrete suffix-adjoint terminal observable error accounting."""
    a, x = np.asarray(transitions, float), np.asarray(approximate_states, float)
    x0, c = np.asarray(initial, float), np.asarray(terminal_observable, float)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError("Transitions must have shape (N,D,D)")
    n, d, _ = a.shape
    if x.shape != (n + 1, d) or x0.shape != (d,) or c.shape != (d,):
        raise ValueError("State or observable shape does not match transitions")
    if not all(np.all(np.isfinite(v)) for v in (a, x, x0, c)):
        raise ValueError("Inputs must be finite")
    exact = adjoints is None
    if exact:
        if adjoint_error_bounds is not None:
            raise ValueError("Bounds are only used with supplied approximate adjoints")
        z = np.empty_like(x)
        z[-1] = c
        for step in range(n - 1, -1, -1):
            z[step] = a[step].T @ z[step + 1]
    else:
        z = np.asarray(adjoints, float)
        if z.shape != x.shape or not np.all(np.isfinite(z)):
            raise ValueError("Adjoints must be finite and have the state trajectory shape")
    defects = np.vstack([x0 - x[0], np.einsum("nij,nj->ni", a, x[:-1]) - x[1:]])
    terms = np.einsum("ni,ni->n", z, defects)
    estimate = float(terms.sum())
    radius = 0.0 if exact else None
    if adjoint_error_bounds is not None:
        bounds = np.asarray(adjoint_error_bounds, float)
        if bounds.shape != x.shape or not np.all(np.isfinite(bounds)) or np.any(bounds < 0):
            raise ValueError("Adjoint bounds must be finite, nonnegative and match the states")
        radius = float(np.sum(np.abs(defects) * bounds))
    return dict(
        signed_error=estimate,
        correction_interval=None if radius is None else (estimate - radius, estimate + radius),
        error_radius=radius,
        absolute_pairing_bound=float(np.abs(terms).sum()),
        norm_pairing_bound=float(
            np.sum(np.sum(np.abs(defects), axis=1) * np.max(np.abs(z), axis=1))
        ),
        terms=terms,
        defects=defects,
        adjoints=z,
        exact_adjoint=exact,
    )


def adjoint_sensitivity_residual(
    transitions,
    approximate_states,
    initial,
    terminal_observable,
    transition_derivatives,
    state_derivatives,
    initial_derivatives,
    observable_derivatives,
):
    """Exact discrete suffix-adjoint gradient-error decomposition."""
    result = adjoint_residual_estimate(
        transitions, approximate_states, initial, terminal_observable
    )
    a, x = np.asarray(transitions, float), np.asarray(approximate_states, float)
    da, dx, di, dc = [
        np.asarray(v, float)
        for v in (
            transition_derivatives,
            state_derivatives,
            initial_derivatives,
            observable_derivatives,
        )
    ]
    if dc.ndim != 2:
        raise ValueError("Derivative inputs must append a parameter axis")
    k = dc.shape[-1]
    if (
        da.shape != a.shape + (k,)
        or dx.shape != x.shape + (k,)
        or di.shape != (a.shape[-1], k)
        or dc.shape != di.shape
    ):
        raise ValueError("Derivative shapes do not match their inputs")
    if not all(np.all(np.isfinite(v)) for v in (da, dx, di, dc)):
        raise ValueError("Derivatives must be finite")
    z, defects = result["adjoints"], result["defects"]
    dz = np.empty_like(dx)
    dz[-1] = dc
    for n in range(len(a) - 1, -1, -1):
        dz[n] = np.einsum("ijk,i->jk", da[n], z[n + 1]) + a[n].T @ dz[n + 1]
    dr = np.concatenate(
        [
            (di - dx[0])[None],
            np.einsum("nijk,nj->nik", da, x[:-1])
            + np.einsum("nij,njk->nik", a, dx[:-1])
            - dx[1:],
        ],
        axis=0,
    )
    terms = np.einsum("nik,ni->nk", dz, defects) + np.einsum("ni,nik->nk", z, dr)
    return dict(
        signed_gradient_error=terms.sum(axis=0),
        terms=terms,
        adjoint_derivatives=dz,
        defect_derivatives=dr,
        sum_absolute_gradient_pairings=np.abs(terms).sum(axis=0),
        output_error=result["signed_error"],
    )
