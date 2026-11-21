"""Scenario-robust A-optimal Fisher experimental design for snapshot allocations."""

import numpy as np
from scipy.optimize import minimize


def largest_remainder(weights, total: int):
    """Round nonnegative allocation weights to an exact integer cell budget."""
    weights = np.asarray(weights, float)
    if (
        weights.ndim != 1
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or weights.sum() <= 0
        or not isinstance(total, (int, np.integer))
        or total < 0
    ):
        raise ValueError("Require finite nonnegative weights and a nonnegative integer budget")
    target = total * weights / weights.sum()
    counts = np.floor(target).astype(int)
    order = np.argsort(-(target - counts), kind="stable")
    counts[order[: int(total - counts.sum())]] += 1
    return counts


def a_optimal_value_gradient(weights, information, pilot_information, budget: int, interest):
    """Average trace of interest-block inverse Fisher information across scenarios."""
    w, info, pilot = map(np.asarray, (weights, information, pilot_information))
    selector = np.eye(info.shape[-1])[:, list(interest)]
    full = pilot + budget * np.einsum("m,bmpq->bpq", w, info)
    solved = np.linalg.solve(
        full, np.broadcast_to(selector, (len(info),) + selector.shape)
    )
    value = np.einsum("pi,bpi->", selector, solved) / len(info)
    gradient = -budget * np.einsum("bpi,bmpq,bqi->m", solved, info, solved) / len(info)
    return float(value), gradient


def optimize_fisher_allocation(
    information,
    pilot_information,
    budget: int,
    interest,
    *,
    initial_weights=None,
    maxiter: int = 800,
    gap_tolerance: float = 2e-5,
):
    """Convex scenario-averaged A-optimal design over a discrete candidate grid."""
    info, pilot = np.asarray(information, float), np.asarray(pilot_information, float)
    if (
        info.ndim != 4
        or info.shape[-1] != info.shape[-2]
        or pilot.shape != info.shape[:1] + info.shape[2:]
    ):
        raise ValueError("Expected scenario/candidate information and matching pilot information")
    if not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("Budget must be a positive integer")
    interest = list(interest)
    if (
        not interest
        or len(set(interest)) != len(interest)
        or min(interest) < 0
        or max(interest) >= info.shape[-1]
    ):
        raise ValueError("Interest indices must be nonempty, distinct, and valid")
    if not np.all(np.isfinite(info)) or not np.all(np.isfinite(pilot)):
        raise ValueError("Information must be finite")
    if not np.allclose(info, info.swapaxes(-1, -2), atol=1e-10) or not np.allclose(
        pilot, pilot.swapaxes(-1, -2), atol=1e-10
    ):
        raise ValueError("Information must be symmetric")
    if np.linalg.eigvalsh(info).min() < -1e-8 or np.linalg.eigvalsh(pilot).min() <= 0:
        raise ValueError("Candidate information must be PSD and pilot information positive definite")

    m = info.shape[1]
    initial = (
        np.full(m, 1.0 / m)
        if initial_weights is None
        else np.asarray(initial_weights, float)
    )
    if (
        initial.shape != (m,)
        or np.any(initial < 0)
        or not np.all(np.isfinite(initial))
        or initial.sum() <= 0
    ):
        raise ValueError("Invalid initial simplex weights")
    initial = initial / initial.sum()
    scale = a_optimal_value_gradient(initial, info, pilot, budget, interest)[0]

    def objective(w):
        value, gradient = a_optimal_value_gradient(w, info, pilot, budget, interest)
        return value / scale, gradient / scale

    fit = minimize(
        objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints=[
            dict(type="eq", fun=lambda w: w.sum() - 1.0, jac=lambda w: np.ones(m))
        ],
        options=dict(maxiter=maxiter, ftol=1e-12),
    )
    weights = np.maximum(fit.x, 0.0)
    weights /= weights.sum()
    value, gradient = a_optimal_value_gradient(weights, info, pilot, budget, interest)
    gap = max(float(weights @ gradient - np.min(gradient)), 0.0)
    counts = largest_remainder(weights, budget)
    rounded = a_optimal_value_gradient(counts / budget, info, pilot, budget, interest)[0]
    if gap / value > gap_tolerance:
        raise RuntimeError(
            f"Allocation optimality gap {gap / value:.3g} exceeds tolerance: {fit.message}"
        )
    return dict(
        weights=weights,
        counts=counts,
        criterion=value,
        initial_criterion=scale,
        rounded_criterion=rounded,
        relative_rounding_loss=rounded / value - 1.0,
        convex_gap=gap,
        relative_convex_gap=gap / value,
        iterations=int(fit.nit),
        success=bool(fit.success),
        message=str(fit.message),
    )
