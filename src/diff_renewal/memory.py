"""Auxiliary-state exponential memory solver and static-preserving fast-mode cutoffs."""

from collections.abc import Callable
from dataclasses import dataclass

import diffrax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class MemoryCutoff:
    rates: np.ndarray
    weights: np.ndarray
    direct: float
    cutoff: float
    removed_mass: float

    def screen_bound(self, radius: float) -> float:
        if not 0 <= radius < self.cutoff:
            raise ValueError("require 0 <= radius < cutoff")
        return radius / (self.cutoff - radius) * self.removed_mass

    def log_rate_derivative_bound(self, radius: float) -> float:
        self.screen_bound(radius)
        return (
            self.removed_mass
            * radius
            * (2.0 * self.cutoff + radius)
            / (self.cutoff - radius) ** 2
        )


def fixed_partition_cutoff(
    rates, weights, retained_indices: tuple[int, ...], *, direct: float = 0.0
):
    """Differentiable static-preserving mode elimination with frozen index partition."""
    rates, weights = jnp.asarray(rates), jnp.asarray(weights)
    if rates.ndim != 1 or rates.shape != weights.shape:
        raise ValueError("rates and weights must be equal-length 1D arrays")
    if (
        not isinstance(retained_indices, tuple)
        or any(not isinstance(i, int) or i < 0 or i >= rates.size for i in retained_indices)
        or len(set(retained_indices)) != len(retained_indices)
    ):
        raise ValueError("retained_indices must be a static tuple of distinct valid indices")
    dropped = tuple(i for i in range(rates.size) if i not in retained_indices)
    keep, drop = jnp.array(retained_indices, dtype=int), jnp.array(dropped, dtype=int)
    return rates[keep], weights[keep], direct + jnp.sum(weights[drop] / rates[drop])


def static_preserving_cutoff(rates, weights, cutoff: float, *, direct: float = 0.0):
    rates, weights = np.asarray(rates, dtype=float), np.asarray(weights, dtype=float)
    if rates.ndim != 1 or rates.shape != weights.shape:
        raise ValueError("rates and weights must be equal-length 1D arrays")
    if (
        not np.all(np.isfinite(rates))
        or not np.all(np.isfinite(weights))
        or not np.isfinite(cutoff)
        or not np.isfinite(direct)
        or np.any(rates <= 0)
        or np.any(weights < 0)
        or cutoff <= 0
    ):
        raise ValueError("require finite positive rates/cutoff and nonnegative weights")
    keep = rates <= cutoff
    mass = float(np.sum(weights[~keep] / rates[~keep]))
    return MemoryCutoff(rates[keep], weights[keep], direct + mass, cutoff, mass)


def fit_positive_soe(times, values, rates, *, relative_floor: float = 1e-3):
    """Nonnegative weighted least-squares sum-of-exponentials fit on a fixed rate grid."""
    times, values, rates = map(
        lambda x: np.asarray(x, dtype=float), (times, values, rates)
    )
    if (
        times.ndim != 1
        or values.shape != times.shape
        or rates.ndim != 1
        or times.size == 0
        or rates.size == 0
        or not all(np.all(np.isfinite(v)) for v in (times, values, rates))
        or np.any(times < 0)
        or np.any(values < 0)
        or np.any(rates <= 0)
        or not np.isfinite(relative_floor)
        or relative_floor <= 0
    ):
        raise ValueError("require finite nonnegative samples, positive rates and floor")
    scale = np.maximum(np.abs(values), relative_floor)
    design = np.exp(-times[:, None] * rates[None, :])
    weights, _ = nnls(design / scale[:, None], values / scale, maxiter=50 * rates.size)
    return weights


def solve_memory(
    rhs: Callable,
    source: Callable,
    y0,
    ts,
    rates,
    weights,
    args=None,
    *,
    z0=None,
    direct: float = 0.0,
    clock: Callable | None = None,
    killing: Callable | None = None,
    solver=None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int = 100_000,
):
    """Solve y' = rhs(t, y, memory, args) with augmented exponential memory states."""
    ts, rates, weights = map(jnp.asarray, (ts, rates, weights))
    y0 = jnp.asarray(y0)
    y0 = y0.astype(jnp.result_type(y0, rates, weights, 1.0))
    if rates.ndim != 1 or weights.shape != rates.shape:
        raise ValueError("rates and weights must be equal-length 1D arrays")
    if ts.ndim != 1 or ts.shape[0] < 2:
        raise ValueError("ts must be a strictly increasing 1D grid of length >= 2")
    zshape = (rates.size,) + y0.shape
    z0 = jnp.zeros(zshape, dtype=y0.dtype) if z0 is None else jnp.asarray(z0)
    if z0.shape != zshape:
        raise ValueError(f"z0 must have shape {zshape}")

    def field(t, state, packed):
        y, z = state
        lam, w, local, params = packed
        u = jnp.asarray(source(t, y, params))
        memory = jnp.tensordot(w, z, axes=1) + local * u
        c = 1.0 if clock is None else clock(t, y, params)
        q = 0.0 if killing is None else killing(t, y, params)
        decay = (c * lam + q).reshape((lam.size,) + (1,) * y.ndim)
        return rhs(t, y, memory, params), u[None, ...] - decay * z

    return diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5() if solver is None else solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=(y0, z0),
        args=(rates, weights, direct, args),
        saveat=diffrax.SaveAt(ts=ts, fn=lambda t, state, args: state[0]),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=max_steps,
    )
