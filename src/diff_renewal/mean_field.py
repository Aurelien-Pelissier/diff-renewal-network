"""Differentiable Mean-Field Renewal Networks (McKean-Vlasov semi-Markov ODEs).

Extends linear phase-type renewal networks to interacting populations where
transition rates or routing probabilities depend smoothly on the instantaneous
macroscopic node occupancy bar_p(t) = C x(t) in Delta^{V-1}:

    dx/dt = (T(bar_p(t), t) + B(bar_p(t), t) J(bar_p(t), t)) x(t),

preserving exact non-negativity (x(t) >= 0), simplex mass conservation
(1^T x(t) = 1), and internal residence-age phase memory across nonlinear
feedback such as epidemic contagion or queueing congestion.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from .phase_type import mixed_erlang_components, smooth_erlang_logits


def solve_mean_field_renewal(
    x0: jax.Array,
    ts: jax.Array,
    vector_field_fn,
    readout_fn,
    *,
    dt0: float = 0.05,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int = 8192,
):
    """Integrate a nonlinear mean-field semi-Markov ODE via Diffrax Tsit5."""
    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    ts_arr = jnp.asarray(ts, dtype=jnp.float64)

    def rhs(t, x, args):
        x_pos = jnp.maximum(x, 0.0)
        p_bar = readout_fn(x_pos)
        return vector_field_fn(t, x_pos, p_bar)

    term = diffrax.ODETerm(rhs)
    solver = diffrax.Tsit5()
    stepsize = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=ts_arr)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=ts_arr[-1],
        dt0=dt0,
        y0=x0_arr,
        saveat=saveat,
        stepsize_controller=stepsize,
        max_steps=max_steps,
    )
    return jax.vmap(readout_fn)(sol.ys), sol.ys


def seicr_mean_field_predict(
    beta: jax.Array,
    m_e: jax.Array,
    k_e: jax.Array,
    m_i: jax.Array,
    k_i: jax.Array,
    m_c: jax.Array,
    k_c: jax.Array,
    i0: jax.Array,
    e0: jax.Array,
    ts: jax.Array,
    *,
    stages: int = 8,
    infectious_in_exposed: float = 0.0,
):
    """Differentiable Mean-Field SEICR Renewal Network (S -> E -> I_bed -> C_conv -> R).

    State vector layout (length 3 * stages + 2):
      - index 0: Susceptible S(t)
      - indices [1 : 1 + K]: Exposed E phases (smooth mixed-Erlang(m_e, k_e))
      - indices [1 + K : 1 + 2K]: Bed-confined / Infectious I phases (m_i, k_i)
      - indices [1 + 2K : 1 + 3K]: Convalescent C phases (m_c, k_c)
      - index [1 + 3K]: Recovered R(t)

    Endogenous McKean-Vlasov contagion drives S -> E at rate beta * (I(t) + eps * E(t)),
    entering E with exact mixed-Erlang entry vector b_E. All phase blocks are Metzler
    and column-sum zero, guaranteeing x(t) >= 0 and sum(x(t)) == 1 for all t >= 0.
    """
    k_st = int(stages)
    r_e, b_e = mixed_erlang_components(m_e, smooth_erlang_logits(k_e, stages=k_st))
    r_i, b_i = mixed_erlang_components(m_i, smooth_erlang_logits(k_i, stages=k_st))
    r_c, b_c = mixed_erlang_components(m_c, smooth_erlang_logits(k_c, stages=k_st))

    s0 = jnp.maximum(1.0 - i0 - e0, 1e-12)
    x0 = jnp.concatenate(
        [
            jnp.array([s0]),
            e0 * b_e,
            i0 * b_i,
            jnp.zeros(k_st),
            jnp.array([0.0]),
        ]
    )

    def readout(x):
        s_val = x[0]
        e_val = jnp.sum(x[1 : 1 + k_st])
        i_val = jnp.sum(x[1 + k_st : 1 + 2 * k_st])
        c_val = jnp.sum(x[1 + 2 * k_st : 1 + 3 * k_st])
        r_val = x[1 + 3 * k_st]
        return jnp.stack([s_val, e_val, i_val, c_val, r_val])

    def chain_dx(seg, rate, inflow, entry_vec):
        # Forward suffix chain: phase 0 -> 1 -> ... -> K-1 -> exit
        dx = -rate * seg + inflow * entry_vec
        dx = dx.at[1:].add(rate * seg[:-1])
        exit_flux = rate * seg[-1]
        return dx, exit_flux

    def vector_field(_t, x, p_bar):
        s_val = x[0]
        e_seg = x[1 : 1 + k_st]
        i_seg = x[1 + k_st : 1 + 2 * k_st]
        c_seg = x[1 + 2 * k_st : 1 + 3 * k_st]

        inf_pressure = p_bar[2] + infectious_in_exposed * p_bar[1]
        inc_flux = beta * s_val * inf_pressure

        de, flux_e_to_i = chain_dx(e_seg, r_e, inc_flux, b_e)
        di, flux_i_to_c = chain_dx(i_seg, r_i, flux_e_to_i, b_i)
        dc, flux_c_to_r = chain_dx(c_seg, r_c, flux_i_to_c, b_c)

        ds = jnp.array([-inc_flux])
        dr = jnp.array([flux_c_to_r])
        return jnp.concatenate([ds, de, di, dc, dr])

    p_traj, x_traj = solve_mean_field_renewal(x0, ts, vector_field, readout)
    return p_traj, x_traj


def simulate_interacting_seicr_gillespie(
    beta: float,
    m_e: float,
    k_e: float,
    m_i: float,
    k_i: float,
    m_c: float,
    k_c: float,
    i0: float,
    e0: float,
    ts: np.ndarray,
    n_agents: int,
    *,
    stages: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Exact interacting phase-lifted Gillespie simulation of M finite agents.

    Used to verify Kurtz's O(M^{-1/2}) propagation-of-chaos convergence of the
    finite interacting semi-Markov population to `seicr_mean_field_predict`.
    """
    rng = np.random.default_rng(seed)
    k_st = int(stages)
    r_e, b_e = map(np.asarray, mixed_erlang_components(jnp.array(m_e), smooth_erlang_logits(jnp.array(k_e), stages=k_st)))
    r_i, b_i = map(np.asarray, mixed_erlang_components(jnp.array(m_i), smooth_erlang_logits(jnp.array(k_i), stages=k_st)))
    r_c, b_c = map(np.asarray, mixed_erlang_components(jnp.array(m_c), smooth_erlang_logits(jnp.array(k_c), stages=k_st)))
    r_e, r_i, r_c = float(r_e), float(r_i), float(r_c)
    b_e = np.maximum(b_e, 0.0)
    b_e /= b_e.sum()
    b_i = np.maximum(b_i, 0.0)
    b_i /= b_i.sum()
    b_c = np.maximum(b_c, 0.0)
    b_c /= b_c.sum()

    n_i0 = max(1, int(round(i0 * n_agents)))
    n_e0 = int(round(e0 * n_agents))
    n_s0 = n_agents - n_i0 - n_e0

    # Counts per phase
    s_cnt = int(n_s0)
    e_cnt = rng.multinomial(n_e0, b_e)
    i_cnt = rng.multinomial(n_i0, b_i)
    c_cnt = np.zeros(k_st, dtype=np.int64)
    r_cnt = 0

    ts_arr = np.asarray(ts, dtype=float)
    out = np.zeros((len(ts_arr), 5), dtype=float)
    t_now = 0.0
    out_idx = 0

    def record():
        out[out_idx, 0] = s_cnt / n_agents
        out[out_idx, 1] = e_cnt.sum() / n_agents
        out[out_idx, 2] = i_cnt.sum() / n_agents
        out[out_idx, 3] = c_cnt.sum() / n_agents
        out[out_idx, 4] = r_cnt / n_agents

    while out_idx < len(ts_arr) and ts_arr[out_idx] <= t_now:
        record()
        out_idx += 1

    t_end = float(ts_arr[-1])
    while out_idx < len(ts_arr):
        i_tot = int(i_cnt.sum())
        e_tot = int(e_cnt.sum())
        c_tot = int(c_cnt.sum())
        rate_inf = beta * s_cnt * (i_tot / n_agents)
        rate_e = r_e * e_tot
        rate_i = r_i * i_tot
        rate_c = r_c * c_tot
        tot = rate_inf + rate_e + rate_i + rate_c
        if tot <= 1e-15:
            while out_idx < len(ts_arr):
                record()
                out_idx += 1
            break
        dt = rng.exponential(1.0 / tot)
        t_next = t_now + dt
        while out_idx < len(ts_arr) and ts_arr[out_idx] < t_next:
            record()
            out_idx += 1
        if t_next > t_end and out_idx >= len(ts_arr):
            break
        t_now = t_next

        u = rng.uniform(0.0, tot)
        if u < rate_inf:
            s_cnt -= 1
            ph = int(rng.choice(k_st, p=b_e))
            e_cnt[ph] += 1
        elif u < rate_inf + rate_e:
            ph = int(rng.choice(k_st, p=e_cnt / e_tot))
            e_cnt[ph] -= 1
            if ph < k_st - 1:
                e_cnt[ph + 1] += 1
            else:
                ph_i = int(rng.choice(k_st, p=b_i))
                i_cnt[ph_i] += 1
        elif u < rate_inf + rate_e + rate_i:
            ph = int(rng.choice(k_st, p=i_cnt / i_tot))
            i_cnt[ph] -= 1
            if ph < k_st - 1:
                i_cnt[ph + 1] += 1
            else:
                ph_c = int(rng.choice(k_st, p=b_c))
                c_cnt[ph_c] += 1
        else:
            ph = int(rng.choice(k_st, p=c_cnt / c_tot))
            c_cnt[ph] -= 1
            if ph < k_st - 1:
                c_cnt[ph + 1] += 1
            else:
                r_cnt += 1

    return out
