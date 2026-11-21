"""Modern continuous-time neural population baselines on renewal networks.

Implements parameter-matched, multi-start population architectures for comparison
against Conditional Neural Clocks:
1. Tuned Population Latent ODE (Rubanova et al., 2019)
2. Liquid Time-Constant Population ODE (LTC / CfC; Hasani et al., 2021, 2022)
3. Diagonal Structured State-Space Population ODE (S4D; Gu et al., 2022)
4. Time-Inhomogeneous Neural Master Equation (Neural-CME; clock-time generator A(t, x, u))
"""

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np


def init_population_baseline(kind: str, seed: int = 17):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    if kind == "latent_ode":
        width, latent = 16, 6  # 387 parameters
        return (
            jax.random.normal(keys[0], (6, latent)) * 0.35,
            jnp.zeros(latent),
            jax.random.normal(keys[1], (3 + latent + 2, width)) * 0.32,
            jnp.zeros(width),
            jax.random.normal(keys[2], (width, latent + 3)) * 0.16,
            jnp.zeros(latent + 3),
        )
    if kind == "ltc_ode":
        # Liquid Time-Constant ODE: state- and input-modulated time constants tau(y,x,u)
        width, latent = 12, 6  # 378 parameters
        return (
            jax.random.normal(keys[0], (6, latent)) * 0.35,
            jnp.zeros(latent),
            jax.random.normal(keys[1], (3 + latent + 2, width)) * 0.30,
            jnp.zeros(width),
            jax.random.normal(keys[2], (width, latent)) * 0.20,  # target attractor A(y,u)
            jnp.zeros(latent),
            jax.random.normal(keys[3], (width, latent)) * 0.20,  # liquid rate gate
            jnp.zeros(latent),
            jax.random.normal(keys[4], (latent + 5, 3)) * 0.18,  # positive cycle flux head
            jnp.zeros(3),
        )
    if kind == "s4d_ode":
        # Diagonal Structured State-Space (S4D) with multi-timescale log-spaced decay bank
        latent = 12  # 435 parameters
        log_dt_rates = jnp.linspace(jnp.log(0.15), jnp.log(4.5), latent)
        return (
            jax.random.normal(keys[0], (6, latent)) * 0.35,  # age/prep encoder
            jnp.zeros(latent),
            log_dt_rates,  # trainable diagonal S4D log decay rates
            jax.random.normal(keys[1], (latent,)) * 0.25,  # oscillatory/skew coupling
            jax.random.normal(keys[2], (5, latent)) * 0.28,  # input B projection (p, trait, u)
            jnp.zeros(latent),
            jax.random.normal(keys[3], (latent + 5, 12)) * 0.25,  # nonlinear readout mixer
            jnp.zeros(12),
            jax.random.normal(keys[4], (12, 3)) * 0.18,
            jnp.zeros(3),
        )
    if kind == "neural_cme":
        # Time-Inhomogeneous Neural Master Equation: dp/dt = A_phi(t, trait, base, u) p(t)
        # Isolates clock-time inhomogeneity A(t) from true semi-Markov residence-age memory!
        width = 24  # 7*24 + 24 + 24*6 + 6 = 342 parameters
        return (
            jax.random.normal(keys[0], (7, width)) * 0.35,
            jnp.zeros(width),
            jax.random.normal(keys[1], (width, 6)) * 0.20,  # 6 directed off-diagonal rates on 3-node graph
            jnp.zeros(6),
        )
    raise ValueError(f"Unknown baseline kind: {kind}")


def _rk4_scan(field, state, inputs, sample, read, period: float, substeps: int):
    dt = period / substeps

    def interval(y, u):
        def substep(_, v):
            k1 = field(v, u)
            k2 = field(v + 0.5 * dt * k1, u)
            k3 = field(v + 0.5 * dt * k2, u)
            k4 = field(v + dt * k3, u)
            return v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        y_next = jax.lax.fori_loop(0, substeps, substep, y)
        return y_next, read(y_next)

    obs = jax.lax.scan(interval, state, inputs)[1]
    return obs[sample].transpose(1, 0, 2, 3)


def predict_population_baseline(kind: str, params, plan, *, period: float = 0.25, substeps: int = 4):
    traits, base, age = map(jnp.asarray, (plan.traits, plan.baseline, plan.ages))
    schedule = jnp.asarray(plan.exposure)
    prep = jnp.broadcast_to(jnp.eye(3), (len(traits), 3, 3))

    def bcast(v):
        return jnp.broadcast_to(v[:, None, None], prep.shape[:-1] + (1,))

    encoded = jnp.concatenate((prep, bcast(traits), bcast(base), bcast(age)), axis=-1)

    if kind == "latent_ode":
        enc, eb, w, b, out, ob = params
        state = jnp.concatenate((prep, jnp.tanh(encoded @ enc + eb)), axis=-1)

        def field(y, u):
            p = y[..., :3]
            inp = jnp.concatenate((y, bcast(traits), bcast(u)), axis=-1)
            raw = jnp.tanh(inp @ w + b) @ out + ob
            flux = p * (0.005 + 5.0 * jax.nn.sigmoid(raw[..., :3] - 1.8))
            return jnp.concatenate((jnp.roll(flux, 1, axis=-1) - flux, jnp.tanh(raw[..., 3:])), axis=-1)

        return _rk4_scan(field, state, schedule, plan.sample, lambda y: y[..., :3], period, substeps)

    if kind == "ltc_ode":
        enc, eb, w_h, b_h, w_a, b_a, w_t, b_t, w_f, b_f = params
        z0 = jnp.tanh(encoded @ enc + eb)
        state = jnp.concatenate((prep, z0), axis=-1)

        def field(y, u):
            p, z = y[..., :3], y[..., 3:]
            inp = jnp.concatenate((p, z, bcast(traits), bcast(u)), axis=-1)
            h = jnp.tanh(inp @ w_h + b_h)
            target = jnp.tanh(h @ w_a + b_a)
            tau_inv = 0.15 + 4.5 * jax.nn.sigmoid(h @ w_t + b_t)
            dz = -tau_inv * (z - target)
            f_inp = jnp.concatenate((p, z, bcast(traits), bcast(u)), axis=-1)
            flux_raw = f_inp @ w_f + b_f
            flux = p * (0.005 + 5.0 * jax.nn.sigmoid(flux_raw - 1.8))
            dp = jnp.roll(flux, 1, axis=-1) - flux
            return jnp.concatenate((dp, dz), axis=-1)

        return _rk4_scan(field, state, schedule, plan.sample, lambda y: y[..., :3], period, substeps)

    if kind == "s4d_ode":
        enc, eb, log_rates, omega, w_b, b_b, w_m, b_m, w_o, b_o = params
        z0 = jnp.tanh(encoded @ enc + eb)
        state = jnp.concatenate((prep, z0), axis=-1)
        decay = jnp.exp(log_rates)
        # Skew-symmetric block coupling among pairs of diagonal modes
        half = z0.shape[-1] // 2

        def field(y, u):
            p, z = y[..., :3], y[..., 3:]
            drv_in = jnp.concatenate((p, bcast(traits), bcast(u)), axis=-1)
            drive = drv_in @ w_b + b_b
            z1, z2 = z[..., :half], z[..., half:]
            w_skew = omega[:half]
            rot = jnp.concatenate((-w_skew * z2, w_skew * z1), axis=-1)
            dz = -decay * z + rot + drive
            mix_in = jnp.concatenate((p, z, bcast(traits), bcast(u)), axis=-1)
            hid = jax.nn.gelu(mix_in @ w_m + b_m)
            flux_raw = hid @ w_o + b_o
            flux = p * (0.005 + 5.0 * jax.nn.sigmoid(flux_raw - 1.8))
            dp = jnp.roll(flux, 1, axis=-1) - flux
            return jnp.concatenate((dp, dz), axis=-1)

        return _rk4_scan(field, state, schedule, plan.sample, lambda y: y[..., :3], period, substeps)

    if kind == "neural_cme":
        w1, b1, w2, b2 = params
        dt = period / substeps
        n_intervals = schedule.shape[0]

        def interval(carry, step_in):
            p, t_cur = carry
            u = step_in

            def substep(i_sub, p_st):
                t_now = t_cur + (i_sub + 0.5) * dt
                t_norm = t_now / 4.5
                t_feats = jnp.stack(
                    [
                        jnp.broadcast_to(t_norm, traits.shape),
                        jnp.broadcast_to(jnp.sin(2.0 * jnp.pi * t_norm), traits.shape),
                        jnp.broadcast_to(jnp.cos(2.0 * jnp.pi * t_norm), traits.shape),
                        traits,
                        base,
                        age,
                        u,
                    ],
                    axis=-1,
                )  # (n_plans, 7)
                h = jax.nn.gelu(t_feats @ w1 + b1)
                rates = 0.005 + 5.0 * jax.nn.sigmoid(h @ w2 + b2 - 1.5)  # (n_plans, 6)

                def eval_field(p_cur):
                    # 3 forward cycle edges (0->1, 1->2, 2->0) + 3 reverse/skip edges
                    f_fwd = p_cur * rates[:, None, :3]
                    f_rev = p_cur * rates[:, None, 3:]
                    return (
                        jnp.roll(f_fwd, 1, axis=-1)
                        - f_fwd
                        + jnp.roll(f_rev, -1, axis=-1)
                        - f_rev
                    )

                k1 = eval_field(p_st)
                k2 = eval_field(p_st + 0.5 * dt * k1)
                k3 = eval_field(p_st + 0.5 * dt * k2)
                k4 = eval_field(p_st + dt * k3)
                return p_st + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            p_next = jax.lax.fori_loop(0, substeps, substep, p)
            return (p_next, t_cur + period), p_next

        obs = jax.lax.scan(interval, (prep, jnp.array(0.0, dtype=prep.dtype)), schedule)[1]
        return obs[plan.sample].transpose(1, 0, 2, 3)

    raise ValueError(kind)


def make_baseline_engine(kind: str, train_plan, test_plan, *, seed: int = 17, reg: float = 1e-4):
    params = init_population_baseline(kind, seed)
    flat, unpack = ravel_pytree(params)

    def loss(w, counts):
        p = predict_population_baseline(kind, unpack(w), train_plan)
        nll = -jnp.sum(counts * jnp.log(jnp.maximum(p, 1e-30))) / jnp.sum(counts)
        return nll + reg * jnp.sum((w - flat) ** 2)

    return (
        np.asarray(flat),
        unpack,
        jax.jit(jax.value_and_grad(loss)),
        jax.jit(lambda w: predict_population_baseline(kind, unpack(w), test_plan)),
    )
