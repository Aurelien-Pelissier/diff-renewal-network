"""Minimal runnable demonstration of Differentiable Renewal Networks."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diff_renewal import (
    compact_network_state_size,
    moment_matched_logits,
    multinomial_nll,
    network_state_size,
    solve_compact_renewal_network,
    solve_renewal_network,
)


def main() -> None:
    edges = ((0, 1), (0, 2), (1, 3), (2, 3))
    stages = 16
    full_dim = network_state_size(edges, 4, stages)
    compact_dim = compact_network_state_size(edges, 4, stages)

    means = jnp.array([1.8, 2.4, 1.5, 2.1])
    shapes = jnp.array([6.0, 8.5, 5.0, 9.0])
    ts = jnp.linspace(0.5, 6.0, 12)
    initial = jnp.array([1.0, 0.0, 0.0, 0.0])

    def loss_fn(theta):
        m = jnp.exp(theta[:4])
        k = jnp.exp(theta[4:])
        logits = moment_matched_logits(k)
        probs = solve_compact_renewal_network(m, logits, edges, initial, ts).ys
        return multinomial_nll(probs, 200.0 * probs)

    theta0 = jnp.concatenate([jnp.log(means), jnp.log(shapes)])
    logits0 = moment_matched_logits(shapes)
    y_full = solve_renewal_network(means, logits0, edges, initial, ts).ys
    y_comp = solve_compact_renewal_network(means, logits0, edges, initial, ts).ys
    grad = jax.grad(loss_fn)(theta0)

    print(f"Branching diamond state count: {full_dim} (joint tensor) -> {compact_dim} (marked race)")
    print(f"Max trajectory discrepancy:   {float(jnp.max(jnp.abs(y_full - y_comp))):.3e}")
    print(f"Parameter gradient L2 norm:   {float(jnp.linalg.norm(grad)):.3e}")


if __name__ == "__main__":
    main()
