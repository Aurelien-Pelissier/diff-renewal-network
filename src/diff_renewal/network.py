"""Independent-particle renewal networks with competing, freshly reset clocks."""

import diffrax
import jax
import jax.numpy as jnp

from .phase_type import mixed_erlang_components


def network_layout(edges, n_nodes: int):
    """Validate static topology and return outgoing edge indices per node."""
    if not isinstance(n_nodes, int) or isinstance(n_nodes, bool) or n_nodes < 1:
        raise ValueError("n_nodes must be a positive integer")
    edges = tuple(tuple(edge) for edge in edges)
    for edge in edges:
        if len(edge) != 2 or any(
            not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= n_nodes
            for i in edge
        ):
            raise ValueError("edges must contain valid integer node pairs")
    return edges, tuple(
        tuple(e for e, (source, _) in enumerate(edges) if source == i)
        for i in range(n_nodes)
    )


def network_state_size(edges, n_nodes: int, stages: int) -> int:
    """Number of joint tensor phase states: sum_i K**outdegree(i), including sinks."""
    if not isinstance(stages, int) or isinstance(stages, bool) or stages < 1:
        raise ValueError("stages must be a positive integer")
    _, outgoing = network_layout(edges, n_nodes)
    return sum(stages ** len(group) for group in outgoing)


def solve_renewal_network(
    means,
    logits,
    edges,
    initial,
    ts,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """Differentiate mean occupancies of a finite renewal network via K**d tensors."""
    means, logits = jnp.asarray(means), jnp.asarray(logits)
    initial, ts = jnp.asarray(initial), jnp.asarray(ts)
    if initial.ndim != 1 or initial.size < 1:
        raise ValueError("initial must be a nonempty 1D array")
    edges, outgoing = network_layout(edges, initial.size)
    if (
        not edges
        or means.shape != (len(edges),)
        or logits.ndim != 2
        or logits.shape[0] != len(edges)
        or logits.shape[1] < 1
    ):
        raise ValueError("require E edges, means (E,), and logits (E,K), E,K >= 1")
    if ts.ndim != 1 or ts.size < 1:
        raise ValueError("ts must be a nonempty 1D array")
    rates, alpha = jax.vmap(mixed_erlang_components)(means, logits)
    entries = []
    for group in outgoing:
        entry = jnp.array(1.0, dtype=alpha.dtype)
        for e in group:
            entry = jnp.tensordot(entry, alpha[e], axes=0)
        entries.append(entry)
    entries = tuple(entries)
    y0 = tuple(initial[i] * entry for i, entry in enumerate(entries))

    def field(t, state, args):
        rates_arg, entries_arg = args
        incoming = [jnp.array(0.0, dtype=rates_arg.dtype) for _ in outgoing]
        derivatives = []
        for occupancy, group in zip(state, outgoing):
            derivative = jnp.zeros_like(occupancy)
            for axis, e in enumerate(group):
                flux = rates_arg[e] * occupancy
                shifted = jnp.roll(flux, 1, axis=axis)
                first = [slice(None)] * occupancy.ndim
                first[axis] = 0
                shifted = shifted.at[tuple(first)].set(0.0)
                derivative = derivative + shifted - flux
                exit_flux = jnp.sum(jnp.take(flux, -1, axis=axis))
                destination = edges[e][1]
                incoming[destination] = incoming[destination] + exit_flux
            derivatives.append(derivative)
        return tuple(
            dp + arrival * entry
            for dp, arrival, entry in zip(derivatives, incoming, entries_arg)
        )

    return diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=ts[-1],
        dt0=None,
        y0=y0,
        args=(rates, entries),
        saveat=diffrax.SaveAt(
            ts=ts, fn=lambda t, y, args: jnp.stack([jnp.sum(p) for p in y])
        ),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=100_000,
    )
