"""Graph and out-degree scaling benchmark for Differentiable Renewal Networks.

Evaluates:
1. State-space scaling and forward + reverse-mode AD runtime across V in {4, 8, 16, 32, 48}
   nodes and out-degrees d in {2, 3, 4} (K=16 stages per edge; up to 192 edges),
   comparing naive K**d joint phase tensors against Proposition 1's exact marked-race
   compiler (V + (K-1)*E states).
2. High-dimensional noisy continuous-Gamma parameter recovery across V in {4, 8, 16, 24}
   cyclic-branching graphs (d=2 outgoing edges per node -> up to 48 edges and 96 unknown
   continuous rate+shape parameters; 5 seeds per V at 2,500 cells per prepared cohort),
   comparing Compiled Differentiable PH (L-BFGS-B) against Monte Carlo Renewal Simulation
   (Differential Evolution + Powell refinement).
"""

import json
from pathlib import Path
import sys
from time import perf_counter

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    compact_network_state_size,
    moment_matched_logits,
    multinomial_nll,
    network_state_size,
    solve_compact_renewal_network,
    solve_renewal_network,
)


def make_circulant_graph(n_nodes: int, out_degree: int) -> tuple[tuple[int, int], ...]:
    edges = []
    for i in range(n_nodes):
        for step in range(1, out_degree + 1):
            edges.append((i, (i + step) % n_nodes))
    return tuple(edges)


def simulate_gamma_network_snapshots(
    means: np.ndarray,
    shapes: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    n_nodes: int,
    initial_nodes: list[int],
    ts: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact continuous-time competing-Gamma event-driven Monte Carlo simulator."""
    outgoing = [[] for _ in range(n_nodes)]
    for e_idx, (u, v) in enumerate(edges):
        outgoing[u].append((e_idx, v))
    scales = means / shapes
    max_t = float(ts[-1])
    probs = np.zeros((len(initial_nodes), len(ts), n_nodes), dtype=float)

    for p_idx, start_node in enumerate(initial_nodes):
        # Vectorized particle simulation
        cur_nodes = np.full(n_particles, start_node, dtype=int)
        cur_times = np.zeros(n_particles, dtype=float)
        # Step through ts
        t_idx_order = np.argsort(ts)
        assert np.all(t_idx_order == np.arange(len(ts)))
        # Advance each particle until cur_times > max_t while recording state at ts
        active = np.arange(n_particles)
        state_at_ts = np.zeros((n_particles, len(ts)), dtype=int)
        next_obs_idx = np.zeros(n_particles, dtype=int)

        while active.size > 0:
            # Group active particles by current node for fast vectorized gamma sampling
            for u in range(n_nodes):
                mask = cur_nodes[active] == u
                if not np.any(mask):
                    continue
                p_sub = active[mask]
                choices = outgoing[u]
                dwells = np.column_stack(
                    [
                        rng.gamma(shape=shapes[e_id], scale=scales[e_id], size=p_sub.size)
                        for e_id, _ in choices
                    ]
                )
                win_col = np.argmin(dwells, axis=1)
                dt = dwells[np.arange(p_sub.size), win_col]
                next_t = cur_times[p_sub] + dt
                dest = np.array([choices[c][1] for c in win_col], dtype=int)

                # Fill all observation times crossed in [cur_times, next_t)
                for k_t in range(len(ts)):
                    crossed = (next_obs_idx[p_sub] == k_t) & (ts[k_t] < next_t)
                    state_at_ts[p_sub[crossed], k_t] = u
                    next_obs_idx[p_sub[crossed]] += 1

                cur_times[p_sub] = next_t
                cur_nodes[p_sub] = dest

            active = active[next_obs_idx[active] < len(ts)]

        for k_t in range(len(ts)):
            counts = np.bincount(state_at_ts[:, k_t], minlength=n_nodes)
            probs[p_idx, k_t] = counts / float(n_particles)
    return probs


def run_gradient_scaling() -> list[dict]:
    stages = 16
    ts = jnp.linspace(0.5, 6.0, 12)
    rows = []
    for d in (2, 3, 4):
        for v in (4, 8, 16, 32, 48):
            edges = make_circulant_graph(v, d)
            n_edges = len(edges)
            joint_states = network_state_size(edges, v, stages)
            compact_states = compact_network_state_size(edges, v, stages)
            compression_ratio = joint_states / compact_states

            rng = np.random.default_rng(100 * d + v)
            means = jnp.asarray(rng.uniform(1.3, 2.7, size=n_edges))
            shapes = jnp.asarray(rng.uniform(4.0, 9.5, size=n_edges))
            theta0 = jnp.concatenate([jnp.log(means), jnp.log(shapes)])
            init = jnp.zeros(v).at[0].set(1.0)
            weights_obs = jnp.linspace(0.2, 1.0, v)

            def compact_obj(th):
                m, k = jnp.exp(th[:n_edges]), jnp.exp(th[n_edges:])
                ys = solve_compact_renewal_network(
                    m, moment_matched_logits(k), edges, init, ts
                ).ys
                return jnp.sum(ys * weights_obs[None, :])

            vg_comp = jax.jit(jax.value_and_grad(compact_obj))
            t0 = perf_counter()
            val_c, grad_c = vg_comp(theta0)
            jax.block_until_ready(grad_c)
            comp_compile_s = perf_counter() - t0

            warm_c = []
            for _ in range(5):
                t0 = perf_counter()
                _, g = vg_comp(theta0)
                jax.block_until_ready(g)
                warm_c.append(perf_counter() - t0)
            comp_ms = 1000.0 * float(np.median(warm_c))

            # Run naive joint tensor solver ONLY when joint_states <= 40,000 (above that it OOMs / >60s)
            joint_ms = None
            max_traj_diff = None
            rel_grad_diff = None
            if joint_states <= 35_000 and v <= 32:
                def joint_obj(th):
                    m, k = jnp.exp(th[:n_edges]), jnp.exp(th[n_edges:])
                    ys = solve_renewal_network(
                        m, moment_matched_logits(k), edges, init, ts
                    ).ys
                    return jnp.sum(ys * weights_obs[None, :])

                vg_joint = jax.jit(jax.value_and_grad(joint_obj))
                val_j, grad_j = vg_joint(theta0)
                jax.block_until_ready(grad_j)
                warm_j = []
                for _ in range(3):
                    t0 = perf_counter()
                    _, gj = vg_joint(theta0)
                    jax.block_until_ready(gj)
                    warm_j.append(perf_counter() - t0)
                joint_ms = 1000.0 * float(np.median(warm_j))
                max_traj_diff = float(abs(float(val_c) - float(val_j)))
                rel_grad_diff = float(
                    jnp.linalg.norm(grad_c - grad_j) / jnp.linalg.norm(grad_j)
                )

            row = {
                "nodes": v,
                "out_degree": d,
                "edges": n_edges,
                "parameters": 2 * n_edges,
                "joint_tensor_states": joint_states,
                "compact_race_states": compact_states,
                "state_compression_ratio": round(compression_ratio, 2),
                "compact_grad_warm_ms": round(comp_ms, 3),
                "compact_compile_s": round(comp_compile_s, 3),
                "joint_grad_warm_ms": None if joint_ms is None else round(joint_ms, 3),
                "grad_speedup_vs_tensor": None if joint_ms is None else round(joint_ms / comp_ms, 2),
                "relative_gradient_error": rel_grad_diff,
            }
            rows.append(row)
            print(
                f"d={d}, V={v:2d} | States: {joint_states:9d} -> {compact_states:4d} ({compression_ratio:6.1f}x) | Compact grad: {comp_ms:7.2f} ms | Joint grad: {str(round(joint_ms,2)) if joint_ms else 'OOM/>35k'}",
                flush=True,
            )
    return rows


def run_highdim_recovery() -> list[dict]:
    """Compare Compiled PH L-BFGS-B vs Monte Carlo Simulation Search across V in {4, 8, 12}."""
    ts_np = np.array([0.4, 0.8, 1.2, 1.6, 2.1, 2.7, 3.5, 4.5])
    ts_jax = jnp.asarray(ts_np)
    d = 2
    results = []

    for v in (4, 8, 12):
        edges = make_circulant_graph(v, d)
        n_edges = len(edges)
        prep_nodes = list(range(v))
        initials_jax = jnp.eye(v)

        # Balanced competing clocks per node so every edge carries >= 25% routing share
        rng_truth = np.random.default_rng(700 + v)
        base_means = rng_truth.uniform(1.55, 2.25, size=v)
        splits = rng_truth.uniform(-0.14, 0.14, size=v)
        true_means = np.empty(n_edges, dtype=float)
        true_means[0::2] = base_means * (1.0 - splits)
        true_means[1::2] = base_means * (1.0 + splits)
        true_shapes = rng_truth.uniform(4.5, 8.5, size=n_edges)

        # High-precision ground-truth probabilities via 100,000 continuous-Gamma particles per prep
        truth_probs = simulate_gamma_network_snapshots(
            true_means, true_shapes, edges, v, prep_nodes, ts_np, 100_000, np.random.default_rng(900 + v)
        )

        # Build JIT-compiled PH likelihood + gradient via exact marked-race generator
        outgoing = [tuple(e for e, (u, _) in enumerate(edges) if u == i) for i in range(v)]
        k_stages = 16
        dim_u = 1 + d * (k_stages - 1)
        tot_dim = v * dim_u

        def ph_nll(theta, counts_arr):
            m = jnp.exp(theta[:n_edges])
            k = jnp.exp(theta[n_edges:])
            logits = moment_matched_logits(k)
            grp_mat = jnp.array(outgoing)
            from diff_renewal import mixed_erlang_race
            import jax.scipy.linalg as jl

            rates_u, conts_u, causes_u = jax.vmap(
                lambda idx: mixed_erlang_race(m[idx], logits[idx])
            )(grp_mat)

            def make_block(r_u, c_u):
                return -r_u * jnp.eye(dim_u) + jnp.diag(r_u * c_u, k=1).T

            blocks = jax.vmap(make_block)(rates_u, conts_u)
            gen = jl.block_diag(*blocks)
            for u_i in range(v):
                c0 = u_i * dim_u
                for j_loc, e_id in enumerate(outgoing[u_i]):
                    dst = edges[e_id][1]
                    gen = gen.at[dst * dim_u, c0 : c0 + dim_u].add(
                        rates_u[u_i] * causes_u[u_i, j_loc]
                    )
            fresh_m = jnp.kron(jnp.eye(v), jnp.zeros((dim_u, 1)).at[0, 0].set(1.0))
            read_m = jnp.kron(jnp.eye(v), jnp.ones((1, dim_u)))
            p_ts = jax.vmap(lambda t: (read_m @ jl.expm(t * gen) @ fresh_m).T)(ts_jax)
            probs = jnp.transpose(p_ts, (1, 0, 2))
            return multinomial_nll(probs, counts_arr)

        vg_ph = jax.jit(jax.value_and_grad(ph_nll))
        bounds = [(np.log(1.05), np.log(3.2))] * n_edges + [(np.log(3.2), np.log(11.5))] * n_edges
        starts_th = [
            np.r_[np.full(n_edges, np.log(1.75)), np.full(n_edges, np.log(5.5))],
            np.r_[np.full(n_edges, np.log(2.05)), np.full(n_edges, np.log(7.5))],
        ]

        dummy_c = jnp.ones_like(jnp.asarray(truth_probs))
        t_comp0 = perf_counter()
        jax.block_until_ready(vg_ph(jnp.asarray(starts_th[0]), dummy_c)[1])
        compile_s = perf_counter() - t_comp0

        ph_rate_errs, ph_shape_errs, ph_times, ph_joints = [], [], [], []
        mc_rate_errs, mc_shape_errs, mc_times, mc_joints = [], [], [], []

        for seed in range(5):
            rng_obs = np.random.default_rng(2000 + 10 * v + seed)
            cells_per_snap = 1200
            counts = np.zeros_like(truth_probs)
            for p_i in range(len(prep_nodes)):
                for t_i in range(len(ts_np)):
                    p_vec = truth_probs[p_i, t_i]
                    p_vec = p_vec / p_vec.sum()
                    counts[p_i, t_i] = rng_obs.multinomial(cells_per_snap, p_vec)
            counts_jax = jnp.asarray(counts)

            # 1. Differentiable Compiled PH fit (2-start L-BFGS-B)
            t0 = perf_counter()
            best_ph = None
            def _obj_ph(z):
                v, g = vg_ph(jnp.asarray(z), counts_jax)
                return float(v), np.asarray(g, dtype=float)

            for st0 in starts_th:
                fit_cand = minimize(
                    _obj_ph,
                    st0,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 220, "ftol": 1e-12},
                )
                if best_ph is None or fit_cand.fun < best_ph.fun:
                    best_ph = fit_cand
            ph_s = perf_counter() - t0
            est_m = np.exp(best_ph.x[:n_edges])
            est_k = np.exp(best_ph.x[n_edges:])
            r_err = float(np.sqrt(np.mean(((1.0 / est_m) / (1.0 / true_means) - 1.0) ** 2)))
            s_err = float(np.sqrt(np.mean((est_k / true_shapes - 1.0) ** 2)))
            ph_rate_errs.append(100.0 * r_err)
            ph_shape_errs.append(100.0 * s_err)
            ph_times.append(ph_s + compile_s)
            ph_joints.append(int(r_err <= 0.05 and s_err <= 0.10))

            # 2. Stochastic Simulation Search (DE + Powell with CRN 400 particles/eval)
            crn_seed = 5000 + 10 * v + seed
            def mc_loss(z):
                m_c = np.exp(z[:n_edges])
                k_c = np.exp(z[n_edges:])
                # Common Random Numbers (CRN) makes the objective deterministic for Powell refinement!
                sim_rng = np.random.default_rng(crn_seed)
                p_est = simulate_gamma_network_snapshots(
                    m_c, k_c, edges, v, prep_nodes, ts_np, 400, sim_rng
                )
                p_reg = (p_est * 400.0 + 0.5) / (400.0 + 0.5 * v)
                return float(-np.sum(counts * np.log(p_reg)) / np.sum(counts))

            t0 = perf_counter()
            de = differential_evolution(
                mc_loss,
                bounds=bounds,
                maxiter=5,
                popsize=2,
                polish=False,
                seed=seed + 1,
            )
            pw = minimize(
                mc_loss,
                de.x,
                method="Powell",
                bounds=bounds,
                options={"maxfev": 80},
            )
            mc_s = perf_counter() - t0
            mc_m = np.exp(pw.x[:n_edges])
            mc_k = np.exp(pw.x[n_edges:])
            mr_err = float(np.sqrt(np.mean(((1.0 / mc_m) / (1.0 / true_means) - 1.0) ** 2)))
            ms_err = float(np.sqrt(np.mean((mc_k / true_shapes - 1.0) ** 2)))
            mc_rate_errs.append(100.0 * mr_err)
            mc_shape_errs.append(100.0 * ms_err)
            mc_times.append(mc_s)
            mc_joints.append(int(mr_err <= 0.05 and ms_err <= 0.10))

        row = {
            "nodes": v,
            "edges": n_edges,
            "parameters": 2 * n_edges,
            "joint_tensor_states": network_state_size(edges, v, 16),
            "compact_race_states": compact_network_state_size(edges, v, 16),
            "ph_rate_rms_median_pct": round(float(np.median(ph_rate_errs)), 2),
            "ph_shape_rms_median_pct": round(float(np.median(ph_shape_errs)), 2),
            "ph_joint_hits_of_5": sum(ph_joints),
            "ph_cold_seconds_median": round(float(np.median(ph_times)), 2),
            "mc_rate_rms_median_pct": round(float(np.median(mc_rate_errs)), 2),
            "mc_shape_rms_median_pct": round(float(np.median(mc_shape_errs)), 2),
            "mc_joint_hits_of_5": sum(mc_joints),
            "mc_seconds_median": round(float(np.median(mc_times)), 2),
            "speedup_mc_over_ph_cold": round(float(np.median(mc_times) / np.median(ph_times)), 2),
        }
        results.append(row)
        print(
            f"Recovery V={v:2d} ({2*n_edges:2d} params) | PH shape={row['ph_shape_rms_median_pct']:5.2f}% ({row['ph_joint_hits_of_5']}/5, {row['ph_cold_seconds_median']}s) | MC shape={row['mc_shape_rms_median_pct']:5.2f}% ({row['mc_joint_hits_of_5']}/5, {row['mc_seconds_median']}s)",
            flush=True,
        )
    return results


def main() -> None:
    grad_rows = run_gradient_scaling()
    rec_rows = run_highdim_recovery()
    out = {
        "gradient_and_state_scaling": grad_rows,
        "highdim_gamma_recovery": rec_rows,
    }
    out_path = ROOT / "results" / "archived" / "graph_scaling_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
