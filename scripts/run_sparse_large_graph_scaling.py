"""Matrix-free SparseDRN Scaling Benchmark up to V = 512..1,024 Nodes (>15,000 States).

Demonstrates linear O(V + K*E) memory and runtime scaling of exact forward
propagation and reverse-mode automatic differentiation without materializing
dense N x N generator matrices, plus exact verification against dense
solve_compact_renewal_network at V = 16.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    moment_matched_logits,
    solve_compact_renewal_network,
    solve_sparse_renewal_network,
)


def circulant_dst(v_nodes: int, out_deg: int) -> np.ndarray:
    dst = np.zeros((v_nodes, out_deg), dtype=np.int32)
    for u in range(v_nodes):
        for r in range(out_deg):
            dst[u, r] = (u + r + 1) % v_nodes
    return dst


def main():
    stages = 16
    out_deg = 2
    ts = jnp.linspace(0.5, 4.0, 8)
    rng = np.random.default_rng(77)

    # 1. Exact verification vs dense solve_compact_renewal_network at V = 16
    v_chk = 16
    dst_chk = circulant_dst(v_chk, out_deg)
    edges_chk = tuple((u, int(dst_chk[u, r])) for u in range(v_chk) for r in range(out_deg))
    m_chk = jnp.asarray(rng.uniform(1.2, 2.8, size=(v_chk, out_deg)))
    k_chk = jnp.asarray(rng.uniform(4.0, 10.0, size=(v_chk, out_deg)))
    init_chk = jnp.zeros(v_chk).at[0].set(1.0)
    target_w = jnp.asarray(rng.uniform(0.2, 1.5, size=(len(ts), v_chk)))

    def loss_dense(th):
        m = jnp.exp(th[: v_chk * out_deg])
        k = jnp.exp(th[v_chk * out_deg :])
        ys = solve_compact_renewal_network(m, moment_matched_logits(k), edges_chk, init_chk, ts).ys
        return jnp.sum(ys * target_w)

    def loss_sparse(th):
        m = jnp.exp(th[: v_chk * out_deg]).reshape(v_chk, out_deg)
        k = jnp.exp(th[v_chk * out_deg :]).reshape(v_chk, out_deg)
        logits = moment_matched_logits(k)
        ys = solve_sparse_renewal_network(m, logits, dst_chk, init_chk, ts, rtol=1e-8, atol=1e-10)
        return jnp.sum(ys * target_w)

    th_chk = jnp.concatenate([jnp.log(m_chk.reshape(-1)), jnp.log(k_chk.reshape(-1))])
    v_d, g_d = jax.value_and_grad(loss_dense)(th_chk)
    v_s, g_s = jax.value_and_grad(loss_sparse)(th_chk)
    val_err = float(abs(v_d - v_s))
    grad_rel = float(jnp.linalg.norm(g_d - g_s) / jnp.linalg.norm(g_d))
    print(f"V=16 Dense vs SparseDRN exact check: val_err={val_err:.2e}, grad_rel_err={grad_rel:.2e}")
    assert val_err < 1e-6 and grad_rel < 1e-6

    # 2. Scale matrix-free SparseDRN across V in {16, 64, 128, 256, 512, 1024} (d=2, K=16)
    rows = []
    for v_nodes in [16, 64, 128, 256, 512, 1024]:
        dst = circulant_dst(v_nodes, out_deg)
        m_v = jnp.asarray(rng.uniform(1.2, 2.8, size=(v_nodes, out_deg)))
        k_v = jnp.asarray(rng.uniform(4.0, 10.0, size=(v_nodes, out_deg)))
        th_v = jnp.concatenate([jnp.log(m_v.reshape(-1)), jnp.log(k_v.reshape(-1))])
        init_v = jnp.zeros(v_nodes).at[0].set(1.0)
        w_v = jnp.asarray(rng.uniform(0.2, 1.5, size=(len(ts), v_nodes)))

        @jax.jit
        def fwd_fn(th):
            m = jnp.exp(th[: v_nodes * out_deg]).reshape(v_nodes, out_deg)
            k = jnp.exp(th[v_nodes * out_deg :]).reshape(v_nodes, out_deg)
            logits = moment_matched_logits(k)
            return solve_sparse_renewal_network(m, logits, dst, init_v, ts, rtol=1e-6, atol=1e-8)

        @jax.jit
        def vg_fn(th):
            def obj(t_in):
                ys = fwd_fn(t_in)
                return jnp.sum(ys * w_v)

            return jax.value_and_grad(obj)(th)

        # Warmup + measure forward and reverse-mode gradient wall time
        ys_out = fwd_fn(th_v)
        ys_out.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(5):
            fwd_fn(th_v).block_until_ready()
        fwd_ms = 1000.0 * (time.perf_counter() - t0) / 5.0

        val0, g0 = vg_fn(th_v)
        g0.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(5):
            vg_fn(th_v)[1].block_until_ready()
        grad_ms = 1000.0 * (time.perf_counter() - t0) / 5.0

        compiled_states = v_nodes * (1 + out_deg * (stages - 1))
        tensor_states = v_nodes * (stages**out_deg)
        dense_mat_mb = (compiled_states**2) * 8.0 / (1024.0**2)
        sparse_arr_mb = compiled_states * (2 + out_deg) * 8.0 / (1024.0**2)
        mass_err = float(jnp.max(jnp.abs(ys_out.sum(axis=-1) - 1.0)))
        min_prob = float(jnp.min(ys_out))
        assert mass_err < 1e-6 and min_prob >= -1e-9

        row = {
            "nodes": v_nodes,
            "out_degree": out_deg,
            "stages": stages,
            "parameters": int(th_v.size),
            "tensor_states": int(tensor_states),
            "compiled_states": int(compiled_states),
            "dense_matrix_mb": float(dense_mat_mb),
            "sparse_operator_mb": float(sparse_arr_mb),
            "memory_compression_vs_dense": float(dense_mat_mb / sparse_arr_mb),
            "forward_warm_ms": float(fwd_ms),
            "grad_warm_ms": float(grad_ms),
            "simplex_mass_error": mass_err,
        }
        print(
            f"V={v_nodes:4d} ({row['parameters']:4d}p) | states={compiled_states:6d} (tensor {tensor_states:7d}) | "
            f"mem={sparse_arr_mb:5.2f}MB vs {dense_mat_mb:8.1f}MB dense | fwd={fwd_ms:6.2f}ms | grad={grad_ms:6.2f}ms"
        )
        rows.append(row)

    out = {
        "dense_vs_sparse_v16_check": {
            "value_max_abs_error": val_err,
            "gradient_relative_l2_error": grad_rel,
        },
        "scaling_rows": rows,
    }
    out_path = ROOT / "results" / "archived" / "sparse_large_graph_scaling.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
