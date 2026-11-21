"""End-to-End Large-Graph SparseDRN Training (V=64, 256 params) & Peak Autodiff RSS Audit.

1. Measures actual peak resident process memory (Linux VmHWM) in isolated
   subprocesses during reverse-mode Diffrax differentiation of SparseDRN across
   V in {16, 64, 256, 512, 1024} (d=2, K=16 -> up to 31,744 compiled states),
   contrasting static operator storage (0.97 MB) with actual peak autodiff RSS.
2. Trains SparseDRN end-to-end with L-BFGS-B on a heterogeneous V=64, d=2
   non-Markovian network (128 directed edges, 256 continuous mean & shape
   parameters, 16,384 tensor states -> 1,984 compiled sparse states) from
   destructive population snapshots, recovering rates and shapes accurately while
   dramatically outperforming a memoryless CTMC on held-out snapshot KL.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import moment_matched_logits, solve_sparse_renewal_network  # noqa: E402


def circulant_dst(v_nodes: int, out_deg: int) -> np.ndarray:
    return np.array([[(u + r + 1) % v_nodes for r in range(out_deg)] for u in range(v_nodes)], dtype=np.int32)


def measure_peak_rss_subprocess(v_nodes: int) -> dict:
    """Measure isolated subprocess peak resident memory (VmHWM in MB) for SparseDRN backprop."""
    code = f"""
import os, sys, time
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp, numpy as np
sys.path.insert(0, '{ROOT / "src"}')
from diff_renewal import moment_matched_logits, solve_sparse_renewal_network

def read_vmhwm_mb():
    for line in open('/proc/self/status'):
        if line.startswith('VmHWM:'):
            return float(line.split()[1]) / 1024.0
    return 0.0

base_mb = read_vmhwm_mb()
v = {v_nodes}
deg, stages = 2, 16
dst = np.array([[(u + 1) % v, (u + 2) % v] for u in range(v)], dtype=np.int32)
rng = np.random.default_rng(77)
th = jnp.concatenate([jnp.log(rng.uniform(1.2, 2.8, size=v * deg)), jnp.log(rng.uniform(4.0, 10.0, size=v * deg))])
init_v = jnp.zeros(v).at[0].set(1.0)
ts = jnp.linspace(0.5, 4.0, 8)
w_v = jnp.asarray(rng.uniform(0.2, 1.5, size=(8, v)))

@jax.jit
def vg(t_in):
    def obj(z):
        m = jnp.exp(z[:v * deg]).reshape(v, deg)
        k = jnp.exp(z[v * deg:]).reshape(v, deg)
        ys = solve_sparse_renewal_network(m, moment_matched_logits(k), dst, init_v, ts, rtol=1e-6, atol=1e-8)
        return jnp.sum(ys * w_v)
    return jax.value_and_grad(obj)(t_in)

val, g = vg(th); g.block_until_ready()
t0 = time.perf_counter()
for _ in range(3):
    vg(th)[1].block_until_ready()
grad_ms = 1000.0 * (time.perf_counter() - t0) / 3.0
peak_mb = read_vmhwm_mb()
print(f'RESULT {{base_mb:.2f}} {{peak_mb:.2f}} {{peak_mb - base_mb:.2f}} {{grad_ms:.2f}}')
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
    )
    for ln in proc.stdout.splitlines():
        if ln.startswith("RESULT "):
            _, b, pk, delta, gms = ln.split()
            compiled = v_nodes * (1 + 2 * 15)
            return {
                "nodes": v_nodes,
                "compiled_states": compiled,
                "static_operator_mb": round(compiled * 4 * 8 / (1024**2), 3),
                "baseline_jax_rss_mb": float(b),
                "peak_total_rss_mb": float(pk),
                "peak_autodiff_rss_delta_mb": float(delta),
                "grad_warm_ms": float(gms),
            }
    raise RuntimeError(f"Failed to parse subprocess RSS output: {proc.stdout}\n{proc.stderr}")


def run_e2e_v64_optimization() -> dict:
    """Train SparseDRN end-to-end on a heterogeneous V=64, d=2 (256-parameter) network."""
    v_nodes, deg = 64, 2
    dst = circulant_dst(v_nodes, deg)
    rng = np.random.default_rng(2026)

    # Heterogeneous true node-edge means in [1.1, 2.4] and non-exponential shapes in [4.5, 9.5]
    # Use 8 node-cluster blocks (each 8 nodes) with per-edge heterogeneous offsets -> 256 free params!
    true_m = rng.uniform(1.2, 2.3, size=(v_nodes, deg))
    true_k = rng.uniform(5.0, 9.0, size=(v_nodes, deg))

    # 8 distinct single-node entry preparations spaced around the 64-node ring + 8 observation times
    prep_nodes = np.arange(0, v_nodes, 8)  # 8 preparations -> covers all 64 nodes cleanly
    inits = jnp.eye(v_nodes)[prep_nodes]  # (8, 64)
    ts_train = jnp.array([0.8, 1.6, 2.5, 3.6, 5.0, 6.8, 9.0, 12.0], dtype=jnp.float64)
    ts_test = jnp.array([1.2, 2.0, 3.0, 4.3, 6.0, 8.0, 10.5, 14.0], dtype=jnp.float64)

    def predict_batch(means_arr, shapes_arr, ts_eval):
        logits = moment_matched_logits(shapes_arr)
        return jax.vmap(
            lambda init_vec: solve_sparse_renewal_network(
                means_arr, logits, dst, init_vec, ts_eval, rtol=1e-6, atol=1e-8
            )
        )(inits)  # (8, 8, 64)

    p_true_tr = np.asarray(predict_batch(jnp.asarray(true_m), jnp.asarray(true_k), ts_train))
    p_true_te = np.asarray(predict_batch(jnp.asarray(true_m), jnp.asarray(true_k), ts_test))

    # Sample destructive multinomial snapshots (10,000 cells per preparation-time stratum)
    n_cells = 10_000
    counts_tr = np.zeros_like(p_true_tr)
    for pi in range(p_true_tr.shape[0]):
        for ti in range(p_true_tr.shape[1]):
            pr = np.clip(p_true_tr[pi, ti], 0.0, 1.0)
            pr /= pr.sum()
            counts_tr[pi, ti] = rng.multinomial(n_cells, pr)
    counts_tr_j = jnp.asarray(counts_tr, dtype=jnp.float64)
    tot_cells = float(counts_tr.sum())

    # Fit 256-parameter SparseDRN end-to-end via L-BFGS-B
    @jax.jit
    def vg_sparse(th):
        def loss_fn(w):
            m = jnp.exp(jnp.clip(w[: v_nodes * deg], -1.0, 2.0)).reshape(v_nodes, deg)
            k = (3.0 + 8.5 * jax.nn.sigmoid(w[v_nodes * deg :])).reshape(v_nodes, deg)
            p = jnp.clip(predict_batch(m, k, ts_train), 1e-12, 1.0)
            nll = -jnp.sum(counts_tr_j * jnp.log(p)) / tot_cells
            return nll + 1e-5 * jnp.mean(w**2)

        return jax.value_and_grad(loss_fn)(th)

    # Fit 128-parameter memoryless CTMC baseline on identical V=64 snapshots
    @jax.jit
    def vg_ctmc(th):
        def loss_fn(w):
            rates = jnp.exp(jnp.clip(w, -4.0, 3.0)).reshape(v_nodes, deg)
            # Matrix-free CTMC (K=1): compile via sparse operator or small 64x64 expm
            gen = jnp.zeros((v_nodes, v_nodes), dtype=jnp.float64)
            for r in range(deg):
                gen = gen.at[dst[:, r], jnp.arange(v_nodes)].add(rates[:, r])
                gen = gen.at[jnp.arange(v_nodes), jnp.arange(v_nodes)].add(-rates[:, r])
            p_all = jax.vmap(lambda t: (jax.scipy.linalg.expm(t * gen) @ inits.T).T)(ts_train)
            p_all = jnp.swapaxes(p_all, 0, 1)  # (8, 8, 64)
            nll = -jnp.sum(counts_tr_j * jnp.log(jnp.clip(p_all, 1e-12, 1.0))) / tot_cells
            return nll + 1e-5 * jnp.mean(w**2)

        return jax.value_and_grad(loss_fn)(th)

    # Warm JIT compile then time full end-to-end optimization
    x0_sp = np.concatenate([np.full(v_nodes * deg, np.log(1.7)), np.zeros(v_nodes * deg)])
    _ = vg_sparse(jnp.asarray(x0_sp))
    t0 = time.perf_counter()
    res_sp = minimize(
        lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_sparse(jnp.asarray(x)))),
        x0_sp,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 80},
    )
    opt_s = time.perf_counter() - t0

    m_hat = np.exp(np.clip(res_sp.x[: v_nodes * deg], -1.0, 2.0)).reshape(v_nodes, deg)
    k_hat = (3.0 + 8.5 / (1.0 + np.exp(-res_sp.x[v_nodes * deg :]))).reshape(v_nodes, deg)
    p_sp_te = np.asarray(predict_batch(jnp.asarray(m_hat), jnp.asarray(k_hat), ts_test))

    # Optimize CTMC
    res_ct = minimize(
        lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_ctmc(jnp.asarray(x)))),
        np.full(v_nodes * deg, -0.5),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 120},
    )
    rates_ct = np.exp(np.clip(res_ct.x, -4.0, 3.0)).reshape(v_nodes, deg)
    gen_ct = np.zeros((v_nodes, v_nodes), dtype=float)
    for r in range(deg):
        for u in range(v_nodes):
            gen_ct[dst[u, r], u] += rates_ct[u, r]
            gen_ct[u, u] -= rates_ct[u, r]
    from scipy.linalg import expm

    p_ct_te = np.stack([(expm(float(t) * gen_ct) @ np.asarray(inits).T).T for t in np.asarray(ts_test)], axis=1)

    def mean_kl(p_pred, p_true):
        pp = np.clip(p_pred, 1e-12, 1.0)
        pp /= pp.sum(axis=-1, keepdims=True)
        return float(np.mean(np.sum(np.where(p_true > 0, p_true * np.log(np.maximum(p_true, 1e-30) / pp), 0.0), axis=-1)))

    kl_sp = mean_kl(p_sp_te, p_true_te)
    kl_ct = mean_kl(p_ct_te, p_true_te)
    # Initial vs final mean/shape RMS error across all 128 edges (256 parameters!)
    init_m_rms = float(np.sqrt(np.mean(((1.7 - true_m) / true_m) ** 2)))
    final_m_rms = float(np.sqrt(np.mean(((m_hat - true_m) / true_m) ** 2)))
    init_k_rms = float(np.sqrt(np.mean(((7.25 - true_k) / true_k) ** 2)))
    final_k_rms = float(np.sqrt(np.mean(((k_hat - true_k) / true_k) ** 2)))

    out_e2e = {
        "nodes": v_nodes,
        "edges": v_nodes * deg,
        "parameters": 2 * v_nodes * deg,
        "naive_tensor_states": v_nodes * (16**deg),
        "compiled_sparse_states": v_nodes * (1 + deg * 15),
        "lbfgs_iterations": int(res_sp.nit),
        "optimization_wall_seconds": round(opt_s, 2),
        "heldout_kl_sparse_drn_nats": round(kl_sp, 6),
        "heldout_kl_ctmc_nats": round(kl_ct, 6),
        "kl_reduction_vs_ctmc": round(kl_ct / max(kl_sp, 1e-12), 1),
        "initial_mean_rms_pct": round(100.0 * init_m_rms, 2),
        "recovered_mean_rms_pct": round(100.0 * final_m_rms, 2),
        "initial_shape_rms_pct": round(100.0 * init_k_rms, 2),
        "recovered_shape_rms_pct": round(100.0 * final_k_rms, 2),
    }
    print("End-to-End V=64 (256-param) SparseDRN Optimization:", json.dumps(out_e2e, indent=2), flush=True)
    return out_e2e


def main():
    rss_rows = []
    for v_nodes in [16, 64, 256, 512, 1024]:
        row = measure_peak_rss_subprocess(v_nodes)
        print(
            f"RSS V={v_nodes:4d} ({row['compiled_states']:5d} st) | static={row['static_operator_mb']:5.2f}MB | "
            f"autodiff_delta={row['peak_autodiff_rss_delta_mb']:6.1f}MB | total_peak={row['peak_total_rss_mb']:6.1f}MB | grad={row['grad_warm_ms']:6.1f}ms",
            flush=True,
        )
        rss_rows.append(row)

    e2e = run_e2e_v64_optimization()
    out = {
        "peak_rss_scaling": rss_rows,
        "e2e_v64_256param_optimization": e2e,
    }
    out_path = ROOT / "results" / "archived" / "sparse_e2e_and_memory.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
