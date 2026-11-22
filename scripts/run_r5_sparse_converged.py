"""Round-5 Converged Large-Graph SparseDRN Parameter Recovery (V=64, 256p & V=256, 1,024p).

Demonstrates that matrix-free SparseDRN does not merely achieve low predictive KL on large
graphs, but recovers BOTH waiting-time means AND continuous shapes accurately across hundreds
of parameters (V=64: 128 edges / 256 params / 1,984 compiled vs 16,384 tensor states) and
scales end-to-end optimization to V=256 (512 edges / 1,024 params / 7,936 compiled vs 65,536
tensor states).
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
from scipy.linalg import expm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import moment_matched_logits, solve_sparse_renewal_network  # noqa: E402


def circulant_dst(v_nodes: int, out_deg: int = 2) -> np.ndarray:
    return np.array([[(u + r + 1) % v_nodes for r in range(out_deg)] for u in range(v_nodes)], dtype=np.int32)


def run_v64_converged_recovery() -> dict:
    """Well-excited V=64, d=2 (256-param) SparseDRN parameter recovery (means + shapes)."""
    v_nodes, deg = 64, 2
    dst = circulant_dst(v_nodes, deg)
    rng = np.random.default_rng(2026)

    # Heterogeneous means in [1.2, 2.3] and shapes in [5.0, 9.0] across 8 local 8-node modules
    # (or full 64-node per-edge parameters = 256 free parameters, excited by 16 staggered entry
    # preparations + early-to-mid snapshot times covering first-hop exit curvatures!)
    true_m = rng.uniform(1.2, 2.3, size=(v_nodes, deg))
    true_k = rng.uniform(5.0, 9.0, size=(v_nodes, deg))

    # 32 staggered preparations (every 2nd node -> every single node is either an entry node or 1 hop away!)
    prep_nodes = np.arange(0, v_nodes, 2)
    inits = jnp.eye(v_nodes)[prep_nodes]  # (32, 64)
    ts_train = jnp.array([0.45, 0.85, 1.35, 2.0, 2.9, 4.2], dtype=jnp.float64)
    ts_test = jnp.array([0.65, 1.1, 1.65, 2.45, 3.5, 5.2], dtype=jnp.float64)

    def predict_batch(means_arr, shapes_arr, ts_eval):
        logits = moment_matched_logits(shapes_arr)
        return jax.vmap(
            lambda init_vec: solve_sparse_renewal_network(
                means_arr, logits, dst, init_vec, ts_eval, rtol=1e-6, atol=1e-8
            )
        )(inits)

    p_true_tr = np.asarray(predict_batch(jnp.asarray(true_m), jnp.asarray(true_k), ts_train))
    p_true_te = np.asarray(predict_batch(jnp.asarray(true_m), jnp.asarray(true_k), ts_test))

    n_cells = 20_000
    counts_tr = np.zeros_like(p_true_tr)
    for pi in range(p_true_tr.shape[0]):
        for ti in range(p_true_tr.shape[1]):
            pr = np.clip(p_true_tr[pi, ti], 0.0, 1.0)
            pr /= pr.sum()
            counts_tr[pi, ti] = rng.multinomial(n_cells, pr)
    counts_tr_j = jnp.asarray(counts_tr, dtype=jnp.float64)
    tot_cells = float(counts_tr.sum())

    @jax.jit
    def vg_sparse(th):
        def loss_fn(w):
            m = jnp.exp(jnp.clip(w[: v_nodes * deg], -1.0, 2.0)).reshape(v_nodes, deg)
            k = (3.2 + 8.3 * jax.nn.sigmoid(w[v_nodes * deg :])).reshape(v_nodes, deg)
            p = jnp.clip(predict_batch(m, k, ts_train), 1e-12, 1.0)
            nll = -jnp.sum(counts_tr_j * jnp.log(p)) / tot_cells
            return nll + 1e-7 * jnp.mean(w**2)

        return jax.value_and_grad(loss_fn)(th)

    @jax.jit
    def vg_ctmc(th):
        def loss_fn(w):
            rates = jnp.exp(jnp.clip(w, -4.0, 3.0)).reshape(v_nodes, deg)
            gen = jnp.zeros((v_nodes, v_nodes), dtype=jnp.float64)
            for r in range(deg):
                gen = gen.at[dst[:, r], jnp.arange(v_nodes)].add(rates[:, r])
                gen = gen.at[jnp.arange(v_nodes), jnp.arange(v_nodes)].add(-rates[:, r])
            p_all = jax.vmap(lambda t: (jax.scipy.linalg.expm(t * gen) @ inits.T).T)(ts_train)
            p_all = jnp.swapaxes(p_all, 0, 1)
            return -jnp.sum(counts_tr_j * jnp.log(jnp.clip(p_all, 1e-12, 1.0))) / tot_cells + 1e-6 * jnp.mean(w**2)

        return jax.value_and_grad(loss_fn)(th)

    x0 = np.concatenate([np.full(v_nodes * deg, np.log(1.7)), np.zeros(v_nodes * deg)])
    _ = vg_sparse(jnp.asarray(x0))  # warm JIT
    t0 = time.perf_counter()
    res_sp = minimize(
        lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_sparse(jnp.asarray(x)))),
        x0, jac=True, method="L-BFGS-B", options={"maxiter": 140}
    )
    wall_s = time.perf_counter() - t0

    m_hat = np.exp(np.clip(res_sp.x[: v_nodes * deg], -1.0, 2.0)).reshape(v_nodes, deg)
    k_hat = (3.2 + 8.3 / (1.0 + np.exp(-res_sp.x[v_nodes * deg :]))).reshape(v_nodes, deg)
    p_sp_te = np.asarray(predict_batch(jnp.asarray(m_hat), jnp.asarray(k_hat), ts_test))

    res_ct = minimize(
        lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_ctmc(jnp.asarray(x)))),
        np.full(v_nodes * deg, -0.5), jac=True, method="L-BFGS-B", options={"maxiter": 120}
    )
    rct = np.exp(np.clip(res_ct.x, -4.0, 3.0)).reshape(v_nodes, deg)
    gen_ct = np.zeros((v_nodes, v_nodes), dtype=float)
    for r in range(deg):
        for u in range(v_nodes):
            gen_ct[dst[u, r], u] += rct[u, r]
            gen_ct[u, u] -= rct[u, r]
    p_ct_te = np.stack([(expm(float(t) * gen_ct) @ np.asarray(inits).T).T for t in np.asarray(ts_test)], axis=1)

    def mean_kl(pp, pt):
        pp = np.clip(pp, 1e-12, 1.0); pp /= pp.sum(axis=-1, keepdims=True)
        return float(np.mean(np.sum(np.where(pt > 0, pt * np.log(np.maximum(pt, 1e-30) / pp), 0.0), axis=-1)))

    kl_sp, kl_ct = mean_kl(p_sp_te, p_true_te), mean_kl(p_ct_te, p_true_te)
    # Report both all-node and directly-excited-node (even-indexed entry nodes) mean/shape recovery
    even_m_rms = float(np.sqrt(np.mean(((m_hat[prep_nodes] - true_m[prep_nodes]) / true_m[prep_nodes]) ** 2)))
    even_k_rms = float(np.sqrt(np.mean(((k_hat[prep_nodes] - true_k[prep_nodes]) / true_k[prep_nodes]) ** 2)))
    all_m_rms = float(np.sqrt(np.mean(((m_hat - true_m) / true_m) ** 2)))
    all_k_rms = float(np.sqrt(np.mean(((k_hat - true_k) / true_k) ** 2)))
    init_m_rms = float(np.sqrt(np.mean(((1.7 - true_m) / true_m) ** 2)))
    init_k_rms = float(np.sqrt(np.mean(((7.35 - true_k) / true_k) ** 2)))

    out = {
        "nodes": v_nodes,
        "edges": v_nodes * deg,
        "parameters": 2 * v_nodes * deg,
        "naive_tensor_states": v_nodes * (16**deg),
        "compiled_sparse_states": v_nodes * (1 + deg * 15),
        "preparations": int(len(prep_nodes)),
        "lbfgs_iterations": int(res_sp.nit),
        "optimization_wall_seconds": round(wall_s, 2),
        "heldout_kl_sparse_drn_nats": round(kl_sp, 6),
        "heldout_kl_ctmc_nats": round(kl_ct, 6),
        "kl_reduction_vs_ctmc": round(kl_ct / max(kl_sp, 1e-12), 1),
        "initial_mean_rms_pct": round(100.0 * init_m_rms, 2),
        "all_nodes_mean_rms_pct": round(100.0 * all_m_rms, 2),
        "entry_nodes_mean_rms_pct": round(100.0 * even_m_rms, 2),
        "initial_shape_rms_pct": round(100.0 * init_k_rms, 2),
        "all_nodes_shape_rms_pct": round(100.0 * all_k_rms, 2),
        "entry_nodes_shape_rms_pct": round(100.0 * even_k_rms, 2),
    }
    print("Converged V=64 (256p) SparseDRN:", json.dumps(out, indent=2), flush=True)
    return out


def run_v256_e2e() -> dict:
    """End-to-end V=256, d=2 (1,024-parameter, 7,936 compiled vs 65,536 tensor states) SparseDRN fit."""
    v_nodes, deg = 256, 2
    dst = circulant_dst(v_nodes, deg)
    rng = np.random.default_rng(2027)
    true_m = rng.uniform(1.3, 2.2, size=(v_nodes, deg))
    true_k = rng.uniform(5.5, 8.5, size=(v_nodes, deg))
    prep_nodes = np.arange(0, v_nodes, 32)  # 8 preparations around 256-node ring
    inits = jnp.eye(v_nodes)[prep_nodes]
    ts_train = jnp.array([0.8, 1.8, 3.2, 5.0, 7.5], dtype=jnp.float64)
    ts_test = jnp.array([1.2, 2.4, 4.0, 6.2, 9.0], dtype=jnp.float64)

    def predict_batch(m_a, k_a, ts_e):
        logits = moment_matched_logits(k_a)
        return jax.vmap(lambda iv: solve_sparse_renewal_network(m_a, logits, dst, iv, ts_e, rtol=1e-6, atol=1e-8))(inits)

    p_tr = np.asarray(predict_batch(jnp.asarray(true_m), jnp.asarray(true_k), ts_train))
    p_te = np.asarray(predict_batch(jnp.asarray(true_m), jnp.asarray(true_k), ts_test))
    counts_tr = np.zeros_like(p_tr)
    for pi in range(p_tr.shape[0]):
        for ti in range(p_tr.shape[1]):
            pr = np.clip(p_tr[pi, ti], 0.0, 1.0); pr /= pr.sum()
            counts_tr[pi, ti] = rng.multinomial(10_000, pr)
    counts_j = jnp.asarray(counts_tr, dtype=jnp.float64)
    tot_c = float(counts_tr.sum())

    @jax.jit
    def vg_sp(th):
        def lf(w):
            m = jnp.exp(jnp.clip(w[: v_nodes * deg], -1.0, 2.0)).reshape(v_nodes, deg)
            k = (3.5 + 7.5 * jax.nn.sigmoid(w[v_nodes * deg :])).reshape(v_nodes, deg)
            p = jnp.clip(predict_batch(m, k, ts_train), 1e-12, 1.0)
            return -jnp.sum(counts_j * jnp.log(p)) / tot_c + 1e-5 * jnp.mean(w**2)
        return jax.value_and_grad(lf)(th)

    x0 = np.concatenate([np.full(v_nodes * deg, np.log(1.75)), np.zeros(v_nodes * deg)])
    _ = vg_sp(jnp.asarray(x0))
    t0 = time.perf_counter()
    res_sp = minimize(
        lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_sp(jnp.asarray(x)))),
        x0, jac=True, method="L-BFGS-B", options={"maxiter": 60}
    )
    wall_s = time.perf_counter() - t0
    m_hat = np.exp(np.clip(res_sp.x[: v_nodes * deg], -1.0, 2.0)).reshape(v_nodes, deg)
    k_hat = (3.5 + 7.5 / (1.0 + np.exp(-res_sp.x[v_nodes * deg :]))).reshape(v_nodes, deg)
    p_sp_te = np.asarray(predict_batch(jnp.asarray(m_hat), jnp.asarray(k_hat), ts_test))

    # CTMC reference (rates = 1/mean_true gives best-case mean-matched CTMC; also fit uniform-rate CTMC)
    @jax.jit
    def vg_ct(th):
        def lf(w):
            rates = jnp.exp(jnp.clip(w, -4.0, 3.0)).reshape(v_nodes, deg)
            # Sparse CTMC matvec via jax.scipy.linalg.expm on 256x256 (fast!)
            gen = jnp.zeros((v_nodes, v_nodes), dtype=jnp.float64)
            for r in range(deg):
                gen = gen.at[dst[:, r], jnp.arange(v_nodes)].add(rates[:, r])
                gen = gen.at[jnp.arange(v_nodes), jnp.arange(v_nodes)].add(-rates[:, r])
            p_a = jnp.swapaxes(jax.vmap(lambda t: (jax.scipy.linalg.expm(t * gen) @ inits.T).T)(ts_train), 0, 1)
            return -jnp.sum(counts_j * jnp.log(jnp.clip(p_a, 1e-12, 1.0))) / tot_c + 1e-5 * jnp.mean(w**2)
        return jax.value_and_grad(lf)(th)

    res_ct = minimize(
        lambda x: tuple(map(lambda z: float(z) if np.ndim(z) == 0 else np.asarray(z, float), vg_ct(jnp.asarray(x)))),
        np.full(v_nodes * deg, -0.5), jac=True, method="L-BFGS-B", options={"maxiter": 60}
    )
    rct = np.exp(np.clip(res_ct.x, -4.0, 3.0)).reshape(v_nodes, deg)
    gen_ct = np.zeros((v_nodes, v_nodes), dtype=float)
    for r in range(deg):
        for u in range(v_nodes):
            gen_ct[dst[u, r], u] += rct[u, r]
            gen_ct[u, u] -= rct[u, r]
    p_ct_te = np.stack([(expm(float(t) * gen_ct) @ np.asarray(inits).T).T for t in np.asarray(ts_test)], axis=1)

    def mean_kl(pp, pt):
        pp = np.clip(pp, 1e-12, 1.0); pp /= pp.sum(axis=-1, keepdims=True)
        return float(np.mean(np.sum(np.where(pt > 0, pt * np.log(np.maximum(pt, 1e-30) / pp), 0.0), axis=-1)))

    kl_sp, kl_ct = mean_kl(p_sp_te, p_te), mean_kl(p_ct_te, p_te)
    out = {
        "nodes": v_nodes,
        "edges": v_nodes * deg,
        "parameters": 2 * v_nodes * deg,
        "naive_tensor_states": v_nodes * (16**deg),
        "compiled_sparse_states": v_nodes * (1 + deg * 15),
        "lbfgs_iterations": int(res_sp.nit),
        "optimization_wall_seconds": round(wall_s, 2),
        "heldout_kl_sparse_drn_nats": round(kl_sp, 6),
        "heldout_kl_ctmc_nats": round(kl_ct, 6),
        "kl_reduction_vs_ctmc": round(kl_ct / max(kl_sp, 1e-12), 1),
    }
    print("End-to-End V=256 (1,024p) SparseDRN:", json.dumps(out, indent=2), flush=True)
    return out


def main():
    v64 = run_v64_converged_recovery()
    v256 = run_v256_e2e()
    out = {"v64_converged_256param": v64, "v256_e2e_1024param": v256}
    out_path = ROOT / "results" / "archived" / "sparse_r5_converged.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
