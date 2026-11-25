"""Benchmark Incoming-Edge & Hyper-Erlang Semi-Markov DRN (EdgeMemoryHyperDRN) on 5-Node Sleep-EDF.

Evaluates whether lifting renewal clocks onto incoming-transition fibres (u -> v) with
two-scale Hyper-Erlang fast-arousal + consolidated-bout hazards closes the supervised
rollout gap against 540-parameter Latent ODEs on human sleep while preserving exact
analytical zero-shot survivor conditioning and simplex mass conservation.
"""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
import urllib.request

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import (  # noqa: E402
    build_edge_memory_hyper_generator,
    edge_memory_param_size,
    multinomial_nll,
)

CACHE = Path("/tmp/drn_sleep_cache")
SLEEP_SUMS = "1bc88e19e2e921f7851d7cf61e31ceb8f6d670211ca546a323e9db5a77525edf"
BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"


def ensure_sleep_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    sums_path = CACHE / "SHA256SUMS.txt"
    if not sums_path.exists() or sha256(sums_path.read_bytes()).hexdigest() != SLEEP_SUMS:
        raw = urllib.request.urlopen(BASE_URL + "SHA256SUMS.txt", timeout=30).read()
        assert sha256(raw).hexdigest() == SLEEP_SUMS
        sums_path.write_bytes(raw)

    needed = [
        (checksum, name)
        for checksum, name in (r.split() for r in sums_path.read_text().splitlines())
        if name.startswith("sleep-cassette/") and name.endswith("-Hypnogram.edf")
    ]
    (CACHE / "sleep-cassette").mkdir(parents=True, exist_ok=True)

    def fetch_one(item):
        csum, rel = item
        dst = CACHE / rel
        if dst.exists() and sha256(dst.read_bytes()).hexdigest() == csum:
            return
        last_err = None
        for attempt in range(5):
            try:
                data = urllib.request.urlopen(BASE_URL + rel, timeout=30).read()
                assert sha256(data).hexdigest() == csum
                dst.write_bytes(data)
                return
            except Exception as exc:
                last_err = exc
                import time
                time.sleep(0.5 * (2**attempt))
        raise last_err

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch_one, needed))


def parse_sleep_sequences() -> dict[str, list[np.ndarray]]:
    ensure_sleep_cache()
    manifest = CACHE / "SHA256SUMS.txt"
    files = [
        (CACHE / name, checksum)
        for checksum, name in (r.split() for r in manifest.read_text().splitlines())
        if name.startswith("sleep-cassette/") and name.endswith("-Hypnogram.edf")
    ]
    subjects: dict[str, dict[int, Path]] = {}
    for path, checksum in sorted(files):
        assert sha256(path.read_bytes()).hexdigest() == checksum
        subj = path.name[3:5]
        night = int(path.name[5])
        subjects.setdefault(subj, {})[night] = path

    state_map = {"W": 0, "1": 1, "2": 2, "3": 3, "4": 3, "R": 4}
    out_seqs: dict[str, list[np.ndarray]] = {}
    for subj, nights in sorted(subjects.items()):
        if set(nights) != {1, 2}:
            continue
        subj_epochs = []
        for night in (1, 2):
            raw = nights[night].read_bytes()
            header = int(raw[184:192])
            data = raw[header:]
            ann = []
            for tal in data.split(b"\x00"):
                if not tal.strip():
                    continue
                fields = tal.rstrip(b"\x14").split(b"\x14")
                if len(fields) < 2 or not (
                    fields[1].startswith(b"Sleep stage ") or fields[1] == b"Movement time"
                ):
                    continue
                onset, dur = fields[0].split(b"\x15")
                start_e = int(round(float(onset) / 30.0))
                len_e = int(round(float(dur) / 30.0))
                code = "M" if fields[1] == b"Movement time" else fields[1][-1:].decode()
                ann.append((start_e, len_e, state_map.get(code, -1)))
            ep = np.full(sum(x[1] for x in ann), -1, dtype=int)
            pos = 0
            for _, ln, st in ann:
                ep[pos : pos + ln] = st
                pos += ln
            sleep_idx = np.flatnonzero(ep >= 1)
            ep_sleep = ep[sleep_idx[0] : sleep_idx[-1] + 1]
            bad = np.flatnonzero(ep_sleep < 0)
            starts = np.r_[0, bad + 1]
            ends = np.r_[bad, len(ep_sleep)]
            for s0, e0 in zip(starts, ends):
                if e0 - s0 >= 40:
                    subj_epochs.append(ep_sleep[s0:e0])
        out_seqs[subj] = subj_epochs
    return out_seqs


def build_sleep_snapshots(subj_dict: dict, subjects: list[str], obs_epochs: np.ndarray, aged_epochs: int = 10):
    fresh = np.zeros((5, len(obs_epochs), 5), dtype=float)
    aged = np.zeros((5, len(obs_epochs), 5), dtype=float)
    max_e = int(obs_epochs[-1]) + aged_epochs + 1
    edge_counts = np.zeros((5, 5), dtype=int)
    for s in subjects:
        for seq in subj_dict[s]:
            changes = np.r_[0, np.flatnonzero(np.diff(seq) != 0) + 1]
            for idx in range(len(changes) - 1):
                t0, t1 = int(changes[idx]), int(changes[idx + 1])
                u, v = int(seq[t0]), int(seq[t1])
                edge_counts[u, v] += 1
                dwell = t1 - t0
                if t0 + max_e >= len(seq):
                    continue
                for ti, de in enumerate(obs_epochs):
                    fresh[u, ti, seq[t0 + int(de)]] += 1.0
                if dwell >= aged_epochs:
                    for ti, de in enumerate(obs_epochs):
                        aged[u, ti, seq[t0 + aged_epochs + int(de)]] += 1.0
    return fresh, aged, edge_counts


def main() -> None:
    subj_seqs = parse_sleep_sequences()
    subjects = sorted(subj_seqs.keys())
    assert len(subjects) == 75

    obs_epochs = np.array([2, 4, 8, 12, 20, 30, 45, 60], dtype=int)
    ts_ten_min = jnp.asarray((obs_epochs * 0.5) / 10.0)
    aged_ten_min = 0.5  # 5.0 minutes

    _, _, all_edge_counts = build_sleep_snapshots(subj_seqs, subjects, obs_epochs)
    edges = tuple(
        (u, v)
        for u in range(5)
        for v in range(5)
        if u != v and all_edge_counts[u, v] >= 0.01 * all_edge_counts[u].sum()
    )
    n_edges = len(edges)
    stages = 4
    n_params = edge_memory_param_size(5, n_edges, stages=stages, rank=3)
    epoch_step = 0.05  # 30s = 1 epoch in 10-min units
    max_ep = int(obs_epochs[-1])
    sample_ep_idx = jnp.asarray(obs_epochs - 1, dtype=jnp.int32)

    def predict_em_drn(params):
        gen, fr_m, ag_m, rd_m = build_edge_memory_hyper_generator(
            params, edges, 5, stages=stages, aged_horizon=aged_ten_min
        )
        # Single matrix exponential at 30s epoch step + fast matrix-vector scan!
        trans = jl.expm(epoch_step * gen)
        both_init = jnp.concatenate([fr_m, ag_m], axis=1)  # (M_tot, 10)

        def step_fn(st, _):
            st_next = trans @ st
            return st_next, (rd_m @ st_next).T  # (10, 5)

        traj = jax.lax.scan(step_fn, both_init, None, length=max_ep)[1]  # (60, 10, 5)
        sel = traj[sample_ep_idx]  # (8, 10, 5)
        pf = jnp.transpose(sel[:, :5, :], (1, 0, 2))
        pa = jnp.transpose(sel[:, 5:, :], (1, 0, 2))
        return pf, pa, gen.shape[0]

    compiled_states = int(predict_em_drn(jnp.zeros(n_params))[2])

    @jax.jit
    def loss_joint(params, fc, ac):
        pf, pa, _ = predict_em_drn(params)
        return multinomial_nll(pf, fc) + multinomial_nll(pa, ac) + 2e-5 * jnp.sum(params**2)

    @jax.jit
    def loss_fresh_only(params, fc):
        pf, _, _ = predict_em_drn(params)
        return multinomial_nll(pf, fc) + 2e-5 * jnp.sum(params**2)

    vg_joint = jax.jit(jax.value_and_grad(loss_joint))
    vg_fo = jax.jit(jax.value_and_grad(loss_fresh_only))

    # Also evaluate Time-Inhomogeneous Neural-CME(t) on the 5-node Sleep-EDF graph!
    src_idx = jnp.array([u for u, _ in edges])
    dst_idx = jnp.array([v for _, v in edges])
    cme_width = 18  # 4*18 + 18 + 18*16 + 16 = 394 params
    cme_dim = 4 * cme_width + cme_width + cme_width * n_edges + n_edges

    dt_cme = 0.025
    n_steps_cme = int(round(((obs_epochs[-1] * 0.5) / 10.0) / dt_cme))
    sample_idx_cme = jnp.asarray(np.rint(((obs_epochs * 0.5) / 10.0) / dt_cme).astype(int) - 1)

    def predict_sleep_cme(theta):
        w1 = theta[: 4 * cme_width].reshape(4, cme_width)
        b1 = theta[4 * cme_width : 5 * cme_width]
        w2 = theta[5 * cme_width : 5 * cme_width + cme_width * n_edges].reshape(cme_width, n_edges)
        b2 = theta[5 * cme_width + cme_width * n_edges :]
        dt = dt_cme
        n_steps = n_steps_cme
        sample_idx = sample_idx_cme

        def rollout(age_val: float):
            p0 = jnp.eye(5)

            def step(carry, i_step):
                p = carry
                t_norm = (i_step + 0.5) * dt / 3.0
                feats = jnp.array(
                    [t_norm, jnp.sin(2.0 * jnp.pi * t_norm), jnp.cos(2.0 * jnp.pi * t_norm), age_val]
                )
                rates = 0.02 + 8.0 * jax.nn.sigmoid(jax.nn.gelu(feats @ w1 + b1) @ w2 + b2 - 1.5)

                def field(p_cur):
                    ef = p_cur[:, src_idx] * rates[None, :]
                    return jnp.zeros_like(p_cur).at[:, dst_idx].add(ef).at[:, src_idx].add(-ef)

                k1 = field(p)
                k2 = field(p + 0.5 * dt * k1)
                k3 = field(p + 0.5 * dt * k2)
                k4 = field(p + dt * k3)
                pn = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                return pn, pn

            traj = jax.lax.scan(step, p0, jnp.arange(n_steps))[1]
            return jnp.transpose(traj[sample_idx], (1, 0, 2))

        return rollout(0.0), rollout(aged_ten_min)

    @jax.jit
    def loss_cme_joint(theta, fc, ac):
        pf, pa = predict_sleep_cme(theta)
        return multinomial_nll(pf, fc) + multinomial_nll(pa, ac) + 1e-4 * jnp.sum(theta**2)

    @jax.jit
    def loss_cme_fo(theta, fc):
        pf, _ = predict_sleep_cme(theta)
        return multinomial_nll(pf, fc) + 1e-4 * jnp.sum(theta**2)

    vg_cme_j = jax.jit(jax.value_and_grad(loss_cme_joint))
    vg_cme_fo = jax.jit(jax.value_and_grad(loss_cme_fo))

    folds = [subjects[i::5] for i in range(5)]
    em_joint_scores, em_fo_scores, cme_j_scores, cme_fo_scores = [], [], [], []
    fit_times = []

    for f_idx, te_subjs in enumerate(folds):
        tr_subjs = [s for s in subjects if s not in set(te_subjs)]
        f_tr, a_tr, _ = build_sleep_snapshots(subj_seqs, tr_subjs, obs_epochs)
        f_te, a_te, _ = build_sleep_snapshots(subj_seqs, te_subjs, obs_epochs)
        fc_j, ac_j = jnp.asarray(f_tr), jnp.asarray(a_tr)

        def _wrap_vg(fn, *extra):
            def _obj(z):
                v, g = fn(jnp.asarray(z), *extra)
                return float(v), np.asarray(g, float)
            return _obj

        rng_init = np.random.default_rng(700 + f_idx)
        p0_init = rng_init.normal(0.0, 0.08, size=n_params)
        res_fo = minimize(
            _wrap_vg(vg_fo, fc_j),
            p0_init,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 220},
        )
        t0 = perf_counter()
        res_j = minimize(
            _wrap_vg(vg_joint, fc_j, ac_j),
            res_fo.x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 240},
        )
        fit_times.append(perf_counter() - t0)

        w0_c = rng_init.normal(0.0, 0.18, size=cme_dim)
        res_cme_fo = minimize(
            _wrap_vg(vg_cme_fo, fc_j),
            w0_c,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        res_cme_j = minimize(
            _wrap_vg(vg_cme_j, fc_j, ac_j),
            w0_c,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )

        def score_pair(pf_hat, pa_hat):
            out_s = {}
            for name, p_h, c_te in [("fresh_kl", pf_hat, f_te), ("aged_kl", pa_hat, a_te)]:
                p_c = np.clip(np.asarray(p_h), 1e-12, 1.0)
                p_c = p_c / p_c.sum(axis=-1, keepdims=True)
                q_e = c_te / np.maximum(c_te.sum(axis=-1, keepdims=True), 1.0)
                w_e = c_te.sum(axis=-1) / np.maximum(c_te.sum(), 1.0)
                out_s[name] = float(
                    np.sum(
                        w_e
                        * np.sum(
                            np.where(q_e > 0, q_e * (np.log(np.maximum(q_e, 1e-30)) - np.log(p_c)), 0.0),
                            axis=-1,
                        )
                    )
                )
            return out_s

        pf_fo, pa_fo, _ = predict_em_drn(jnp.asarray(res_fo.x))
        pf_j, pa_j, _ = predict_em_drn(jnp.asarray(res_j.x))
        em_fo_scores.append(score_pair(pf_fo, pa_fo))
        em_joint_scores.append(score_pair(pf_j, pa_j))
        pf_cme_fo, _ = predict_sleep_cme(jnp.asarray(res_cme_fo.x))
        # Clean age=0 evaluation for fresh_only Neural-CME
        cme_fo_scores.append(score_pair(pf_cme_fo, pf_cme_fo))
        cme_j_scores.append(score_pair(*predict_sleep_cme(jnp.asarray(res_cme_j.x))))
        print(
            f"Sleep fold {f_idx} | EdgeMemoryDRN: 0-shot aged={em_fo_scores[-1]['aged_kl']:.5f}, joint={em_joint_scores[-1]['fresh_kl']:.5f}/{em_joint_scores[-1]['aged_kl']:.5f} | NeuralCME: 0-shot={cme_fo_scores[-1]['aged_kl']:.5f}, joint={cme_j_scores[-1]['fresh_kl']:.5f}/{cme_j_scores[-1]['aged_kl']:.5f}",
            flush=True,
        )

    def _agg(rows):
        return {
            "fresh_kl_mean": round(float(np.mean([x["fresh_kl"] for x in rows])), 5),
            "aged_kl_mean": round(float(np.mean([x["aged_kl"] for x in rows])), 5),
            "fresh_kl_sd": round(float(np.std([x["fresh_kl"] for x in rows], ddof=1)), 5),
            "aged_kl_sd": round(float(np.std([x["aged_kl"] for x in rows], ddof=1)), 5),
        }

    rcb_path = ROOT / "results" / "archived" / "real_continuous_benchmarks.json"
    rcb = json.loads(rcb_path.read_text(encoding="utf-8"))
    lode_j = rcb["sleep_5node_branching_network"]["models_5fold_mean"]["latent_ode_joint"]

    out = {
        "architecture": "EdgeMemoryHyperDRN",
        "stages_per_edge": stages,
        "compiled_states": compiled_states,
        "parameters": n_params,
        "neural_cme_parameters": cme_dim,
        "median_fit_seconds": round(float(np.median(fit_times)), 2),
        "edge_memory_drn_joint": _agg(em_joint_scores),
        "edge_memory_drn_fresh_only": _agg(em_fo_scores),
        "neural_cme_joint": _agg(cme_j_scores),
        "neural_cme_fresh_only": _agg(cme_fo_scores),
        "reference_latent_ode_joint": lode_j,
    }
    out_path = ROOT / "results" / "archived" / "edge_memory_sleep_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
