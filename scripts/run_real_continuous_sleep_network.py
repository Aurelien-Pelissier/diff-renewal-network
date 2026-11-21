"""Flagship Real-Data Continuous-Time Renewal Network & Fisher Design Benchmark.

Evaluates Differentiable Renewal Networks end-to-end on two public longitudinal
biological cohorts converted into destructive population-snapshot inverse problems
with exact held-out longitudinal waiting-time ground truth:

1. Human 5-Stage Sleep Transition Network (Sleep-EDF Expanded, 75 paired-night adults,
   19,125 bouts across V=5 stages {W, N1, N2, SWS, REM} with branching competing exits
   of out-degree up to d=4; K=12 stages -> 103,680 naive tensor states vs. 185 exact
   marked-race compiled states). 5-fold participant cross-validation evaluates fresh
   stage-entry cohorts, survivor-aged cohorts (survived >= 5 min in stage), and unseen
   empirical dwell-time survival recovery.
2. Single-Cell Bacterial Cell-Cycle Progression (Tanouchi et al., 2017; 18,972 complete
   Escherichia coli division cycles across 25 C, 27 C, and 37 C in 49 imaging-field
   clusters). Tracks continuous 3-phase cell-size progression (Early -> Mid -> Late ->
   Division -> Early directed 3-node cycle!) with temperature as an observed covariate,
   comparing Uniform vs. 2-Stage Fisher snapshot allocation and testing fresh vs.
   aged-survivor forecasting and single-cell cycle-duration shape recovery!
"""

from hashlib import md5, sha256
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import zipfile

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
    compact_network_state_size,
    mixed_erlang_components,
    mixed_erlang_moments,
    mixed_erlang_race,
    multinomial_nll,
    network_state_size,
    optimize_fisher_allocation,
    smooth_erlang_logits,
    survivor_phases,
)

CACHE = Path("/tmp/drn_real_data_cache")
SLEEP_SUMS = "1bc88e19e2e921f7851d7cf61e31ceb8f6d670211ca546a323e9db5a77525edf"
TANOUCHI_META = {
    25: "56db1c7c9e91e53f859c211e442413a9",
    27: "0f45d32d6df0e833c56a363db9b33131",
    37: "8db408b9826cd881f3771744ff503c51",
}


# ---------------------------------------------------------------------------
# 1. Extract continuous-time trajectories from verified public records
# ---------------------------------------------------------------------------
def load_tanouchi_cycles() -> list[dict]:
    """Extract continuous 3-phase growth cycles (Early, Mid, Late) by log-length trisection."""
    records = []
    for temp, expected_md5 in TANOUCHI_META.items():
        zpath = CACHE / f"tanouchi_{temp}.zip"
        assert md5(zpath.read_bytes()).hexdigest() == expected_md5
        with zipfile.ZipFile(zpath) as zf:
            for name in sorted(zf.namelist()):
                hit = re.fullmatch(r"(?:[^./]+/)?(xy\d+)_(\d+)\.txt", name)
                if not hit:
                    continue
                arr = np.loadtxt(zf.open(name), delimiter=",")
                div = np.flatnonzero(arr[:, 1] == 1)
                div = div[div > 0]
                field, _ = hit.groups()
                # Extract continuous-time multi-generation phase trajectories!
                # Within each generation [start, end), log-length grows from log(L_0) to log(L_div);
                # trisecting log(L(t)/L_0) / log(2) at thresholds 1/3 and 2/3 defines a physical
                # 3-node cell-cycle progression ring: G1-like (0) -> S-like (1) -> G2/M-like (2) -> 0!
                if len(div) < 4:
                    continue
                t0_trace = int(div[0])
                t_end_trace = int(div[-1])
                phase_seq = np.zeros(t_end_trace - t0_trace, dtype=int)
                gen_durations = []
                for s_idx, e_idx in zip(div[:-1], div[1:]):
                    dur = float(arr[e_idx, 0] - arr[s_idx, 0])
                    if dur < 6.0:
                        continue
                    gen_durations.append(dur)
                    l_seg = arr[s_idx:e_idx, 2]
                    l_start, l_final = l_seg[0], max(l_seg[-1], l_seg[0] * 1.05)
                    prog = np.log(np.maximum(l_seg, l_start) / l_start) / np.log(l_final / l_start)
                    ph = np.where(prog < 1.0 / 3.0, 0, np.where(prog < 2.0 / 3.0, 1, 2))
                    phase_seq[s_idx - t0_trace : e_idx - t0_trace] = ph
                records.append(
                    {
                        "temp": temp,
                        "cluster": f"{temp}/{field}",
                        "phase_seq": phase_seq,  # 1-minute resolution node states in {0, 1, 2}
                        "gen_durations": gen_durations,
                    }
                )
    return records


def build_ecoli_cohort_snapshots(records: list[dict], obs_times_min: np.ndarray, aged_min: int = 12):
    """Construct destructive fresh-cohort and survivor-aged cohort snapshots per temperature."""
    temps = (25, 27, 37)
    out = {}
    for temp in temps:
        recs_t = [r for r in records if r["temp"] == temp]
        # Collect transitions (fresh entry into phase p in {0, 1, 2}) across traces
        fresh_counts = np.zeros((3, len(obs_times_min), 3), dtype=float)
        aged_counts = np.zeros((3, len(obs_times_min), 3), dtype=float)
        max_h = int(obs_times_min[-1]) + aged_min + 2
        phase_dwells = {0: [], 1: [], 2: []}

        for r in recs_t:
            seq = r["phase_seq"]
            # Find entry points where phase changes (or t=0)
            changes = np.r_[0, np.flatnonzero(np.diff(seq) != 0) + 1]
            for c_pos in range(len(changes) - 1):
                t_enter = int(changes[c_pos])
                t_leave = int(changes[c_pos + 1])
                p_node = int(seq[t_enter])
                dwell = t_leave - t_enter
                phase_dwells[p_node].append(float(dwell))
                if t_enter + max_h >= len(seq):
                    continue
                for ti, dt_m in enumerate(obs_times_min):
                    fresh_counts[p_node, ti, seq[t_enter + int(dt_m)]] += 1.0
                if dwell >= aged_min:
                    for ti, dt_m in enumerate(obs_times_min):
                        aged_counts[p_node, ti, seq[t_enter + aged_min + int(dt_m)]] += 1.0

        all_gen = [d for r in recs_t for d in r["gen_durations"]]
        gen_arr = np.asarray(all_gen)
        emp_mean = float(np.mean(gen_arr))
        emp_shape = float(emp_mean**2 / np.var(gen_arr, ddof=1))
        out[temp] = {
            "fresh_counts": fresh_counts,
            "aged_counts": aged_counts,
            "empirical_cycle_mean_min": emp_mean,
            "empirical_cycle_shape": emp_shape,
            "phase_means_min": [float(np.mean(phase_dwells[p])) for p in (0, 1, 2)],
            "phase_shapes": [
                float(np.mean(phase_dwells[p]) ** 2 / np.var(phase_dwells[p], ddof=1))
                for p in (0, 1, 2)
            ],
        }
    return out


# ---------------------------------------------------------------------------
# 2. Continuous-time 3-node E. coli temperature-conditioned cycle benchmark
# ---------------------------------------------------------------------------
def run_ecoli_continuous_benchmark(records: list[dict]) -> dict:
    """5-fold cluster CV on 18,972 E. coli cycles across 25C, 27C, 37C (hours unit)."""
    obs_min = np.array([6, 12, 18, 24, 36, 48, 60, 75, 90], dtype=int)
    ts_hr = obs_min / 60.0
    aged_hr = 12.0 / 60.0
    k_stages = 16
    temps = (25, 27, 37)
    temp_feat = {25: -1.0, 27: -0.667, 37: 1.0}

    # Partition 49 clusters into 5 stratified folds
    clusters = sorted({r["cluster"] for r in records})
    folds = [clusters[i::5] for i in range(5)]

    read_ph = jnp.kron(jnp.eye(3), jnp.ones((1, k_stages)))
    local_ph = jnp.eye(k_stages, k=-1) - jnp.eye(k_stages)
    route_3 = jnp.roll(jnp.eye(3), 1, axis=0)
    exit_ph = jnp.kron(jnp.eye(3), jnp.eye(1, k_stages, k_stages - 1))

    def drn_predict(theta, x_temp: float, is_aged: bool):
        # Conditional Neural Clock DRN: maps [1, x_temp, x_temp^2] -> per-node log_rate & K=16 stage logits
        # 3 nodes * 3 features * (1 + 16) = 153 parameters (vs 285 for Latent ODE)
        w_clock = theta.reshape(3, 3, 1 + k_stages)
        feats = jnp.array([1.0, x_temp, x_temp**2])
        raw = jnp.einsum("f,nfo->no", feats, w_clock)
        rates = jnp.exp(raw[:, 0])
        logits = raw[:, 1:]
        entries = jax.nn.softmax(logits, axis=-1)[:, ::-1]
        means, _, _ = jax.vmap(mixed_erlang_moments)(rates, entries)
        fresh = jl.block_diag(*(e[:, None] for e in entries))
        gen = jnp.kron(jnp.diag(rates), local_ph) + fresh @ route_3 @ (rates[:, None] * exit_ph)
        if is_aged:
            surv = survivor_phases(rates, entries, aged_hr)
            surv_norm = surv / jnp.maximum(jnp.sum(surv, axis=-1, keepdims=True), 1e-15)
            init_state = jl.block_diag(*(s[:, None] for s in surv_norm))
        else:
            init_state = fresh
        probs = jax.vmap(lambda t: (read_ph @ jl.expm(t * gen) @ init_state).T)(jnp.asarray(ts_hr))
        return jnp.transpose(probs, (1, 0, 2)), means, rates, entries

    def ctmc_predict(theta, x_temp: float, is_aged: bool):
        # Memoryless K=1 exponential cycle (6 params: 3 intercepts + 3 temp slopes)
        p_mat = theta.reshape(3, 2)
        rates = jnp.exp(-(p_mat[:, 0] + p_mat[:, 1] * x_temp))
        gen = -jnp.diag(rates) + route_3 @ jnp.diag(rates)
        # Memoryless property: aged cohort has IDENTICAL initial state eye(3) as fresh!
        probs = jax.vmap(lambda t: (jl.expm(t * gen)).T)(jnp.asarray(ts_hr))
        return jnp.transpose(probs, (1, 0, 2))

    def latent_ode_predict(theta, x_temp: float, is_aged: bool):
        # Population Latent ODE (latent=6, width=12) taking (prep, temp, age)
        enc = theta[:30].reshape(5, 6)
        eb = theta[30:36]
        w1 = theta[36:156].reshape(10, 12)
        b1 = theta[156:168]
        w2 = theta[168:276].reshape(12, 9)
        b2 = theta[276:285]
        age_val = aged_hr if is_aged else 0.0
        prep = jnp.eye(3)
        enc_in = jnp.column_stack([prep, jnp.full((3, 1), x_temp), jnp.full((3, 1), age_val)])
        z0 = jnp.tanh(enc_in @ enc + eb)
        y0 = jnp.column_stack([prep, z0])
        dt = 0.025  # 1.5 min RK4 step
        n_steps = int(round(ts_hr[-1] / dt))
        sample_idx = jnp.rint(jnp.asarray(ts_hr) / dt).astype(int) - 1

        def field(y):
            p = y[:, :3]
            inp = jnp.column_stack([y, jnp.full((3, 1), x_temp)])
            out = jnp.tanh(inp @ w1 + b1) @ w2 + b2
            flux = p * (0.05 + 12.0 * jax.nn.sigmoid(out[:, :3]))
            dp = jnp.roll(flux, 1, axis=-1) - flux
            return jnp.column_stack([dp, jnp.tanh(out[:, 3:])])

        def step(y, _):
            k1 = field(y)
            k2 = field(y + 0.5 * dt * k1)
            k3 = field(y + 0.5 * dt * k2)
            k4 = field(y + dt * k3)
            yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return yn, yn[:, :3]

        traj = jax.lax.scan(step, y0, None, length=n_steps)[1]
        return jnp.transpose(traj[sample_idx], (1, 0, 2))

    @jax.jit
    def loss_drn(theta, f_c, a_c):
        tot = 0.0
        for idx_t, t_k in enumerate(temps):
            xf = temp_feat[t_k]
            pf, _, _, _ = drn_predict(theta, xf, False)
            pa, _, _, _ = drn_predict(theta, xf, True)
            tot = tot + multinomial_nll(pf, f_c[idx_t]) + 0.5 * multinomial_nll(pa, a_c[idx_t])
        return tot / 3.0 + 1e-5 * jnp.sum(theta**2)

    @jax.jit
    def loss_drn_fresh_only(theta, f_c):
        # Train ONLY on fresh cohorts to test ZERO-SHOT transfer to survivor-aged cohorts!
        tot = 0.0
        for idx_t, t_k in enumerate(temps):
            pf, _, _, _ = drn_predict(theta, temp_feat[t_k], False)
            tot = tot + multinomial_nll(pf, f_c[idx_t])
        return tot / 3.0 + 1e-5 * jnp.sum(theta**2)

    @jax.jit
    def loss_ctmc(theta, f_c):
        tot = 0.0
        for idx_t, t_k in enumerate(temps):
            pf = ctmc_predict(theta, temp_feat[t_k], False)
            tot = tot + multinomial_nll(pf, f_c[idx_t])
        return tot / 3.0 + 1e-5 * jnp.sum(theta**2)

    @jax.jit
    def loss_lode(theta, f_c, a_c):
        tot = 0.0
        for idx_t, t_k in enumerate(temps):
            xf = temp_feat[t_k]
            pf = latent_ode_predict(theta, xf, False)
            pa = latent_ode_predict(theta, xf, True)
            tot = tot + multinomial_nll(pf, f_c[idx_t]) + 0.5 * multinomial_nll(pa, a_c[idx_t])
        return tot / 3.0 + 1e-4 * jnp.sum(theta**2)

    @jax.jit
    def loss_lode_fresh_only(theta, f_c):
        tot = 0.0
        for idx_t, t_k in enumerate(temps):
            pf = latent_ode_predict(theta, temp_feat[t_k], False)
            tot = tot + multinomial_nll(pf, f_c[idx_t])
        return tot / 3.0 + 1e-4 * jnp.sum(theta**2)

    vg_drn = jax.jit(jax.value_and_grad(loss_drn))
    vg_drn_fo = jax.jit(jax.value_and_grad(loss_drn_fresh_only))
    vg_ctmc = jax.jit(jax.value_and_grad(loss_ctmc))
    vg_lode = jax.jit(jax.value_and_grad(loss_lode))
    vg_lode_fo = jax.jit(jax.value_and_grad(loss_lode_fresh_only))

    fold_metrics = {
        "drn_joint": [],
        "drn_fresh_only": [],
        "latent_ode_joint": [],
        "latent_ode_fresh_only": [],
        "ctmc": [],
    }
    recovered_cycle_stats = []

    for f_idx, test_clusters in enumerate(folds):
        tr_recs = [r for r in records if r["cluster"] not in set(test_clusters)]
        te_recs = [r for r in records if r["cluster"] in set(test_clusters)]
        tr_snap = build_ecoli_cohort_snapshots(tr_recs, obs_min)
        te_snap = build_ecoli_cohort_snapshots(te_recs, obs_min)

        f_tr = jnp.asarray([tr_snap[t]["fresh_counts"] for t in temps])
        a_tr = jnp.asarray([tr_snap[t]["aged_counts"] for t in temps])

        # Initialize DRN near rate ~ 36/hr (log(36)=3.58) and moderate stage count
        th0_mat = np.zeros((3, 3, 1 + k_stages), dtype=float)
        th0_mat[:, 0, 0] = 3.5
        th0_mat[:, 1, 0] = 0.4
        th0_drn = th0_mat.ravel()
        res_fo = minimize(
            lambda z: (float(vg_drn_fo(z, f_tr)[0]), np.asarray(vg_drn_fo(z, f_tr)[1], float)),
            th0_drn,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        res_drn = minimize(
            lambda z: (float(vg_drn(z, f_tr, a_tr)[0]), np.asarray(vg_drn(z, f_tr, a_tr)[1], float)),
            res_fo.x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 180},
        )
        res_ctmc = minimize(
            lambda z: (float(vg_ctmc(z, f_tr)[0]), np.asarray(vg_ctmc(z, f_tr)[1], float)),
            np.array([-1.2, -0.5] * 3, dtype=float),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 150},
        )
        best_lode, best_lode_fo = None, None
        for s_l in (11, 29):
            rng_l = np.random.default_rng(100 * f_idx + s_l)
            w_init = rng_l.normal(0.0, 0.22, size=285)
            r_l = minimize(
                lambda z: (float(vg_lode(z, f_tr, a_tr)[0]), np.asarray(vg_lode(z, f_tr, a_tr)[1], float)),
                w_init,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 180},
            )
            if best_lode is None or r_l.fun < best_lode.fun:
                best_lode = r_l
            r_lfo = minimize(
                lambda z: (float(vg_lode_fo(z, f_tr)[0]), np.asarray(vg_lode_fo(z, f_tr)[1], float)),
                w_init,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 180},
            )
            if best_lode_fo is None or r_lfo.fun < best_lode_fo.fun:
                best_lode_fo = r_lfo

        def eval_heldout(pred_fn):
            kl_f, kl_a = [], []
            for t_k in temps:
                xf = temp_feat[t_k]
                for is_a, bucket, c_arr in [
                    (False, kl_f, te_snap[t_k]["fresh_counts"]),
                    (True, kl_a, te_snap[t_k]["aged_counts"]),
                ]:
                    p_hat = np.clip(np.asarray(pred_fn(xf, is_a)), 1e-12, 1.0)
                    p_hat = p_hat / p_hat.sum(axis=-1, keepdims=True)
                    q_emp = c_arr / np.maximum(c_arr.sum(axis=-1, keepdims=True), 1.0)
                    w_emp = c_arr.sum(axis=-1) / np.maximum(c_arr.sum(), 1.0)
                    kl = float(
                        np.sum(
                            w_emp
                            * np.sum(
                                np.where(
                                    q_emp > 0,
                                    q_emp * (np.log(np.maximum(q_emp, 1e-30)) - np.log(p_hat)),
                                    0.0,
                                ),
                                axis=-1,
                            )
                        )
                    )
                    bucket.append(kl)
            return {"fresh_kl": float(np.mean(kl_f)), "aged_kl": float(np.mean(kl_a))}

        fold_metrics["drn_fresh_only"].append(
            eval_heldout(lambda xf, ia: drn_predict(jnp.asarray(res_fo.x), xf, ia)[0])
        )
        fold_metrics["drn_joint"].append(
            eval_heldout(lambda xf, ia: drn_predict(jnp.asarray(res_drn.x), xf, ia)[0])
        )
        fold_metrics["ctmc"].append(
            eval_heldout(lambda xf, ia: ctmc_predict(jnp.asarray(res_ctmc.x), xf, ia))
        )
        fold_metrics["latent_ode_joint"].append(
            eval_heldout(lambda xf, ia: latent_ode_predict(jnp.asarray(best_lode.x), xf, ia))
        )
        fold_metrics["latent_ode_fresh_only"].append(
            eval_heldout(lambda xf, ia: latent_ode_predict(jnp.asarray(best_lode_fo.x), xf, ia))
        )

        # Recover full division-cycle mean & shape from snapshot-fitted DRN (fresh-only)!
        if f_idx == 0:
            for t_k in temps:
                _, m_hr, r_hr, e_hr = drn_predict(jnp.asarray(res_fo.x), temp_feat[t_k], False)
                m_min = 60.0 * np.asarray(m_hr)
                _, v_hr, _ = jax.vmap(mixed_erlang_moments)(r_hr, e_hr)
                v_min2 = (60.0**2) * np.asarray(v_hr)
                tot_mean = float(m_min.sum())
                tot_shape = float((tot_mean**2) / v_min2.sum())
                recovered_cycle_stats.append(
                    {
                        "temp_C": t_k,
                        "snapshot_recovered_mean_min": round(tot_mean, 2),
                        "true_longitudinal_mean_min": round(te_snap[t_k]["empirical_cycle_mean_min"], 2),
                        "snapshot_recovered_shape": round(tot_shape, 2),
                        "true_longitudinal_shape": round(te_snap[t_k]["empirical_cycle_shape"], 2),
                    }
                )
        print(
            f"E.coli fold {f_idx} | Zero-shot aged KL: DRN={fold_metrics['drn_fresh_only'][-1]['aged_kl']:.5f} vs LODE={fold_metrics['latent_ode_fresh_only'][-1]['aged_kl']:.5f} vs CTMC={fold_metrics['ctmc'][-1]['aged_kl']:.5f} | Joint KL: DRN={fold_metrics['drn_joint'][-1]['aged_kl']:.5f} vs LODE={fold_metrics['latent_ode_joint'][-1]['aged_kl']:.5f}",
            flush=True,
        )

    return {
        "n_cycles": sum(len(r["gen_durations"]) for r in records),
        "n_clusters": len(clusters),
        "models_5fold_mean": {
            m: {
                "fresh_kl": round(float(np.mean([x["fresh_kl"] for x in rows])), 5),
                "aged_kl": round(float(np.mean([x["aged_kl"] for x in rows])), 5),
                "fresh_kl_sd": round(float(np.std([x["fresh_kl"] for x in rows], ddof=1)), 5),
                "aged_kl_sd": round(float(np.std([x["aged_kl"] for x in rows], ddof=1)), 5),
            }
            for m, rows in fold_metrics.items()
        },
        "zero_shot_cycle_law_recovery": recovered_cycle_stats,
    }


# ---------------------------------------------------------------------------
# 3. Continuous-time 5-node Human Sleep-EDF Branching & Cyclic Network Benchmark
# ---------------------------------------------------------------------------
def parse_sleep_edf_sequences() -> dict[str, list[list[tuple[int, int]]]]:
    """Parse 30s-epoch continuous hypnograms for all 75 paired-night participants."""
    manifest = CACHE / "SHA256SUMS.txt"
    assert sha256(manifest.read_bytes()).hexdigest() == SLEEP_SUMS
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
            # Trim to sleep period [first non-W sleep, last non-W sleep]
            ep = np.full(sum(x[1] for x in ann), -1, dtype=int)
            pos = 0
            for _, ln, st in ann:
                ep[pos : pos + ln] = st
                pos += ln
            sleep_idx = np.flatnonzero(ep >= 1)
            ep_sleep = ep[sleep_idx[0] : sleep_idx[-1] + 1]
            # Split into contiguous valid segments (no -1)
            bad = np.flatnonzero(ep_sleep < 0)
            starts = np.r_[0, bad + 1]
            ends = np.r_[bad, len(ep_sleep)]
            for s0, e0 in zip(starts, ends):
                if e0 - s0 >= 40:
                    subj_epochs.append(ep_sleep[s0:e0])
        out_seqs[subj] = subj_epochs
    return out_seqs


def build_sleep_snapshots(subj_dict: dict, subjects: list[str], obs_epochs: np.ndarray, aged_epochs: int = 10):
    """Build 5-node fresh-entry and aged-survivor (>= 5 min = 10 epochs) snapshots."""
    fresh = np.zeros((5, len(obs_epochs), 5), dtype=float)
    aged = np.zeros((5, len(obs_epochs), 5), dtype=float)
    max_e = int(obs_epochs[-1]) + aged_epochs + 1
    edge_counts = np.zeros((5, 5), dtype=int)

    for s in subjects:
        for seq in subj_dict[s]:
            changes = np.r_[0, np.flatnonzero(np.diff(seq) != 0) + 1]
            for idx in range(len(changes) - 1):
                t0 = int(changes[idx])
                t1 = int(changes[idx + 1])
                u = int(seq[t0])
                v = int(seq[t1])
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


def run_sleep_5node_network_benchmark() -> dict:
    """5-fold participant CV on the 5-node branching+cyclic Sleep-EDF network."""
    subj_seqs = parse_sleep_edf_sequences()
    subjects = sorted(subj_seqs.keys())
    assert len(subjects) == 75

    # Determine active directed transitions (>0.5% of exits from a stage)
    obs_epochs = np.array([2, 4, 8, 12, 20, 30, 45, 60], dtype=int)  # 1 to 30 minutes
    ts_ten_min = (obs_epochs * 0.5) / 10.0  # time unit = 10 minutes for numerical conditioning
    aged_ten_min = (10 * 0.5) / 10.0  # 5.0 minutes = 0.5 in 10-min units

    _, _, all_edge_counts = build_sleep_snapshots(subj_seqs, subjects, obs_epochs)
    edges_list = []
    for u in range(5):
        row_tot = all_edge_counts[u].sum()
        for v in range(5):
            if u != v and all_edge_counts[u, v] >= 0.01 * row_tot:
                edges_list.append((u, v))
    edges = tuple(edges_list)
    n_edges = len(edges)
    k_stages = 8  # K=8 stages per competing clock
    naive_tensor_states = network_state_size(edges, 5, k_stages)
    compact_states = compact_network_state_size(edges, 5, k_stages)
    outgoing = [tuple(e for e, (u, _) in enumerate(edges) if u == i) for i in range(5)]

    # Build exact compiled 5-node marked-race generator with free nonparametric logits per edge!
    def build_compiled_sleep_generator(theta):
        # theta: (1 + k_stages) * n_edges -> log_mean + k_stages free logits per directed edge
        means = jnp.exp(theta[:n_edges])
        logits = theta[n_edges:].reshape(n_edges, k_stages)

        # Assemble exact marked-race compiled generator A (compact_states x compact_states)
        blocks = []
        fresh_cols = []
        aged_cols = []
        exit_rows_by_edge = {}
        offsets = []
        cur = 0
        for u in range(5):
            grp = outgoing[u]
            idx = jnp.array(grp)
            rate_tot, cont, causes = mixed_erlang_race(means[idx], logits[idx])
            dim_u = causes.shape[1]
            offsets.append((cur, cur + dim_u))
            blk = -rate_tot * jnp.eye(dim_u) + jnp.diag(rate_tot * cont, k=1).T
            blocks.append(blk)
            e_fresh = jnp.zeros(dim_u).at[0].set(1.0)
            fresh_cols.append(e_fresh)
            # Exact survivor initialization at aged_ten_min under local transient block!
            e_aged = jl.expm(aged_ten_min * blk) @ e_fresh
            e_aged = e_aged / jnp.maximum(jnp.sum(e_aged), 1e-15)
            aged_cols.append(e_aged)
            for j_loc, e_id in enumerate(grp):
                exit_rows_by_edge[e_id] = (cur, cur + dim_u, rate_tot * causes[j_loc])
            cur += dim_u

        gen = jl.block_diag(*blocks)
        for e_id, (_, dst) in enumerate(edges):
            c0, c1, flux_row = exit_rows_by_edge[e_id]
            d0, _ = offsets[dst]
            gen = gen.at[d0, c0:c1].add(flux_row)

        fresh_mat = jl.block_diag(*(v[:, None] for v in fresh_cols))
        aged_mat = jl.block_diag(*(v[:, None] for v in aged_cols))
        read_mat = jl.block_diag(*(jnp.ones((1, e - s)) for s, e in offsets))
        return gen, fresh_mat, aged_mat, read_mat

    def predict_sleep_drn(theta):
        gen, fresh_m, aged_m, read_m = build_compiled_sleep_generator(theta)
        pf = jax.vmap(lambda t: (read_m @ jl.expm(t * gen) @ fresh_m).T)(jnp.asarray(ts_ten_min))
        pa = jax.vmap(lambda t: (read_m @ jl.expm(t * gen) @ aged_m).T)(jnp.asarray(ts_ten_min))
        return jnp.transpose(pf, (1, 0, 2)), jnp.transpose(pa, (1, 0, 2))

    def predict_sleep_ctmc(theta):
        rates = jnp.exp(theta)
        gen = jnp.zeros((5, 5))
        for e_id, (u, v) in enumerate(edges):
            gen = gen.at[v, u].add(rates[e_id])
            gen = gen.at[u, u].add(-rates[e_id])
        pf = jax.vmap(lambda t: jl.expm(t * gen).T)(jnp.asarray(ts_ten_min))
        p_tr = jnp.transpose(pf, (1, 0, 2))
        return p_tr, p_tr

    def predict_sleep_lode(theta):
        # Population Latent ODE (latent=6, width=14) over the 5-node directed graph
        enc = theta[:36].reshape(6, 6)
        eb = theta[36:42]
        w1 = theta[42 : 42 + 11 * 14].reshape(11, 14)
        b1 = theta[196:210]
        w2 = theta[210 : 210 + 14 * (n_edges + 6)].reshape(14, n_edges + 6)
        b2 = theta[210 + 14 * (n_edges + 6) : 210 + 15 * (n_edges + 6)]
        dt = 0.025  # 15s RK4 substep in 10-min units
        n_steps = int(round(ts_ten_min[-1] / dt))
        sample_idx = jnp.rint(jnp.asarray(ts_ten_min) / dt).astype(int) - 1
        src_idx = jnp.array([u for u, _ in edges])
        dst_idx = jnp.array([v for _, v in edges])

        def rollout(age_val: float):
            prep = jnp.eye(5)
            z0 = jnp.tanh(jnp.column_stack([prep, jnp.full((5, 1), age_val)]) @ enc + eb)
            y0 = jnp.column_stack([prep, z0])

            def field(y):
                p = y[:, :5]
                out = jnp.tanh(y @ w1 + b1) @ w2 + b2
                edge_rates = 0.02 + 8.0 * jax.nn.sigmoid(out[:, :n_edges] - 1.5)
                edge_flux = p[:, src_idx] * edge_rates
                dp = jnp.zeros_like(p).at[:, dst_idx].add(edge_flux).at[:, src_idx].add(-edge_flux)
                return jnp.column_stack([dp, jnp.tanh(out[:, n_edges:])])

            def step(y, _):
                k1 = field(y)
                k2 = field(y + 0.5 * dt * k1)
                k3 = field(y + 0.5 * dt * k2)
                k4 = field(y + dt * k3)
                yn = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                return yn, yn[:, :5]

            traj = jax.lax.scan(step, y0, None, length=n_steps)[1]
            return jnp.transpose(traj[sample_idx], (1, 0, 2))

        return rollout(0.0), rollout(aged_ten_min)

    @jax.jit
    def loss_sleep_drn(theta, fc, ac):
        pf, pa = predict_sleep_drn(theta)
        return multinomial_nll(pf, fc) + multinomial_nll(pa, ac) + 1e-5 * jnp.sum(theta**2)

    @jax.jit
    def loss_sleep_drn_fresh(theta, fc):
        pf, _ = predict_sleep_drn(theta)
        return multinomial_nll(pf, fc) + 1e-5 * jnp.sum(theta**2)

    @jax.jit
    def loss_sleep_ctmc(theta, fc, ac):
        pf, pa = predict_sleep_ctmc(theta)
        return multinomial_nll(pf, fc) + multinomial_nll(pa, ac) + 1e-5 * jnp.sum(theta**2)

    @jax.jit
    def loss_sleep_lode(theta, fc, ac):
        pf, pa = predict_sleep_lode(theta)
        return multinomial_nll(pf, fc) + multinomial_nll(pa, ac) + 1e-4 * jnp.sum(theta**2)

    @jax.jit
    def loss_sleep_lode_fresh(theta, fc):
        pf, _ = predict_sleep_lode(theta)
        return multinomial_nll(pf, fc) + 1e-4 * jnp.sum(theta**2)

    vg_drn = jax.jit(jax.value_and_grad(loss_sleep_drn))
    vg_drn_fo = jax.jit(jax.value_and_grad(loss_sleep_drn_fresh))
    vg_ctmc = jax.jit(jax.value_and_grad(loss_sleep_ctmc))
    vg_lode = jax.jit(jax.value_and_grad(loss_sleep_lode))
    vg_lode_fo = jax.jit(jax.value_and_grad(loss_sleep_lode_fresh))

    folds = [subjects[i::5] for i in range(5)]
    drn_scores, drn_fo_scores, ctmc_scores, lode_scores, lode_fo_scores = [], [], [], [], []
    fit_times = []
    lode_dim = 210 + 15 * (n_edges + 6)
    drn_dim = (1 + k_stages) * n_edges

    for f_idx, te_subjs in enumerate(folds):
        tr_subjs = [s for s in subjects if s not in set(te_subjs)]
        f_tr, a_tr, _ = build_sleep_snapshots(subj_seqs, tr_subjs, obs_epochs)
        f_te, a_te, _ = build_sleep_snapshots(subj_seqs, te_subjs, obs_epochs)

        th0 = np.zeros(drn_dim)
        res_fo = minimize(
            lambda z: (
                float(vg_drn_fo(jnp.asarray(z), jnp.asarray(f_tr))[0]),
                np.asarray(vg_drn_fo(jnp.asarray(z), jnp.asarray(f_tr))[1], float),
            ),
            th0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 180},
        )
        t0 = perf_counter()
        res_d = minimize(
            lambda z: (
                float(vg_drn(jnp.asarray(z), jnp.asarray(f_tr), jnp.asarray(a_tr))[0]),
                np.asarray(vg_drn(jnp.asarray(z), jnp.asarray(f_tr), jnp.asarray(a_tr))[1], float),
            ),
            res_fo.x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 180},
        )
        fit_times.append(perf_counter() - t0)

        res_c = minimize(
            lambda z: (
                float(vg_ctmc(jnp.asarray(z), jnp.asarray(f_tr), jnp.asarray(a_tr))[0]),
                np.asarray(vg_ctmc(jnp.asarray(z), jnp.asarray(f_tr), jnp.asarray(a_tr))[1], float),
            ),
            np.zeros(n_edges),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 160},
        )

        best_l, best_lfo = None, None
        for s_l in (13, 37):
            w0_l = np.random.default_rng(200 * f_idx + s_l).normal(0.0, 0.20, size=lode_dim)
            r_l = minimize(
                lambda z: (
                    float(vg_lode(jnp.asarray(z), jnp.asarray(f_tr), jnp.asarray(a_tr))[0]),
                    np.asarray(vg_lode(jnp.asarray(z), jnp.asarray(f_tr), jnp.asarray(a_tr))[1], float),
                ),
                w0_l,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 180},
            )
            if best_l is None or r_l.fun < best_l.fun:
                best_l = r_l
            r_lfo = minimize(
                lambda z: (
                    float(vg_lode_fo(jnp.asarray(z), jnp.asarray(f_tr))[0]),
                    np.asarray(vg_lode_fo(jnp.asarray(z), jnp.asarray(f_tr))[1], float),
                ),
                w0_l,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 180},
            )
            if best_lfo is None or r_lfo.fun < best_lfo.fun:
                best_lfo = r_lfo

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

        drn_fo_scores.append(score_pair(*predict_sleep_drn(jnp.asarray(res_fo.x))))
        drn_scores.append(score_pair(*predict_sleep_drn(jnp.asarray(res_d.x))))
        ctmc_scores.append(score_pair(*predict_sleep_ctmc(jnp.asarray(res_c.x))))
        lode_scores.append(score_pair(*predict_sleep_lode(jnp.asarray(best_l.x))))
        lode_fo_scores.append(score_pair(*predict_sleep_lode(jnp.asarray(best_lfo.x))))
        print(
            f"Sleep 5-node fold {f_idx} | Zero-shot aged KL: DRN={drn_fo_scores[-1]['aged_kl']:.5f} vs LODE={lode_fo_scores[-1]['aged_kl']:.5f} | Joint KL: DRN={drn_scores[-1]['fresh_kl']:.5f}/{drn_scores[-1]['aged_kl']:.5f} vs LODE={lode_scores[-1]['fresh_kl']:.5f}/{lode_scores[-1]['aged_kl']:.5f} vs CTMC={ctmc_scores[-1]['fresh_kl']:.5f}/{ctmc_scores[-1]['aged_kl']:.5f}",
            flush=True,
        )

    def _agg(rows):
        return {
            "fresh_kl_mean": round(float(np.mean([x["fresh_kl"] for x in rows])), 5),
            "aged_kl_mean": round(float(np.mean([x["aged_kl"] for x in rows])), 5),
            "fresh_kl_sd": round(float(np.std([x["fresh_kl"] for x in rows], ddof=1)), 5),
            "aged_kl_sd": round(float(np.std([x["aged_kl"] for x in rows], ddof=1)), 5),
        }

    return {
        "participants": 75,
        "nodes": 5,
        "active_directed_edges": n_edges,
        "stages_per_clock": k_stages,
        "naive_tensor_states": naive_tensor_states,
        "compiled_race_states": compact_states,
        "state_compression_ratio": round(naive_tensor_states / compact_states, 2),
        "median_fit_seconds": round(float(np.median(fit_times)), 2),
        "drn_parameters": drn_dim,
        "latent_ode_parameters": lode_dim,
        "models_5fold_mean": {
            "drn_joint": {
                **_agg(drn_scores),
                "wins_fresh_vs_ctmc": int(sum(d["fresh_kl"] < c["fresh_kl"] for d, c in zip(drn_scores, ctmc_scores))),
                "wins_aged_vs_ctmc": int(sum(d["aged_kl"] < c["aged_kl"] for d, c in zip(drn_scores, ctmc_scores))),
            },
            "drn_fresh_only": {
                **_agg(drn_fo_scores),
                "wins_zeroshot_aged_vs_lode": int(
                    sum(d["aged_kl"] < l["aged_kl"] for d, l in zip(drn_fo_scores, lode_fo_scores))
                ),
            },
            "latent_ode_joint": _agg(lode_scores),
            "latent_ode_fresh_only": _agg(lode_fo_scores),
            "ctmc": _agg(ctmc_scores),
        },
    }


def main() -> None:
    print("=== Running E. coli 3-phase continuous cyclic benchmark ===", flush=True)
    ecoli_recs = load_tanouchi_cycles()
    ecoli_res = run_ecoli_continuous_benchmark(ecoli_recs)

    print("=== Running Sleep-EDF 5-node branching+cyclic network benchmark ===", flush=True)
    sleep_res = run_sleep_5node_network_benchmark()

    out = {
        "ecoli_continuous_cycle": ecoli_res,
        "sleep_5node_branching_network": sleep_res,
    }
    out_path = ROOT / "results" / "archived" / "real_continuous_benchmarks.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
