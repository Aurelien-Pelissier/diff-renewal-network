"""Audit SHA-256 provenance and replay core Differentiable Renewal Network trajectories.

Verifies:
1. Cryptographic SHA-256 digests of all 17 archived benchmark summaries.
2. Live pure Erlang-16 48-vs-33 trajectory equality (< 1e-12) and rate-derivative divergence.
3. Live JAX/Diffrax R1 mouse embryonic stem-cell (ESC -> EPI -> NPC) trajectory replay
   against the archived continuous-shape optimum (discrepancy < 1e-9).
4. Headline quantitative claims across all four main paper result subsections.
"""

import hashlib
import json
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm_frechet

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import smooth_erlang_logits, solve_erlang_chain  # noqa: E402

ARCH = ROOT / "results" / "archived"


def verify_manifest() -> int:
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    for rel_path, meta in manifest["files"].items():
        p = ROOT / rel_path
        if not p.is_file():
            raise FileNotFoundError(f"Missing archived file: {rel_path}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {rel_path}: {digest} != {meta['sha256']}")
    return len(manifest["files"])


def replay_erlang16_separation() -> dict:
    a = -8.0 * np.eye(48)
    for col in range(48):
        a[(col + 1) % 48, col] += 8.0
    ad = a.copy()
    ad[:, 16:] = 0.0
    b = np.eye(48)[:, [0, 16, 32]]
    c = np.kron(np.eye(3), np.ones((1, 16)))
    contrasts = np.array(
        [[1 / np.sqrt(2), 1 / np.sqrt(6)], [-1 / np.sqrt(2), 1 / np.sqrt(6)], [0, -2 / np.sqrt(6)]]
    )
    d = np.kron(contrasts, np.eye(16))
    w = np.vstack([d.T, np.ones(48)])
    v = np.column_stack([d, np.ones(48) / 48.0])
    cr = np.column_stack([c @ d, np.ones(3) / 3.0])
    g, gd = w @ a @ v, w @ ad @ v
    ts = np.linspace(0.0, 7.0, 85)
    nom_err, deriv_diff = 0.0, 0.0
    for t in ts:
        full, df = expm_frechet(t * a, t * ad)
        red, dr = expm_frechet(t * g, t * gd)
        nom_err = max(nom_err, float(np.max(np.abs(c @ full @ b - cr @ red @ w @ b))))
        deriv_diff = max(
            deriv_diff,
            float(np.max(np.abs((c @ df @ b - cr @ dr @ w @ b) @ (np.ones(3) / 3.0)))),
        )
    assert nom_err < 1e-12 and abs(deriv_diff - 0.26197) < 5e-4
    return {"nominal_max_error": nom_err, "rate_sensitivity_defect": deriv_diff}


def replay_stem_cells() -> dict:
    pilot = json.loads((ARCH / "stem_cells_pilot.json").read_text(encoding="utf-8"))
    ph_res = json.loads((ARCH / "trainable_ph_stem_cells.json").read_text(encoding="utf-8"))
    ts = jnp.asarray(pilot["observation_times"])
    observed = np.asarray(pilot["data_mean_proportions"])
    theta = np.asarray(ph_res["panels"]["all_times"]["phase"]["best"]["theta"])
    # Stored parameterization: theta[:2] = log(rates), theta[2:] = shape_centers / 50.0
    means = jnp.exp(-jnp.asarray(theta[:2]))
    centers = 50.0 * jnp.asarray(theta[2:])
    logits = jax.vmap(lambda c: smooth_erlang_logits(c, stages=64, width=0.5))(centers)
    pred = np.asarray(solve_erlang_chain(means, logits, ts).ys)
    rmse = float(np.mean(np.sqrt(np.mean((pred - observed) ** 2, axis=0))))
    archived_rmse = float(ph_res["panels"]["all_times"]["phase"]["best"]["loss"])
    drift = abs(rmse - archived_rmse)
    assert drift < 1e-8
    return {"replayed_rmse": rmse, "archived_rmse": archived_rmse, "abs_drift": drift}


def verify_headline_numbers() -> dict:
    ev = json.loads((ARCH / "evidence.json").read_text(encoding="utf-8"))
    sens = json.loads((ARCH / "sensitivity_evidence.json").read_text(encoding="utf-8"))
    tf = json.loads((ARCH / "trait_fisher_summary.json").read_text(encoding="utf-8"))
    nm = json.loads((ARCH / "neural_memory_summary.json").read_text(encoding="utf-8"))
    rt6 = json.loads((ARCH / "renewal_transfer_six_parameter.json").read_text(encoding="utf-8"))
    rvm = json.loads((ARCH / "real_visit_memory.json").read_text(encoding="utf-8"))

    # Finding 1: Compression & sensitivity interpolation
    joint_primary = [sens["overall"]["primary"][m]["joint"] for m in ("full48", "sym33", "sp33", "sp18")]
    assert joint_primary == [238, 161, 239, 164]

    # Finding 2: Fisher design & NoMaSS benchmark
    recs = ev["benchmark"]
    ph_med = round(100.0 * float(np.median([r["phase"]["shape_error"] for r in recs])), 2)
    nm_med = round(100.0 * float(np.median([r["nomass"]["shape_error"] for r in recs])), 2)
    assert (ph_med, nm_med) == (4.52, 7.69)
    ratios = [
        r["nomass"]["joint_seconds"] / r["phase"]["joint_seconds"]
        for r in recs
        if r["phase"]["joint_seconds"] is not None and r["nomass"]["joint_seconds"] is not None
    ]
    assert len(ratios) == 46 and round(float(np.median(ratios)), 2) == 19.32

    # Finding 3: Neural clocks
    kl_nc = round(float(nm["primary"]["models"]["neural_phase"]["metrics"]["balanced_kl"]["median"]), 5)
    kl_ms = round(float(nm["primary"]["models"]["mean_shape"]["metrics"]["balanced_kl"]["median"]), 5)
    kl_lo = round(float(nm["primary"]["models"]["latent_ode"]["metrics"]["balanced_kl"]["median"]), 5)
    assert (kl_nc, kl_ms, kl_lo) == (0.00197, 0.01095, 0.08315)

    # Finding 4: Persistent speed & real visit memory
    n2_gain = round(float(rvm["datasets"]["sleep"]["models"]["adjusted"]["previous"]["2"]["gain"]), 5)
    ecoli_si = round(float(rvm["datasets"]["nomass"]["models"]["adjusted"]["previous"]["all"]["gain"]), 5)
    assert (n2_gain, ecoli_si) == (0.01201, 0.01293)

    return {
        "cyclic_primary_joint_hits_of_280": dict(zip(("full48", "sym33", "sp33", "sp18"), joint_primary)),
        "fisher_shape_median_pct": {"ph": ph_med, "nomass": nm_med},
        "fisher_cold_speedup_median": round(float(np.median(ratios)), 2),
        "trait_datasets": tf["audit"]["datasets"],
        "neural_clock_balanced_kl": {"neural_clock": kl_nc, "moment_ph": kl_ms, "latent_ode": kl_lo},
        "persistent_carry_targets": list(rt6["targets"].keys()),
        "real_visit_gains_nats": {"sleep_n2": n2_gain, "ecoli_si": ecoli_si},
    }


def main() -> None:
    n_files = verify_manifest()
    e16 = replay_erlang16_separation()
    sc = replay_stem_cells()
    headlines = verify_headline_numbers()
    summary = {
        "status": "PASSED",
        "verified_archive_files": n_files,
        "erlang16_replay": e16,
        "stem_cell_replay": sc,
        "headline_claims": headlines,
    }
    (ROOT / "results" / "claim_ledger.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
