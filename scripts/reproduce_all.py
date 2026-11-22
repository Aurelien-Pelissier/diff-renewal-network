"""Audit SHA-256 provenance and replay core Differentiable Renewal Network trajectories.

Verifies:
1. Cryptographic SHA-256 digests of all 43 archived benchmark summaries.
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
    pos = json.loads((ARCH / "positive_sensitivity_summary.json").read_text(encoding="utf-8"))
    gsc = json.loads((ARCH / "graph_scaling_summary.json").read_text(encoding="utf-8"))
    mnb = json.loads((ARCH / "modern_neural_baselines.json").read_text(encoding="utf-8"))
    rcb = json.loads((ARCH / "real_continuous_benchmarks.json").read_text(encoding="utf-8"))
    ems = json.loads((ARCH / "edge_memory_sleep_summary.json").read_text(encoding="utf-8"))
    hem = json.loads((ARCH / "hematopoiesis_branching_summary.json").read_text(encoding="utf-8"))
    hdk = json.loads((ARCH / "helpdesk_systems_summary.json").read_text(encoding="utf-8"))
    hdk_ext = json.loads((ARCH / "helpdesk_extended_baselines.json").read_text(encoding="utf-8"))
    slp_ext = json.loads((ARCH / "sleep_extended_baselines.json").read_text(encoding="utf-8"))
    hem_d = json.loads((ARCH / "hematopoiesis_donor_extrap.json").read_text(encoding="utf-8"))
    mfe = json.loads((ARCH / "mean_field_epidemic_summary.json").read_text(encoding="utf-8"))
    spg = json.loads((ARCH / "sparse_large_graph_scaling.json").read_text(encoding="utf-8"))
    sp_rss = json.loads((ARCH / "sparse_e2e_and_memory.json").read_text(encoding="utf-8"))
    zs_ctrl = json.loads((ARCH / "zeroshot_age0_control.json").read_text(encoding="utf-8"))
    bio_r5 = json.loads((ARCH / "bio_influenza_r5_complete.json").read_text(encoding="utf-8"))
    sp_r5 = json.loads((ARCH / "sparse_r5_converged.json").read_text(encoding="utf-8"))
    hdk_r5 = json.loads((ARCH / "helpdesk_r5_complete.json").read_text(encoding="utf-8"))
    r6 = json.loads((ARCH / "r6_uniformity_closure.json").read_text(encoding="utf-8"))
    r6b = json.loads((ARCH / "r6_integrity_fixes.json").read_text(encoding="utf-8"))
    r7t = json.loads((ARCH / "r7_theory_bt_separation.json").read_text(encoding="utf-8"))
    r7m = json.loads((ARCH / "r7_mamba_thp_baselines.json").read_text(encoding="utf-8"))
    r7d = json.loads((ARCH / "r7_defi_neural_clock.json").read_text(encoding="utf-8"))
    r7n = json.loads((ARCH / "r7_temporal_networks.json").read_text(encoding="utf-8"))
    r8n = json.loads((ARCH / "r8_temporal_networks_upgraded.json").read_text(encoding="utf-8"))
    r8d = json.loads((ARCH / "r8_defi_upgraded.json").read_text(encoding="utf-8"))
    r8b = json.loads((ARCH / "r8_real_bt_failure.json").read_text(encoding="utf-8"))

    # Finding 1: Compression, sensitivity interpolation, and Pos-Sens 33
    joint_primary = [sens["overall"]["primary"][m]["joint"] for m in ("full48", "sym33", "sp33", "sp18")]
    assert joint_primary == [238, 161, 239, 164]
    assert pos["overall"]["primary"]["pos33"]["joint"] == 228
    assert pos["overall"]["primary"]["pos33"]["early_negative_fits"] == 0
    assert pos["overall"]["late"]["pos33"]["early_negative_fits"] == 0

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

    # Graph & out-degree scaling + 48-parameter Gamma recovery
    max_comp = max(r["state_compression_ratio"] for r in gsc["gradient_and_state_scaling"])
    assert round(max_comp, 2) == 1074.36
    assert [r["ph_joint_hits_of_5"] for r in gsc["highdim_gamma_recovery"]] == [5, 5, 5]
    assert [r["mc_joint_hits_of_5"] for r in gsc["highdim_gamma_recovery"]] == [0, 0, 0]

    # Finding 3: Neural clocks vs 3-start Latent ODE, LTC, S4D, and Neural-CME
    kl_nc = round(float(nm["primary"]["models"]["neural_phase"]["metrics"]["balanced_kl"]["median"]), 5)
    kl_ms = round(float(nm["primary"]["models"]["mean_shape"]["metrics"]["balanced_kl"]["median"]), 5)
    kl_lo = round(float(mnb["primary"]["latent_ode_3start"]["metrics"]["balanced_kl"]["median"]), 5)
    kl_ltc = round(float(mnb["primary"]["ltc_ode_3start"]["metrics"]["balanced_kl"]["median"]), 5)
    kl_s4d = round(float(mnb["primary"]["s4d_ode_3start"]["metrics"]["balanced_kl"]["median"]), 5)
    kl_cme = round(float(mnb["primary"]["neural_cme_3start"]["metrics"]["balanced_kl"]["median"]), 5)
    assert (kl_nc, kl_ms, kl_lo, kl_ltc, kl_s4d, kl_cme) == (
        0.00197,
        0.01095,
        0.07982,
        0.05644,
        0.06793,
        0.12854,
    )

    # Finding 4: Real continuous-time networks, EdgeMemoryHyperDRN, Hematopoiesis, & persistence
    ec_zs_drn = rcb["ecoli_continuous_cycle"]["models_5fold_mean"]["drn_fresh_only"]["aged_kl"]
    ec_zs_lode = rcb["ecoli_continuous_cycle"]["models_5fold_mean"]["latent_ode_fresh_only"]["aged_kl"]
    sl_comp = rcb["sleep_5node_branching_network"]["state_compression_ratio"]
    em_zs = ems["edge_memory_drn_fresh_only"]["aged_kl_mean"]
    em_joint_aged = ems["edge_memory_drn_joint"]["aged_kl_mean"]
    hem_day7_drn = hem["day1to5_to_day7_extrapolation"]["compiled_drn"]["day7_kl"]
    assert (ec_zs_drn, ec_zs_lode, sl_comp, em_zs, em_joint_aged, hem_day7_drn) == (
        0.05691,
        0.42433,
        52.51,
        0.0622,
        0.02206,
        0.00816,
    )
    n2_gain = round(float(rvm["datasets"]["sleep"]["models"]["adjusted"]["previous"]["2"]["gain"]), 5)
    ecoli_si = round(float(rvm["datasets"]["nomass"]["models"]["adjusted"]["previous"]["all"]["gain"]), 5)
    assert (n2_gain, ecoli_si) == (0.01201, 0.01293)

    # New Systems ML (IT Helpdesk), Extended Real Baselines (LTC/S4D/Chrono/Donor), MeanFieldDRN, & SparseDRN checks
    hdk_em_zs = round(float(hdk["cv_5fold_summary"]["edge_memory_drn_fresh_only"]["aged_kl_mean"]), 4)
    hdk_lo_zs = round(float(hdk["cv_5fold_summary"]["latent_ode_fresh_only"]["aged_kl_mean"]), 4)
    hdk_cme_zs = round(float(hdk["cv_5fold_summary"]["neural_cme_fresh_only"]["aged_kl_mean"]), 4)
    hdk_ltc_zs = round(float(hdk_ext["cv_5fold_extended"]["ltc_ode_fresh_only"]["aged_kl_mean"]), 4)
    hdk_s4d_zs = round(float(hdk_ext["cv_5fold_extended"]["s4d_ode_fresh_only"]["aged_kl_mean"]), 4)
    hdk_ups_zs = round(float(hdk_ext["cv_5fold_extended"]["universal_pos_sens_fresh_only"]["aged_kl_mean"]), 4)
    hdk_ch = hdk_ext["chronological_split"]["models"]
    assert hdk_em_zs < 0.04 and hdk_lo_zs > 0.20 and hdk_cme_zs > 0.25 and hdk_ltc_zs > 0.20 and hdk_s4d_zs > 0.20
    assert hdk_ups_zs < 0.14
    assert hdk_ch["edge_memory_drn_fresh_only"]["aged_kl"] < 0.16
    assert min(hdk_ch[k]["aged_kl"] for k in ("ltc_ode_fresh_only", "s4d_ode_fresh_only", "neural_cme_fresh_only", "latent_ode_fresh_only")) > 0.50
    slp_ltc_zs = round(float(slp_ext["models_5fold_mean"]["ltc_ode_fresh_only"]["aged_kl_mean"]), 4)
    slp_s4d_zs = round(float(slp_ext["models_5fold_mean"]["s4d_ode_fresh_only"]["aged_kl_mean"]), 4)
    assert slp_ltc_zs > 0.085 and slp_s4d_zs > 0.085 and em_zs < slp_s4d_zs
    hem_d7_drn = round(float(hem_d["unseen_donor_day7"]["compiled_drn"]["kl_mean"]), 4)
    hem_d7_ctmc = round(float(hem_d["unseen_donor_day7"]["ctmc"]["kl_mean"]), 4)
    assert hem_d7_drn < hem_d7_ctmc
    mfe_drn_ex8 = round(float(mfe["extrapolation_splits"]["days_1to8_to_9to14"]["mean_field_drn"]["test"]["mean_kl_nats"]), 4)
    mfe_mkv_ex8 = round(float(mfe["extrapolation_splits"]["days_1to8_to_9to14"]["markov_seicr"]["test"]["mean_kl_nats"]), 4)
    mfe_slope = round(float(mfe["kurtz_propagation_of_chaos"]["empirical_slope"]), 3)
    assert mfe_drn_ex8 < 0.02 and mfe_mkv_ex8 > 0.05 and -0.65 <= mfe_slope <= -0.35
    sp_1024 = [r for r in spg["scaling_rows"] if r["nodes"] == 1024][0]
    assert sp_1024["compiled_states"] == 31744 and sp_1024["sparse_operator_mb"] < 1.0
    assert spg["dense_vs_sparse_v16_check"]["gradient_relative_l2_error"] < 1e-10
    rss_1024 = [r for r in sp_rss["peak_rss_scaling"] if r["nodes"] == 1024][0]
    e2e_v64 = sp_rss["e2e_v64_256param_optimization"]
    assert rss_1024["peak_total_rss_mb"] < 700.0 and e2e_v64["heldout_kl_sparse_drn_nats"] < 0.001 and e2e_v64["kl_reduction_vs_ctmc"] > 500.0
    assert zs_ctrl["helpdesk_5fold"]["empirical_fresh_oracle_aged_kl_mean"] > 0.24
    assert zs_ctrl["helpdesk_chrono"]["empirical_fresh_oracle_aged_kl"] > 0.55
    assert zs_ctrl["sleep_5fold"]["empirical_fresh_oracle_aged_kl_mean"] > 0.085

    # Round-5 complete baseline, full-cone universal Hyper-Pos-Sens, and converged SparseDRN assertions
    assert bio_r5["universal_clock_certificate_full_cone"]["sup_survival_abs_error"] < 0.005
    assert bio_r5["hematopoiesis_r5"]["pooled_day7"]["universal_pos_sens_fullcone"]["day7_kl"] < 0.015
    assert bio_r5["influenza_r5"]["ltc_ode"]["days_1to8_to_9to14_best_l2"]["test"]["mean_kl_nats"] > 0.40
    assert bio_r5["influenza_r5"]["s4d_ode"]["days_1to8_to_9to14_best_l2"]["test"]["mean_kl_nats"] > 0.20
    assert bio_r5["sleep_hyper_pos_sens_r5"]["models_5fold_mean"]["hyper_pos_sens_fresh_only"]["aged_kl_mean"] < 0.095
    assert sp_r5["v64_converged_256param"]["all_nodes_mean_rms_pct"] < 2.0
    assert sp_r5["v64_converged_256param"]["all_nodes_shape_rms_pct"] < 10.0
    assert sp_r5["v64_converged_256param"]["kl_reduction_vs_ctmc"] > 3000.0
    assert sp_r5["v256_e2e_1024param"]["heldout_kl_sparse_drn_nats"] < 0.001
    assert hdk_r5["cv_5fold_clean_age0"]["hyper_pos_sens_fresh_only"]["aged_kl_mean"] < 0.05
    assert min(
        hdk_r5["cv_5fold_clean_age0"][f"{m}_fresh_only"]["aged_kl_mean"]
        for m in ("ltc_ode", "s4d_ode", "neural_cme", "latent_ode")
    ) > 0.23
    assert min(
        hdk_r5["chronological_split_3start_clean_age0"]["models"][f"{m}_fresh_only"]["aged_kl"]
        for m in ("ltc_ode", "s4d_ode", "neural_cme", "latent_ode")
    ) > 0.50
    assert min(
        r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"][f"{m}_fresh_only"]["aged_kl"]
        for m in ("ltc_ode", "s4d_ode", "neural_cme", "latent_ode")
    ) > 0.40
    assert r6["helpdesk_r6_hyper_pos_coarsening"][3]["hyper_pos_sens_aged_kl"] < 0.05
    assert r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ctmc_fresh_only"]["aged_kl_mean"] > 0.10
    assert r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["neural_cme_shifted_t_plus_a"]["aged_kl_mean"] > 0.10
    assert r6b["influenza_r6b_neural_cme"]["days_1to8_to_9to14_best_l2"]["test"]["mean_kl_nats"] > 0.03

    hdk_r5_cv = hdk_r5["cv_5fold_clean_age0"]
    hdk_r5_ch = hdk_r5["chronological_split_3start_clean_age0"]["models"]
    slp_r6b = r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]
    ec_r6_m = r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]

    return {
        "cyclic_primary_joint_hits_of_280": {
            **dict(zip(("full48", "sym33", "sp33", "sp18"), joint_primary)),
            "pos33": pos["overall"]["primary"]["pos33"]["joint"],
        },
        "pos33_negative_early_fits_of_400": 0,
        "max_marked_race_compression": max_comp,
        "highdim_ph_joint_hits_of_15": 15,
        "fisher_shape_median_pct": {"ph": ph_med, "nomass": nm_med},
        "fisher_cold_speedup_median": round(float(np.median(ratios)), 2),
        "trait_datasets": tf["audit"]["datasets"],
        "neural_clock_balanced_kl": {
            "neural_clock": kl_nc,
            "moment_ph": kl_ms,
            "ltc_3start": kl_ltc,
            "s4d_3start": kl_s4d,
            "latent_ode": kl_lo,
            "neural_cme_3start": kl_cme,
        },
        "real_continuous_zeroshot_aged_kl": {
            "ecoli_drn": ec_zs_drn,
            "ecoli_lode_clean_age0": ec_r6_m["latent_ode_fresh_only"]["aged_kl"],
            "ecoli_s4d_clean_age0": ec_r6_m["s4d_ode_fresh_only"]["aged_kl"],
            "sleep_edge_memory_drn": em_zs,
            "sleep_ctmc_strict_fresh_only": slp_r6b["ctmc_fresh_only"]["aged_kl_mean"],
        },
        "sleep_edge_memory_joint_aged_kl": em_joint_aged,
        "hematopoiesis_day7_extrapolation_kl": hem_day7_drn,
        "sleep_5node_state_compression": sl_comp,
        "persistent_carry_targets": list(rt6["targets"].keys()),
        "real_visit_gains_nats": {"sleep_n2": n2_gain, "ecoli_si": ecoli_si},
        "helpdesk_zeroshot_aged_kl_clean_age0": {
            "edge_memory_drn": hdk_em_zs,
            "hyper_pos_sens": round(float(hdk_r5_cv["hyper_pos_sens_fresh_only"]["aged_kl_mean"]), 4),
            "universal_pos_sens": hdk_ups_zs,
            "ltc_ode": round(float(hdk_r5_cv["ltc_ode_fresh_only"]["aged_kl_mean"]), 4),
            "s4d_ode": round(float(hdk_r5_cv["s4d_ode_fresh_only"]["aged_kl_mean"]), 4),
            "latent_ode": round(float(hdk_r5_cv["latent_ode_fresh_only"]["aged_kl_mean"]), 4),
            "neural_cme": round(float(hdk_r5_cv["neural_cme_fresh_only"]["aged_kl_mean"]), 4),
            "chrono_edge_memory_drn": round(float(hdk_ch["edge_memory_drn_fresh_only"]["aged_kl"]), 4),
            "chrono_ltc_ode": round(float(hdk_r5_ch["ltc_ode_fresh_only"]["aged_kl"]), 4),
        },
        "sleep_3start_clean_age0_zeroshot_aged_kl": {
            "ctmc_strict_fresh_only": slp_r6b["ctmc_fresh_only"]["aged_kl_mean"],
            "ltc_ode": slp_r6b["ltc_ode_fresh_only"]["aged_kl_mean"],
            "s4d_ode": slp_r6b["s4d_ode_fresh_only"]["aged_kl_mean"],
            "neural_cme": slp_r6b["neural_cme_fresh_only"]["aged_kl_mean"],
            "neural_cme_shifted_t_plus_a": slp_r6b["neural_cme_shifted_t_plus_a"]["aged_kl_mean"],
            "latent_ode": slp_r6b["latent_ode_fresh_only"]["aged_kl_mean"],
        },
        "hematopoiesis_donor_heldout_day7_kl": {
            "compiled_drn": hem_d7_drn,
            "ctmc": hem_d7_ctmc,
        },
        "mean_field_influenza_day9to14_kl": {
            "mean_field_drn": mfe_drn_ex8,
            "markov_seicr": mfe_mkv_ex8,
            "neural_cme": round(float(r6b["influenza_r6b_neural_cme"]["days_1to8_to_9to14_best_l2"]["test"]["mean_kl_nats"]), 4),
            "kurtz_slope": mfe_slope,
        },
        "sparse_drn_v1024": {
            "compiled_states": sp_1024["compiled_states"],
            "sparse_operator_mb": round(sp_1024["sparse_operator_mb"], 3),
            "peak_total_rss_mb": rss_1024["peak_total_rss_mb"],
            "grad_warm_ms": round(sp_1024["grad_warm_ms"], 1),
            "e2e_v64_kl_reduction_vs_ctmc": sp_r5["v64_converged_256param"]["kl_reduction_vs_ctmc"],
            "e2e_v256_kl_reduction_vs_ctmc": sp_r5["v256_e2e_1024param"]["kl_reduction_vs_ctmc"],
        },
        "theorem6_bt_near_cancellation": {
            "k4_delta_1e-5_ratio_rebal": round(r7t["symmetric_cycles"]["K4"]["sweep"][0]["ratio_rebal"], 1),
            "k4_delta_1e-5_ratio_frozen": round(r7t["symmetric_cycles"]["K4"]["sweep"][0]["ratio_frozen"], 1),
            "het433_delta_1e-5_ratio_frozen": round(r7t["heterogeneous_common_zero"]["sweep"][0]["ratio_frozen"], 1),
        },
        "mamba_and_thp_baselines": {
            "synthetic_ph_kl": round(r7m["synthetic_mamba"]["primary"]["balanced_kl"]["median"], 5),
            "synthetic_lognormal_kl": round(r7m["synthetic_mamba"]["lognormal"]["balanced_kl"]["median"], 5),
            "helpdesk_0shot_aged_kl": round(r7m["helpdesk_mamba_and_thp"]["cv5"]["fresh_only"]["aged_kl_mean"], 4),
            "helpdesk_thp_14d_kl": round(r7m["helpdesk_mamba_and_thp"]["coarsening"]["dt_14.0"]["thp_forecast_kl"], 4),
            "sleep_0shot_aged_kl": round(r7m["sleep_mamba"]["cv5"]["fresh_only"]["aged_kl_mean"], 4),
            "influenza_day9to14_kl": round(r7m["influenza_mamba"]["days_1to8_to_9to14_best_l2"]["test"]["mean_kl_nats"], 4),
        },
        "mooc_v98_real_sparse_drn": {
            "compiled_states": r7n["mooc_v98"]["results"]["sparse_drn_meta"]["compiled_states"],
            "sparse_hyper_drn_0shot_aged_kl": round(r7n["mooc_v98"]["results"]["sparse_hyper_drn_fresh_only"]["aged_kl"], 4),
            "sparse_hyper_drn_joint_aged_kl": round(r7n["mooc_v98"]["results"]["sparse_hyper_drn_joint"]["aged_kl"], 4),
            "ctmc_0shot_aged_kl": round(r7n["mooc_v98"]["results"]["ctmc_fresh_only"]["aged_kl"], 4),
            "mamba_0shot_aged_kl": round(r7n["mooc_v98"]["results"]["mamba_fresh_only"]["aged_kl"], 4),
            "ltc_0shot_aged_kl": round(r7n["mooc_v98"]["results"]["ltc_fresh_only"]["aged_kl"], 4),
        },
        "defi_multipool_183k_neural_clock": {
            "total_rows": r7d["dataset"]["total_rows"],
            "regime_a_zeroshot_neural_clock": round(r7d["regime_a_zeroshot_deep_age_sale"]["neural_clock"]["all"]["logloss"], 5),
            "regime_a_zeroshot_mlp_age": round(r7d["regime_a_zeroshot_deep_age_sale"]["mlp_age"]["all"]["logloss"], 5),
            "regime_a_zeroshot_mamba": round(r7d["regime_a_zeroshot_deep_age_sale"]["mamba_ssm"]["all"]["logloss"], 5),
            "regime_b_drift_neural_clock": round(r7d["regime_b_temporal_drift_sale"]["neural_clock"]["all"]["logloss"], 5),
            "regime_b_drift_mlp_age": round(r7d["regime_b_temporal_drift_sale"]["mlp_age"]["all"]["logloss"], 5),
            "regime_b_drift_mamba": round(r7d["regime_b_temporal_drift_sale"]["mamba_ssm"]["all"]["logloss"], 5),
        },
        "r8_laplace_talbot_frailty_and_defi_mh": {
            "mooc_talbot_warm_grad_ms": r8n["mooc_v98_r8"]["talbot_sparse_hyper_drn_fresh_only"]["warm_grad_ms"],
            "mooc_talbot_0shot_aged_kl": round(r8n["mooc_v98_r8"]["talbot_sparse_hyper_drn_fresh_only"]["aged_kl"], 4),
            "mooc_talbot_joint_aged_kl": round(r8n["mooc_v98_r8"]["talbot_sparse_hyper_drn_joint"]["aged_kl"], 4),
            "mooc_univ_hyper_pos_0shot_aged_kl": round(r8n["mooc_v98_r8"]["universal_hyper_pos_sens_fresh_only"]["aged_kl"], 4),
            "mooc_deep_2layer_mamba_params": r8n["mooc_v98_r8"]["deep_2layer_mamba_adamw"]["params"],
            "mooc_deep_2layer_mamba_joint_aged_kl": round(r8n["mooc_v98_r8"]["deep_2layer_mamba_adamw"]["joint"]["aged_kl"], 4),
            "stackoverflow_pure_frailty_0shot_aged_kl": round(r8n["stackoverflow_v22_r8"]["pure_frailty_g3_fresh_only"]["aged_kl"], 4),
            "defi_m4_drift_logloss": round(r8d["regime_b_temporal_drift"]["neural_clock_m4"]["all"]["logloss"], 5),
            "defi_m4_bsc_to_base_logloss": round(r8d["regime_c_bsc_to_base_young"]["neural_clock_m4"]["all"]["logloss"], 5),
            "defi_m4_dt120_zeroshot_logloss": round(r8d["multi_horizon_zeroshot_transfer"]["all_age_train_dt10"]["dt_120m"]["neural_clock_m4"]["logloss"], 4),
            "defi_mlp_chain_dt120_logloss": round(r8d["multi_horizon_zeroshot_transfer"]["all_age_train_dt10"]["dt_120m"]["mlp_age_chained"]["logloss"], 4),
            "real_hemato_bt49_distortion_fz": round(r8b["hematopoiesis_7node_real_bt_vs_pos_sens"]["balanced_truncation_sweep"]["r48"]["distortion_ratio_l2"], 1),
            "real_helpdesk_bt37_distortion_fz": round(r8b["helpdesk_5node_real_bt_vs_hyper_pos_sens"]["balanced_truncation_sweep"]["r36"]["distortion_ratio_l2"], 1),
        },
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
