"""End-to-end reproducibility test checking SHA-256 archives and figure assets."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_reproduce_all_script():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce_all.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(proc.stdout)
    assert summary["status"] == "PASSED"
    assert summary["verified_archive_files"] == 43
    assert summary["erlang16_replay"]["nominal_max_error"] < 1e-12
    assert summary["stem_cell_replay"]["abs_drift"] < 1e-8
    assert summary["headline_claims"]["mooc_v98_real_sparse_drn"]["sparse_hyper_drn_0shot_aged_kl"] < 0.30
    r8 = summary["headline_claims"]["r8_laplace_talbot_frailty_and_defi_mh"]
    assert r8["mooc_talbot_0shot_aged_kl"] < 0.20
    assert r8["mooc_talbot_n12_expm_kl_diff"] < 1e-5
    assert r8["mooc_talbot_joint_aged_kl"] < r8["mooc_deep_2layer_mamba_joint_aged_kl"]
    assert r8["mooc_talbot_warm_grad_ms"] < 300.0
    assert r8["mooc_univ_hyper_pos_0shot_aged_kl"] < 0.28
    assert r8["stackoverflow_pure_frailty_0shot_aged_kl"] < 0.12
    assert r8["stackoverflow_pure_me_g1_0shot_aged_kl"] < 0.09
    assert r8["defi_m4_dt10_logloss"] < r8["defi_mlp_dt10_logloss"]
    assert r8["real_hemato_bt13_min_prob"] < 0.0
    assert r8["real_helpdesk_bt13_min_prob"] < 0.0
    assert (
        summary["headline_claims"]["defi_multipool_183k_neural_clock"]["regime_b_drift_neural_clock"]
        < summary["headline_claims"]["defi_multipool_183k_neural_clock"]["regime_b_drift_mlp_age"]
    )
