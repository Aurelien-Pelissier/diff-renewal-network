"""Unit tests for 33-vs-48 sensitivity separation and interpolatory reduction."""

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from diff_renewal import (
    ANCHORS_DEFAULT,
    SHIFTS_DEFAULT,
    build_interpolatory_basis,
    make_cycle_engine,
    make_interpolatory_engine,
    resolvent_sensitivity_diagnostic,
)


def test_symmetry_closure_nominal_match_and_gradient_defect():
    full_eval = make_cycle_engine(reduced=False)
    red_eval = make_cycle_engine(reduced=True)
    theta_sym = np.log([2.0, 2.0, 2.0, 6.0, 6.0, 6.0])
    p_full, j_full = full_eval(theta_sym)
    p_red, j_red = red_eval(theta_sym)

    # Identical nominal predictions at symmetry
    np.testing.assert_allclose(p_full, p_red, atol=1e-12)

    # Common direction has zero defect, contrast direction has nonzero defect
    common_mean = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) / np.sqrt(3.0)
    contrast_mean = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)
    err_common = np.max(np.abs((j_full - j_red) @ common_mean))
    err_contrast = np.max(np.abs((j_full - j_red) @ contrast_mean))
    assert err_common < 1e-13
    assert err_contrast > 0.10


def test_interpolatory_basis_matches_transfer_and_sensitivities():
    q33, info33 = build_interpolatory_basis(33)
    assert q33.shape == (48, 33)
    assert info33["relative_span_residual"] < 1e-14
    assert info33["orthogonality_error"] < 1e-14

    diag = resolvent_sensitivity_diagnostic(q33, ANCHORS_DEFAULT[0], SHIFTS_DEFAULT[1])
    assert diag["transfer_max_abs"] < 1e-13
    assert diag["sensitivity_max_abs"] < 1e-13

    interp_eval = make_interpolatory_engine(q33)
    full_eval = make_cycle_engine(reduced=False)
    p_f, j_f = full_eval(ANCHORS_DEFAULT[0])
    p_i, j_i = interp_eval(ANCHORS_DEFAULT[0])
    assert np.max(np.abs(p_f - p_i)) < 1e-4
    assert np.max(np.abs(j_f - j_i)) < 6e-3
    assert np.max(np.abs(j_f[2:] - j_i[2:])) < 5e-3
