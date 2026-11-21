"""Differentiable Renewal Networks (diff_renewal).

Core JAX and Diffrax primitives for differentiable phase-type renewal networks,
exact marked-race compilation, sensitivity-interpolating model reduction,
conditional neural waiting-time clocks, persistent individual traits, and
scenario-robust Fisher observation design.
"""

from .baselines import (
    init_population_baseline,
    make_baseline_engine,
    predict_population_baseline,
)
from .capping import (
    cap_mixed_erlang,
    capped_clock_bank_gradient_bound,
    capped_network_tv_bound,
)
from .design import (
    a_optimal_value_gradient,
    largest_remainder,
    optimize_fisher_allocation,
)
from .diagnostics import (
    adjoint_residual_estimate,
    adjoint_sensitivity_residual,
    coarsened_fisher,
    efficient_fisher,
    multinomial_score_design,
)
from .memory import (
    MemoryCutoff,
    fit_positive_soe,
    fixed_partition_cutoff,
    solve_memory,
    static_preserving_cutoff,
)
from .network import (
    network_layout,
    network_state_size,
    solve_renewal_network,
)
from .neural_clock import (
    ClockNetwork,
    init_clock_network,
    neural_clock,
    survivor_phases,
)
from .observations import multinomial_nll
from .phase_type import (
    coxian_generator,
    erlang_generator,
    mixed_erlang_components,
    mixed_erlang_moments,
    phase_type_dynamics,
    smooth_erlang_logits,
)
from .race import (
    compact_network_state_size,
    mixed_erlang_race,
    solve_compact_renewal_network,
)
from .reduction import (
    ANCHORS_DEFAULT,
    K_POS_RED,
    READ_POS33,
    SHIFTS_DEFAULT,
    build_interpolatory_basis,
    calibrate_positive_33_surrogate,
    cycle_symmetry_matrices,
    make_cycle_engine,
    make_interpolatory_engine,
    make_positive_33_engine,
    pos33_cycle_matrices,
    resolvent_sensitivity_diagnostic,
)
from .renewal import solve_erlang_chain
from .traits import (
    endpoint_coefficients,
    moment_matched_logits,
    trait_mean_shape,
    trait_network_probabilities,
)

__all__ = [
    "ANCHORS_DEFAULT",
    "ClockNetwork",
    "K_POS_RED",
    "MemoryCutoff",
    "READ_POS33",
    "SHIFTS_DEFAULT",
    "a_optimal_value_gradient",
    "adjoint_residual_estimate",
    "adjoint_sensitivity_residual",
    "build_interpolatory_basis",
    "calibrate_positive_33_surrogate",
    "cap_mixed_erlang",
    "capped_clock_bank_gradient_bound",
    "capped_network_tv_bound",
    "coarsened_fisher",
    "compact_network_state_size",
    "coxian_generator",
    "cycle_symmetry_matrices",
    "efficient_fisher",
    "endpoint_coefficients",
    "erlang_generator",
    "fit_positive_soe",
    "fixed_partition_cutoff",
    "init_clock_network",
    "init_population_baseline",
    "largest_remainder",
    "make_baseline_engine",
    "make_cycle_engine",
    "make_interpolatory_engine",
    "make_positive_33_engine",
    "mixed_erlang_components",
    "mixed_erlang_moments",
    "mixed_erlang_race",
    "moment_matched_logits",
    "multinomial_nll",
    "multinomial_score_design",
    "network_layout",
    "network_state_size",
    "neural_clock",
    "optimize_fisher_allocation",
    "phase_type_dynamics",
    "pos33_cycle_matrices",
    "predict_population_baseline",
    "resolvent_sensitivity_diagnostic",
    "smooth_erlang_logits",
    "solve_compact_renewal_network",
    "solve_erlang_chain",
    "solve_memory",
    "solve_renewal_network",
    "static_preserving_cutoff",
    "survivor_phases",
    "trait_mean_shape",
    "trait_network_probabilities",
]
