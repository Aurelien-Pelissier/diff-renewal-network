"""Differentiable Renewal Networks (diff_renewal).

Core JAX and Diffrax primitives for differentiable phase-type renewal networks,
exact mixed- and Hyper-Erlang marked-race compilation, positivity-preserving
Fisher-sensitivity reduction, incoming-edge semi-Markov clocks, nonlinear
mean-field renewal ODEs, matrix-free sparse graph solvers, and Fisher design.
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
from .mean_field import (
    seicr_mean_field_predict,
    simulate_interacting_seicr_gillespie,
    solve_mean_field_renewal,
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
    hyper_erlang_race,
    hyper_race_state_size,
    mixed_erlang_race,
    solve_compact_renewal_network,
)
from .reduction import (
    ANCHORS_DEFAULT,
    K_POS_RED,
    READ_POS33,
    SHIFTS_DEFAULT,
    build_interpolatory_basis,
    build_universal_hyper_pos_network_generator,
    build_universal_pos_network_generator,
    calibrate_positive_33_surrogate,
    calibrate_universal_pos_clock,
    cycle_symmetry_matrices,
    make_cycle_engine,
    make_interpolatory_engine,
    make_positive_33_engine,
    pos33_cycle_matrices,
    resolvent_sensitivity_diagnostic,
    universal_pos_clock,
)
from .renewal import solve_erlang_chain
from .semi_markov import (
    build_edge_memory_hyper_generator,
    edge_memory_param_size,
)
from .sparse_network import (
    apply_sparse_generator,
    compile_sparse_regular_network,
    solve_sparse_renewal_network,
)
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
    "apply_sparse_generator",
    "build_edge_memory_hyper_generator",
    "build_interpolatory_basis",
    "build_universal_hyper_pos_network_generator",
    "build_universal_pos_network_generator",
    "calibrate_positive_33_surrogate",
    "calibrate_universal_pos_clock",
    "cap_mixed_erlang",
    "capped_clock_bank_gradient_bound",
    "capped_network_tv_bound",
    "coarsened_fisher",
    "compact_network_state_size",
    "compile_sparse_regular_network",
    "coxian_generator",
    "cycle_symmetry_matrices",
    "edge_memory_param_size",
    "efficient_fisher",
    "endpoint_coefficients",
    "erlang_generator",
    "fit_positive_soe",
    "fixed_partition_cutoff",
    "hyper_erlang_race",
    "hyper_race_state_size",
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
    "seicr_mean_field_predict",
    "simulate_interacting_seicr_gillespie",
    "smooth_erlang_logits",
    "solve_compact_renewal_network",
    "solve_erlang_chain",
    "solve_mean_field_renewal",
    "solve_memory",
    "solve_renewal_network",
    "solve_sparse_renewal_network",
    "static_preserving_cutoff",
    "survivor_phases",
    "trait_mean_shape",
    "trait_network_probabilities",
    "universal_pos_clock",
]
