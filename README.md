# Differentiable Renewal Networks (`diff_renewal`)

Clean, standalone, and reproducible JAX/Diffrax implementation accompanying the manuscript **Differentiable Renewal Networks: When Predictive Minimality Fails Parameter Learning**.

## Overview

Many physical, operations-queueing, epidemiological, and biological processes remember elapsed residence time in a state, from IT ticketing workflows and epidemic incubation cascades to embryonic stem-cell differentiation, bacterial division lineages, and human sleep stages. **Differentiable Renewal Networks** enable exact gradient-based parameter inference, positivity- and sensitivity-preserving state compression, incoming-edge semi-Markov neural clocks, nonlinear mean-field contagion/congestion ODEs, matrix-free $O(V + KE)$ scaling, and Fisher experimental design from coarse population snapshots:

1. **Exact Mixed- & Hyper-Erlang Marked-Race Compilation (`diff_renewal.race`, `diff_renewal.sparse_network`)**: Compiles $d$ competing common-rate mixed-Erlang clocks ($K^d \to 1 + \sum_e (K_e - 1)$, Proposition 1) and multi-rate $CV > 1$ Hyper-Erlang mixtures ($(MK)^d \to M^d(1 + d(K - 1))$, Proposition 2; $1{,}074.4\times$ state reduction at $d=4, K=16$), plus a matrix-free sparse block-operator solver (`SparseDRN`) scaling exact reverse-mode Diffrax gradients to $V = 1{,}024$ nodes ($31{,}744$ compiled states, $4{,}096$ parameters in $0.97\,\mathrm{MB}$ / $452\,\mathrm{ms}$, matching dense compilation to $2.8 \times 10^{-14}$).
2. **Positivity-Preserving & Topology-Agnostic Model Reduction (`diff_renewal.reduction`)**: Implements stationary-profile symmetry continuations, two-sided Laplace resolvent sensitivity interpolation (Baur et al., 2011), Fisher-calibrated positive Coxian reduction (`Pos-Sens 33`, $0/400$ negative probabilities vs. $400/400$ for signed projections), and a scale-equivariant 1D universal `Pos-Sens` clock compressor ($\sup |\tilde{S} - S| < 0.004$) that plugs zero-shot into arbitrary directed networks $(V, E)$ alongside sharp Aldous–Shepp impossibility lower bounds (Theorem 5).
3. **Conditional Neural Clocks, Incoming-Edge Semi-Markov Fibres, Nonlinear Mean-Field ODEs, and Modern Baselines (`diff_renewal.neural_clock`, `diff_renewal.semi_markov`, `diff_renewal.mean_field`, `diff_renewal.baselines`)**: Conditional neural clocks with exact Poisson-tail survivor conditioning, incoming-transition lifted two-scale Hyper-Erlang networks (`EdgeMemoryHyperDRN`), nonlinear McKean–Vlasov mean-field renewal networks (`MeanFieldDRN`) with verified Kurtz $O(M^{-1/2})$ convergence, and multi-start continuous-time baselines (`Latent ODE`, `LTC / CfC`, `S4D`, Time-Inhomogeneous `Neural-CME`, and monotone cumulative-hazard `Neural Competing-Risks`).
4. **Scenario-Robust Fisher Experimental Design (`diff_renewal.design`, `diff_renewal.diagnostics`)**: Two-stage pilot-and-follow-up A-optimal snapshot allocation over preparations, observation times, and observed traits with nuisance-parameter accounting and integer largest-remainder rounding.

## Repository Structure

```text
diff-renewal-network/
├── src/diff_renewal/
│   ├── phase_type.py     # Erlang, smooth mixed-Erlang, moment calculation, Coxian generators
│   ├── renewal.py        # Sparse shared-suffix sequential mixed-Erlang chain solver
│   ├── network.py        # Full K**d tensor renewal network solver in Diffrax
│   ├── race.py           # Exact mixed- & Hyper-Erlang marked-race compiler & vectorized solver
│   ├── sparse_network.py # Matrix-free O(V + K*E) SparseDRN solver up to V = 1,024 nodes
│   ├── mean_field.py     # Nonlinear McKean-Vlasov MeanFieldDRN ODE + interacting Gillespie
│   ├── reduction.py      # Symmetry closure, 2-sided interpolation, Pos-Sens 33 & universal 1D
│   ├── traits.py         # Persistent individual trait regressions & custom-JVP shape matching
│   ├── neural_clock.py   # Conditional neural waiting-time clocks & survivor phase conditioning
│   ├── semi_markov.py    # Incoming-edge lifted two-scale Hyper-Erlang semi-Markov DRN
│   ├── baselines.py      # Multi-start Latent ODE, LTC, S4D, & Time-Inhomogeneous Neural-CME
│   ├── memory.py         # Augmented exponential-kernel solver & static-preserving cutoffs
│   ├── capping.py        # Rate-preserving phase capping & path-TV / latent-Fisher bounds
│   ├── observations.py   # Count-weighted multinomial negative log likelihood
│   ├── design.py         # Convex scenario-averaged A-optimal Fisher allocation & rounding
│   └── diagnostics.py    # Coarsened/efficient Fisher matrices & adjoint sensitivity residuals
├── scripts/
│   ├── verify_theory.py                        # Symbolic (SymPy) & numerical proof audit
│   ├── reproduce_all.py                        # SHA-256 audit of 26 archives + ODE replays
│   ├── build_paper_assets.py                   # Rebuilds all 5 vector figures & 13 LaTeX tables
│   ├── run_positive_sensitivity_reduction.py   # 400-dataset Pos-Sens 33 cyclic benchmark
│   ├── run_graph_scaling_benchmark.py          # V=4..48, d=2..4 scaling & 48-param Gamma recovery
│   ├── run_sparse_large_graph_scaling.py       # Matrix-free SparseDRN V=16..1,024 benchmark
│   ├── run_modern_neural_baselines.py          # Multi-start Latent ODE, LTC, & S4D comparisons
│   ├── run_neural_cme_baseline.py              # 3-start Time-Inhomogeneous Neural-CME benchmark
│   ├── run_real_continuous_sleep_network.py    # Continuous-time E. coli & 5-node Sleep-EDF
│   ├── run_edge_memory_sleep_benchmark.py      # EdgeMemoryHyperDRN & Neural-CME on Sleep-EDF
│   ├── run_hematopoiesis_branching_benchmark.py # 7-node branching human hematopoiesis FACS
│   ├── run_helpdesk_systems_benchmark.py       # 4,576-ticket IT Helpdesk CV & Neural-CR sweep
│   └── run_mean_field_epidemic_benchmark.py    # 1978 Influenza MeanFieldDRN & Kurtz O(M^-1/2)
├── examples/
│   └── minimal_renewal_network.py  # Compact runnable marked-race & gradient example
├── results/archived/     # 26 SHA-256-fingerprinted benchmark archives across all paper studies
├── tests/                # Pytest unit and end-to-end reproducibility suite
├── pyproject.toml
└── requirements.txt
```

## Quick Start & Reproducibility

```bash
pip install -e ".[dev]"
pytest -q
python scripts/verify_theory.py
python scripts/reproduce_all.py
python scripts/build_paper_assets.py
```

## Public Empirical Datasets

All real-world evaluations analyze publicly released, checksum-verified datasets spanning **operations/systems engineering, epidemiology, and biology**:

- **IT Helpdesk software-support queueing workflow (`Assign -> Active <-> Wait -> Resolved -> Closed`)**: 4,580 tickets (21,348 timestamped events, MD5 `a91d74377c55be5cbdfda05dae69db30`) from Polato (4TU.ResearchData, 2017; DOI `10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb`).
- **1978 English Boarding School H1N1 Influenza Outbreak (`S -> E -> In-Bed -> Convalescent -> R`)**: Daily multi-compartment prevalence snapshots ($N = 763$ boys, 14 days) from Anonymous (*BMJ*, 1978, 1(6112):587).
- **Primary human hematopoietic branching lineage hierarchy (`HSC -> MPP -> {MLP, CMP} -> {GMP, MEP} -> Mature`)**: 78 FACS snapshots across 12 healthy bone-marrow donor cultures over Days 1--7 from Bast et al. (*iScience*, 2021).
- **Mouse embryonic stem-cell differentiation (`ESC -> EPI -> NPC`)**: R1 cell line from Stumpf et al. (*Cell Systems*, 2017) as curated in Pélissier et al. (*Briefings in Bioinformatics*, 2026).
- **Single-cell *Escherichia coli* mother-machine lineages**: Si et al. (*Current Biology*, 2019; 9,449 cycles) and Tanouchi et al. (*Scientific Data*, 2017; 18,972 cycles).
- **Human multi-night sleep staging**: Sleep-EDF Expanded (Kemp et al., *IEEE TBME*, 2000; 75 paired-night adults), BIDSleep (Song et al., *IEEE TBME*, 2026; 253 home nights), and Ying et al. (*Scientific Data*, 2025; 58 triple-scorer recordings).

## License

Released under the [MIT License](LICENSE).
