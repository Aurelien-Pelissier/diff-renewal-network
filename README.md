# Differentiable Renewal Networks (`diff_renewal`)

Clean, standalone, and reproducible JAX/Diffrax implementation accompanying the manuscript **Differentiable Renewal Networks: When Predictive Minimality Fails Parameter Learning**.

## Overview

Many temporal interaction, financial-market, operations-queueing, epidemiological, and biological processes remember elapsed residence time in a state, from MOOC student workflows, on-chain DeFi liquidity pools, and IT ticketing queues to epidemic incubation cascades, embryonic stem-cell differentiation, bacterial division lineages, and human sleep stages. **Differentiable Renewal Networks** enable exact gradient-based parameter inference, positivity- and sensitivity-preserving state compression, incoming-edge semi-Markov neural clocks, nonlinear mean-field contagion/congestion ODEs, matrix-free $O(V + KE)$ scaling, and Fisher experimental design from coarse population snapshots:

1. **Exact Mixed- & Hyper-Erlang Marked-Race Compilation, Matrix-Free Scaling, & Stiffness-Free `Laplace-Schur-Talbot` Solvers (`diff_renewal.race`, `diff_renewal.sparse_network`, `diff_renewal.laplace_solver`)**: Compiles $d$ competing common-rate mixed-Erlang clocks ($K^d \to 1 + \sum_e (K_e - 1)$, Proposition 1) and multi-rate $\mathrm{CV} > 1$ Hyper-Erlang mixtures ($(MK)^d \to M^d(1 + d(K - 1))$, Proposition 2; $1{,}074.4\times$ state reduction at $d=4, K=16$). Matrix-free sparse block-operator solvers (`SparseDRN` & `SparseHyperDRN`) scale exact reverse-mode Diffrax gradients to $V = 1{,}024$ synthetic nodes ($31{,}744$ compiled states, $4{,}096$ parameters in $0.97\,\mathrm{MB}$ operator storage / $644\,\mathrm{MB}$ peak autodiff RSS / $452\,\mathrm{ms}$, matching dense compilation to $2.8 \times 10^{-14}$). Our stiffness- and stage-count-independent **`Laplace-Schur-Talbot`** solver (`diff_renewal.laplace_solver`) inverts the Theorem 3 resolvent via a Weideman–Trefethen contour with $O(V^3)$ node-level Schur complements and exact implicit reverse-mode gradients, achieving **`143.3×` faster gradients** on real **MOOC ($V=98$, $411{,}749$ events, $1{,}355$ sparse edges; `142.4 ms/grad` vs. `20.4 s/grad` `Tsit5`)**, uncapping fast burst rates to $200.4\,\mathrm{h}^{-1}$ and cutting zero-shot aged-student KL to **`0.1903 nats`** (`7.8×` below a val-tuned `136,331`-param 2-layer AdamW Mamba `1.4937` and `9.5×–10.2×` lower than exact CTMC and all five parameter-matched neural ODE/SSM baselines, while also beating `136k`-param 2L-Mamba jointly `0.1487` vs. `0.1722` with `78×` fewer parameters). It also operationalizes Proposition 3 via persistent latent-speed **`Frailty-DRN`** (`Pure Frailty-DRN`, $G=3$), cutting **StackOverflow ($V=22$)** fresh KL by `14.6%` (`0.0222`) and zero-shot aged KL by `24.5%` (`0.1119`).
2. **Positivity-Preserving, Quantitative Near-Cancellation, & Topology-Agnostic Model Reduction (`diff_renewal.reduction`)**: Implements stationary-profile symmetry continuations, two-sided Laplace resolvent sensitivity interpolation (Baur et al., 2011), exact Kato Riesz-projection balanced-truncation analysis verifying Theorem 6 ($O(\delta)$ forward $H_\infty$ error alongside $\Omega(1)$ sensitivity floor and up to $4.10 \times 10^6\times$ $\Omega(\delta^{-1})$ synthetic / **`23.3M×` real Hematopoiesis & `9.79M×` real IT Helpdesk** distortion blow-up with positivity violations), Fisher-calibrated positive Coxian reduction (`Pos-Sens 33`, $0/400$ negative probabilities vs. $400/400$ for signed projections), a scale-equivariant 1D universal `Pos-Sens` clock compressor certified over $k \in [1.4, 7.8]$ ($\sup |\tilde{S} - S| = 0.00385$), and a two-branch `Universal Hyper-Pos-Sens` ($K_r + 1$ states/active node) extending data-free universal reduction zero-shot to real $V=98$ MOOC (`0.2690` zero-shot aged KL in `179.2 ms/grad`, `6.7×–7.2×` below baselines) alongside sharp Aldous–Shepp impossibility lower bounds (Theorem 5) and generalized heterogeneous sensitivity-pole theory (Theorem 3, $d_{\mathrm{sens}} = \sum_i n_i$).
3. **Conditional Neural Clocks, Incoming-Edge Semi-Markov Fibres, Nonlinear Mean-Field ODEs, and Modern Baselines (`diff_renewal.neural_clock`, `diff_renewal.semi_markov`, `diff_renewal.mean_field`, `diff_renewal.baselines`)**: Conditional neural clocks with exact analytical survivor conditioning (proportional-scale $M=4$ four-decade clock across `183,881` real on-chain BSC/Base DeFi liquidity opportunities: `0.31412` on 5-month drift beating Weibull/Markov/2L-Mamba and deep-age strata, #1 `0.31566` on cross-chain `BSC -> Base`, and **#1 `0.4171`** on zero-shot $12\times$ multi-horizon quiet-spell survival transfer $\Delta t=10 \to 120\,\mathrm{min}$ beating autoregressive age-chained `MLP-chain` `0.4378`, $p<10^{-3}$, and hazard-compounded `MLP-comp` `0.4323`, $p=0.008$, under 1,000-resample UTC-day block bootstrap), incoming-transition lifted two-scale Hyper-Erlang networks (`EdgeMemoryHyperDRN`), nonlinear McKean–Vlasov mean-field renewal networks (`MeanFieldDRN`) with verified Kurtz $O(M^{-1/2})$ convergence, and multi-start continuous-time baselines (`Mamba / Selective-SSM` + high-capacity 2-layer AdamW Mamba, `Latent ODE`, `LTC / CfC`, `S4D`, Time-Inhomogeneous `Neural-CME`, monotone cumulative-hazard `Neural Competing-Risks`, and continuous-time `Neural Hawkes / THP`).
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
│   ├── laplace_solver.py # Stiffness-free Laplace-Schur-Talbot, Frailty-DRN, & Coxian Talbot solvers
│   ├── mean_field.py     # Nonlinear McKean-Vlasov MeanFieldDRN ODE + interacting Gillespie
│   ├── reduction.py      # Symmetry closure, 2-sided interpolation, Pos-Sens 33 & universal 1D
│   ├── traits.py         # Persistent individual trait regressions & custom-JVP shape matching
│   ├── neural_clock.py   # Conditional neural waiting-time clocks & survivor phase conditioning
│   ├── semi_markov.py    # Incoming-edge lifted two-scale Hyper-Erlang semi-Markov DRN
│   ├── baselines.py      # Multi-start Mamba/Selective-SSM, Latent ODE, LTC, S4D, & Neural-CME
│   ├── memory.py         # Augmented exponential-kernel solver & static-preserving cutoffs
│   ├── capping.py        # Rate-preserving phase capping & path-TV / latent-Fisher bounds
│   ├── observations.py   # Count-weighted multinomial negative log likelihood
│   ├── design.py         # Convex scenario-averaged A-optimal Fisher allocation & rounding
│   └── diagnostics.py    # Coarsened/efficient Fisher matrices & adjoint sensitivity residuals
├── scripts/
│   ├── verify_theory.py                        # Symbolic (SymPy) & numerical proof audit (all 7 theorems)
│   ├── reproduce_all.py                        # SHA-256 audit of 43 archives + ODE replays
│   ├── build_paper_assets.py                   # Rebuilds all 6 vector figures & 15 LaTeX tables
│   ├── run_positive_sensitivity_reduction.py   # 400-dataset Pos-Sens 33 cyclic benchmark
│   ├── run_graph_scaling_benchmark.py          # V=4..48, d=2..4 scaling & 48-param Gamma recovery
│   ├── run_sparse_large_graph_scaling.py       # Matrix-free SparseDRN V=16..1,024 benchmark
│   ├── run_sparse_e2e_and_memory.py            # Peak autodiff RSS audit & V=64 (256p) E2E fit
│   ├── run_r5_sparse_converged.py              # Converged V=64 (256p) recovery & V=256 (1,024p) E2E
│   ├── run_modern_neural_baselines.py          # Multi-start Latent ODE, LTC, & S4D comparisons
│   ├── run_neural_cme_baseline.py              # 3-start Time-Inhomogeneous Neural-CME benchmark
│   ├── run_real_continuous_sleep_network.py    # Continuous-time E. coli & 5-node Sleep-EDF
│   ├── run_edge_memory_sleep_benchmark.py      # EdgeMemoryHyperDRN & Neural-CME on Sleep-EDF
│   ├── run_sleep_extended_baselines.py         # Real Sleep-EDF LTC, S4D, & Universal Pos-Sens
│   ├── run_hematopoiesis_branching_benchmark.py # 7-node branching human hematopoiesis FACS
│   ├── run_hematopoiesis_donor_extrap.py       # Strict 5-fold donor-held-out Day-7 extrap.
│   ├── run_helpdesk_systems_benchmark.py       # 4,576-ticket IT Helpdesk CV & Neural-CR sweep
│   ├── run_helpdesk_extended_baselines.py      # Helpdesk LTC, S4D, Univ. Pos-Sens & Chrono split
│   ├── run_mean_field_epidemic_benchmark.py    # 1978 Influenza MeanFieldDRN & Kurtz O(M^-1/2)
│   ├── run_r5_bio_and_influenza_complete.py    # Complete LTC/S4D Bio+Influenza & Hyper-Pos-Sens
│   ├── run_r5_helpdesk_complete.py             # Equalized 3-start clean age=0 Helpdesk & Coarsening
│   ├── run_r6_uniformity_closure.py            # Complete E. coli neural ODEs & Hyper-Pos coarsening
│   ├── run_r6_integrity_fixes.py               # Sleep fresh CTMC / 3-start age=0, Influenza CME, Plain-ME
│   ├── run_r7_theory_bt_separation.py          # Theorem 6 exact Balanced-Truncation & off-symmetry MLE
│   ├── run_r7_mamba_and_thp_baselines.py       # Mamba/Selective-SSM & Neural Hawkes (THP) benchmarks
│   ├── run_r7_defi_neural_clock.py             # 183,881-opp multi-chain DeFi Conditional Neural Clock
│   ├── run_r7_temporal_networks.py             # Real large-V MOOC (V=98, 411k) & StackOverflow (V=22)
│   ├── run_r8_real_bt_failure.py               # Real Hematopoiesis & Helpdesk Theorem 6 BT failure audit
│   ├── run_r8_temporal_networks_upgraded.py    # Talbot 143x MOOC, Frailty-DRN, Univ-H 0-shot, THP, 2L-Mamba
│   └── run_r8_defi_upgraded.py                 # M=4 DeFi clock, 12x multi-horizon transfer, UTC-day bootstrap
├── examples/
│   └── minimal_renewal_network.py  # Compact runnable marked-race & gradient example
├── results/archived/     # 43 SHA-256-fingerprinted benchmark archives across all paper studies
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

All real-world evaluations analyze publicly released, checksum-verified datasets spanning **temporal interaction networks, on-chain DeFi markets, operations/systems engineering, epidemiology, and biology**:

- **MOOC Student Interaction Temporal Network ($V = 98$: 97 course activities + absorbing Dropout)**: 7,047 students, 411,749 interactions, 1,355 sparse edges ($\mathrm{CV} = 5.54$) from Kumar et al. (*KDD*, 2019; JODIE benchmark).
- **StackOverflow Badge Award Temporal Network ($V = 22$, near-memoryless boundary case)**: 2,203 users, 142,777 badge events ($\mathrm{CV} = 1.22$, 31% self-loops; official EasyTPP splits) from Mei & Eisner (*NeurIPS*, 2017).
- **Multi-Chain DeFi Quiet-Pool Liquidity Panels (BSC & Base, Jan--Jul 2025)**: 16 on-chain panels across 35 pools and 5 token pairs (183,881 opportunities; 164,376 unseen Mar--Jul test opportunities), SHA-256 pinned in `results/archived/r7_defi_neural_clock.json`.
- **IT Helpdesk software-support queueing workflow (`Assign -> Active <-> Wait -> Resolved -> Closed`)**: 4,580 tickets (21,348 timestamped events, MD5 `a91d74377c55be5cbdfda05dae69db30`) from Polato (4TU.ResearchData, 2017; DOI `10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb`).
- **1978 English Boarding School H1N1 Influenza Outbreak (`S -> E -> In-Bed -> Convalescent -> R`)**: Daily multi-compartment prevalence snapshots ($N = 763$ boys, 14 days) from Anonymous (*BMJ*, 1978, 1(6112):587).
- **Primary human hematopoietic branching lineage hierarchy (`HSC -> MPP -> {MLP, CMP} -> {GMP, MEP} -> Mature`)**: 78 FACS snapshots across 10 healthy bone-marrow donor cultures over Days 1--7 from Bast et al. (*iScience*, 2021).
- **Mouse embryonic stem-cell differentiation (`ESC -> EPI -> NPC`)**: R1 cell line from Stumpf et al. (*Cell Systems*, 2017) as curated in Pélissier et al. (*Briefings in Bioinformatics*, 2026).
- **Single-cell *Escherichia coli* mother-machine lineages**: Si et al. (*Current Biology*, 2019; 9,449 cycles) and Tanouchi et al. (*Scientific Data*, 2017; 18,972 cycles).
- **Human multi-night sleep staging**: Sleep-EDF Expanded (Kemp et al., *IEEE TBME*, 2000; 75 paired-night adults), BIDSleep (Song et al., *IEEE TBME*, 2026; 253 home nights), and Ying et al. (*Scientific Data*, 2025; 58 triple-scorer recordings).

## License

Released under the [MIT License](LICENSE).
