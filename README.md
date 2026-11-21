# Differentiable Renewal Networks (`diff_renewal`)

Clean, standalone, and reproducible JAX/Diffrax implementation accompanying the manuscript **Differentiable Renewal Networks: When Predictive Minimality Fails Parameter Learning**.

## Overview

Many physical and biological processes remember elapsed residence time in a state, from multi-step molecular cascades in embryonic stem-cell differentiation and bacterial division lineages to stage transitions in human sleep. **Differentiable Renewal Networks** enable exact gradient-based parameter inference, sensitivity-preserving state compression, conditional neural waiting-time modeling, and Fisher experimental design from destructive population snapshots:

1. **Exact Marked-Race Compilation (`diff_renewal.race`)**: Compiles $d$ competing common-rate mixed-Erlang clocks at a branching node from $K^d$ Cartesian phase tensors into a minimal positive sequential Coxian chain of dimension $1 + \sum_e (K_e - 1)$ via log-space polynomial convolution, preserving joint exit-time and destination laws alongside exact autodiff parameter derivatives.
2. **Sensitivity-Aware Model Reduction (`diff_renewal.reduction`)**: Implements both stationary-profile symmetry continuations and two-sided Laplace resolvent sensitivity interpolation (Baur et al., 2011), demonstrating why trajectory-minimal realizations can corrupt node-specific parameter derivatives while sensitivity-aware projections preserve parameter recovery at reduced state count.
3. **Trainable Parametric, Trait-Conditioned, and Neural Clocks (`diff_renewal.phase_type`, `diff_renewal.traits`, `diff_renewal.neural_clock`)**: Smooth moment-matched mixed-Erlang clocks (with implicit custom-JVP shape-root differentiation), persistent individual trait regressions, conditional neural clocks with exact age-conditioned survival initialization, and multi-visit speed-carry extensions.
4. **Scenario-Robust Fisher Experimental Design (`diff_renewal.design`, `diff_renewal.diagnostics`)**: Two-stage pilot-and-follow-up A-optimal snapshot allocation over preparations, observation times, and observed traits with nuisance-parameter accounting and integer largest-remainder rounding.

## Repository Structure

```text
diff-renewal-network/
├── src/diff_renewal/
│   ├── phase_type.py     # Erlang, smooth mixed-Erlang, moment calculation, Coxian generators
│   ├── renewal.py        # Sparse shared-suffix sequential mixed-Erlang chain solver
│   ├── network.py        # Full K**d tensor renewal network solver in Diffrax
│   ├── race.py           # Exact marked-race log-space compiler & compact network solver
│   ├── reduction.py      # 33-state symmetry closure & two-sided sensitivity interpolation
│   ├── traits.py         # Persistent individual trait regressions & custom-JVP shape matching
│   ├── neural_clock.py   # Conditional neural waiting-time clocks & survivor phase conditioning
│   ├── memory.py         # Augmented exponential-kernel solver & static-preserving cutoffs
│   ├── capping.py        # Rate-preserving phase capping & path-TV / latent-Fisher bounds
│   ├── observations.py   # Count-weighted multinomial negative log likelihood
│   ├── design.py         # Convex scenario-averaged A-optimal Fisher allocation & rounding
│   └── diagnostics.py    # Coarsened/efficient Fisher matrices & adjoint sensitivity residuals
├── scripts/
│   ├── verify_theory.py       # Independent symbolic (SymPy) & numerical (JAX/SciPy) proof audit
│   ├── reproduce_all.py       # SHA-256 archive verification + live double-precision ODE replays
│   └── build_paper_assets.py  # Rebuilds all 5 vector figures (PDF/SVG/PNG) & 10 LaTeX tables
├── examples/
│   └── minimal_renewal_network.py  # Compact runnable marked-race & gradient example
├── results/archived/     # SHA-256-fingerprinted benchmark archives across all paper studies
├── tests/                # Pytest unit and end-to-end reproducibility suite
├── pyproject.toml
└── requirements.txt
```

## Quick Start & Reproducibility

### 1. Installation

```bash
pip install -e ".[dev]"
```

### 2. Run Unit & Reproducibility Tests

```bash
pytest -q
```

### 3. Verify All Mathematical Propositions and Theorems

Runs independent symbolic rational block-Hankel rank checks (Theorems 1--2), Laplace residue evaluations and Erlang-16 pole-restoration checks (Theorem 3), Duhamel vs. Fréchet derivative checks, two-sided resolvent sensitivity interpolation checks, two-preparation Laplace inversion checks (Theorem 4), lag-$\ell$ persistence covariance checks (Proposition 2), and Diffrax marked-race trajectory/gradient parity checks (Proposition 1):

```bash
python scripts/verify_theory.py
```

### 4. Audit SHA-256 Provenance & Replay Trajectories

Verifies cryptographic SHA-256 digests across all 17 archived benchmark files and replays the 48-vs-33 Erlang-16 trajectory/sensitivity separation and R1 mouse embryonic stem-cell differentiation curves (`ESC -> EPI -> NPC`, max drift $< 10^{-11}$):

```bash
python scripts/reproduce_all.py
```

### 5. Rebuild All Publication Figures and LaTeX Tables

```bash
python scripts/build_paper_assets.py
```

## Public Biological Datasets

All empirical biological evaluations analyze publicly available, de-identified datasets pinned by SHA-256 checksums:

- **Mouse embryonic stem-cell differentiation (`ESC -> EPI -> NPC`)**: R1 cell line from Stumpf et al. (*Cell Systems*, 2017) as curated in Pélissier et al. (*Briefings in Bioinformatics*, 2026).
- **Single-cell *Escherichia coli* mother-machine lineages**: Si et al. (*Current Biology*, 2019; 9,449 division cycles across 114 condition-lineage groups) and Tanouchi et al. (*Scientific Data*, 2017; 18,972 division cycles across 49 temperature-field groups).
- **Human multi-night sleep staging**: Sleep-EDF Expanded (Kemp et al., *IEEE TBME*, 2000; PhysioNet v1.0.0, 75 paired-night adults), BIDSleep (Song et al., *IEEE TBME*, 2026; PhysioNet v1.0.1, 253 home nights across 47 adults), and the triple-scorer multi-night dataset of Ying et al. (*Scientific Data*, 2025; 58 recordings across 20 adults).

## License

Released under the [MIT License](LICENSE).
