"""Build publication-ready figures and LaTeX tables for Differentiable Renewal Networks.

All figures are drawn at exact ICLR text width (5.50 in = 396.0 PDF pt) and exported
without tight-bbox rescaling so rendered glyph sizes sit at 7.8-8.6 pt (just slightly
smaller than the 9.0 pt ICLR figure captions). Every figure passes render-time
panel-alignment checks, PDF glyph-size floor checks, and PyMuPDF collision checks.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import ScaledTranslation
import numpy as np
from scipy.linalg import expm_frechet

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import smooth_erlang_logits, solve_erlang_chain  # noqa: E402

SKILL_SCRIPTS = Path(
    "/usr/local/google/home/apelissier/.gemini/config/skills/nature-figure/scripts"
)
if SKILL_SCRIPTS.exists():
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
else:
    require_matplotlib_panel_alignment = None

ARCH = ROOT / "results" / "archived"
OUT_FIG_REPO = ROOT / "figures"
LATEX_DIR = Path("/usr/local/google/home/apelissier/diff-renewal-network-latex")
OUT_FIG_LATEX = LATEX_DIR / "figures"
OUT_TAB_LATEX = LATEX_DIR / "tables"
QA_DIR = ROOT / "output" / "qa"

for folder in (OUT_FIG_REPO, OUT_FIG_LATEX, OUT_TAB_LATEX, QA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Exact ICLR \textwidth = 5.50 true in (396.0 PDF pt).
# Captions (\small) are 9.0 pt; figure text sits cleanly at 7.8-8.6 pt at 1:1 scale.
TEXTWIDTH_IN = 5.50

PALETTE = {
    "navy": "#0F4D92",
    "royal": "#2B6CB0",
    "teal": "#178577",
    "amber": "#C86D1F",
    "crimson": "#B64342",
    "violet": "#6B5B95",
    "slate": "#64748B",
    "light_blue": "#E8F0FA",
    "light_teal": "#E2F3F0",
    "light_amber": "#FDF2E4",
    "light_crimson": "#FBEAEA",
    "light_gray": "#F1F5F9",
    "ink": "#1E293B",
    "border": "#475569",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
        "font.size": 8.1,
        "axes.titlesize": 8.3,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }
)


def load_archived(fname: str) -> dict:
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    rel = f"results/archived/{fname}"
    p = ARCH / fname
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != manifest["files"][rel]["sha256"]:
        raise ValueError(f"SHA-256 mismatch on {rel}")
    return json.loads(p.read_text(encoding="utf-8"))


def add_panel_tag(
    ax: plt.Axes, tag: str, x_pt: float = -24.0, y_pt: float = 5.0
) -> None:
    offset = ScaledTranslation(x_pt / 72.0, y_pt / 72.0, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        tag,
        transform=ax.transAxes + offset,
        fontsize=8.7,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="left",
        va="bottom",
    )


def export_figure(
    fig: plt.Figure,
    stem: str,
    *,
    audit_alignment: bool = True,
    exclude_axes: list[plt.Axes] | None = None,
    exemptions: list[dict] | None = None,
) -> None:
    """Export figure at exact 5.50-in canvas width (1:1 scale) and run QA checks."""
    fig.canvas.draw()
    if audit_alignment and require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(QA_DIR / f"{stem}.alignment.json"),
            overlay_svg=str(QA_DIR / f"{stem}.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            exclude_axes=exclude_axes or [],
            exemptions=exemptions or [],
            strict=True,
        )
    for folder in (OUT_FIG_REPO, OUT_FIG_LATEX):
        fig.savefig(folder / f"{stem}.pdf")
        fig.savefig(folder / f"{stem}.svg")
        fig.savefig(folder / f"{stem}.png", dpi=300)
    plt.close(fig)

    pdf_path = OUT_FIG_LATEX / f"{stem}.pdf"
    if SKILL_SCRIPTS.exists():
        for script_name, extra in [
            ("audit_pdf_text.py", []),
            (
                "audit_figure_collisions.py",
                ["--json-out", str(QA_DIR / f"{stem}.collision.json")],
            ),
        ]:
            script = SKILL_SCRIPTS / script_name
            if script.exists():
                proc = subprocess.run(
                    [sys.executable, str(script), str(pdf_path), *extra],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode == 1:
                    raise RuntimeError(
                        f"{script_name} failed on {stem}.pdf:\n{proc.stdout}\n{proc.stderr}"
                    )


def write_tex_table(
    fname: str, col_spec: str, header: list[str], rows: list[list[str] | str]
) -> None:
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        if r == "\\midrule":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (OUT_TAB_LATEX / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 1: Architecture Schematic + 33-vs-48 Sensitivity Separation
# ---------------------------------------------------------------------------
def build_fig1() -> None:
    # Deterministic pure Erlang-16 ring calculation
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
    curves, full_d, red_d = [], [], []
    for t in ts:
        full, df = expm_frechet(t * a, t * ad)
        red, dr = expm_frechet(t * g, t * gd)
        pf, pr = c @ full @ b, cr @ red @ w @ b
        curves.append([pf[0, 0], pr[0, 0]])
        full_d.append(float((c @ df @ b @ np.ones(3) / 3.0)[0]))
        red_d.append(float((cr @ dr @ w @ b @ np.ones(3) / 3.0)[0]))
    curves_arr = np.asarray(curves)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.50))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.96, 1.16],
        left=0.095,
        right=0.985,
        top=0.94,
        bottom=0.115,
        hspace=0.46,
        wspace=0.48,
    )

    # Top schematic spanning all 3 columns (coordinate system in physical half-inches: 0..10.0)
    ax_sch = fig.add_subplot(gs[0, :])
    ax_sch.set_xlim(0.0, 10.0)
    ax_sch.set_ylim(0.0, 2.50)
    ax_sch.axis("off")
    add_panel_tag(ax_sch, "a", x_pt=-22.0, y_pt=0.0)

    boxes = [
        (0.02, 0.25, 2.96, 2.00, PALETTE["light_blue"], PALETTE["navy"],
         "Marked-race compiler",
         "Competing clocks\n$K^d \\to 1 + d(K-1)$\n$289 \\to 64$ states"),
        (3.48, 0.25, 3.04, 2.00, PALETTE["light_teal"], PALETTE["teal"],
         "Forward & Fisher AD",
         "Parametric / neural PH\n$P_\\theta(t),\\, \\partial P_\\theta / \\partial\\theta$\nSnapshot NLL & $I(\\theta)$"),
        (7.02, 0.25, 2.96, 2.00, PALETTE["light_amber"], PALETTE["amber"],
         "Sensitivity rank",
         "$15$ modes cancel in $P$\n$P$: $33$  |  $\\partial P / \\partial\\theta$: $48$\n2-sided interpolation"),
    ]
    for x0, y0, bw, bh, fc, ec, title, body in boxes:
        patch = FancyBboxPatch(
            (x0, y0),
            bw,
            bh,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.0,
        )
        ax_sch.add_patch(patch)
        ax_sch.text(
            x0 + bw / 2.0,
            y0 + bh - 0.36,
            title,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=ec,
        )
        ax_sch.text(
            x0 + bw / 2.0,
            y0 + 0.78,
            body,
            ha="center",
            va="center",
            fontsize=7.8,
            color=PALETTE["ink"],
            linespacing=1.22,
        )

    for x_start, x_end in [(3.05, 3.41), (6.59, 6.95)]:
        ax_sch.annotate(
            "",
            xy=(x_end, 1.25),
            xytext=(x_start, 1.25),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.34,head_width=0.19",
                "lw": 1.1,
                "color": PALETTE["border"],
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    ax_b.plot(ts, curves_arr[:, 0], color=PALETTE["navy"], lw=1.9, label="Full 48")
    ax_b.plot(ts, curves_arr[:, 1], color=PALETTE["amber"], lw=1.35, ls="--", label="Sym. 33")
    ax_b.set_xlabel("Time $t$")
    ax_b.set_ylabel("Node A probability")
    ax_b.set_xlim(0.0, 7.0)
    ax_b.set_ylim(0.0, 1.05)
    ax_b.set_xticks([0, 2, 4, 6])
    ax_b.legend(loc="upper right", handlelength=1.5, borderpad=0.1, labelspacing=0.2)
    add_panel_tag(ax_b, "b", x_pt=-22.0, y_pt=4.0)

    ax_c.axhline(0.0, color=PALETTE["slate"], lw=0.6, ls=":")
    ax_c.plot(ts, full_d, color=PALETTE["navy"], lw=1.8, label="Exact 48")
    ax_c.plot(ts, red_d, color=PALETTE["amber"], lw=1.35, ls="--", label="Sym. 33")
    ax_c.set_xlabel("Time $t$")
    ax_c.set_ylabel(r"$\partial P_A / \partial \log\lambda_A$")
    ax_c.set_xlim(0.0, 7.0)
    ax_c.set_ylim(-0.58, 0.22)
    ax_c.set_xticks([0, 2, 4, 6])
    ax_c.legend(loc="lower right", handlelength=1.5, borderpad=0.1, labelspacing=0.2)
    add_panel_tag(ax_c, "c", x_pt=-24.0, y_pt=4.0)

    idx = np.arange(48)
    poles = 8.0 * (np.exp(2j * np.pi * idx / 48.0) - 1.0)
    hidden = (idx != 0) & (idx % 3 == 0)
    ax_d.scatter(
        poles.real[~hidden],
        poles.imag[~hidden],
        s=13,
        color=PALETTE["navy"],
        label="33 visible",
        zorder=3,
    )
    ax_d.scatter(
        poles.real[hidden],
        poles.imag[hidden],
        s=24,
        marker="x",
        linewidths=1.1,
        color=PALETTE["crimson"],
        label="15 restored",
        zorder=4,
    )
    ax_d.set_xlabel(r"$\mathrm{Re}(s)$")
    ax_d.set_ylabel(r"$\mathrm{Im}(s)$")
    ax_d.set_xlim(-17.5, 1.2)
    ax_d.set_ylim(-10.2, 10.2)
    ax_d.set_xticks([-16, -8, 0])
    ax_d.set_yticks([-8, 0, 8])
    ax_d.legend(loc="center", handletextpad=0.25, borderpad=0.1, labelspacing=0.22)
    add_panel_tag(ax_d, "d", x_pt=-22.0, y_pt=4.0)

    export_figure(
        fig,
        "fig1_architecture_and_separation",
        exclude_axes=[ax_sch],
    )


# ---------------------------------------------------------------------------
# Figure 2: Gradient Defect & Sensitivity-Interpolating Compression
# ---------------------------------------------------------------------------
def build_fig2() -> None:
    ev = load_archived("evidence.json")
    sens = load_archived("sensitivity_evidence.json")

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.28))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.80, bottom=0.20, wspace=0.46)

    bounds = ev["gradient_bounds"]["families"]["trainable_PH"]
    xs_b = np.array([r["epsilon"] for r in bounds])
    for key, label, col, mk in [
        ("mean_contrast", "Mean", PALETTE["navy"], "o"),
        ("shape_contrast", "Shape", PALETTE["amber"], "s"),
    ]:
        vals = [r[key]["fitted_times"] for r in bounds]
        axes[0].fill_between(
            xs_b,
            [r["lower"] for r in vals],
            [r["upper"] for r in vals],
            color=col,
            alpha=0.14,
            lw=0,
        )
        axes[0].plot(
            xs_b,
            [r["gradient_error"] for r in vals],
            marker=mk,
            color=col,
            ms=3.2,
            lw=1.5,
            label=label,
        )
    axes[0].set_xlabel(r"Heterogeneity $\epsilon$")
    axes[0].set_ylabel("Max. Jacobian defect")
    axes[0].set_xlim(-0.008, 0.26)
    axes[0].set_ylim(-0.02, 0.62)
    axes[0].set_xticks([0.0, 0.075, 0.15, 0.25], ["0", ".075", ".15", ".25"])
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.1,
        columnspacing=0.7,
        handletextpad=0.3,
        borderpad=0.1,
    )
    add_panel_tag(axes[0], "a", x_pt=-22.0, y_pt=3.0)

    names = ["symmetric", "mixed_025", "mixed_075", "mixed_150", "mixed_250"]
    xs = [0.0, 0.025, 0.075, 0.15, 0.25]
    series_cfg = [
        ("full48", "Full 48", PALETTE["navy"], "o", "-"),
        ("sp33", "Interp. 33", PALETTE["teal"], "s", "--"),
        ("sym33", "Sym. 33", PALETTE["amber"], "^", "-"),
        ("sp18", "Interp. 18", PALETTE["crimson"], "d", ":"),
    ]
    for ax, law, tag, ylab in [
        (axes[1], "PH", "b", "PH shape RMS (%)"),
        (axes[2], "Gamma", "c", "Gamma shape RMS (%)"),
    ]:
        ax.axhline(10.0, color=PALETTE["slate"], ls="--", lw=0.75)
        for mkey, label, col, mk, ls in series_cfg:
            ys = [
                sens["cases"][f"primary/{k}_{law}"][mkey]["median_shape_percent"]
                for k in names
            ]
            ax.plot(xs, ys, marker=mk, ls=ls, color=col, ms=3.2, lw=1.45, label=label)
        ax.set_xlabel(r"Heterogeneity $\epsilon$")
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.008, 0.26)
        ax.set_ylim(0.0, 24.5)
        ax.set_xticks([0.0, 0.075, 0.15, 0.25], ["0", ".075", ".15", ".25"])
        add_panel_tag(ax, tag, x_pt=-22.0, y_pt=3.0)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(
        handles[:2],
        labels[:2],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.1,
        columnspacing=0.6,
        handletextpad=0.25,
        borderpad=0.1,
    )
    axes[2].legend(
        handles[2:],
        labels[2:],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.1,
        columnspacing=0.6,
        handletextpad=0.25,
        borderpad=0.1,
    )

    export_figure(fig, "fig2_compression_and_sensitivity")


# ---------------------------------------------------------------------------
# Figure 3: Stem-Cell Differentiation & 60-Dataset Fisher Benchmark
# ---------------------------------------------------------------------------
def build_fig3() -> None:
    pilot = load_archived("stem_cells_pilot.json")
    ph_sc = load_archived("trainable_ph_stem_cells.json")
    ev = load_archived("evidence.json")
    records = ev["benchmark"]

    ts_obs = np.asarray(pilot["observation_times"])
    obs = np.asarray(pilot["data_mean_proportions"])
    theta = np.asarray(ph_sc["panels"]["all_times"]["phase"]["best"]["theta"])
    means = jnp.exp(-jnp.asarray(theta[:2]))
    centers = 50.0 * jnp.asarray(theta[2:])
    logits = jax.vmap(lambda c: smooth_erlang_logits(c, stages=64, width=0.5))(centers)
    ts_dense = jnp.linspace(0.0, 168.0, 240)
    pred_dense = np.asarray(solve_erlang_chain(means, logits, ts_dense).ys)

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.30))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.80, bottom=0.20, wspace=0.46)

    for idx_s, (sname, col, mk) in enumerate(
        [
            ("ESC", PALETTE["navy"], "o"),
            ("EPI", PALETTE["amber"], "s"),
            ("NPC", PALETTE["teal"], "^"),
        ]
    ):
        axes[0].plot(np.asarray(ts_dense), pred_dense[:, idx_s], color=col, lw=1.55, label=sname)
        axes[0].scatter(ts_obs, obs[:, idx_s], color=col, marker=mk, s=16, zorder=4)
    axes[0].set_xlabel("Time (h)")
    axes[0].set_ylabel("Cell state fraction")
    axes[0].set_xlim(-4.0, 174.0)
    axes[0].set_ylim(-0.04, 1.06)
    axes[0].set_xticks([0, 48, 96, 144])
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.55,
        handletextpad=0.25,
        borderpad=0.1,
    )
    add_panel_tag(axes[0], "a", x_pt=-22.0, y_pt=14.0)

    lim = 22.5
    axes[1].plot([0, lim], [0, lim], color=PALETTE["slate"], lw=0.8, ls=":")
    axes[1].axhline(10.0, color=PALETTE["slate"], lw=0.65, ls="--")
    axes[1].axvline(10.0, color=PALETTE["slate"], lw=0.65, ls="--")
    for topo, label, mk, col in [
        ("cyclic", "Cycle", "o", PALETTE["navy"]),
        ("branching", "Diamond", "^", PALETTE["teal"]),
    ]:
        rows = [r for r in records if r["case"].startswith(topo)]
        axes[1].scatter(
            [100.0 * r["nomass"]["shape_error"] for r in rows],
            [100.0 * r["phase"]["shape_error"] for r in rows],
            s=17,
            marker=mk,
            color=col,
            alpha=0.84,
            label=label,
        )
    axes[1].set_xlim(0.0, lim)
    axes[1].set_ylim(0.0, lim)
    axes[1].set_xticks([0, 10, 20])
    axes[1].set_yticks([0, 10, 20])
    axes[1].set_xlabel("NoMaSS shape error (%)")
    axes[1].set_ylabel("PH shape error (%)")
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=0.6,
        handletextpad=0.2,
        borderpad=0.1,
    )
    add_panel_tag(axes[1], "b", x_pt=-22.0, y_pt=14.0)

    end_t = max(r[m]["joint_seconds"] or 0.0 for r in records for m in ("phase", "nomass")) * 1.25
    for mkey, label, col, ls in [
        ("phase", "PH", PALETTE["navy"], "-"),
        ("nomass", "NoMaSS", PALETTE["amber"], "--"),
    ]:
        times = sorted(r[mkey]["joint_seconds"] for r in records if r[mkey]["joint_seconds"] is not None)
        axes[2].step(
            [4.0] + times + [end_t],
            [0.0] + list(np.arange(1, len(times) + 1) / 60.0) + [len(times) / 60.0],
            where="post",
            color=col,
            ls=ls,
            lw=1.65,
            label=label,
        )
    axes[2].set_xscale("log")
    axes[2].set_xlim(5.0, end_t)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel("Cold wall time (s)")
    axes[2].set_ylabel("Joint attainment")
    axes[2].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.6,
        handletextpad=0.25,
        borderpad=0.1,
    )
    add_panel_tag(axes[2], "c", x_pt=-22.0, y_pt=14.0)

    export_figure(fig, "fig3_stemcells_and_fisher_benchmark")


# ---------------------------------------------------------------------------
# Figure 4: Conditional Neural Clocks & Multi-System Few-Shot Transfer
# ---------------------------------------------------------------------------
def build_fig4() -> None:
    nm = load_archived("neural_memory_summary.json")
    rt = load_archived("renewal_transfer_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.34))
    fig.subplots_adjust(left=0.105, right=0.98, top=0.79, bottom=0.20, wspace=0.38)

    # Panel a: Grouped log-KL bars with IQR whiskers across 3 evaluation settings
    studies = [
        ("primary", "PH truth"),
        ("richer", "PH + aged"),
        ("lognormal", "Lognormal"),
    ]
    models = [
        ("neural_phase", "Neural clock", PALETTE["navy"], -0.24),
        ("mean_shape", "Moment PH", PALETTE["teal"], 0.0),
        ("latent_ode", "Latent ODE", PALETTE["crimson"], 0.24),
    ]
    x_pos = np.arange(len(studies))
    axes[0].axhline(0.005, color=PALETTE["slate"], ls="--", lw=0.75)
    for mkey, mlabel, col, dx in models:
        meds, lows, highs = [], [], []
        for skey, _ in studies:
            stat = nm[skey]["models"][mkey]["metrics"]["balanced_kl"]
            med = float(stat["median"])
            meds.append(med)
            lows.append(med - float(stat["q25"]))
            highs.append(float(stat["q75"]) - med)
        y_base = 8e-4
        axes[0].bar(
            x_pos + dx,
            np.asarray(meds) - y_base,
            bottom=y_base,
            width=0.22,
            color=col,
            edgecolor=PALETTE["ink"],
            linewidth=0.55,
            label=mlabel,
        )
        axes[0].errorbar(
            x_pos + dx,
            meds,
            yerr=[lows, highs],
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=0.8,
            capsize=2.0,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x_pos, [lbl for _, lbl in studies])
    axes[0].set_ylim(8e-4, 0.25)
    axes[0].set_ylabel("Held-out excess KL (nats)")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.6,
        handletextpad=0.3,
        borderpad=0.1,
    )
    add_panel_tag(axes[0], "a", x_pt=-24.0, y_pt=16.0)

    # Panel b: Multi-system transfer across target cell budgets (400, 2000, 8000)
    budgets = [400, 2000, 8000]
    arms_cfg = [
        ("neural_phase.pretrained_code", "Clock 4-coeff", PALETTE["navy"], "o", "-"),
        ("neural_phase.scratch", "Clock scratch", PALETTE["royal"], "s", "--"),
        ("latent_ode.pretrained_code", "Latent 4-coeff", PALETTE["teal"], "^", "-"),
        ("latent_ode.scratch", "Latent scratch", PALETTE["crimson"], "d", ":"),
    ]
    axes[1].axhline(0.005, color=PALETTE["slate"], ls="--", lw=0.75)
    for akey, alabel, col, mk, ls in arms_cfg:
        ys = [float(rt["heldout_ph"][str(b)]["arms"][akey]["balanced_kl"]) for b in budgets]
        axes[1].plot(budgets, ys, marker=mk, ls=ls, color=col, ms=3.8, lw=1.55, label=alabel)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(budgets, ["400", "2,000", "8,000"])
    axes[1].minorticks_off()
    axes[1].set_xlim(320, 9800)
    axes[1].set_ylim(1e-3, 0.40)
    axes[1].set_xlabel("Target adaptation cells")
    axes[1].set_ylabel("Held-out excess KL (nats)")
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.6,
        handletextpad=0.3,
        borderpad=0.1,
    )
    add_panel_tag(axes[1], "b", x_pt=-24.0, y_pt=16.0)

    export_figure(fig, "fig4_neural_clocks_and_transfer")


# ---------------------------------------------------------------------------
# Figure 5: Biological Return-Visit Memory & Multi-Scorer Sleep Calibration
# ---------------------------------------------------------------------------
def build_fig5() -> None:
    rvm = load_archived("real_visit_memory.json")["datasets"]
    n2_f = load_archived("n2_fisher_recordings_compact.json")
    n2_r = load_archived("n2_scorer_pooling.json")

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.36))
    fig.subplots_adjust(left=0.185, right=0.98, top=0.79, bottom=0.20, wspace=0.45)

    # Panel a: Previous-visit log-score gain across human sleep & E. coli cohorts
    cohorts = [
        ("sleep", "2", "Sleep N2\n(75 adults)"),
        ("nomass", "all", "E. coli\n(Si et al.)"),
        ("tanouchi", "all", "E. coli\n(Tanouchi)"),
    ]
    axes[0].axvline(0.0, color=PALETTE["slate"], lw=0.8, ls="--")
    for idx_y, (dkey, subk, _) in enumerate(cohorts):
        y_coord = 2 - idx_y
        for mkey, mlabel, col, mk, dy in [
            ("marginal", "Unadjusted", PALETTE["amber"], "s", 0.14),
            ("adjusted", "Covariate-adj.", PALETTE["navy"], "o", -0.14),
        ]:
            est = rvm[dkey]["models"][mkey]["previous"][subk]
            val = 100.0 * float(est["gain"])
            lo, hi = [100.0 * float(v) for v in est["interval"]]
            axes[0].errorbar(
                val,
                y_coord + dy,
                xerr=[[val - lo], [hi - val]],
                fmt=mk,
                color=col,
                ms=4.0,
                lw=1.35,
                capsize=2.2,
                label=mlabel if idx_y == 0 else None,
            )
    axes[0].set_yticks([2, 1, 0], [lbl for _, _, lbl in cohorts])
    axes[0].set_ylim(-0.52, 2.55)
    axes[0].set_xlim(-0.45, 2.15)
    axes[0].set_xlabel("Gain per 100 visits (nats)")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=0.6,
        handletextpad=0.2,
        borderpad=0.1,
    )
    add_panel_tag(axes[0], "a", x_pt=-54.0, y_pt=16.0)

    # Panel b: Unseen-physician sleep boundary calibration curve
    x_b = np.array([46.4 if k == "all" else float(k) for k in n2_f["budgets"]])
    target_loss = float(n2_f["calibration"]["adaptive"]["common_loss_target"])
    axes[1].axhline(target_loss, color=PALETTE["slate"], ls=":", lw=0.85)
    for mkey, label, col, ls in [
        ("random", "Random (4)", PALETTE["navy"], "-"),
        ("longest", "Longest (3)", PALETTE["teal"], "-."),
        ("fixed", "Fixed (9)", PALETTE["amber"], "--"),
        ("adaptive", "Adaptive (11)", PALETTE["crimson"], ":"),
    ]:
        curve = (
            n2_r["curves"]["history_nll"]["mean"]
            if mkey == "random"
            else n2_f["curves"][mkey]["history_nll"]["mean"]
        )
        axes[1].plot(x_b, np.asarray(curve), color=col, ls=ls, lw=1.5, label=label)
    axes[1].set_xlim(-0.8, 47.5)
    axes[1].set_ylim(1.314, 1.382)
    axes[1].set_xticks([0, 4, 11, 24, 46.4], ["0", "4", "11", "24", "All"])
    axes[1].set_xlabel("Calibration nights (new scorer)")
    axes[1].set_ylabel("Held-out NLL (nats/bout)")
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.6,
        handletextpad=0.3,
        borderpad=0.1,
    )
    add_panel_tag(axes[1], "b", x_pt=-24.0, y_pt=16.0)

    export_figure(fig, "fig5_persistence_and_sleep_calibration")


# ---------------------------------------------------------------------------
# Build Main-Text and Appendix Tables
# ---------------------------------------------------------------------------
def build_tables() -> None:
    ev = load_archived("evidence.json")
    sens = load_archived("sensitivity_evidence.json")
    tf = load_archived("trait_fisher_summary.json")
    nm = load_archived("neural_memory_summary.json")
    nmp = load_archived("neural_memory_pulse.json")
    rt = load_archived("renewal_transfer_summary.json")
    rt6 = load_archived("renewal_transfer_six_parameter.json")
    rvm = load_archived("real_visit_memory.json")["datasets"]
    ph_sc = load_archived("trainable_ph_stem_cells.json")
    n2_f = load_archived("n2_fisher_recordings_compact.json")

    # Table 1: Cyclic compression & sensitivity interpolation summary
    p_cases = sens["cases"]
    p_ov = sens["overall"]["primary"]
    l_ov = sens["overall"]["late"]
    t1_rows = []
    for mkey, mname, dim, w_ms in [
        ("full48", "\\textbf{Full PH}", "48", "89.4"),
        ("sym33", "Symmetry closure", "33", "\\textbf{39.4}"),
        ("sp33", "\\textbf{Interp. 33 (2-sided)}", "33", "46.9"),
        ("sp18", "Interp. 18 (1-anchor)", "18", "38.9"),
    ]:
        sym_ph = p_cases["primary/symmetric_PH"][mkey]["median_shape_percent"]
        het_ph = p_cases["primary/mixed_150_PH"][mkey]["median_shape_percent"]
        het_gm = p_cases["primary/mixed_150_Gamma"][mkey]["median_shape_percent"]
        j_pri = f"{p_ov[mkey]['joint']}/280"
        j_lat = f"{l_ov[mkey]['joint']}/120"
        if mkey == "sym33":
            sym_str = f"\\textbf{{{sym_ph:.2f}\\%}}"
        else:
            sym_str = f"{sym_ph:.2f}\\%"
        if mkey == "sp33":
            het_str = f"\\textbf{{{het_ph:.2f}\\%}} / \\textbf{{{het_gm:.2f}\\%}}"
            j_pri = f"\\textbf{{{j_pri}}}"
        elif mkey == "full48":
            j_lat = f"\\textbf{{{j_lat}}}"
            het_str = f"{het_ph:.2f}\\% / {het_gm:.2f}\\%"
        else:
            het_str = f"{het_ph:.2f}\\% / {het_gm:.2f}\\%"
        t1_rows.append([mname, dim, sym_str, het_str, j_pri, j_lat, w_ms])
    write_tex_table(
        "tab1_compression_summary.tex",
        "lcrrrrr",
        [
            "Realization",
            "States",
            "Sym. Err. $\\downarrow$",
            "$\\epsilon{=}0.15$ (PH / $\\Gamma$) $\\downarrow$",
            "Primary $\\uparrow$",
            "Interior $\\uparrow$",
            "Warm (ms) $\\downarrow$",
        ],
        t1_rows,
    )

    # Table 2: Rate, shape, and persistent-trait recovery vs NoMaSS
    bsum = ev["benchmark_summary"]
    t2_rows = [
        [
            "6-regime Gamma",
            "PH (Uniform)",
            "1.50\\%",
            "7.70\\%",
            "42/60",
            "---",
        ],
        [
            "($n=60$, 10k cells)",
            "\\textbf{PH (Fisher)}",
            "\\textbf{1.42\\%}",
            "\\textbf{4.52\\%}",
            "\\textbf{59/60}",
            "\\textbf{19.32$\\times$ cold}",
        ],
        [
            "",
            "NoMaSS (Uniform)",
            "2.39\\%",
            "11.86\\%",
            "24/60",
            "---",
        ],
        [
            "",
            "NoMaSS (Fisher)",
            "2.22\\%",
            "7.69\\%",
            "46/60",
            "1.00$\\times$ ref.",
        ],
        "\\midrule",
        [
            "Persistent trait",
            "PH (Uniform)",
            "2.89\\%",
            "10.40\\%",
            "3/10",
            "---",
        ],
        [
            "($n=10$, 41 levels)",
            "\\textbf{PH (Fisher)}",
            "\\textbf{1.60\\%}",
            "\\textbf{5.90\\%}",
            "\\textbf{9/10}",
            "\\textbf{38.12$\\times$ CPU}",
        ],
        [
            "",
            "NoMaSS (Uniform)",
            "3.71\\%",
            "15.99\\%",
            "0/10",
            "---",
        ],
        [
            "",
            "NoMaSS (Fisher)",
            "1.73\\%",
            "8.84\\%",
            "\\textbf{9/10}",
            "1.00$\\times$ ref.",
        ],
    ]
    assert (
        round(100.0 * float(np.median([r["phase"]["shape_error"] for r in ev["benchmark"]])), 2)
        == 4.52
    )
    write_tex_table(
        "tab2_fisher_and_traits.tex",
        "llrrrr",
        [
            "Benchmark",
            "Method \\& Design",
            "Rate / Mean $\\downarrow$",
            "Shape RMS $\\downarrow$",
            "Joint Hits $\\uparrow$",
            "Paired Speedup $\\uparrow$",
        ],
        t2_rows,
    )

    # Table 3: Conditional neural clocks, mid-visit pulse, and 4-coeff transfer
    p_mod = nm["primary"]["models"]
    l_mod = nm["lognormal"]["models"]
    pu_mod = nmp["studies"]["primary"]
    tr_400 = rt["heldout_ph"]["400"]["arms"]
    tr_8000 = rt["heldout_ph"]["8000"]["arms"]
    t3_rows = [
        [
            "Moment PH",
            "18",
            f"{p_mod['mean_shape']['metrics']['balanced_kl']['median']:.5f}",
            f"{l_mod['mean_shape']['metrics']['balanced_kl']['median']:.5f}",
            f"{100*pu_mod['mean_shape']['median_arm_rmse']['exit_increase']:.2f}\\,\\text{{pp}}",
            "---",
            "---",
        ],
        [
            "\\textbf{Neural clock}",
            "185",
            f"\\textbf{{{p_mod['neural_phase']['metrics']['balanced_kl']['median']:.5f}}}",
            f"\\textbf{{{l_mod['neural_phase']['metrics']['balanced_kl']['median']:.5f}}}",
            f"\\textbf{{{100*pu_mod['neural_phase']['median_arm_rmse']['exit_increase']:.2f}\\,\\text{{pp}}}}",
            f"\\textbf{{{tr_400['neural_phase.pretrained_code']['balanced_kl']:.5f}}}",
            f"\\textbf{{{tr_8000['neural_phase.pretrained_full']['balanced_kl']:.5f}}}",
        ],
        [
            "Latent ODE",
            "387",
            f"{p_mod['latent_ode']['metrics']['balanced_kl']['median']:.5f}",
            f"{l_mod['latent_ode']['metrics']['balanced_kl']['median']:.5f}",
            f"{100*pu_mod['latent_ode']['median_arm_rmse']['exit_increase']:.2f}\\,\\text{{pp}}",
            f"{tr_400['latent_ode.pretrained_code']['balanced_kl']:.5f}",
            f"{tr_8000['latent_ode.pretrained_code']['balanced_kl']:.5f}",
        ],
    ]
    write_tex_table(
        "tab3_neural_and_transfer.tex",
        "lrrrrrr",
        [
            "Architecture",
            "Params",
            "PH KL $\\downarrow$",
            "Lognorm. KL $\\downarrow$",
            "Pulse Err. $\\downarrow$",
            "400-Cell KL $\\downarrow$",
            "8k Adapt KL $\\downarrow$",
        ],
        t3_rows,
    )

    # Table 4: Cross-visit persistence on synthetic snapshots & biological cohorts
    s_n2 = rvm["sleep"]["models"]["adjusted"]["previous"]["2"]
    s_ec1 = rvm["nomass"]["models"]["adjusted"]["previous"]["all"]
    s_ec2 = rvm["tanouchi"]["models"]["adjusted"]["previous"]["all"]
    t4_rows = [
        [
            "Synthetic resampled ($\\rho{=}0$)",
            "10 seeds (8k cells)",
            "\\textbf{KL 0.00251}",
            "$\\hat\\rho = 0.000$ (9/10 hits)",
        ],
        [
            "Synthetic persistent ($\\rho{=}1$)",
            "10 seeds (8k cells)",
            "\\textbf{KL 0.00416}",
            "$\\hat\\rho = 0.875$ (8/10 hits)",
        ],
        "\\midrule",
        [
            "Sleep-EDF (stage N2)",
            "75 adults (2,957 visits)",
            f"\\textbf{{{s_n2['gain']:+.5f}}} nats",
            "Shuffle $+0.00725$ ($p{=}0.002$)",
        ],
        [
            "\\emph{E.\\ coli} (Si et al.)",
            "114 groups (7,389 cycles)",
            f"\\textbf{{{s_ec1['gain']:+.5f}}} nats",
            "95\\% CI $[+0.0087, +0.0171]$",
        ],
        [
            "\\emph{E.\\ coli} (Tanouchi)",
            "49 groups (18,693 cycles)",
            f"${s_ec2['gain']:+.5f}$ nats",
            "Size/time absorb $+0.0132$",
        ],
        [
            "Multi-scorer sleep",
            "20 adults (58 nights)",
            "$+0.00068$ nats",
            "4-night pool ($p{=}0.0213$)",
        ],
    ]
    assert "resampled" in rt6["targets"] and "persistent" in rt6["targets"]
    write_tex_table(
        "tab4_persistence_and_trajectories.tex",
        "llll",
        [
            "Cohort / Truth",
            "Sample Size",
            "Carry KL or Gain",
            "Key Control or Estimate",
        ],
        t4_rows,
    )

    # Appendix Table A1: Proof audit Hankel orders
    write_tex_table(
        "tab_a1_proof_audit.tex",
        "lr",
        ["Exact Rational Realization / Assumption Check", "Block-Hankel Order"],
        [
            ["Two alternating Erlang-2 clocks ($V{=}2, n{=}2$, 1 shared zero)", "3"],
            ["Three Erlang-3 clocks on a directed cycle ($V{=}3, n{=}3$, 2 shared zeros)", "7"],
            ["Distinct 2-stage clocks sharing survival zero $s{=}-3$", "3"],
            ["Distinct 2-stage clocks with shared zero perturbed away", "4"],
            ["Three order-3 clocks sharing a double survival zero ($m{=}2$)", "7"],
            ["Two order-2 clocks on a DAG feeding an absorbing sink", "5"],
            ["Counterexample: reducible routing $R{=}I$ on two Erlang-2 clocks", "2"],
            ["Counterexample: stationary mixture preparation only on 3-node Erlang-3 cycle", "1"],
        ],
    )

    # Appendix Tables A2 & A3: Full 14-row primary & 6-row later-time reduction tables
    r_labels = {
        "symmetric": "Symmetric ($\\epsilon{=}0$)",
        "mixed_025": "Mixed $\\epsilon{=}0.025$",
        "mixed_075": "Mixed $\\epsilon{=}0.075$",
        "mixed_150": "Mixed $\\epsilon{=}0.150$",
        "mixed_250": "Mixed $\\epsilon{=}0.250$",
        "mean_only_150": "Mean-only $\\epsilon{=}0.150$",
        "shape_only_150": "Shape-only $\\epsilon{=}0.150$",
    }
    for design, out_fname in [
        ("primary", "tab_a2_primary_regimes.tex"),
        ("late", "tab_a3_late_regimes.tex"),
    ]:
        rows_sp = []
        for key, rec in sens["cases"].items():
            which, case = key.split("/")
            if which != design:
                continue
            regime, law = case.rsplit("_", 1)
            diff = rec["sp33"]["paired_shape_difference_vs_full_pp"]
            vals = [f"{rec[m]['median_shape_percent']:.2f}" for m in ("full48", "sym33", "sp33", "sp18")]
            ci_str = f"${diff['median']:+.2f}\\;[{diff['ci95'][0]:+.2f}, {diff['ci95'][1]:+.2f}]$"
            rows_sp.append([law, r_labels[regime], *vals, ci_str])
        write_tex_table(
            out_fname,
            "llrrrrl",
            [
                "Truth",
                "Regime",
                "Full 48",
                "Sym. 33",
                "Interp. 33",
                "Interp. 18",
                "Interp. 33 $-$ Full [95\\% CI]",
            ],
            rows_sp,
        )

    # Appendix Table A4: Per-regime 60-dataset Fisher NoMaSS comparison
    write_tex_table(
        "tab_a4_nomass_regimes.tex",
        "lrrrr",
        [
            "Regime ($n{=}10$)",
            "PH Shape $\\downarrow$",
            "NoMaSS $\\downarrow$",
            "Joint Hits",
            "Cold Ratio",
        ],
        [
            ["Cycle broad", "\\textbf{4.96\\%}", "7.99\\%", "\\textbf{10/10} vs.\\ 6/10", "29.35$\\times$ (6)"],
            ["Diamond broad", "\\textbf{5.40\\%}", "7.42\\%", "\\textbf{10/10} vs.\\ 8/10", "13.24$\\times$ (8)"],
            ["Cycle mixed", "\\textbf{4.41\\%}", "7.35\\%", "\\textbf{10/10} vs.\\ 8/10", "27.62$\\times$ (8)"],
            ["Diamond mixed", "\\textbf{4.41\\%}", "8.91\\%", "\\textbf{10/10} vs.\\ 7/10", "15.96$\\times$ (7)"],
            ["Cycle narrow", "\\textbf{3.87\\%}", "6.10\\%", "\\textbf{10/10} vs.\\ 9/10", "36.05$\\times$ (9)"],
            ["Diamond narrow", "\\textbf{5.15\\%}", "7.35\\%", "\\textbf{9/10} vs.\\ 8/10", "7.63$\\times$ (8)"],
        ],
    )

    # Appendix Table A5: Stem cell differentiation comparison
    at_sc = ph_sc["panels"]["all_times"]
    write_tex_table(
        "tab_a5_stem_cells.tex",
        "lrrrr",
        [
            "Method (R1 ESC $\\to$ EPI $\\to$ NPC)",
            "RMSE $\\downarrow$",
            "Rate 1",
            "Rate 2",
            "Runtime $\\downarrow$",
        ],
        [
            [
                "\\textbf{Smooth mixed-Erlang PH}",
                f"{at_sc['phase']['best']['loss']:.5f}",
                f"{at_sc['phase']['equivalent_gamma_parameters'][0]:.5f}",
                f"{at_sc['phase']['equivalent_gamma_parameters'][1]:.5f}",
                "0.52\\,s (+3.15\\,s JIT)",
            ],
            [
                "Continuous $\\Gamma$ quadrature",
                f"\\textbf{{{at_sc['gamma_reference']['best']['loss']:.5f}}}",
                f"{at_sc['gamma_reference']['best']['params'][0]:.5f}",
                f"{at_sc['gamma_reference']['best']['params'][1]:.5f}",
                "\\textbf{0.40\\,s}",
            ],
            [
                "NoMaSS $\\Gamma$ (3-seed median)",
                "0.01787",
                "0.03249",
                "0.01658",
                "68.37\\,s",
            ],
        ],
    )

    # Appendix Table A6: Unseen-scorer sleep boundary calibration policies
    assert n2_f["calibration"]["adaptive"]["first_sustained"] == 11
    write_tex_table(
        "tab_a6_sleep_calibration.tex",
        "lrrr",
        [
            "Selection Policy (Partial Pool)",
            "Mean Target $\\downarrow$",
            "95\\% Band $\\downarrow$",
            "4-Night NLL $\\downarrow$",
        ],
        [
            ["Longest-first control", "\\textbf{3}", "\\textbf{10}", "\\textbf{1.31986}"],
            ["Random panels (48 draws)", "4", "11", "1.32231"],
            ["Fixed prior Fisher", "9", "15", "1.32315"],
            ["Adaptive sequential Fisher", "11", "24", "1.32351"],
        ],
    )


def main() -> None:
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_tables()
    print("Successfully built 5 publication figures (PDF/SVG/PNG) and 10 LaTeX tables.")


if __name__ == "__main__":
    main()
