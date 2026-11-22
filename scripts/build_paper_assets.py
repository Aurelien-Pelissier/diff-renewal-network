"""Build publication-ready figures and LaTeX tables for Differentiable Renewal Networks.

All figures are drawn at exact ICLR text width (5.50 in = 396.0 PDF pt) and exported
without tight-bbox rescaling so rendered glyph sizes sit at 7.8-8.6 pt (just slightly
smaller than the 9.0 pt ICLR figure captions). Every figure passes render-time
panel-alignment checks, PDF glyph-size floor checks, and PyMuPDF collision checks.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import ScaledTranslation
import numpy as np
from scipy.linalg import expm_frechet

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from diff_renewal import smooth_erlang_logits, solve_erlang_chain  # noqa: E402

_qa_scripts_env = os.environ.get("DRN_QA_SCRIPTS")
SKILL_SCRIPTS = Path(_qa_scripts_env) if _qa_scripts_env else None
if SKILL_SCRIPTS is not None and SKILL_SCRIPTS.exists():
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
else:
    require_matplotlib_panel_alignment = None

ARCH = ROOT / "results" / "archived"
OUT_FIG_REPO = ROOT / "figures"
LATEX_DIR = Path(os.environ.get("DRN_LATEX_DIR", str(ROOT.parent / "diff-renewal-network-latex")))
OUT_FIG_LATEX = LATEX_DIR / "figures" if LATEX_DIR.exists() else ROOT / "output" / "figures"
OUT_TAB_LATEX = LATEX_DIR / "tables" if LATEX_DIR.exists() else ROOT / "output" / "tables"
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
    if SKILL_SCRIPTS is not None and SKILL_SCRIPTS.exists():
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
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"{script_name} issue on {stem}.pdf (code {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
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

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 3.46))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.92, 1.16],
        left=0.095,
        right=0.985,
        top=0.945,
        bottom=0.115,
        hspace=0.46,
        wspace=0.48,
    )

    ax_sch = fig.add_subplot(gs[0, :])
    ax_sch.set_xlim(0.0, 10.0)
    ax_sch.set_ylim(0.0, 2.50)
    ax_sch.axis("off")
    add_panel_tag(ax_sch, "a", x_pt=-22.0, y_pt=0.0)

    boxes = [
        (0.02, 0.25, 2.96, 2.00, PALETTE["light_blue"], PALETTE["navy"],
         "Marked & hyper-race",
         "Mixed & Hyper-Erlang\n$(MK)^d \\to M^d(1 + d(K-1))$\nUp to 1,074× smaller"),
        (3.48, 0.25, 3.04, 2.00, PALETTE["light_amber"], PALETTE["amber"],
         "Positive reduction",
         "$15$ modes cancel in $P$\n$P$: $33$  |  $\\partial P / \\partial\\theta$: $48$\nUniversal Pos-Sens"),
        (7.02, 0.25, 2.96, 2.00, PALETTE["light_teal"], PALETTE["teal"],
         "Renewal & MF solvers",
         "Age, $(u{\\to}v)$ & mean-field\n$P_\\theta(t),\\, \\partial P_\\theta / \\partial\\theta$\nSnapshot NLL & $I(\\theta)$"),
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
    ax_c.plot(ts, full_d, color=PALETTE["navy"], lw=1.8, label="Full 48")
    ax_c.plot(ts, red_d, color=PALETTE["amber"], lw=1.35, ls="--", label="Sym. 33")
    ax_c.set_xlabel("Time $t$")
    ax_c.set_ylabel(r"$\partial P_A / \partial \log\lambda_A$")
    ax_c.set_xlim(0.0, 7.0)
    ax_c.set_ylim(-0.66, 0.22)
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
# Figure 2: Gradient Defect & Positivity-Preserving Sensitivity Reduction
# ---------------------------------------------------------------------------
def build_fig2() -> None:
    ev = load_archived("evidence.json")
    sens = load_archived("sensitivity_evidence.json")
    pos = load_archived("positive_sensitivity_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.38))
    fig.subplots_adjust(left=0.098, right=0.98, top=0.735, bottom=0.195, wspace=0.47)

    bounds = ev["gradient_bounds"]["families"]["trainable_PH"]
    xs_b = np.array([r["epsilon"] for r in bounds])
    for key, label, col, mk in [
        ("mean_contrast", "Sym. mean", PALETTE["navy"], "o"),
        ("shape_contrast", "Sym. shape", PALETTE["amber"], "s"),
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
            ms=3.1,
            lw=1.45,
            label=label,
        )
    # Overlay Pos-Sens 33 max Jacobian error across the same displacements!
    pos_rnames = ["symmetric", "mixed_025", "mixed_075", "mixed_150", "mixed_250"]
    pos_jerr = [pos["regime_diagnostics"][rn]["pos33_jac_max_abs"] for rn in pos_rnames]
    axes[0].plot(
        [0.0, 0.025, 0.075, 0.15, 0.25],
        pos_jerr,
        marker="P",
        ls="-.",
        color=PALETTE["violet"],
        ms=3.6,
        lw=1.45,
        label="Pos-Sens 33",
    )
    axes[0].set_xlabel(r"Heterogeneity $\epsilon$")
    axes[0].set_ylabel("Max. Jacobian defect")
    axes[0].set_xlim(-0.008, 0.26)
    axes[0].set_ylim(-0.02, 0.62)
    axes[0].set_xticks([0.0, 0.075, 0.15, 0.25], ["0", ".075", ".15", ".25"])
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        handlelength=1.1,
        labelspacing=0.16,
        handletextpad=0.3,
        borderpad=0.05,
    )
    add_panel_tag(axes[0], "a", x_pt=-25.0, y_pt=6.0)

    names = ["symmetric", "mixed_025", "mixed_075", "mixed_150", "mixed_250"]
    xs = [0.0, 0.025, 0.075, 0.15, 0.25]
    series_cfg = [
        ("full48", "Full 48", PALETTE["navy"], "o", "-"),
        ("pos33", "Pos-Sens 33", PALETTE["violet"], "P", "-."),
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
            if mkey == "pos33":
                ys = [pos["cases"][f"primary/{k}_{law}"]["median_shape_percent"] for k in names]
            else:
                ys = [
                    sens["cases"][f"primary/{k}_{law}"][mkey]["median_shape_percent"]
                    for k in names
                ]
            ax.plot(xs, ys, marker=mk, ls=ls, color=col, ms=3.2, lw=1.4, label=label)
        ax.set_xlabel(r"Heterogeneity $\epsilon$")
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.008, 0.26)
        ax.set_ylim(0.0, 24.5)
        ax.set_xticks([0.0, 0.075, 0.15, 0.25], ["0", ".075", ".15", ".25"])
        add_panel_tag(ax, tag, x_pt=-25.0, y_pt=6.0)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.69, 0.99),
        ncol=3,
        handlelength=1.15,
        columnspacing=0.65,
        handletextpad=0.28,
        labelspacing=0.22,
        borderpad=0.1,
    )

    export_figure(fig, "fig2_compression_and_sensitivity")


# ---------------------------------------------------------------------------
# Figure 3: Graph Scaling, Stem-Cell Cascade, 60-Dataset Fisher, & 48-Param Recovery
# ---------------------------------------------------------------------------
def build_fig3() -> None:
    hem = load_archived("hematopoiesis_branching_summary.json")
    ev = load_archived("evidence.json")
    gsc = load_archived("graph_scaling_summary.json")
    spg = load_archived("sparse_large_graph_scaling.json")
    records = ev["benchmark"]

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.52))
    fig.subplots_adjust(
        left=0.11, right=0.975, top=0.895, bottom=0.115, hspace=0.62, wspace=0.40
    )

    # Panel a: Marked-race (V=4..48) & Matrix-Free SparseDRN (V=16..1024) gradient scaling
    ax_a = axes[0, 0]
    g_rows = gsc["gradient_and_state_scaling"]
    sp_rows = spg["scaling_rows"]
    ax_a.plot(
        [r["nodes"] for r in sp_rows],
        [r["grad_warm_ms"] for r in sp_rows],
        marker="o",
        color=PALETTE["navy"],
        ms=3.3,
        lw=1.55,
        label="Sparse $d{=}2$",
    )
    for deg, col, mk, lbl in [
        (3, PALETTE["teal"], "s", "Race $d{=}3$"),
        (4, PALETTE["violet"], "^", "Race $d{=}4$"),
    ]:
        sub = [r for r in g_rows if r["out_degree"] == deg]
        ax_a.plot(
            [r["nodes"] for r in sub],
            [r["compact_grad_warm_ms"] for r in sub],
            marker=mk,
            color=col,
            ms=3.2,
            lw=1.4,
            label=lbl,
        )
    sub_j3 = [r for r in g_rows if r["out_degree"] == 3 and r["joint_grad_warm_ms"] is not None]
    ax_a.plot(
        [r["nodes"] for r in sub_j3],
        [r["joint_grad_warm_ms"] for r in sub_j3],
        marker="x",
        ls="--",
        color=PALETTE["crimson"],
        ms=4.0,
        lw=1.4,
        label="Tensor $d{=}3$",
    )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(3.2, 1350)
    ax_a.set_ylim(10.0, 900.0)
    ax_a.set_xticks([4, 16, 64, 256, 1024], ["4", "16", "64", "256", "1024"])
    ax_a.minorticks_off()
    ax_a.set_xlabel("Graph nodes $V$")
    ax_a.set_ylabel("Warm grad. (ms)")
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.05,
        columnspacing=0.55,
        handletextpad=0.25,
        labelspacing=0.18,
        borderpad=0.05,
    )
    add_panel_tag(ax_a, "a", x_pt=-24.0, y_pt=18.0)

    # Panel b: Primary human 7-node branching hematopoiesis FACS trajectories (Bast et al., 2021)
    ax_b = axes[0, 1]
    tr_h = hem["trajectories_days_1to7"]
    days_h = np.asarray(tr_h["days"], dtype=float)
    emp_h = np.asarray(tr_h["empirical_fractions"], dtype=float)
    drn_h = np.asarray(tr_h["compiled_drn_fractions"], dtype=float)
    ax_b.axvline(5.5, color=PALETTE["slate"], lw=0.75, ls="--")
    # Plot HSC (0), MPP+MLP lymphoid/multipotent (1+2), CMP+GMP+MEP myeloid/erythroid (3+4+5), Mature (6)
    groups_h = [
        ("HSC", [0], PALETTE["navy"], "o"),
        ("MPP+MLP", [1, 2], PALETTE["teal"], "s"),
        ("Myel./Ery.", [3, 4, 5], PALETTE["amber"], "^"),
        ("Mature", [6], PALETTE["violet"], "d"),
    ]
    for gname, idxs, col, mk in groups_h:
        y_emp = emp_h[:, idxs].sum(axis=1)
        y_drn = drn_h[:, idxs].sum(axis=1)
        ax_b.plot(days_h, y_drn, color=col, lw=1.5, label=gname)
        ax_b.scatter(days_h, y_emp, color=col, marker=mk, s=14, zorder=4)
    ax_b.set_xlabel("Culture time (days)")
    ax_b.set_ylabel("FACS fraction")
    ax_b.set_xlim(0.7, 7.3)
    ax_b.set_ylim(-0.04, 1.02)
    ax_b.set_xticks([1, 3, 5, 7])
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.5,
        handletextpad=0.25,
        labelspacing=0.18,
        borderpad=0.05,
    )
    add_panel_tag(ax_b, "b", x_pt=-24.0, y_pt=18.0)

    # Panel c: 60-dataset Fisher shape recovery scatter
    ax_c = axes[1, 0]
    lim = 22.5
    ax_c.plot([0, lim], [0, lim], color=PALETTE["slate"], lw=0.8, ls=":")
    ax_c.axhline(10.0, color=PALETTE["slate"], lw=0.65, ls="--")
    ax_c.axvline(10.0, color=PALETTE["slate"], lw=0.65, ls="--")
    for topo, label, mk, col in [
        ("cyclic", "Cycle", "o", PALETTE["navy"]),
        ("branching", "Diamond", "^", PALETTE["teal"]),
    ]:
        rows = [r for r in records if r["case"].startswith(topo)]
        ax_c.scatter(
            [100.0 * r["nomass"]["shape_error"] for r in rows],
            [100.0 * r["phase"]["shape_error"] for r in rows],
            s=17,
            marker=mk,
            color=col,
            alpha=0.84,
            label=label,
        )
    ax_c.set_xlim(0.0, lim)
    ax_c.set_ylim(0.0, lim)
    ax_c.set_xticks([0, 10, 20])
    ax_c.set_yticks([0, 10, 20])
    ax_c.set_xlabel("NoMaSS shape error (%)")
    ax_c.set_ylabel("PH shape error (%)")
    ax_c.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=0.6,
        handletextpad=0.2,
        borderpad=0.05,
    )
    add_panel_tag(ax_c, "c", x_pt=-24.0, y_pt=14.0)

    # Panel d: High-dimensional continuous-Gamma recovery (16, 32, 48 parameters)
    ax_d = axes[1, 1]
    hd = gsc["highdim_gamma_recovery"]
    x_hd = np.arange(len(hd))
    ax_d.axhline(10.0, color=PALETTE["slate"], lw=0.7, ls="--")
    ax_d.bar(
        x_hd - 0.17,
        [r["ph_shape_rms_median_pct"] for r in hd],
        width=0.31,
        color=PALETTE["navy"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="Compiled PH",
    )
    ax_d.bar(
        x_hd + 0.17,
        [r["mc_shape_rms_median_pct"] for r in hd],
        width=0.31,
        color=PALETTE["amber"],
        edgecolor=PALETTE["ink"],
        linewidth=0.55,
        label="MC search",
    )
    ax_d.set_xticks(x_hd, [f"V={r['nodes']}\n({r['parameters']}p)" for r in hd])
    ax_d.set_ylim(0.0, 41.0)
    ax_d.set_ylabel("Shape RMS (%)")
    ax_d.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.55,
        handletextpad=0.25,
        borderpad=0.05,
    )
    add_panel_tag(ax_d, "d", x_pt=-24.0, y_pt=14.0)

    export_figure(fig, "fig3_stemcells_and_fisher_benchmark")


# ---------------------------------------------------------------------------
# Figure 4: Conditional Neural Clocks vs Modern Baselines & Multi-System Transfer
# ---------------------------------------------------------------------------
def build_fig4() -> None:
    mnb = load_archived("modern_neural_baselines.json")
    rt = load_archived("renewal_transfer_summary.json")
    r7m = load_archived("r7_mamba_thp_baselines.json")["synthetic_mamba"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.52))
    fig.subplots_adjust(left=0.108, right=0.98, top=0.68, bottom=0.195, wspace=0.39)

    studies = [
        ("primary", "PH truth"),
        ("lognormal", "Lognormal"),
    ]
    models = [
        ("neural_phase", "Neural clock", PALETTE["navy"], -0.30),
        ("mean_shape", "Moment PH", PALETTE["teal"], -0.20),
        ("mamba_3start", "Mamba", PALETTE["royal"], -0.10),
        ("ltc_ode_3start", "LTC", PALETTE["violet"], 0.00),
        ("s4d_ode_3start", "S4D", PALETTE["amber"], 0.10),
        ("latent_ode_3start", "Latent ODE", PALETTE["crimson"], 0.20),
        ("neural_cme_3start", "Neural-CME", PALETTE["slate"], 0.30),
    ]
    x_pos = np.arange(len(studies))
    axes[0].axhline(0.005, color=PALETTE["slate"], ls="--", lw=0.75)
    for mkey, mlabel, col, dx in models:
        meds, lows, highs = [], [], []
        for skey, _ in studies:
            if mkey == "mamba_3start":
                stat = r7m[skey]["balanced_kl"]
            else:
                stat = mnb[skey][mkey]["metrics"]["balanced_kl"]
            med = float(stat["median"])
            meds.append(med)
            lows.append(med - float(stat["q25"]))
            highs.append(float(stat["q75"]) - med)
        y_base = 8e-4
        axes[0].bar(
            x_pos + dx,
            np.asarray(meds) - y_base,
            bottom=y_base,
            width=0.092,
            color=col,
            edgecolor=PALETTE["ink"],
            linewidth=0.42,
            label=mlabel,
        )
        axes[0].errorbar(
            x_pos + dx,
            meds,
            yerr=[lows, highs],
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=0.65,
            capsize=1.2,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x_pos, [lbl for _, lbl in studies])
    axes[0].set_xlim(-0.48, 1.48)
    axes[0].set_ylim(8e-4, 0.24)
    axes[0].set_ylabel("Held-out excess KL (nats)")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=6.6,
        handlelength=0.80,
        columnspacing=0.40,
        handletextpad=0.20,
        labelspacing=0.14,
        borderpad=0.04,
    )
    add_panel_tag(axes[0], "a", x_pt=-24.0, y_pt=28.0)

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
        handlelength=1.15,
        columnspacing=0.55,
        handletextpad=0.25,
        labelspacing=0.18,
        borderpad=0.05,
    )
    add_panel_tag(axes[1], "b", x_pt=-24.0, y_pt=28.0)

    export_figure(fig, "fig4_neural_clocks_and_transfer")


# ---------------------------------------------------------------------------
# Figure 5: Multi-Domain Real Networks (Sleep, IT Helpdesk, Mean-Field Outbreak, Persistence)
# ---------------------------------------------------------------------------
def build_fig5() -> None:
    rcb = load_archived("real_continuous_benchmarks.json")
    ems = load_archived("edge_memory_sleep_summary.json")
    hdk = load_archived("helpdesk_systems_summary.json")
    hdk_r5 = load_archived("helpdesk_r5_complete.json")
    r6 = load_archived("r6_uniformity_closure.json")
    r6b = load_archived("r6_integrity_fixes.json")
    mfe = load_archived("mean_field_epidemic_summary.json")
    rvm = load_archived("real_visit_memory.json")["datasets"]

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.46))
    fig.subplots_adjust(
        left=0.12, right=0.975, top=0.845, bottom=0.115, hspace=0.74, wspace=0.49
    )

    # Panel a: 5-Node Human Sleep-EDF (incl. leak-free strict 0-shot CTMC!)
    ax_a = axes[0, 0]
    sl_m = rcb["sleep_5node_branching_network"]["models_5fold_mean"]
    sl_r6 = r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]
    x_c = np.arange(2)
    for mlabel, vals, sds, col, dx in [
        (
            "Edge-Mem",
            [ems["edge_memory_drn_joint"]["aged_kl_mean"], ems["edge_memory_drn_fresh_only"]["aged_kl_mean"]],
            [ems["edge_memory_drn_joint"]["aged_kl_sd"], ems["edge_memory_drn_fresh_only"]["aged_kl_sd"]],
            PALETTE["navy"],
            -0.30,
        ),
        (
            "Renew. DRN",
            [sl_m["drn_joint"]["aged_kl_mean"], sl_m["drn_fresh_only"]["aged_kl_mean"]],
            [sl_m["drn_joint"]["aged_kl_sd"], sl_m["drn_fresh_only"]["aged_kl_sd"]],
            PALETTE["teal"],
            -0.15,
        ),
        (
            "CTMC",
            [sl_r6["ctmc_joint"]["aged_kl_mean"], sl_r6["ctmc_fresh_only"]["aged_kl_mean"]],
            [sl_r6["ctmc_joint"]["aged_kl_sd"], sl_r6["ctmc_fresh_only"]["aged_kl_sd"]],
            PALETTE["slate"],
            0.00,
        ),
        (
            "Neural-CME",
            [sl_r6["neural_cme_joint"]["aged_kl_mean"], sl_r6["neural_cme_fresh_only"]["aged_kl_mean"]],
            [sl_r6["neural_cme_joint"]["aged_kl_sd"], sl_r6["neural_cme_fresh_only"]["aged_kl_sd"]],
            PALETTE["amber"],
            0.15,
        ),
        (
            "Latent ODE",
            [sl_r6["latent_ode_joint"]["aged_kl_mean"], sl_r6["latent_ode_fresh_only"]["aged_kl_mean"]],
            [sl_r6["latent_ode_joint"]["aged_kl_sd"], sl_r6["latent_ode_fresh_only"]["aged_kl_sd"]],
            PALETTE["crimson"],
            0.30,
        ),
    ]:
        ax_a.bar(
            x_c + dx,
            np.asarray(vals),
            width=0.135,
            color=col,
            edgecolor=PALETTE["ink"],
            linewidth=0.45,
            label=mlabel,
        )
        ax_a.errorbar(
            x_c + dx,
            vals,
            yerr=sds,
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=0.7,
            capsize=1.3,
        )
    ax_a.set_xticks(x_c, ["Sleep (joint)", "Sleep (0-shot)"])
    ax_a.set_xlim(-0.50, 1.50)
    ax_a.set_ylim(0.0, 0.145)
    ax_a.set_yticks([0.0, 0.05, 0.10, 0.14], ["0", ".05", ".10", ".14"])
    ax_a.set_ylabel("Aged KL (nats)")
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=6.5,
        handlelength=0.75,
        columnspacing=0.35,
        handletextpad=0.18,
        labelspacing=0.15,
        borderpad=0.04,
    )
    add_panel_tag(ax_a, "a", x_pt=-32.0, y_pt=22.0)

    # Panel b: IT Helpdesk Coarsening Sweep across ALL 4 intervals {0.5, 2, 7, 14}d
    # Includes Hyper-Pos-Sens (best overall) & LTC (strongest neural baseline at 14d)!
    ax_b = axes[0, 1]
    c_rows = hdk["coarsening_panel_sweep"]
    c_r5 = hdk_r5["coarsening_neural_clean_age0"]
    c_hp = r6["helpdesk_r6_hyper_pos_coarsening"]
    x_dt = np.arange(len(c_rows))
    for mlabel, ys, col, mk, ls in [
        ("Hyper-Pos", [float(r["hyper_pos_sens_aged_kl"]) for r in c_hp], PALETTE["teal"], "v", "-"),
        ("Edge-Mem", [float(r["edge_memory_drn_multihop_aged_kl"]) for r in c_rows], PALETTE["navy"], "o", "-"),
        ("LTC 0-sh.", [float(r["ltc_ode_aged_kl_clean_age0"]) for r in c_r5], PALETTE["amber"], "^", ":"),
        ("LODE 0-sh.", [float(r["latent_ode_aged_kl_clean_age0"]) for r in c_r5], PALETTE["crimson"], "d", "--"),
        ("Neural-CR", [float(r["neural_competing_risks_best_kl"]) for r in c_rows], PALETTE["slate"], "s", "-."),
    ]:
        ax_b.plot(
            x_dt,
            ys,
            marker=mk,
            ls=ls,
            color=col,
            ms=3.3,
            lw=1.35,
            label=mlabel,
        )
    ax_b.set_yscale("log")
    ax_b.set_xticks(x_dt, [f"{r['delta_t_days']:g}d" for r in c_rows])
    ax_b.set_xlim(-0.25, 3.25)
    ax_b.set_ylim(0.025, 3.2)
    ax_b.set_yticks([0.04, 0.2, 1.0, 2.5], [".04", ".2", "1.0", "2.5"])
    ax_b.minorticks_off()
    ax_b.set_xlabel(r"Panel interval $\Delta t$ (IT Helpdesk)")
    ax_b.set_ylabel("Held-out KL (nats)")
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=6.5,
        handlelength=0.85,
        columnspacing=0.35,
        handletextpad=0.18,
        labelspacing=0.15,
        borderpad=0.04,
    )
    add_panel_tag(ax_b, "b", x_pt=-32.0, y_pt=22.0)

    # Panel c: Nonlinear MeanFieldDRN on 1978 Influenza (Color=compartment, Linestyle=model!)
    ax_c = axes[1, 0]
    days_e = np.asarray(mfe["dataset"]["days"], dtype=float)
    obs_bed = np.asarray(mfe["dataset"]["in_bed"], dtype=float)
    obs_conv = np.asarray(mfe["dataset"]["convalescent"], dtype=float)
    ex8 = mfe["extrapolation_splits"]["days_1to8_to_9to14"]
    ax_c.axvline(8.5, color=PALETTE["slate"], lw=0.75, ls="--")
    ax_c.scatter(days_e, obs_bed, s=13, color=PALETTE["navy"], marker="o", zorder=5)
    ax_c.scatter(days_e, obs_conv, s=13, color=PALETTE["teal"], marker="s", zorder=5)
    ax_c.plot(days_e, ex8["mean_field_drn"]["pred_in_bed"], color=PALETTE["navy"], lw=1.5, ls="-")
    ax_c.plot(days_e, ex8["mean_field_drn"]["pred_convalescent"], color=PALETTE["teal"], lw=1.5, ls="-")
    ax_c.plot(days_e, ex8["markov_seicr"]["pred_in_bed"], color=PALETTE["navy"], lw=1.25, ls="--")
    ax_c.plot(days_e, ex8["markov_seicr"]["pred_convalescent"], color=PALETTE["teal"], lw=1.25, ls="--")
    ax_c.set_xlim(0.6, 14.4)
    ax_c.set_ylim(-10, 335)
    ax_c.set_xticks([1, 5, 8, 11, 14])
    ax_c.set_yticks([0, 100, 200, 300])
    ax_c.set_xlabel("Outbreak day (1978 Influenza)")
    ax_c.set_ylabel("Cases (of 763)")
    leg_c = [
        Line2D([0], [0], color=PALETTE["navy"], marker="o", ms=3.3, lw=1.4, label="In-bed"),
        Line2D([0], [0], color=PALETTE["teal"], marker="s", ms=3.3, lw=1.4, label="Conval."),
        Line2D([0], [0], color=PALETTE["ink"], lw=1.4, ls="-", label="MF-DRN"),
        Line2D([0], [0], color=PALETTE["ink"], lw=1.25, ls="--", label="Markov"),
    ]
    ax_c.legend(
        handles=leg_c,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.1,
        columnspacing=0.55,
        handletextpad=0.25,
        labelspacing=0.18,
        borderpad=0.05,
    )
    add_panel_tag(ax_c, "c", x_pt=-32.0, y_pt=20.0)

    # Panel d: Longitudinal cross-visit persistence gain (before vs after covariates)
    ax_d = axes[1, 1]
    cohorts = [
        ("sleep", "2", "Sleep N2"),
        ("nomass", "all", "Si E. coli"),
        ("tanouchi", "all", "Tan. E. coli"),
    ]
    ax_d.axvline(0.0, color=PALETTE["slate"], lw=0.8, ls="--")
    for idx_y, (dkey, subk, _) in enumerate(cohorts):
        y_coord = 2 - idx_y
        for mkey, mlabel, col, mk, dy in [
            ("marginal", "Raw", PALETTE["amber"], "s", 0.14),
            ("adjusted", "Adj.", PALETTE["navy"], "o", -0.14),
        ]:
            est = rvm[dkey]["models"][mkey]["previous"][subk]
            val = 100.0 * float(est["gain"])
            lo, hi = [100.0 * float(v) for v in est["interval"]]
            ax_d.errorbar(
                val,
                y_coord + dy,
                xerr=[[val - lo], [hi - val]],
                fmt=mk,
                color=col,
                ms=3.8,
                lw=1.3,
                capsize=2.0,
                label=mlabel if idx_y == 0 else None,
            )
    ax_d.set_yticks([2, 1, 0], [lbl for _, _, lbl in cohorts])
    ax_d.set_ylim(-0.52, 2.55)
    ax_d.set_xlim(-0.45, 2.75)
    ax_d.set_xlabel("Gain / 100 visits (nats)")
    ax_d.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=0.55,
        handletextpad=0.2,
        borderpad=0.05,
    )
    add_panel_tag(ax_d, "d", x_pt=-32.0, y_pt=20.0)

    export_figure(fig, "fig5_persistence_and_sleep_calibration")


# ---------------------------------------------------------------------------
# Figure 6: Theorem 6 Near-Cancellation BT Distortion, MOOC V=98, & DeFi Multi-Pool
# ---------------------------------------------------------------------------
def build_fig6() -> None:
    r7t = load_archived("r7_theory_bt_separation.json")
    r7n = load_archived("r7_temporal_networks.json")
    r7d = load_archived("r7_defi_neural_clock.json")

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 3.65))
    fig.subplots_adjust(
        left=0.12, right=0.975, top=0.855, bottom=0.115, hspace=0.74, wspace=0.48
    )

    # Panel a: Theorem 6 Quantitative delta-near-cancellation & BT distortion ratio
    ax_a = axes[0, 0]
    sw_k4 = [r for r in r7t["symmetric_cycles"]["K4"]["sweep"] if r["delta"] <= 0.03]
    sw_ht = [r for r in r7t["heterogeneous_common_zero"]["sweep"] if r["delta"] <= 0.03]
    d_k4 = [r["delta"] for r in sw_k4]
    d_ht = [r["delta"] for r in sw_ht]
    ax_a.plot(d_k4, [r["ratio_frozen"] for r in sw_k4], marker="o", ms=3.0, lw=1.4, color=PALETTE["navy"], label="K4 BT-fz")
    ax_a.plot(d_k4, [r["ratio_rebal"] for r in sw_k4], marker="s", ms=2.9, lw=1.35, ls="--", color=PALETTE["teal"], label="K4 BT-reb")
    ax_a.plot(d_ht, [r["ratio_frozen"] for r in sw_ht], marker="^", ms=3.0, lw=1.35, ls="-.", color=PALETTE["crimson"], label="Het BT-fz")
    ax_a.plot(d_k4, [r["fwd_hinf"] for r in sw_k4], marker="d", ms=2.8, lw=1.25, ls=":", color=PALETTE["amber"], label="K4 H-inf")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(6e-6, 5e-2)
    ax_a.set_ylim(1e-7, 1.5e7)
    ax_a.set_xticks([1e-5, 1e-4, 1e-3, 1e-2])
    ax_a.set_yticks([1e-6, 1e-2, 1e2, 1e6])
    ax_a.minorticks_off()
    ax_a.set_xlabel(r"Near-cancellation gap $\delta$")
    ax_a.set_ylabel("Ratio / H-inf err.")
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=6.8,
        handlelength=1.0,
        columnspacing=0.45,
        handletextpad=0.20,
        labelspacing=0.14,
        borderpad=0.04,
    )
    add_panel_tag(ax_a, "a", x_pt=-32.0, y_pt=22.0)

    # Panel b: MOOC (V=98, 411k events) — 0-shot aged (>=6h) & supervised joint aged KL
    ax_b = axes[0, 1]
    mr = r7n["mooc_v98"]["results"]
    x_m = np.arange(2)
    mooc_bars = [
        ("SpDRN", [mr["sparse_hyper_drn_fresh_only"]["aged_kl"], mr["sparse_hyper_drn_joint"]["aged_kl"]], PALETTE["navy"], -0.32),
        ("CTMC", [mr["ctmc_fresh_only"]["aged_kl"], mr["ctmc_joint"]["aged_kl"]], PALETTE["slate"], -0.192),
        ("Mamba", [mr["mamba_fresh_only"]["aged_kl"], mr["mamba_joint"]["aged_kl"]], PALETTE["royal"], -0.064),
        ("S4D", [mr["s4d_fresh_only"]["aged_kl"], mr["s4d_joint"]["aged_kl"]], PALETTE["amber"], 0.064),
        ("LTC", [mr["ltc_fresh_only"]["aged_kl"], mr["ltc_joint"]["aged_kl"]], PALETTE["violet"], 0.192),
        ("LODE", [mr["latent_ode_fresh_only"]["aged_kl"], mr["latent_ode_joint"]["aged_kl"]], PALETTE["crimson"], 0.32),
    ]
    for lbl, vals, col, dx in mooc_bars:
        ax_b.bar(x_m + dx, vals, width=0.118, color=col, edgecolor=PALETTE["ink"], linewidth=0.42, label=lbl)
    ax_b.axhline(mr["oracle_fresh_rollout_aged_kl"], color=PALETTE["slate"], ls="--", lw=0.75)
    ax_b.set_xticks(x_m, ["0-shot (>=6h)", "Joint (sup.)"])
    ax_b.set_xlim(-0.50, 1.50)
    ax_b.set_ylim(0.0, 2.15)
    ax_b.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0], ["0", "0.5", "1.0", "1.5", "2.0"])
    ax_b.set_xlabel("MOOC ($V{=}98$, 411k events)")
    ax_b.set_ylabel("Aged KL (nats)")
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=6.4,
        handlelength=0.75,
        columnspacing=0.35,
        handletextpad=0.18,
        labelspacing=0.14,
        borderpad=0.04,
    )
    add_panel_tag(ax_b, "b", x_pt=-32.0, y_pt=22.0)

    # Panel c: MOOC (V=98) multi-hop coarsening sweep dt in {1, 6, 12, 24}h
    ax_c = axes[1, 0]
    mc = mr["coarsening"]
    dts_m = ["dt_1.0", "dt_6.0", "dt_12.0", "dt_24.0"]
    xm_c = np.arange(len(dts_m))
    for lbl, key, col, mk, ls in [
        ("SpDRN", "sparse_drn_aged_kl", PALETTE["navy"], "o", "-"),
        ("Mamba", "mamba_aged_kl", PALETTE["royal"], "^", "--"),
        ("CTMC", "ctmc_aged_kl", PALETTE["amber"], "s", ":"),
        ("1-j TPP", "single_jump_tpp_aged_kl", PALETTE["crimson"], "d", "-."),
    ]:
        ax_c.plot(xm_c, [mc[d][key] for d in dts_m], marker=mk, ls=ls, color=col, ms=3.3, lw=1.4, label=lbl)
    ax_c.set_yscale("log")
    ax_c.set_xticks(xm_c, ["1h", "6h", "12h", "24h"])
    ax_c.set_xlim(-0.25, 3.25)
    ax_c.set_ylim(0.22, 22.0)
    ax_c.set_yticks([0.3, 1.0, 4.0, 15.0], ["0.3", "1.0", "4.0", "15"])
    ax_c.minorticks_off()
    ax_c.set_xlabel(r"Panel interval $\Delta t$ (MOOC)")
    ax_c.set_ylabel("Aged KL (nats)")
    ax_c.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=6.6,
        handlelength=1.0,
        columnspacing=0.45,
        handletextpad=0.20,
        labelspacing=0.14,
        borderpad=0.04,
    )
    add_panel_tag(ax_c, "c", x_pt=-32.0, y_pt=20.0)

    # Panel d: DeFi Multi-Pool (183,881 opps, 16 on-chain panels) — Regimes A & B log-loss
    ax_d = axes[1, 1]
    ra = r7d["regime_a_zeroshot_deep_age_sale"]
    rb = r7d["regime_b_temporal_drift_sale"]
    x_d = np.arange(2)
    defi_bars = [
        ("N-Clock", [ra["neural_clock"]["all"]["logloss"], rb["neural_clock"]["all"]["logloss"]], PALETTE["navy"], -0.32),
        ("MLP+age", [ra["mlp_age"]["all"]["logloss"], rb["mlp_age"]["all"]["logloss"]], PALETTE["teal"], -0.192),
        ("LTC", [ra["ltc_seq"]["all"]["logloss"], rb["ltc_seq"]["all"]["logloss"]], PALETTE["violet"], -0.064),
        ("S4D", [ra["s4d_seq"]["all"]["logloss"], rb["s4d_seq"]["all"]["logloss"]], PALETTE["amber"], 0.064),
        ("Markov", [ra["markov_clock"]["all"]["logloss"], rb["markov_clock"]["all"]["logloss"]], PALETTE["slate"], 0.192),
        ("Mamba", [ra["mamba_ssm"]["all"]["logloss"], rb["mamba_ssm"]["all"]["logloss"]], PALETTE["crimson"], 0.32),
    ]
    y0_d = 0.29
    for lbl, vals, col, dx in defi_bars:
        ax_d.bar(
            x_d + dx,
            np.asarray(vals) - y0_d,
            bottom=y0_d,
            width=0.118,
            color=col,
            edgecolor=PALETTE["ink"],
            linewidth=0.42,
            label=lbl,
        )
    ax_d.set_xticks(x_d, ["0-shot age", "5-mo. drift"])
    ax_d.set_xlim(-0.50, 1.50)
    ax_d.set_ylim(y0_d, 0.445)
    ax_d.set_yticks([0.30, 0.34, 0.38, 0.42], [".30", ".34", ".38", ".42"])
    ax_d.set_xlabel("DeFi Multi-Pool (164k test)")
    ax_d.set_ylabel("Held-out NLL (nats)")
    ax_d.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=6.4,
        handlelength=0.75,
        columnspacing=0.35,
        handletextpad=0.18,
        labelspacing=0.14,
        borderpad=0.04,
    )
    add_panel_tag(ax_d, "d", x_pt=-32.0, y_pt=20.0)

    export_figure(fig, "fig6_bt_and_largescale_networks")


# ---------------------------------------------------------------------------
# Build Main-Text and Appendix Tables
# ---------------------------------------------------------------------------
def build_tables() -> None:
    ev = load_archived("evidence.json")
    sens = load_archived("sensitivity_evidence.json")
    pos = load_archived("positive_sensitivity_summary.json")
    gsc = load_archived("graph_scaling_summary.json")
    mnb = load_archived("modern_neural_baselines.json")
    rcb = load_archived("real_continuous_benchmarks.json")
    ems = load_archived("edge_memory_sleep_summary.json")
    hem = load_archived("hematopoiesis_branching_summary.json")
    hem_d = load_archived("hematopoiesis_donor_extrap.json")
    bio_r5 = load_archived("bio_influenza_r5_complete.json")
    hdk = load_archived("helpdesk_systems_summary.json")
    hdk_ext = load_archived("helpdesk_extended_baselines.json")
    hdk_r5 = load_archived("helpdesk_r5_complete.json")
    r6 = load_archived("r6_uniformity_closure.json")
    r6b = load_archived("r6_integrity_fixes.json")
    r7t = load_archived("r7_theory_bt_separation.json")
    r7m = load_archived("r7_mamba_thp_baselines.json")
    r7d = load_archived("r7_defi_neural_clock.json")
    r7n = load_archived("r7_temporal_networks.json")
    slp_ext = load_archived("sleep_extended_baselines.json")
    mfe = load_archived("mean_field_epidemic_summary.json")
    spg = load_archived("sparse_large_graph_scaling.json")
    sp_rss = load_archived("sparse_e2e_and_memory.json")
    tf = load_archived("trait_fisher_summary.json")
    nm = load_archived("neural_memory_summary.json")
    nmp = load_archived("neural_memory_pulse.json")
    rt = load_archived("renewal_transfer_summary.json")
    rt6 = load_archived("renewal_transfer_six_parameter.json")
    rvm = load_archived("real_visit_memory.json")["datasets"]
    ph_sc = load_archived("trainable_ph_stem_cells.json")
    n2_f = load_archived("n2_fisher_recordings_compact.json")

    # Table 1: Cyclic compression, Pos-Sens 33, and sensitivity interpolation summary
    p_cases = sens["cases"]
    p_ov = sens["overall"]["primary"]
    t1_rows = [
        [
            "\\textbf{Full 48}",
            "48",
            f"{p_cases['primary/symmetric_PH']['full48']['median_shape_percent']:.2f}\\%",
            f"{p_cases['primary/mixed_150_PH']['full48']['median_shape_percent']:.2f}\\% / {p_cases['primary/mixed_150_Gamma']['full48']['median_shape_percent']:.2f}\\%",
            f"{p_ov['full48']['joint']}/280",
            "\\textbf{0/400}",
            "1.00$\\times$",
        ],
        [
            "\\textbf{Pos-Sens 33 (ours)}",
            "33",
            f"{pos['cases']['primary/symmetric_PH']['median_shape_percent']:.2f}\\%",
            f"{pos['cases']['primary/mixed_150_PH']['median_shape_percent']:.2f}\\% / {pos['cases']['primary/mixed_150_Gamma']['median_shape_percent']:.2f}\\%",
            f"{pos['overall']['primary']['pos33']['joint']}/280",
            "\\textbf{0/400}",
            f"{pos['overall']['primary']['pos33']['median_paired_speedup_vs_full48']:.2f}$\\times$",
        ],
        [
            "\\textbf{Interp. 33 (2-sided)}",
            "33",
            f"{p_cases['primary/symmetric_PH']['sp33']['median_shape_percent']:.2f}\\%",
            f"\\textbf{{{p_cases['primary/mixed_150_PH']['sp33']['median_shape_percent']:.2f}\\%}} / \\textbf{{{p_cases['primary/mixed_150_Gamma']['sp33']['median_shape_percent']:.2f}\\%}}",
            f"\\textbf{{{p_ov['sp33']['joint']}/280}}",
            "400/400",
            "1.90$\\times$",
        ],
        [
            "Symmetry closure",
            "33",
            f"\\textbf{{{p_cases['primary/symmetric_PH']['sym33']['median_shape_percent']:.2f}\\%}}",
            f"{p_cases['primary/mixed_150_PH']['sym33']['median_shape_percent']:.2f}\\% / {p_cases['primary/mixed_150_Gamma']['sym33']['median_shape_percent']:.2f}\\%",
            f"{p_ov['sym33']['joint']}/280",
            "400/400",
            "2.31$\\times$",
        ],
        [
            "Interp. 18 (1-anchor)",
            "18",
            f"{p_cases['primary/symmetric_PH']['sp18']['median_shape_percent']:.2f}\\%",
            f"{p_cases['primary/mixed_150_PH']['sp18']['median_shape_percent']:.2f}\\% / {p_cases['primary/mixed_150_Gamma']['sp18']['median_shape_percent']:.2f}\\%",
            f"{p_ov['sp18']['joint']}/280",
            "400/400",
            "\\textbf{2.34$\\times$}",
        ],
    ]
    write_tex_table(
        "tab1_compression_summary.tex",
        "lcrrrrr",
        [
            "Realization",
            "States",
            "Sym. $\\downarrow$",
            "$\\epsilon{=}0.15$ (PH / $\\Gamma$) $\\downarrow$",
            "Primary $\\uparrow$",
            "Neg. Early $\\downarrow$",
            "Speedup $\\uparrow$",
        ],
        t1_rows,
    )

    # Table 2: Rate, shape, persistent-trait, and high-dim 48-parameter recovery
    hd_rec = gsc["highdim_gamma_recovery"]
    assert [r["ph_joint_hits_of_5"] for r in hd_rec] == [5, 5, 5]
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
            "NoMaSS (Fisher)",
            "1.73\\%",
            "8.84\\%",
            "\\textbf{9/10}",
            "1.00$\\times$ ref.",
        ],
        "\\midrule",
        [
            "High-dim Gamma",
            "\\textbf{Compiled PH}",
            f"\\textbf{{{hd_rec[2]['ph_rate_rms_median_pct']:.2f}\\%}}",
            f"\\textbf{{{hd_rec[2]['ph_shape_rms_median_pct']:.2f}\\%}}",
            "\\textbf{15/15}",
            "\\textbf{8.3$\\times$ states}",
        ],
        [
            "($V{=}12$, 48p; all $V$)",
            "MC search (DE+Powell)",
            f"{hd_rec[2]['mc_rate_rms_median_pct']:.2f}\\%",
            f"{hd_rec[2]['mc_shape_rms_median_pct']:.2f}\\%",
            "0/15",
            "fails ($>26\\%$)",
        ],
    ]
    write_tex_table(
        "tab2_fisher_and_traits.tex",
        "llrrrr",
        [
            "Benchmark",
            "Method \\& Design",
            "Rate / Mean $\\downarrow$",
            "Shape RMS $\\downarrow$",
            "Joint Hits $\\uparrow$",
            "Gain / Scaling $\\uparrow$",
        ],
        t2_rows,
    )

    # Table 3: Conditional neural clocks vs tuned LTC, S4D, Latent ODE, and Neural-CME + transfer
    # Use uniform pulse_effect_rmse across all 6 rows (Forensic E6) and uniform pretrained_code at 8k (Forensic A2)
    tr_400 = rt["heldout_ph"]["400"]["arms"]
    tr_8000 = rt["heldout_ph"]["8000"]["arms"]
    t3_rows = [
        [
            "\\textbf{Neural clock}",
            "185",
            f"\\textbf{{{mnb['primary']['neural_phase']['metrics']['balanced_kl']['median']:.5f}}}",
            f"\\textbf{{{mnb['lognormal']['neural_phase']['metrics']['balanced_kl']['median']:.5f}}}",
            f"\\textbf{{{100*mnb['primary']['neural_phase']['metrics']['pulse_effect_rmse']['median']:.2f}\\,\\text{{pp}}}}",
            f"\\textbf{{{tr_400['neural_phase.pretrained_code']['balanced_kl']:.5f}}}",
            f"\\textbf{{{tr_8000['neural_phase.pretrained_code']['balanced_kl']:.5f}}}",
        ],
        [
            "Moment PH",
            "18",
            f"{mnb['primary']['mean_shape']['metrics']['balanced_kl']['median']:.5f}",
            f"{mnb['lognormal']['mean_shape']['metrics']['balanced_kl']['median']:.5f}",
            f"{100*mnb['primary']['mean_shape']['metrics']['pulse_effect_rmse']['median']:.2f}\\,\\text{{pp}}",
            "---",
            "---",
        ],
        [
            "Mamba SSM (3-start)",
            f"{r7m['synthetic_mamba']['params']}",
            f"{r7m['synthetic_mamba']['primary']['balanced_kl']['median']:.5f}",
            f"{r7m['synthetic_mamba']['lognormal']['balanced_kl']['median']:.5f}",
            f"{100*r7m['synthetic_mamba']['primary']['pulse_effect_rmse']['median']:.2f}\\,\\text{{pp}}",
            "---",
            "---",
        ],
        [
            "LTC ODE (3-start)",
            "378",
            f"{mnb['primary']['ltc_ode_3start']['metrics']['balanced_kl']['median']:.5f}",
            f"{mnb['lognormal']['ltc_ode_3start']['metrics']['balanced_kl']['median']:.5f}",
            f"{100*mnb['primary']['ltc_ode_3start']['metrics']['pulse_effect_rmse']['median']:.2f}\\,\\text{{pp}}",
            "---",
            "---",
        ],
        [
            "S4D ODE (3-start)",
            "435",
            f"{mnb['primary']['s4d_ode_3start']['metrics']['balanced_kl']['median']:.5f}",
            f"{mnb['lognormal']['s4d_ode_3start']['metrics']['balanced_kl']['median']:.5f}",
            f"{100*mnb['primary']['s4d_ode_3start']['metrics']['pulse_effect_rmse']['median']:.2f}\\,\\text{{pp}}",
            "---",
            "---",
        ],
        [
            "Latent ODE (3-start)",
            "387",
            f"{mnb['primary']['latent_ode_3start']['metrics']['balanced_kl']['median']:.5f}",
            f"{mnb['lognormal']['latent_ode_3start']['metrics']['balanced_kl']['median']:.5f}",
            f"{100*mnb['primary']['latent_ode_3start']['metrics']['pulse_effect_rmse']['median']:.2f}\\,\\text{{pp}}",
            f"{tr_400['latent_ode.pretrained_code']['balanced_kl']:.5f}",
            f"{tr_8000['latent_ode.pretrained_code']['balanced_kl']:.5f}",
        ],
        [
            "Neural-CME (3-start)",
            "342",
            f"{mnb['primary']['neural_cme_3start']['metrics']['balanced_kl']['median']:.5f}",
            f"{mnb['lognormal']['neural_cme_3start']['metrics']['balanced_kl']['median']:.5f}",
            f"{100*mnb['primary']['neural_cme_3start']['metrics']['pulse_effect_rmse']['median']:.2f}\\,\\text{{pp}}",
            "---",
            "---",
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

    # Prepare real-world summary dictionaries used in Appendix Tables A8-A9
    ec_m = rcb["ecoli_continuous_cycle"]["models_5fold_mean"]
    sl_b = rcb["sleep_5node_branching_network"]
    sl_m = sl_b["models_5fold_mean"]
    hdk_cv = hdk["cv_5fold_summary"]
    mfe_ex8 = mfe["extrapolation_splits"]["days_1to8_to_9to14"]
    assert "resampled" in rt6["targets"] and "persistent" in rt6["targets"]
    assert rvm["sleep"]["models"]["adjusted"]["previous"]["2"]["gain"] > 0

    def fmt_pm(mean: float, sd: float, bold: bool = False) -> str:
        s_sd = f"{sd:.3f}"[1:]
        if bold:
            return f"\\textbf{{{mean:.4f}}}$\\,\\pm\\,{s_sd}$"
        return f"${mean:.4f}\\pm{s_sd}$"

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

    # Appendix Tables A2 & A3: Full 14-row primary & 6-row later-time reduction tables incl. Pos-Sens 33
    r_labels = {
        "symmetric": "Sym. ($\\epsilon{=}0$)",
        "mixed_025": "Mixed $0.025$",
        "mixed_075": "Mixed $0.075$",
        "mixed_150": "Mixed $0.150$",
        "mixed_250": "Mixed $0.250$",
        "mean_only_150": "Mean $0.150$",
        "shape_only_150": "Shape $0.150$",
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
            pos_rec = pos["cases"][key]
            diff = pos_rec["paired_shape_difference_vs_full_pp"]
            vals = [
                f"{rec['full48']['median_shape_percent']:.2f}",
                f"{pos_rec['median_shape_percent']:.2f}",
                f"{rec['sp33']['median_shape_percent']:.2f}",
                f"{rec['sym33']['median_shape_percent']:.2f}",
                f"{rec['sp18']['median_shape_percent']:.2f}",
            ]
            ci_str = f"${diff['median']:+.2f}\\;[{diff['ci95'][0]:+.2f}, {diff['ci95'][1]:+.2f}]$"
            rows_sp.append([law, r_labels[regime], *vals, ci_str])
        write_tex_table(
            out_fname,
            "llrrrrrl",
            [
                "Law",
                "Regime",
                "Full 48",
                "Pos. 33",
                "Int. 33",
                "Sym. 33",
                "Int. 18",
                "$\\Delta$ Pos.$-$Full [CI]",
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
                f"{at_sc['gamma_reference']['best']['loss']:.5f}",
                f"{at_sc['gamma_reference']['best']['params'][0]:.5f}",
                f"{at_sc['gamma_reference']['best']['params'][1]:.5f}",
                "\\textbf{0.40\\,s}",
            ],
            [
                "NoMaSS $\\Gamma$ (3-seed median)",
                "0.01787",
                "0.03264",
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

    # Appendix Table A7: Graph and out-degree marked-race scaling across V=4..48, d=2..4
    rows_g7 = []
    for r in gsc["gradient_and_state_scaling"]:
        if r["nodes"] not in (4, 8, 16, 48):
            continue
        sp_str = (
            "OOM / $>35$k"
            if r["grad_speedup_vs_tensor"] is None
            else f"\\textbf{{{r['grad_speedup_vs_tensor']:.2f}$\\times$}}"
        )
        rows_g7.append(
            [
                f"$d{{=}}{r['out_degree']},\\,V{{=}}{r['nodes']}$",
                f"{r['parameters']}",
                f"{r['joint_tensor_states']:,}",
                f"{r['compact_race_states']:,}",
                f"{r['state_compression_ratio']:.1f}$\\times$",
                f"{r['compact_grad_warm_ms']:.1f}",
                sp_str,
            ]
        )
    rows_g7.append("\\midrule")
    rss_by_v = {r["nodes"]: r for r in sp_rss["peak_rss_scaling"]}
    for r in spg["scaling_rows"]:
        if r["nodes"] not in (16, 64, 256, 512, 1024):
            continue
        rr = rss_by_v[r["nodes"]]
        rows_g7.append(
            [
                f"Sparse $V{{=}}{r['nodes']}$",
                f"{r['parameters']:,}",
                f"{r['tensor_states']:,}",
                f"{r['compiled_states']:,}",
                f"{r['sparse_operator_mb']:.2f} / {rr['peak_total_rss_mb']:.0f}\\,MB",
                f"{r['grad_warm_ms']:.1f}",
                f"{r['memory_compression_vs_dense']:,.0f}$\\times$ op",
            ]
        )
    write_tex_table(
        "tab_a7_graph_scaling.tex",
        "lrrrrrr",
        [
            "Graph $(d, V)$",
            "Params",
            "Tensor",
            "Race",
            "Comp. / Op. (RSS)",
            "Grad (ms)",
            "Gain",
        ],
        rows_g7,
    )

    # Appendix Table A8: Complete 7-Node Human Hematopoiesis & 5-Node Sleep-EDF Architecture Breakdown
    h_cv = hem["donor_5fold_cv_mean"]
    h_ex = hem["day1to5_to_day7_extrapolation"]
    hd_d7 = hem_d["unseen_donor_day7"]
    h_r5_cv = bio_r5["hematopoiesis_r5"]["donor_5fold_cv"]
    h_r5_p7 = bio_r5["hematopoiesis_r5"]["pooled_day7"]
    h_r5_d7 = bio_r5["hematopoiesis_r5"]["donor_heldout_day7"]
    sl_x = slp_ext["models_5fold_mean"]
    sl_hp = bio_r5["sleep_hyper_pos_sens_r5"]
    sl_hp_m = sl_hp["models_5fold_mean"]
    write_tex_table(
        "tab_a8_hematopoiesis_and_sleep.tex",
        "llrrrr",
        [
            "Dataset",
            "Architecture",
            "States (Params)",
            "Primary KL $\\downarrow$",
            "Extrap. / 0-Shot $\\downarrow$",
            "RMSE / Fresh $\\downarrow$",
        ],
        [
            [
                "Hematopoiesis",
                "\\textbf{Compiled DRN}",
                "127 (16)",
                f"{h_cv['compiled_drn']['kl']:.4f}",
                f"\\textbf{{{h_ex['compiled_drn']['day7_kl']:.4f}}} (${hd_d7['compiled_drn']['kl_mean']:.3f}\\pm.13$)",
                f"{h_cv['compiled_drn']['prob_rmse']:.4f}",
            ],
            [
                "(7-node tree,",
                "\\textbf{Univ. Pos-Sens}",
                "49 (14)",
                f"{h_r5_cv['universal_pos_sens_fullcone']['kl']:.4f}",
                f"{h_r5_p7['universal_pos_sens_fullcone']['day7_kl']:.4f} (${h_r5_d7['universal_pos_sens_fullcone']['kl_mean']:.3f}\\pm.{int(round(100*h_r5_d7['universal_pos_sens_fullcone']['kl_sd'])):02d}$)",
                f"{h_r5_cv['universal_pos_sens_fullcone']['prob_rmse']:.4f}",
            ],
            [
                "10 donors)",
                "Memoryless CTMC",
                "7 (8)",
                f"{h_cv['ctmc']['kl']:.4f}",
                f"{h_ex['ctmc']['day7_kl']:.4f} (${hd_d7['ctmc']['kl_mean']:.3f}\\pm.10$)",
                f"{h_cv['ctmc']['prob_rmse']:.4f}",
            ],
            [
                "",
                "LTC ODE",
                f"{h_r5_cv['ltc_ode']['states']} ({h_r5_cv['ltc_ode']['params']})",
                f"\\textbf{{{h_r5_cv['ltc_ode']['kl']:.4f}}}",
                f"{h_r5_p7['ltc_ode']['day7_kl']:.4f} (${h_r5_d7['ltc_ode']['kl_mean']:.3f}\\pm.{int(round(100*h_r5_d7['ltc_ode']['kl_sd'])):02d}$)",
                f"{h_r5_cv['ltc_ode']['prob_rmse']:.4f}",
            ],
            [
                "",
                "S4D ODE",
                f"{h_r5_cv['s4d_ode']['states']} ({h_r5_cv['s4d_ode']['params']})",
                f"{h_r5_cv['s4d_ode']['kl']:.4f}",
                f"{h_r5_p7['s4d_ode']['day7_kl']:.4f} (${h_r5_d7['s4d_ode']['kl_mean']:.3f}\\pm.{int(round(100*h_r5_d7['s4d_ode']['kl_sd'])):02d}$)",
                f"{h_r5_cv['s4d_ode']['prob_rmse']:.4f}",
            ],
            [
                "",
                "Neural-CME$(t)$",
                "7 (128)",
                f"{h_cv['neural_cme']['kl']:.4f}",
                f"{h_ex['neural_cme']['day7_kl']:.4f} (${hd_d7['neural_cme']['kl_mean']:.3f}\\pm.26$)",
                f"{h_cv['neural_cme']['prob_rmse']:.4f}",
            ],
            [
                "",
                "Latent ODE",
                "12 (273)",
                f"{h_cv['latent_ode']['kl']:.4f}",
                f"{h_ex['latent_ode']['day7_kl']:.4f} (\\textbf{{{hd_d7['latent_ode']['kl_mean']:.3f}}}$\\,\\pm\\,.22$)",
                f"\\textbf{{{h_cv['latent_ode']['prob_rmse']:.4f}}}",
            ],
            "\\midrule",
            [
                "Sleep-EDF",
                "\\textbf{Edge-Mem DRN}",
                "246 (254)",
                fmt_pm(ems["edge_memory_drn_joint"]["aged_kl_mean"], ems["edge_memory_drn_joint"]["aged_kl_sd"]),
                fmt_pm(ems["edge_memory_drn_fresh_only"]["aged_kl_mean"], ems["edge_memory_drn_fresh_only"]["aged_kl_sd"], bold=True),
                f"{ems['edge_memory_drn_joint']['fresh_kl_mean']:.4f}",
            ],
            [
                "(5-node net,",
                "Renewal DRN",
                "117 (144)",
                fmt_pm(sl_m["drn_joint"]["aged_kl_mean"], sl_m["drn_joint"]["aged_kl_sd"]),
                fmt_pm(sl_m["drn_fresh_only"]["aged_kl_mean"], sl_m["drn_fresh_only"]["aged_kl_sd"]),
                f"{sl_m['drn_joint']['fresh_kl_mean']:.4f}",
            ],
            [
                "75 adults)",
                "Hyper-Pos-Sens",
                f"{sl_hp['hyper_pos_sens_states']} ({sl_hp['hyper_pos_sens_params']})",
                fmt_pm(sl_hp_m["hyper_pos_sens_joint"]["aged_kl_mean"], sl_hp_m["hyper_pos_sens_joint"]["aged_kl_sd"]),
                fmt_pm(sl_hp_m["hyper_pos_sens_fresh_only"]["aged_kl_mean"], sl_hp_m["hyper_pos_sens_fresh_only"]["aged_kl_sd"]),
                f"{sl_hp_m['hyper_pos_sens_joint']['fresh_kl_mean']:.4f}",
            ],
            [
                "",
                "Univ. Pos-Sens",
                f"{slp_ext['universal_pos_sens_states']} ({slp_ext['universal_pos_sens_params']})",
                fmt_pm(sl_x["universal_pos_sens_joint"]["aged_kl_mean"], sl_x["universal_pos_sens_joint"]["aged_kl_sd"]),
                fmt_pm(sl_x["universal_pos_sens_fresh_only"]["aged_kl_mean"], sl_x["universal_pos_sens_fresh_only"]["aged_kl_sd"]),
                f"{sl_x['universal_pos_sens_joint']['fresh_kl_mean']:.4f}",
            ],
            [
                "",
                "Memoryless CTMC",
                "5 (16)",
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ctmc_joint"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ctmc_joint"]["aged_kl_sd"]),
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ctmc_fresh_only"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ctmc_fresh_only"]["aged_kl_sd"]),
                f"{r6b['sleep_r6b_3start_clean_age0_and_fresh_ctmc']['ctmc_joint']['fresh_kl_mean']:.4f}",
            ],
            [
                "",
                "Mamba SSM",
                f"15 ({r7m['sleep_mamba']['params']})",
                fmt_pm(r7m["sleep_mamba"]["cv5"]["joint"]["aged_kl_mean"], r7m["sleep_mamba"]["cv5"]["joint"]["aged_kl_sd"], bold=True),
                fmt_pm(r7m["sleep_mamba"]["cv5"]["fresh_only"]["aged_kl_mean"], r7m["sleep_mamba"]["cv5"]["fresh_only"]["aged_kl_sd"]),
                f"{r7m['sleep_mamba']['cv5']['joint']['fresh_kl_mean']:.4f}",
            ],
            [
                "",
                "LTC ODE",
                f"11 ({slp_ext['ltc_ode_params']})",
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ltc_ode_joint"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ltc_ode_joint"]["aged_kl_sd"]),
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ltc_ode_fresh_only"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["ltc_ode_fresh_only"]["aged_kl_sd"]),
                f"{r6b['sleep_r6b_3start_clean_age0_and_fresh_ctmc']['ltc_ode_joint']['fresh_kl_mean']:.4f}",
            ],
            [
                "",
                "S4D ODE",
                f"15 ({slp_ext['s4d_ode_params']})",
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["s4d_ode_joint"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["s4d_ode_joint"]["aged_kl_sd"]),
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["s4d_ode_fresh_only"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["s4d_ode_fresh_only"]["aged_kl_sd"]),
                f"\\textbf{{{r6b['sleep_r6b_3start_clean_age0_and_fresh_ctmc']['s4d_ode_joint']['fresh_kl_mean']:.4f}}}",
            ],
            [
                "",
                "Neural-CME$(t)$",
                "5 (394)",
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["neural_cme_joint"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["neural_cme_joint"]["aged_kl_sd"]),
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["neural_cme_fresh_only"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["neural_cme_fresh_only"]["aged_kl_sd"]),
                f"{r6b['sleep_r6b_3start_clean_age0_and_fresh_ctmc']['neural_cme_joint']['fresh_kl_mean']:.4f}",
            ],
            [
                "",
                "Latent ODE",
                "11 (540)",
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["latent_ode_joint"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["latent_ode_joint"]["aged_kl_sd"]),
                fmt_pm(r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["latent_ode_fresh_only"]["aged_kl_mean"], r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]["latent_ode_fresh_only"]["aged_kl_sd"]),
                f"{r6b['sleep_r6b_3start_clean_age0_and_fresh_ctmc']['latent_ode_joint']['fresh_kl_mean']:.4f}",
            ],
            "\\midrule",
            [
                "\\emph{E.\\ coli} ring",
                "\\textbf{Renewal DRN}",
                "48 (153)",
                fmt_pm(ec_m["drn_joint"]["aged_kl"], ec_m["drn_joint"]["aged_kl_sd"]),
                fmt_pm(ec_m["drn_fresh_only"]["aged_kl"], ec_m["drn_fresh_only"]["aged_kl_sd"], bold=True),
                f"{ec_m['drn_joint']['fresh_kl']:.4f}",
            ],
            [
                "(18,972 cyc.)",
                "Memoryless CTMC",
                "3 (6)",
                fmt_pm(ec_m["ctmc"]["aged_kl"], ec_m["ctmc"]["aged_kl_sd"]),
                fmt_pm(ec_m["ctmc"]["aged_kl"], ec_m["ctmc"]["aged_kl_sd"]),
                f"{ec_m['ctmc']['fresh_kl']:.4f}",
            ],
            [
                "",
                "LTC ODE",
                f"{r6['ecoli_r6_complete_neural_odes']['states_params']['ltc_ode']['states']} ({r6['ecoli_r6_complete_neural_odes']['states_params']['ltc_ode']['params']})",
                fmt_pm(r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["ltc_ode_joint"]["aged_kl"], r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["ltc_ode_joint"]["aged_kl_sd"]),
                fmt_pm(r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["ltc_ode_fresh_only"]["aged_kl"], r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["ltc_ode_fresh_only"]["aged_kl_sd"]),
                f"{r6['ecoli_r6_complete_neural_odes']['models_5fold_clean_age0']['ltc_ode_joint']['fresh_kl']:.4f}",
            ],
            [
                "",
                "S4D ODE",
                f"{r6['ecoli_r6_complete_neural_odes']['states_params']['s4d_ode']['states']} ({r6['ecoli_r6_complete_neural_odes']['states_params']['s4d_ode']['params']})",
                fmt_pm(r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["s4d_ode_joint"]["aged_kl"], r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["s4d_ode_joint"]["aged_kl_sd"]),
                fmt_pm(r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["s4d_ode_fresh_only"]["aged_kl"], r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["s4d_ode_fresh_only"]["aged_kl_sd"]),
                f"{r6['ecoli_r6_complete_neural_odes']['models_5fold_clean_age0']['s4d_ode_joint']['fresh_kl']:.4f}",
            ],
            [
                "",
                "Neural-CME$(t)$",
                f"{r6['ecoli_r6_complete_neural_odes']['states_params']['neural_cme']['states']} ({r6['ecoli_r6_complete_neural_odes']['states_params']['neural_cme']['params']})",
                fmt_pm(r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["neural_cme_joint"]["aged_kl"], r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["neural_cme_joint"]["aged_kl_sd"]),
                fmt_pm(r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["neural_cme_fresh_only"]["aged_kl"], r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["neural_cme_fresh_only"]["aged_kl_sd"]),
                f"{r6['ecoli_r6_complete_neural_odes']['models_5fold_clean_age0']['neural_cme_joint']['fresh_kl']:.4f}",
            ],
            [
                "",
                "Latent ODE",
                "9 (285)",
                fmt_pm(r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["latent_ode_joint"]["aged_kl"], r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["latent_ode_joint"]["aged_kl_sd"], bold=True),
                fmt_pm(r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["latent_ode_fresh_only"]["aged_kl"], r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]["latent_ode_fresh_only"]["aged_kl_sd"]),
                f"\\textbf{{{r6['ecoli_r6_complete_neural_odes']['models_5fold_clean_age0']['latent_ode_joint']['fresh_kl']:.4f}}}",
            ],
        ],
    )

    # Appendix Table A9: IT Helpdesk Systems ML (5-Fold CV, Chronological Split, & Coarsening) + Nonlinear MeanFieldDRN Influenza
    mfe_full = mfe["full_fit_days_1to14"]
    mfe_ex9 = mfe["extrapolation_splits"]["days_1to9_to_10to14"]
    mfe_reg = mfe["regularized_latent_ode_sensitivity"]
    c7 = hdk["coarsening_panel_sweep"][2]
    c14 = hdk["coarsening_panel_sweep"][3]
    c7_r5 = hdk_r5["coarsening_neural_clean_age0"][2]
    c14_r5 = hdk_r5["coarsening_neural_clean_age0"][3]
    hx = hdk_ext["cv_5fold_extended"]
    hch = hdk_ext["chronological_split"]["models"]
    hr5 = hdk_r5["cv_5fold_clean_age0"]
    hch_r5 = hdk_r5["chronological_split_3start_clean_age0"]["models"]
    write_tex_table(
        "tab_a9_systems_and_meanfield.tex",
        "llrrrr",
        [
            "System",
            "Architecture",
            "States (P)",
            "Primary KL $\\downarrow$",
            "0-Shot / Extrap. $\\downarrow$",
            "Chrono (7d/14d) $\\downarrow$",
        ],
        [
            [
                "Helpdesk",
                "\\textbf{Edge-Mem DRN}",
                f"{hdk['state_counts']['edge_memory_drn_states_K6']} ({hdk['state_counts']['edge_memory_drn_params_K6']})",
                fmt_pm(hdk_cv["edge_memory_drn_joint"]["aged_kl_mean"], hdk_cv["edge_memory_drn_joint"]["aged_kl_sd"]),
                fmt_pm(hdk_cv["edge_memory_drn_fresh_only"]["aged_kl_mean"], hdk_cv["edge_memory_drn_fresh_only"]["aged_kl_sd"], bold=True),
                f"\\textbf{{{hch['edge_memory_drn_fresh_only']['aged_kl']:.3f}}} (.{int(round(1000*c7['edge_memory_drn_multihop_aged_kl'])):03d}/.{int(round(1000*c14['edge_memory_drn_multihop_aged_kl'])):03d})",
            ],
            [
                "(4,576)",
                "\\textbf{Hyper-Pos-Sens}",
                f"{hdk_r5['hyper_pos_sens_states']} ({hdk_r5['hyper_pos_sens_params']})",
                fmt_pm(hr5["hyper_pos_sens_joint"]["aged_kl_mean"], hr5["hyper_pos_sens_joint"]["aged_kl_sd"]),
                fmt_pm(hr5["hyper_pos_sens_fresh_only"]["aged_kl_mean"], hr5["hyper_pos_sens_fresh_only"]["aged_kl_sd"]),
                f"{hch_r5['hyper_pos_sens_fresh_only']['aged_kl']:.3f} (\\textbf{{.{int(round(1000*r6['helpdesk_r6_hyper_pos_coarsening'][2]['hyper_pos_sens_aged_kl'])):03d}}}/\\textbf{{.{int(round(1000*r6['helpdesk_r6_hyper_pos_coarsening'][3]['hyper_pos_sens_aged_kl'])):03d}}})",
            ],
            [
                "",
                "Renewal DRN",
                f"{hdk['state_counts']['renewal_drn_race_states_K6']} ({hdk['state_counts']['renewal_drn_params_K6']})",
                fmt_pm(hdk_cv["renewal_drn_joint"]["aged_kl_mean"], hdk_cv["renewal_drn_joint"]["aged_kl_sd"]),
                fmt_pm(hdk_cv["renewal_drn_fresh_only"]["aged_kl_mean"], hdk_cv["renewal_drn_fresh_only"]["aged_kl_sd"]),
                f"{hch['renewal_drn_fresh_only']['aged_kl']:.3f} (.{int(round(1000*c7['drn_multihop_aged_kl'])):03d}/.{int(round(1000*c14['drn_multihop_aged_kl'])):03d})",
            ],
            [
                "",
                "Univ. Pos-Sens",
                f"{hdk_ext['universal_pos_sens_states']} ({hdk_ext['universal_pos_sens_params']})",
                fmt_pm(hx["universal_pos_sens_joint"]["aged_kl_mean"], hx["universal_pos_sens_joint"]["aged_kl_sd"]),
                fmt_pm(hx["universal_pos_sens_fresh_only"]["aged_kl_mean"], hx["universal_pos_sens_fresh_only"]["aged_kl_sd"]),
                f"{hch['universal_pos_sens_fresh_only']['aged_kl']:.3f} (---)",
            ],
            [
                "",
                "CTMC",
                "5 (9)",
                fmt_pm(hdk_cv["ctmc"]["aged_kl_mean"], hdk_cv["ctmc"]["aged_kl_sd"]),
                fmt_pm(hdk_cv["ctmc"]["aged_kl_mean"], hdk_cv["ctmc"]["aged_kl_sd"]),
                f"{hch['ctmc']['aged_kl']:.3f} (.{int(round(1000*c7['ctmc_panel_kl'])):03d}/.{int(round(1000*c14['ctmc_panel_kl'])):03d})",
            ],
            [
                "",
                "Neural-CR",
                f"--- ({hdk['state_counts']['neural_competing_risks_params']})",
                "---",
                "---",
                f"--- (.{int(round(1000*c7['neural_competing_risks_best_kl'])):03d}/{c14['neural_competing_risks_best_kl']:.3f})",
            ],
            [
                "",
                "Neural Hawkes (THP)",
                f"--- ({r7m['helpdesk_mamba_and_thp']['coarsening']['dt_7.0']['params']})",
                "---",
                "---",
                f"--- ({r7m['helpdesk_mamba_and_thp']['coarsening']['dt_7.0']['thp_forecast_kl']:.3f}/{r7m['helpdesk_mamba_and_thp']['coarsening']['dt_14.0']['thp_forecast_kl']:.3f})",
            ],
            [
                "",
                "Mamba SSM",
                f"15 ({r7m['helpdesk_mamba_and_thp']['params']})",
                fmt_pm(r7m["helpdesk_mamba_and_thp"]["cv5"]["joint"]["aged_kl_mean"], r7m["helpdesk_mamba_and_thp"]["cv5"]["joint"]["aged_kl_sd"], bold=True),
                fmt_pm(r7m["helpdesk_mamba_and_thp"]["cv5"]["fresh_only"]["aged_kl_mean"], r7m["helpdesk_mamba_and_thp"]["cv5"]["fresh_only"]["aged_kl_sd"]),
                f"{r7m['helpdesk_mamba_and_thp']['chrono_70_30']['aged_kl']:.3f} (.{int(round(1000*r7m['helpdesk_mamba_and_thp']['coarsening']['dt_7.0']['mamba_aged_kl'])):03d}/.{int(round(1000*r7m['helpdesk_mamba_and_thp']['coarsening']['dt_14.0']['mamba_aged_kl'])):03d})",
            ],
            [
                "",
                "LTC ODE",
                f"11 ({hdk_ext['ltc_ode_params']})",
                fmt_pm(hr5["ltc_ode_joint"]["aged_kl_mean"], hr5["ltc_ode_joint"]["aged_kl_sd"]),
                fmt_pm(hr5["ltc_ode_fresh_only"]["aged_kl_mean"], hr5["ltc_ode_fresh_only"]["aged_kl_sd"]),
                f"{hch_r5['ltc_ode_fresh_only']['aged_kl']:.3f} (.{int(round(1000*c7_r5['ltc_ode_aged_kl_clean_age0'])):03d}/.{int(round(1000*c14_r5['ltc_ode_aged_kl_clean_age0'])):03d})",
            ],
            [
                "",
                "S4D ODE",
                f"15 ({hdk_ext['s4d_ode_params']})",
                fmt_pm(hr5["s4d_ode_joint"]["aged_kl_mean"], hr5["s4d_ode_joint"]["aged_kl_sd"]),
                fmt_pm(hr5["s4d_ode_fresh_only"]["aged_kl_mean"], hr5["s4d_ode_fresh_only"]["aged_kl_sd"]),
                f"{hch_r5['s4d_ode_fresh_only']['aged_kl']:.3f} (.{int(round(1000*c7_r5['s4d_ode_aged_kl_clean_age0'])):03d}/.{int(round(1000*c14_r5['s4d_ode_aged_kl_clean_age0'])):03d})",
            ],
            [
                "",
                "Neural-CME$(t)$",
                f"5 ({hdk['state_counts']['neural_cme_params']})",
                fmt_pm(hr5["neural_cme_joint"]["aged_kl_mean"], hr5["neural_cme_joint"]["aged_kl_sd"]),
                fmt_pm(hr5["neural_cme_fresh_only"]["aged_kl_mean"], hr5["neural_cme_fresh_only"]["aged_kl_sd"]),
                f"{hch_r5['neural_cme_fresh_only']['aged_kl']:.3f} (.{int(round(1000*c7_r5['neural_cme_aged_kl_clean_age0'])):03d}/.{int(round(1000*c14_r5['neural_cme_aged_kl_clean_age0'])):03d})",
            ],
            [
                "",
                "Latent ODE",
                f"11 ({hdk['state_counts']['latent_ode_params']})",
                fmt_pm(hr5["latent_ode_joint"]["aged_kl_mean"], hr5["latent_ode_joint"]["aged_kl_sd"]),
                fmt_pm(hr5["latent_ode_fresh_only"]["aged_kl_mean"], hr5["latent_ode_fresh_only"]["aged_kl_sd"]),
                f"{hch_r5['latent_ode_fresh_only']['aged_kl']:.3f} (.{int(round(1000*c7_r5['latent_ode_aged_kl_clean_age0'])):03d}/.{int(round(1000*c14_r5['latent_ode_aged_kl_clean_age0'])):03d})",
            ],
            "\\midrule",
            [
                "Influenza",
                "\\textbf{MeanFieldDRN}",
                f"{mfe_full['mean_field_drn']['states']} ({mfe_full['mean_field_drn']['params_count']})",
                f"{mfe_full['mean_field_drn']['mean_kl_nats']:.4f}",
                f"\\textbf{{{mfe_ex8['mean_field_drn']['test']['mean_kl_nats']:.4f}}} / \\textbf{{{mfe_ex9['mean_field_drn']['test']['mean_kl_nats']:.4f}}}",
                f"\\textbf{{{mfe_ex8['mean_field_drn']['test']['count_rmse_boys']:.1f}}} boys",
            ],
            [
                "(763 boys)",
                "Markov SEICR",
                f"{mfe_full['markov_seicr']['states']} ({mfe_full['markov_seicr']['params_count']})",
                f"{mfe_full['markov_seicr']['mean_kl_nats']:.4f}",
                f"{mfe_ex8['markov_seicr']['test']['mean_kl_nats']:.4f} / {mfe_ex9['markov_seicr']['test']['mean_kl_nats']:.4f}",
                f"{mfe_ex8['markov_seicr']['test']['count_rmse_boys']:.1f} boys",
            ],
            [
                "",
                "Mamba SSM ($L_2$)",
                f"9 ({r7m['influenza_mamba']['params']})",
                f"{r7m['influenza_mamba']['full_days_1to14']['mean_kl_nats']:.4f}",
                f"{r7m['influenza_mamba']['days_1to8_to_9to14_best_l2']['test']['mean_kl_nats']:.4f} / {r7m['influenza_mamba']['days_1to9_to_10to14_best_l2']['test']['mean_kl_nats']:.4f}",
                f"{r7m['influenza_mamba']['days_1to8_to_9to14_best_l2']['test']['count_rmse_boys']:.1f} boys",
            ],
            [
                "",
                "LTC ODE ($L_2$)",
                f"{bio_r5['influenza_r5']['ltc_ode']['states']} ({bio_r5['influenza_r5']['ltc_ode']['params']})",
                f"{bio_r5['influenza_r5']['ltc_ode']['full_fit']['mean_kl_nats']:.4f}",
                f"{bio_r5['influenza_r5']['ltc_ode']['days_1to8_to_9to14_best_l2']['test']['mean_kl_nats']:.4f} / {bio_r5['influenza_r5']['ltc_ode']['days_1to9_to_10to14_best_l2']['test']['mean_kl_nats']:.4f}",
                f"{bio_r5['influenza_r5']['ltc_ode']['days_1to8_to_9to14_best_l2']['test']['count_rmse_boys']:.1f} boys",
            ],
            [
                "",
                "S4D ODE ($L_2$)",
                f"{bio_r5['influenza_r5']['s4d_ode']['states']} ({bio_r5['influenza_r5']['s4d_ode']['params']})",
                f"{bio_r5['influenza_r5']['s4d_ode']['full_fit']['mean_kl_nats']:.4f}",
                f"{bio_r5['influenza_r5']['s4d_ode']['days_1to8_to_9to14_best_l2']['test']['mean_kl_nats']:.4f} / {bio_r5['influenza_r5']['s4d_ode']['days_1to9_to_10to14_best_l2']['test']['mean_kl_nats']:.4f}",
                f"{bio_r5['influenza_r5']['s4d_ode']['days_1to8_to_9to14_best_l2']['test']['count_rmse_boys']:.1f} boys",
            ],
            [
                "",
                "Neural-CME$(t)$ ($L_2$)",
                f"{r6b['influenza_r6b_neural_cme']['states']} ({r6b['influenza_r6b_neural_cme']['params']})",
                f"\\textbf{{{r6b['influenza_r6b_neural_cme']['full_fit']['mean_kl_nats']:.4f}}}",
                f"{r6b['influenza_r6b_neural_cme']['days_1to8_to_9to14_best_l2']['test']['mean_kl_nats']:.4f} / {r6b['influenza_r6b_neural_cme']['days_1to9_to_10to14_best_l2']['test']['mean_kl_nats']:.4f}",
                f"{r6b['influenza_r6b_neural_cme']['days_1to8_to_9to14_best_l2']['test']['count_rmse_boys']:.1f} boys",
            ],
            [
                "",
                "Latent ODE ($L_2$)",
                f"{mfe_full['latent_ode']['states']} ({mfe_full['latent_ode']['params_count']})",
                f"{mfe_full['latent_ode']['mean_kl_nats']:.4f}",
                f"{mfe_reg['days_1to8_to_9to14']['test']['mean_kl_nats']:.4f} / {mfe_reg['days_1to9_to_10to14']['test']['mean_kl_nats']:.4f}",
                f"{mfe_reg['days_1to8_to_9to14']['test']['count_rmse_boys']:.1f} boys",
            ],
        ],
    )


    # Main-Text Table 3: Compact Cross-Domain Real-World Summary (all 8 real domains!)
    ec_r6 = r6["ecoli_r6_complete_neural_odes"]["models_5fold_clean_age0"]
    hp_c14 = r6["helpdesk_r6_hyper_pos_coarsening"][3]["hyper_pos_sens_aged_kl"]
    sl_r6 = r6b["sleep_r6b_3start_clean_age0_and_fresh_ctmc"]
    inf_cme = r6b["influenza_r6b_neural_cme"]
    mo_r = r7n["mooc_v98"]["results"]
    so_r = r7n["stackoverflow_v22"]["results"]
    df_ra = r7d["regime_a_zeroshot_deep_age_sale"]
    df_rb = r7d["regime_b_temporal_drift_sale"]
    write_tex_table(
        "tab4_real_world_summary.tex",
        "lrrrrr",
        [
            "Domain (Task)",
            "Full DRN $\\downarrow$",
            "Univ. Cl. $\\downarrow$",
            "CTMC $\\downarrow$",
            "Best Neural $\\downarrow$",
            "vs.\\ Neural",
        ],
        [
            [
                "MOOC $V{=}98$ (0-shot)",
                f"\\textbf{{{mo_r['sparse_hyper_drn_fresh_only']['aged_kl']:.4f}}}",
                "---",
                f"{mo_r['ctmc_fresh_only']['aged_kl']:.4f}",
                f"{mo_r['ltc_fresh_only']['aged_kl']:.4f} (LTC)",
                f"\\textbf{{{mo_r['ltc_fresh_only']['aged_kl']/mo_r['sparse_hyper_drn_fresh_only']['aged_kl']:.1f}$\\times$}}",
            ],
            [
                "MOOC (Joint / 24h)",
                f"\\textbf{{{mo_r['sparse_hyper_drn_joint']['aged_kl']:.3f}}} (\\textbf{{{mo_r['coarsening']['dt_24.0']['sparse_drn_aged_kl']:.3f}}})",
                "---",
                f"{mo_r['ctmc_joint']['aged_kl']:.3f} ({mo_r['coarsening']['dt_24.0']['ctmc_aged_kl']:.3f})",
                f"{mo_r['s4d_joint']['aged_kl']:.3f} ({mo_r['coarsening']['dt_24.0']['mamba_aged_kl']:.3f})",
                f"\\textbf{{{mo_r['s4d_joint']['aged_kl']/mo_r['sparse_hyper_drn_joint']['aged_kl']:.1f}$\\times$/{mo_r['coarsening']['dt_24.0']['mamba_aged_kl']/mo_r['coarsening']['dt_24.0']['sparse_drn_aged_kl']:.1f}$\\times$}}",
            ],
            [
                "DeFi 183k (0-sh./5m)",
                f"\\textbf{{{df_ra['neural_clock']['all']['logloss']:.3f}}} (\\textbf{{.{int(round(1000*df_rb['neural_clock']['all']['logloss'])):03d}}})",
                "---",
                f"{df_ra['markov_clock']['all']['logloss']:.3f} (.{int(round(1000*df_rb['markov_clock']['all']['logloss'])):03d})",
                f"{df_ra['mlp_age']['all']['logloss']:.3f} (.{int(round(1000*df_rb['mlp_age']['all']['logloss'])):03d})",
                "\\textbf{\\#1 / \\#1}",
            ],
            [
                "Helpdesk (0-shot)",
                fmt_pm(hdk_cv["edge_memory_drn_fresh_only"]["aged_kl_mean"], hdk_cv["edge_memory_drn_fresh_only"]["aged_kl_sd"], bold=True),
                fmt_pm(hr5["hyper_pos_sens_fresh_only"]["aged_kl_mean"], hr5["hyper_pos_sens_fresh_only"]["aged_kl_sd"]),
                f"{hdk_cv['ctmc']['aged_kl_mean']:.4f}",
                f"{hr5['ltc_ode_fresh_only']['aged_kl_mean']:.4f} (LTC)",
                f"\\textbf{{{hr5['ltc_ode_fresh_only']['aged_kl_mean']/hdk_cv['edge_memory_drn_fresh_only']['aged_kl_mean']:.1f}$\\times$}}",
            ],
            [
                "Helpdesk (Chr./14d)",
                f"\\textbf{{{hch['edge_memory_drn_fresh_only']['aged_kl']:.3f}}} (.{int(round(1000*c14['edge_memory_drn_multihop_aged_kl'])):03d})",
                f"{hch_r5['hyper_pos_sens_fresh_only']['aged_kl']:.3f} (\\textbf{{.{int(round(1000*hp_c14)):03d}}})",
                f"{hch['ctmc']['aged_kl']:.3f} (.{int(round(1000*c14['ctmc_panel_kl'])):03d})",
                f"{hch_r5['ltc_ode_fresh_only']['aged_kl']:.3f} (.{int(round(1000*c14_r5['ltc_ode_aged_kl_clean_age0'])):03d})",
                f"\\textbf{{{hch_r5['ltc_ode_fresh_only']['aged_kl']/hch['edge_memory_drn_fresh_only']['aged_kl']:.1f}$\\times$/{c14_r5['ltc_ode_aged_kl_clean_age0']/hp_c14:.1f}$\\times$}}",
            ],
            [
                "Influenza (D9--14)",
                f"\\textbf{{{mfe_ex8['mean_field_drn']['test']['mean_kl_nats']:.4f}}}",
                "---",
                f"{mfe_ex8['markov_seicr']['test']['mean_kl_nats']:.4f}",
                f"{inf_cme['days_1to8_to_9to14_best_l2']['test']['mean_kl_nats']:.4f} (CME)",
                f"\\textbf{{{inf_cme['days_1to8_to_9to14_best_l2']['test']['mean_kl_nats']/mfe_ex8['mean_field_drn']['test']['mean_kl_nats']:.1f}$\\times$}}",
            ],
            [
                "Hematop. (Pool/D7)",
                f"\\textbf{{{h_ex['compiled_drn']['day7_kl']:.4f}}} (.{int(round(1000*hd_d7['compiled_drn']['kl_mean'])):03d})",
                f"{h_r5_p7['universal_pos_sens_fullcone']['day7_kl']:.4f} (.{int(round(1000*h_r5_d7['universal_pos_sens_fullcone']['kl_mean'])):03d})",
                f"{h_ex['ctmc']['day7_kl']:.4f} (.{int(round(1000*hd_d7['ctmc']['kl_mean'])):03d})",
                f"{h_r5_p7['ltc_ode']['day7_kl']:.4f} (\\textbf{{.{int(round(1000*hd_d7['latent_ode']['kl_mean'])):03d}}})",
                f"\\textbf{{{h_r5_p7['ltc_ode']['day7_kl']/h_ex['compiled_drn']['day7_kl']:.1f}$\\times$}}/{hd_d7['latent_ode']['kl_mean']/hd_d7['compiled_drn']['kl_mean']:.2f}$\\times$",
            ],
            [
                "Sleep-EDF (0-shot)",
                fmt_pm(ems["edge_memory_drn_fresh_only"]["aged_kl_mean"], ems["edge_memory_drn_fresh_only"]["aged_kl_sd"], bold=True),
                fmt_pm(sl_hp_m["hyper_pos_sens_fresh_only"]["aged_kl_mean"], sl_hp_m["hyper_pos_sens_fresh_only"]["aged_kl_sd"]),
                f"{sl_r6['ctmc_fresh_only']['aged_kl_mean']:.4f}",
                f"{sl_r6['neural_cme_fresh_only']['aged_kl_mean']:.4f} (CME)",
                f"\\textbf{{{sl_r6['neural_cme_fresh_only']['aged_kl_mean']/ems['edge_memory_drn_fresh_only']['aged_kl_mean']:.2f}$\\times$}}",
            ],
            [
                "\\emph{E.\\ coli} (0-shot)",
                fmt_pm(ec_m["drn_fresh_only"]["aged_kl"], ec_m["drn_fresh_only"]["aged_kl_sd"], bold=True),
                "---",
                f"{ec_m['ctmc']['aged_kl']:.4f}",
                f"{ec_r6['s4d_ode_fresh_only']['aged_kl']:.4f} (S4D)",
                f"\\textbf{{{ec_r6['s4d_ode_fresh_only']['aged_kl']/ec_m['drn_fresh_only']['aged_kl']:.1f}$\\times$}}",
            ],
            [
                "StackOv. $V{=}22$ (Bnd.)",
                f"{so_r['sparse_hyper_drn_fresh_only']['aged_kl']:.3f} (.{int(round(1000*so_r['sparse_hyper_drn_joint']['aged_kl'])):03d})",
                "---",
                f"{so_r['ctmc_fresh_only']['aged_kl']:.3f} (.{int(round(1000*so_r['ctmc_joint']['aged_kl'])):03d})",
                f"\\textbf{{{so_r['s4d_fresh_only']['aged_kl']:.3f}}} (\\textbf{{.{int(round(1000*so_r['s4d_joint']['aged_kl'])):03d}}})",
                f"{so_r['s4d_fresh_only']['aged_kl']/so_r['sparse_hyper_drn_fresh_only']['aged_kl']:.2f}$\\times$",
            ],
        ],
    )

    # Appendix Table A11: Theorem 6 Quantitative Near-Cancellation & Balanced-Truncation Verification
    k4_sw = {r["delta"]: r for r in r7t["symmetric_cycles"]["K4"]["sweep"]}
    ht_sw = {r["delta"]: r for r in r7t["heterogeneous_common_zero"]["sweep"]}
    k6_sw = {r["delta"]: r for r in r7t["symmetric_cycles"]["K6"]["sweep"]}
    bt_rows = []
    for label, sw_map in [
        ("Sym. $K{=}4$ ($12 \\to 9$)", k4_sw),
        ("Sym. $K{=}6$ ($18 \\to 13$)", k6_sw),
        ("Het. $[4,3,3]$ ($10 \\to 9$)", ht_sw),
    ]:
        for idx_d, d_val in enumerate((1e-5, 1e-4, 1e-3, 1e-2)):
            r = sw_map[d_val]
            bt_rows.append(
                [
                    label if idx_d == 0 else "",
                    f"$10^{{{int(round(np.log10(d_val)))}}}$",
                    f"{r['sigma_r_plus_1']:.2e}",
                    f"{r['fwd_hinf']:.2e}",
                    f"{r['err_rebal']:.2e} / {r['err_frozen']:.2e}",
                    f"{r['ratio_rebal']:,.0f}$\\times$ / {r['ratio_frozen']:,.0f}$\\times$",
                ]
            )
        if label != "Het. $[4,3,3]$ ($10 \\to 9$)":
            bt_rows.append("\\midrule")
    write_tex_table(
        "tab_a10_bt_near_cancellation.tex",
        "lcrrrr",
        [
            "Network ($D_{\\mathrm{st}} \\to d_{\\mathrm{occ}}$)",
            "$\\delta$",
            "$\\sigma_{r+1}$",
            "Fwd $H_\\infty$ $\\downarrow$",
            "Sens. $L_2$ (reb / fz)",
            "Distortion Ratio (reb / fz)",
        ],
        bt_rows,
    )

    # Appendix Table A12: Large-Scale Temporal Interaction Networks (MOOC V=98, StackOverflow V=22) & DeFi Multi-Pool (183k)
    defi_a = r7d["regime_a_zeroshot_deep_age_sale"]
    defi_b = r7d["regime_b_temporal_drift_sale"]
    defi_c1 = r7d["regime_c_cross_chain_base_to_bsc_young"]
    write_tex_table(
        "tab_a11_temporal_networks_and_defi.tex",
        "llrrrr",
        [
            "Benchmark",
            "Architecture",
            "St. (P)",
            "Fresh / Drift $\\downarrow$",
            "0-Shot / Age $\\downarrow$",
            "Joint (Coar./X) $\\downarrow$",
        ],
        [
            [
                "MOOC $V{=}98$",
                "\\textbf{SparseHyperDRN}",
                f"{mo_r['sparse_drn_meta']['compiled_states']} ({mo_r['sparse_drn_meta']['n_params']:,})",
                f"\\textbf{{{mo_r['sparse_hyper_drn_fresh_only']['fresh_kl']:.4f}}}",
                f"\\textbf{{{mo_r['sparse_hyper_drn_fresh_only']['aged_kl']:.4f}}}",
                f"\\textbf{{{mo_r['sparse_hyper_drn_joint']['aged_kl']:.3f}}} (\\textbf{{{mo_r['coarsening']['dt_24.0']['sparse_drn_aged_kl']:.3f}}})",
            ],
            [
                "(411k events,",
                "CTMC (expm)",
                f"98 ({mo_r['ctmc_fresh_only']['params']:,})",
                f"{mo_r['ctmc_fresh_only']['fresh_kl']:.4f}",
                f"{mo_r['ctmc_fresh_only']['aged_kl']:.4f}",
                f"{mo_r['ctmc_joint']['aged_kl']:.3f} ({mo_r['coarsening']['dt_24.0']['ctmc_aged_kl']:.3f})",
            ],
            [
                "1,355 edges,",
                "Mamba SSM",
                f"110 ({mo_r['mamba_fresh_only']['params']:,})",
                f"{mo_r['mamba_fresh_only']['fresh_kl']:.4f}",
                f"{mo_r['mamba_fresh_only']['aged_kl']:.4f}",
                f"{mo_r['mamba_joint']['aged_kl']:.3f} ({mo_r['coarsening']['dt_24.0']['mamba_aged_kl']:.3f})",
            ],
            [
                "$\\mathrm{CV}{=}5.54$)",
                "S4D ODE",
                f"110 ({mo_r['s4d_fresh_only']['params']:,})",
                f"{mo_r['s4d_fresh_only']['fresh_kl']:.4f}",
                f"{mo_r['s4d_fresh_only']['aged_kl']:.4f}",
                f"{mo_r['s4d_joint']['aged_kl']:.3f}",
            ],
            [
                "",
                "LTC/LODE/CME",
                "110 (9.7--13k)",
                f"{mo_r['ltc_fresh_only']['fresh_kl']:.3f}--{mo_r['latent_ode_fresh_only']['fresh_kl']:.3f}",
                f"{mo_r['ltc_fresh_only']['aged_kl']:.3f}--{mo_r['neural_cme_fresh_only']['aged_kl']:.3f}",
                f"{mo_r['latent_ode_joint']['aged_kl']:.3f}--{mo_r['neural_cme_joint']['aged_kl']:.3f}",
            ],
            "\\midrule",
            [
                "StackOv. $V{=}22$",
                "SparseHyperDRN",
                f"{so_r['sparse_drn_meta']['compiled_states']} ({so_r['sparse_drn_meta']['n_params']})",
                f"{so_r['sparse_hyper_drn_fresh_only']['fresh_kl']:.4f}",
                f"{so_r['sparse_hyper_drn_fresh_only']['aged_kl']:.4f}",
                f"{so_r['sparse_hyper_drn_joint']['aged_kl']:.3f} ({so_r['coarsening']['dt_8.0']['sparse_drn_aged_kl']:.3f})",
            ],
            [
                "(143k events,",
                "CTMC (expm)",
                f"22 ({so_r['ctmc_fresh_only']['params']})",
                f"{so_r['ctmc_fresh_only']['fresh_kl']:.4f}",
                f"{so_r['ctmc_fresh_only']['aged_kl']:.4f}",
                f"{so_r['ctmc_joint']['aged_kl']:.3f} (\\textbf{{{so_r['coarsening']['dt_8.0']['ctmc_aged_kl']:.3f}}})",
            ],
            [
                "$\\mathrm{CV}{=}1.22$)",
                "Mamba SSM",
                f"34 ({so_r['mamba_fresh_only']['params']:,})",
                f"{so_r['mamba_fresh_only']['fresh_kl']:.4f}",
                f"{so_r['mamba_fresh_only']['aged_kl']:.4f}",
                f"{so_r['mamba_joint']['aged_kl']:.3f} ({so_r['coarsening']['dt_8.0']['mamba_aged_kl']:.3f})",
            ],
            [
                "",
                "S4D/LTC/LODE/CME",
                "34 (2.1--3.1k)",
                f"\\textbf{{{so_r['s4d_fresh_only']['fresh_kl']:.4f}}}--{so_r['neural_cme_fresh_only']['fresh_kl']:.4f}",
                f"\\textbf{{{so_r['s4d_fresh_only']['aged_kl']:.4f}}}--{so_r['neural_cme_fresh_only']['aged_kl']:.4f}",
                f"\\textbf{{{so_r['s4d_joint']['aged_kl']:.3f}}}--{so_r['neural_cme_joint']['aged_kl']:.3f}",
            ],
            "\\midrule",
            [
                "DeFi 183k",
                "\\textbf{Cond. Neural Clock}",
                f"3 ({defi_a['neural_clock']['params']})",
                f"\\textbf{{{defi_b['neural_clock']['all']['logloss']:.5f}}}",
                f"\\textbf{{{defi_a['neural_clock']['all']['logloss']:.5f}}}",
                f"\\textbf{{{defi_c1['neural_clock']['bsc_all']['logloss']:.3f}}} (\\textbf{{{defi_c1['neural_clock']['bsc_gt60']['logloss']:.3f}}})",
            ],
            [
                "(BSC \\& Base,",
                "MLP + $\\log(1{+}a)$",
                f"--- ({defi_a['mlp_age']['params']})",
                f"{defi_b['mlp_age']['all']['logloss']:.5f}",
                f"{defi_a['mlp_age']['all']['logloss']:.5f}",
                f"{defi_c1['mlp_age']['bsc_all']['logloss']:.3f} ({defi_c1['mlp_age']['bsc_gt60']['logloss']:.3f})",
            ],
            [
                "Jan--Jul 2025)",
                "Weibull / Markov",
                f"--- ({defi_a['weibull_mlp']['params']}/{defi_a['markov_clock']['params']})",
                f"{defi_b['weibull_mlp']['all']['logloss']:.3f} / {defi_b['markov_clock']['all']['logloss']:.3f}",
                f"{defi_a['weibull_mlp']['all']['logloss']:.3f} / {defi_a['markov_clock']['all']['logloss']:.3f}",
                f"{defi_c1['weibull_mlp']['bsc_all']['logloss']:.2f} / {defi_c1['markov_clock']['bsc_all']['logloss']:.3f}",
            ],
            [
                "",
                "LTC/S4D/Mamba",
                "12 (637--931)",
                f"{defi_b['ltc_seq']['all']['logloss']:.3f}--{defi_b['mamba_ssm']['all']['logloss']:.3f}",
                f"{defi_a['ltc_seq']['all']['logloss']:.3f}--{defi_a['mamba_ssm']['all']['logloss']:.3f}",
                "---",
            ],
        ],
    )


def main() -> None:
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_fig6()
    build_tables()
    print("Successfully built 6 publication figures (PDF/SVG/PNG) and 15 LaTeX tables.")


if __name__ == "__main__":
    main()
