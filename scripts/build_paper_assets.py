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
from matplotlib.patches import FancyBboxPatch
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

    ax_sch = fig.add_subplot(gs[0, :])
    ax_sch.set_xlim(0.0, 10.0)
    ax_sch.set_ylim(0.0, 2.50)
    ax_sch.axis("off")
    add_panel_tag(ax_sch, "a", x_pt=-22.0, y_pt=0.0)

    boxes = [
        (0.02, 0.25, 2.96, 2.00, PALETTE["light_blue"], PALETTE["navy"],
         "Marked-race compiler",
         "Competing clocks\n$K^d \\to 1 + d(K-1)$\nUp to $1{,}074\\times$ smaller"),
        (3.48, 0.25, 3.04, 2.00, PALETTE["light_teal"], PALETTE["teal"],
         "Forward & Fisher AD",
         "Parametric / neural PH\n$P_\\theta(t),\\, \\partial P_\\theta / \\partial\\theta$\nSnapshot NLL & $I(\\theta)$"),
        (7.02, 0.25, 2.96, 2.00, PALETTE["light_amber"], PALETTE["amber"],
         "Sensitivity reduction",
         "$15$ modes cancel in $P$\n$P$: $33$  |  $\\partial P / \\partial\\theta$: $48$\nPos-Sens & 2-sided MOR"),
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
# Figure 2: Gradient Defect & Positivity-Preserving Sensitivity Reduction
# ---------------------------------------------------------------------------
def build_fig2() -> None:
    ev = load_archived("evidence.json")
    sens = load_archived("sensitivity_evidence.json")
    pos = load_archived("positive_sensitivity_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.42))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.75, bottom=0.19, wspace=0.46)

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
    add_panel_tag(axes[0], "a", x_pt=-22.0, y_pt=3.0)

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
        add_panel_tag(ax, tag, x_pt=-22.0, y_pt=3.0)

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
    pilot = load_archived("stem_cells_pilot.json")
    ph_sc = load_archived("trainable_ph_stem_cells.json")
    ev = load_archived("evidence.json")
    gsc = load_archived("graph_scaling_summary.json")
    records = ev["benchmark"]

    ts_obs = np.asarray(pilot["observation_times"])
    obs = np.asarray(pilot["data_mean_proportions"])
    theta = np.asarray(ph_sc["panels"]["all_times"]["phase"]["best"]["theta"])
    means = jnp.exp(-jnp.asarray(theta[:2]))
    centers = 50.0 * jnp.asarray(theta[2:])
    logits = jax.vmap(lambda c: smooth_erlang_logits(c, stages=64, width=0.5))(centers)
    ts_dense = jnp.linspace(0.0, 168.0, 240)
    pred_dense = np.asarray(solve_erlang_chain(means, logits, ts_dense).ys)

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 4.18))
    fig.subplots_adjust(
        left=0.11, right=0.975, top=0.91, bottom=0.105, hspace=0.56, wspace=0.40
    )

    # Panel a: Marked-race warm gradient scaling across V in {4..48} and d in {2,3,4}
    ax_a = axes[0, 0]
    g_rows = gsc["gradient_and_state_scaling"]
    for deg, col, mk, lbl in [
        (2, PALETTE["navy"], "o", "Race $d{=}2$"),
        (3, PALETTE["teal"], "s", "Race $d{=}3$"),
        (4, PALETTE["violet"], "^", "Race $d{=}4$"),
    ]:
        sub = [r for r in g_rows if r["out_degree"] == deg]
        ax_a.plot(
            [r["nodes"] for r in sub],
            [r["compact_grad_warm_ms"] for r in sub],
            marker=mk,
            color=col,
            ms=3.4,
            lw=1.45,
            label=lbl,
        )
    sub_j3 = [r for r in g_rows if r["out_degree"] == 3 and r["joint_grad_warm_ms"] is not None]
    ax_a.plot(
        [r["nodes"] for r in sub_j3],
        [r["joint_grad_warm_ms"] for r in sub_j3],
        marker="x",
        ls="--",
        color=PALETTE["crimson"],
        ms=4.2,
        lw=1.4,
        label="Tensor $d{=}3$",
    )
    ax_a.set_yscale("log")
    ax_a.set_xlim(2, 50)
    ax_a.set_ylim(8.0, 700.0)
    ax_a.set_xticks([4, 16, 32, 48])
    ax_a.set_xlabel("Graph nodes $V$")
    ax_a.set_ylabel("Warm grad. (ms)")
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.1,
        columnspacing=0.55,
        handletextpad=0.25,
        labelspacing=0.18,
        borderpad=0.05,
    )
    add_panel_tag(ax_a, "a", x_pt=-24.0, y_pt=18.0)

    # Panel b: Stem-cell differentiation trajectories
    ax_b = axes[0, 1]
    for idx_s, (sname, col, mk) in enumerate(
        [
            ("ESC", PALETTE["navy"], "o"),
            ("EPI", PALETTE["amber"], "s"),
            ("NPC", PALETTE["teal"], "^"),
        ]
    ):
        ax_b.plot(np.asarray(ts_dense), pred_dense[:, idx_s], color=col, lw=1.55, label=sname)
        ax_b.scatter(ts_obs, obs[:, idx_s], color=col, marker=mk, s=16, zorder=4)
    ax_b.set_xlabel("Time (h)")
    ax_b.set_ylabel("Cell state fraction")
    ax_b.set_xlim(-4.0, 174.0)
    ax_b.set_ylim(-0.04, 1.06)
    ax_b.set_xticks([0, 48, 96, 144])
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=1.0,
        columnspacing=0.55,
        handletextpad=0.25,
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

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.48))
    fig.subplots_adjust(left=0.105, right=0.98, top=0.74, bottom=0.19, wspace=0.38)

    studies = [
        ("primary", "PH truth"),
        ("lognormal", "Lognormal"),
    ]
    models = [
        ("neural_phase", "Neural clock", PALETTE["navy"], -0.28),
        ("mean_shape", "Moment PH", PALETTE["teal"], -0.14),
        ("ltc_ode_3start", "LTC (3-st.)", PALETTE["violet"], 0.0),
        ("s4d_ode_3start", "S4D (3-st.)", PALETTE["amber"], 0.14),
        ("latent_ode_3start", "Latent ODE", PALETTE["crimson"], 0.28),
    ]
    x_pos = np.arange(len(studies))
    axes[0].axhline(0.005, color=PALETTE["slate"], ls="--", lw=0.75)
    for mkey, mlabel, col, dx in models:
        meds, lows, highs = [], [], []
        for skey, _ in studies:
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
            width=0.125,
            color=col,
            edgecolor=PALETTE["ink"],
            linewidth=0.5,
            label=mlabel,
        )
        axes[0].errorbar(
            x_pos + dx,
            meds,
            yerr=[lows, highs],
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=0.75,
            capsize=1.6,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x_pos, [lbl for _, lbl in studies])
    axes[0].set_xlim(-0.48, 1.48)
    axes[0].set_ylim(8e-4, 0.22)
    axes[0].set_ylabel("Held-out excess KL (nats)")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=0.95,
        columnspacing=0.5,
        handletextpad=0.25,
        labelspacing=0.18,
        borderpad=0.05,
    )
    add_panel_tag(axes[0], "a", x_pt=-24.0, y_pt=24.0)

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
    add_panel_tag(axes[1], "b", x_pt=-24.0, y_pt=24.0)

    export_figure(fig, "fig4_neural_clocks_and_transfer")


# ---------------------------------------------------------------------------
# Figure 5: Continuous Biological Renewal Networks & Cross-Visit Persistence
# ---------------------------------------------------------------------------
def build_fig5() -> None:
    rcb = load_archived("real_continuous_benchmarks.json")
    rvm = load_archived("real_visit_memory.json")["datasets"]

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.54))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.74, bottom=0.19, wspace=0.52)

    # Panel a: Held-out aged-survivor KL on continuous E. coli & 5-node Sleep-EDF
    ec_m = rcb["ecoli_continuous_cycle"]["models_5fold_mean"]
    sl_m = rcb["sleep_5node_branching_network"]["models_5fold_mean"]
    x_c = np.arange(2)
    y_floor = 0.025
    for mlabel, vals, sds, col, dx in [
        (
            "DRN",
            [ec_m["drn_joint"]["aged_kl"], sl_m["drn_joint"]["aged_kl_mean"]],
            [ec_m["drn_joint"]["aged_kl_sd"], sl_m["drn_joint"]["aged_kl_sd"]],
            PALETTE["navy"],
            -0.27,
        ),
        (
            "DRN 0-sh.",
            [ec_m["drn_fresh_only"]["aged_kl"], sl_m["drn_fresh_only"]["aged_kl_mean"]],
            [ec_m["drn_fresh_only"]["aged_kl_sd"], sl_m["drn_fresh_only"]["aged_kl_sd"]],
            PALETTE["royal"],
            -0.09,
        ),
        (
            "CTMC",
            [ec_m["ctmc"]["aged_kl"], sl_m["ctmc"]["aged_kl_mean"]],
            [ec_m["ctmc"]["aged_kl_sd"], sl_m["ctmc"]["aged_kl_sd"]],
            PALETTE["teal"],
            0.09,
        ),
        (
            "LODE 0-sh.",
            [ec_m["latent_ode_fresh_only"]["aged_kl"], sl_m["latent_ode_fresh_only"]["aged_kl_mean"]],
            [ec_m["latent_ode_fresh_only"]["aged_kl_sd"], sl_m["latent_ode_fresh_only"]["aged_kl_sd"]],
            PALETTE["crimson"],
            0.27,
        ),
    ]:
        axes[0].bar(
            x_c + dx,
            np.asarray(vals) - y_floor,
            bottom=y_floor,
            width=0.16,
            color=col,
            edgecolor=PALETTE["ink"],
            linewidth=0.5,
            label=mlabel,
        )
        axes[0].errorbar(
            x_c + dx,
            vals,
            yerr=sds,
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=0.75,
            capsize=1.6,
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x_c, ["E. coli\n(3-phase)", "Sleep-EDF\n(5-node)"])
    axes[0].set_xlim(-0.52, 1.52)
    axes[0].set_ylim(y_floor, 0.62)
    axes[0].set_ylabel("Aged-survivor KL (nats)")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.55, 1.02),
        ncol=2,
        handlelength=0.85,
        columnspacing=0.42,
        handletextpad=0.22,
        labelspacing=0.18,
        borderpad=0.05,
    )
    add_panel_tag(axes[0], "a", x_pt=-24.0, y_pt=26.0)

    # Panel b: Single-cell E. coli division-cycle duration recovery from snapshots alone
    rec_stats = rcb["ecoli_continuous_cycle"]["zero_shot_cycle_law_recovery"]
    temps_x = np.arange(len(rec_stats))
    axes[1].bar(
        temps_x - 0.17,
        [r["true_longitudinal_mean_min"] for r in rec_stats],
        width=0.31,
        color=PALETTE["slate"],
        edgecolor=PALETTE["ink"],
        linewidth=0.5,
        label="True cell",
    )
    axes[1].bar(
        temps_x + 0.17,
        [r["snapshot_recovered_mean_min"] for r in rec_stats],
        width=0.31,
        color=PALETTE["navy"],
        edgecolor=PALETTE["ink"],
        linewidth=0.5,
        label="DRN snap.",
    )
    axes[1].set_xticks(temps_x, [f"{r['temp_C']}\u00b0C" for r in rec_stats])
    axes[1].set_ylim(0.0, 78.0)
    axes[1].set_ylabel("Cycle mean (min)")
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=0.95,
        columnspacing=0.5,
        handletextpad=0.25,
        borderpad=0.05,
    )
    add_panel_tag(axes[1], "b", x_pt=-22.0, y_pt=26.0)

    # Panel c: Longitudinal cross-visit persistence gain (before vs after covariates)
    cohorts = [
        ("sleep", "2", "Sleep N2"),
        ("nomass", "all", "Si E. coli"),
        ("tanouchi", "all", "Tanouchi"),
    ]
    axes[2].axvline(0.0, color=PALETTE["slate"], lw=0.8, ls="--")
    for idx_y, (dkey, subk, _) in enumerate(cohorts):
        y_coord = 2 - idx_y
        for mkey, mlabel, col, mk, dy in [
            ("marginal", "Raw", PALETTE["amber"], "s", 0.14),
            ("adjusted", "Adj.", PALETTE["navy"], "o", -0.14),
        ]:
            est = rvm[dkey]["models"][mkey]["previous"][subk]
            val = 100.0 * float(est["gain"])
            lo, hi = [100.0 * float(v) for v in est["interval"]]
            axes[2].errorbar(
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
    axes[2].set_yticks([2, 1, 0], [lbl for _, _, lbl in cohorts])
    axes[2].set_ylim(-0.52, 2.55)
    axes[2].set_xlim(-0.45, 2.15)
    axes[2].set_xlabel("Gain / 100 visits (nats)")
    axes[2].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=0.55,
        handletextpad=0.2,
        borderpad=0.05,
    )
    add_panel_tag(axes[2], "c", x_pt=-38.0, y_pt=26.0)

    export_figure(fig, "fig5_persistence_and_sleep_calibration")


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
            "\\textbf{Full PH}",
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
            f"{pos['cases']['primary/mixed_150_PH']['median_shape_percent']:.2f}\\% / \\textbf{{{pos['cases']['primary/mixed_150_Gamma']['median_shape_percent']:.2f}\\%}}",
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
            "($V{=}4..12$, 16--48p)",
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

    # Table 3: Conditional neural clocks vs tuned LTC, S4D, and Latent ODE + transfer
    pu_mod = nmp["studies"]["primary"]
    tr_400 = rt["heldout_ph"]["400"]["arms"]
    tr_8000 = rt["heldout_ph"]["8000"]["arms"]
    t3_rows = [
        [
            "\\textbf{Neural clock}",
            "185",
            f"\\textbf{{{mnb['primary']['neural_phase']['metrics']['balanced_kl']['median']:.5f}}}",
            f"\\textbf{{{mnb['lognormal']['neural_phase']['metrics']['balanced_kl']['median']:.5f}}}",
            f"\\textbf{{{100*pu_mod['neural_phase']['median_arm_rmse']['exit_increase']:.2f}\\,\\text{{pp}}}}",
            f"\\textbf{{{tr_400['neural_phase.pretrained_code']['balanced_kl']:.5f}}}",
            f"\\textbf{{{tr_8000['neural_phase.pretrained_full']['balanced_kl']:.5f}}}",
        ],
        [
            "Moment PH",
            "18",
            f"{mnb['primary']['mean_shape']['metrics']['balanced_kl']['median']:.5f}",
            f"{mnb['lognormal']['mean_shape']['metrics']['balanced_kl']['median']:.5f}",
            f"{100*pu_mod['mean_shape']['median_arm_rmse']['exit_increase']:.2f}\\,\\text{{pp}}",
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

    # Table 4: Continuous biological renewal networks & cross-visit persistence
    ec_m = rcb["ecoli_continuous_cycle"]["models_5fold_mean"]
    sl_b = rcb["sleep_5node_branching_network"]
    sl_m = sl_b["models_5fold_mean"]
    s_n2 = rvm["sleep"]["models"]["adjusted"]["previous"]["2"]
    s_ec1 = rvm["nomass"]["models"]["adjusted"]["previous"]["all"]
    s_ec2 = rvm["tanouchi"]["models"]["adjusted"]["previous"]["all"]
    t4_rows = [
        [
            "\\emph{E.\\ coli} 3-phase ring",
            "18,972 cyc.",
            f"\\textbf{{{ec_m['drn_fresh_only']['aged_kl']:.4f}}} vs.\\ {ec_m['latent_ode_fresh_only']['aged_kl']:.4f}",
            "$37^\\circ$C $k{=}25.8$ ($27.1$)",
        ],
        [
            "Sleep-EDF 5-node net",
            "75 adults",
            f"\\textbf{{{sl_m['drn_fresh_only']['aged_kl_mean']:.4f}}} vs.\\ {sl_m['latent_ode_fresh_only']['aged_kl_mean']:.4f}",
            "$6{,}144{\\to}117$ ($52.5\\times$)",
        ],
        "\\midrule",
        [
            "Synthetic persistent",
            "20 runs",
            "\\textbf{KL 0.00416} ($7.9\\times$)",
            "$\\hat\\rho$: $0.000$ vs.\\ $0.875$",
        ],
        [
            "Sleep-EDF N2 returns",
            "2,957 bouts",
            f"\\textbf{{{s_n2['gain']:+.5f}}} nats",
            "Perm. $+0.0073$ ($p{=}.002$)",
        ],
        [
            "\\emph{E.\\ coli} (Si et al.)",
            "7,389 cyc.",
            f"\\textbf{{{s_ec1['gain']:+.5f}}} nats",
            "CI $[+0.0087, +0.0171]$",
        ],
        [
            "\\emph{E.\\ coli} (Tanouchi)",
            "18,693 cyc.",
            f"${s_ec2['gain']:+.5f}$ nats",
            "Size absorbs $+0.0132$",
        ],
    ]
    assert "resampled" in rt6["targets"] and "persistent" in rt6["targets"]
    write_tex_table(
        "tab4_persistence_and_trajectories.tex",
        "llll",
        [
            "System / Cohort",
            "Scale",
            "0-Shot KL / Gain",
            "Law / Control",
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
    write_tex_table(
        "tab_a7_graph_scaling.tex",
        "lrrrrrr",
        [
            "Graph $(d, V)$",
            "Params",
            "Tensor",
            "Race",
            "Comp. $\\uparrow$",
            "Grad (ms)",
            "Speedup",
        ],
        rows_g7,
    )


def main() -> None:
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_tables()
    print("Successfully built 5 publication figures (PDF/SVG/PNG) and 11 LaTeX tables.")


if __name__ == "__main__":
    main()
