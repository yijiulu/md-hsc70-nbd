#!/usr/bin/env python3
"""Generate publication-grade summary figures and self-contained PyMOL B-factor mapping from an_compare.py output.

Usage:
  python scripts/an_figures.py
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from nbd_defs import ROOT, STATES, INTERFACES  # noqa: E402


def main() -> None:
    out_dir = f"{ROOT}/analysis/summary"
    col = {"atp": "#2a78d6", "adp": "#eb6834"}
    lab = {"atp": "ATP", "adp": "ADP"}
    ink, ink2, muted, grid, axis = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    mark = {"linker": ("D", "Linker crevice"), "SBDb": ("s", "SBDβ face"), "SBDa": ("^", "SBDα face")}

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "axes.edgecolor": axis, "axes.labelcolor": ink, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": grid,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.color": ink2, "ytick.color": ink2, "text.color": ink,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.2,
        "savefig.dpi": 300, "figure.facecolor": "white",
    })

    window, used = open(f"{out_dir}/window.txt").read().split(";")
    skip, until = (float(x) for x in window.replace(" ns", "").split("-"))
    used = ast.literal_eval(used.replace("replicas", "").strip())
    ts = {(s, r): pd.read_csv(f"{ROOT}/analysis/{s}_rep{r}/timeseries.csv") for s, r in used}
    pr = pd.read_csv(f"{out_dir}/per_residue.csv")
    sc = pd.read_csv(f"{out_dir}/scalars.csv")
    nrep = {s: sum(k[0] == s for k in ts) for s in STATES}
    n_pairs = nrep["atp"] * nrep["adp"]
    disp_ok = pr.disp_local_excess.notna().any()
    has_csp = "csp" in pr.columns and pr["csp"].notna().any()

    def win(t):
        return t[(t.time_ns >= skip) & (t.time_ns <= until)]

    def kde_panel(ax, col_name, xlabel, title):
        for (s, r), t in sorted(ts.items()):
            v = win(t)[col_name].values
            x = np.linspace(v.min() - 1, v.max() + 1, 200)
            ax.plot(x, gaussian_kde(v)(x), color=col[s], lw=1.0, alpha=0.9,
                    label=f"{lab[s]} (n = {nrep[s]})" if r == min(k[1] for k in ts if k[0] == s) else None)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="x", visible=False)

    def interface_scatter(ax, y, ylabel, title):
        base = ~pr.if_any
        ax.scatter(pr.dist_pg[base], y[base], s=9, color=muted, alpha=0.6, lw=0, label="Other residues")
        for k, (m, lb) in mark.items():
            sel = pr[f"if_{k}"]
            ax.scatter(pr.dist_pg[sel], y[sel], s=22, marker=m, facecolor=ink, edgecolor="white", lw=0.6, label=lb)
        ax.set_xlabel("Distance from γ-phosphate in 1KAX (Å)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")

    # ---- Main figure ----
    fig = plt.figure(figsize=(7.0, 8.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.95, 1.1], hspace=0.65, wspace=0.45, top=0.93)
    ax = fig.add_subplot(gs[0, 0])
    kde_panel(ax, "delta", "Domain angle δ (°)", "a  Cleft closure")
    hl = sorted(zip(*ax.get_legend_handles_labels()), key=lambda z: not z[1].startswith("ATP"))
    fig.legend(*zip(*hl), loc="upper right", ncol=2, fontsize=7.5, bbox_to_anchor=(0.99, 0.995))

    ax = fig.add_subplot(gs[0, 1])
    kde_panel(ax, "w90_r261", "Trp90 HE1–Arg261 NH (Å)", "b  Trp90–Arg261")

    ax = fig.add_subplot(gs[0, 2])
    for i, s in enumerate(STATES):
        v = sc[sc.state == s].r_lobes_detrended.values
        ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=30, color=col[s], edgecolor="white", lw=0.6, zorder=3)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=ink, lw=1.2, zorder=4)
    ax.axhline(0, color=axis, lw=0.8)
    ax.set_xticks([0, 1], [lab[s] for s in STATES])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("r (IB vs IIB lobe angle)")
    ax.set_title("c  Lobe coupling", loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[1, :])
    d = pr.drmsf_local
    ax.bar(pr.resid, d, width=1.0, color=np.where(d > 0, col["atp"], col["adp"]), lw=0)
    rob = pr.drmsf_local_same_sign
    ax.scatter(pr.resid[rob], d[rob] + np.sign(d[rob]) * 0.04, s=6, color=ink, lw=0,
               label=f"Same sign in all {n_pairs} replica pairs")
    ax.axhline(0, color=axis, lw=0.8)
    ylo, yhi = min(d.min(), -0.2) - 0.05, max(d.max(), 0.2) + 0.1
    step = 0.09 * (yhi - ylo)
    for j, (k, (m, lb)) in enumerate(mark.items()):
        yy = ylo - step * (j + 1)
        ax.scatter(INTERFACES[k], [yy] * len(INTERFACES[k]), marker=m, s=10, color=ink, lw=0)
        ax.text(388, yy, lb, va="center", fontsize=6.5, color=ink2, clip_on=False)
    ax.set_ylim(ylo - step * (len(mark) + 0.8), yhi)
    ax.set_xlim(2, 382)
    ax.set_xlabel("Residue")
    ax.set_ylabel("ΔRMSF, ATP − ADP (Å)")
    ax.set_title("d  Local flexibility (CA RMSF, subdomain fit)", loc="left", fontsize=8, fontweight="bold", pad=20)
    h = [Patch(color=col["atp"], label="More mobile with ATP"), Patch(color=col["adp"], label="More mobile with ADP")]
    if rob.any():
        h.append(ax.collections[0])
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=6.5, ncol=len(h), borderaxespad=0.2)
    ax.grid(axis="x", visible=False)

    sub = gs[2, :].subgridspec(1, 2, wspace=0.3)
    ax = fig.add_subplot(sub[0])
    if disp_ok:
        interface_scatter(ax, pr.disp_local_excess, "Mean-structure shift, ATP vs ADP,\nminus replica noise (Å)", "e  MD: local structural change")
        ax.axhline(0, color=axis, lw=0.8)
    else:
        interface_scatter(ax, pr.drmsf_local.abs(), "|ΔRMSF| (Å)", "e  MD: |ΔRMSF|")
    ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.2)

    ax = fig.add_subplot(sub[1])
    if has_csp:
        tr = pr.csp.notna() & ~pr.csp_untracked
        c = pr.csp.where(tr)
        interface_scatter(ax, c, "AMPPNP − ADP CSP (ppm)", "f  NMR: chemical shift perturbation")
        un = pr.csp_untracked.fillna(False).astype(bool)
        ymax = np.nanmax(c) * 1.1
        ax.scatter(pr.dist_pg[un], [ymax] * un.sum(), s=14, marker="x", color=ink2, lw=0.8, label=f"Peak not followed ({un.sum()})")
        ax.set_ylim(top=ymax * 1.06)
        ax.legend(handles=ax.collections[-1:], loc="upper right", bbox_to_anchor=(1.0, 0.94), fontsize=6.5, handletextpad=0.2)
    else:
        interface_scatter(ax, pr.drmsf_local.abs(), "|ΔRMSF|, ATP − ADP (Å)", "f  MD: local flexibility shift")

    for a_ in fig.axes:
        a_.tick_params(labelsize=7)
    fig.suptitle(f"Hsc70 NBD, ATP vs ADP ({window}, {nrep['atp']} + {nrep['adp']} replicas)", fontsize=9, x=0.01, y=0.995, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_dir}/fig_md_main.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ---- QC figure ----
    rows = [("rmsd_ca", "CA RMSD\nvs 1KAX (Å)"), ("delta", "δ (°)"), ("w90_r261", "Trp90–\nArg261 (Å)"),
            ("crevice", "Linker\ncrevice (Å)"), ("k1_d", "K1 to site\ncentre (Å)"), ("k2_d", "K2 to site\ncentre (Å)"),
            ("nuc_rmsd", "Nucleotide\nRMSD (Å)")]
    fig, axs = plt.subplots(len(rows), 1, figsize=(7.0, 1.15 * len(rows)), sharex=True)
    for ax, (c, lb) in zip(axs, rows):
        for (s, r), t in sorted(ts.items()):
            y = t[c].rolling(50, center=True, min_periods=1).mean()
            ax.plot(t.time_ns, y, color=col[s], lw=0.8, alpha=0.85, label=lab[s] if (c == "rmsd_ca" and r == 1) else None)
        ax.axvspan(0, skip, color=grid, alpha=0.6, lw=0)
        ax.set_ylabel(lb, fontsize=7)
        ax.tick_params(labelsize=7)
    hl = sorted(zip(*axs[0].get_legend_handles_labels()), key=lambda z: z[1] != "ATP")
    axs[0].legend(*zip(*hl), loc="lower right", ncol=2, fontsize=7)
    fig.align_ylabels(axs)
    axs[0].set_title(f"Per-replica time series, 0.5 ns running mean; shaded = equilibration cut ({skip:g} ns)",
                     loc="left", fontsize=8, fontweight="bold")
    axs[-1].set_xlabel("Time (ns)")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_dir}/fig_md_qc.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ---- Self-contained PyMOL B-factor mapping script (relative paths) ----
    for name, col_name in (("md_drmsf_local", "drmsf_local"), ("md_disp_local", "disp_local_excess")):
        v = pr[["resid", col_name]].dropna().copy()
        if name == "md_disp_local":
            v[col_name] = v[col_name].clip(lower=0)
        v.to_csv(f"{out_dir}/{name}.txt", sep="\t", header=False, index=False, float_format="%.4f")

    data_file = "md_disp_local.txt" if disp_ok else "md_drmsf_local.txt"
    v_vals = pr.disp_local_excess.clip(lower=0) if disp_ok else pr.drmsf_local.abs()
    vmax = float(np.nanmax(v_vals))
    with open(f"{out_dir}/md_mapping.pml", "w") as f:
        f.write(f"""# Run from the repository root: pymol analysis/summary/md_mapping.pml
from pymol import cmd
cmd.reinitialize()
cmd.load('structures/3HSC.pdb', '3hsc')
cmd.hide('lines')
cmd.show('cartoon')
cmd.color('white')
cmd.alter('3hsc', 'b=99.9')
for line in open('analysis/summary/{data_file}'):
    resid, val = line.split()
    cmd.alter(f'3hsc and resi {{resid}}', f'b={{float(val):.4f}}')
cmd.spectrum('b', 'yellow_red', '3hsc', '0', '{vmax:.3f}')
cmd.ramp_new('md_shift', '3hsc', '[0,{vmax:.3f}]', '[yellow, red]')
cmd.color('white', '(b=99.9)')
cmd.recolor()
""")
    print(f"[OK] Figures and PyMOL script written to {out_dir}")


if __name__ == "__main__":
    main()
