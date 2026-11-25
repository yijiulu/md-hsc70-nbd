#!/usr/bin/env python3
"""Multi-system lobe-rotation and linker-contact figure generator.

Usage:
  python scripts/fig_rotation.py
Writes analysis/summary/fig_md_rotation.{png,pdf}, dynamically displaying all conditions that have completed
rotation.csv trajectories (isolated NBD, NBD + linker, and optionally linker-removed control).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import lobe_rotation as lr  # noqa: E402
from nbd_defs import ROOT  # noqa: E402


def main() -> None:
    out_dir = f"{ROOT}/analysis/summary"
    for weight in ("Regular", "Bold"):
        fpath = f"{ROOT}/fonts/LiberationSans-{weight}.ttf"
        if os.path.exists(fpath):
            font_manager.fontManager.addfont(fpath)

    col = {"atp": "#2a78d6", "adp": "#eb6834"}
    ink, ink2, muted, grid, axis_c = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7.5,
        "axes.edgecolor": axis_c, "axes.labelcolor": ink, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": grid,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.color": ink2, "ytick.color": ink2, "text.color": ink,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.0,
        "savefig.dpi": 600, "figure.facecolor": "white", "pdf.fonttype": 42,
    })

    axis = lr.docked_axis()
    ref = {
        "3HSC": lr.rotvec({r.id[1]: r["CA"].coord.astype(float) for r in lr.chain_res(f"{ROOT}/structures/3HSC.pdb")}) @ axis,
        "1YUW": lr.rotvec({r.id[1]: r["CA"].coord.astype(float) for r in lr.chain_res(f"{ROOT}/structures/1YUW.pdb", "A")
                           if r.id[1] <= 381}) @ axis,
        "7RAX": lr.rotvec(lr.dnak_ca("7RAX")) @ axis,
    }
    docked = [lr.rotvec(lr.dnak_ca(c)) @ axis for c in lr.DOCKED]

    def load(state, rep, fname="rotation.csv"):
        p = f"{ROOT}/analysis/{state}_rep{rep}/{fname}"
        return pd.read_csv(p) if os.path.exists(p) else None

    all_sys = [
        ("", "Isolated NBD", "Isolated\nNBD", "start 1KAX, 0°", 20, 150),
        ("lk_", "NBD + linker", "NBD +\nlinker", "start docked model, +12°", None, 100),
        ("nl_", "NBD, linker removed", "Linker\nremoved", "start same model, +12°", None, 100),
    ]
    active_sys = [row for row in all_sys if any(load(row[0] + s, r) is not None for s in ("atp", "adp") for r in (1, 2, 3))]
    if not active_sys:
        sys.exit("No rotation.csv files found under analysis/*_rep*/ (run scripts/an_rotation.py first).")

    has_linker = any(load("lk_" + s, r, "linker.csv") is not None for s in ("atp", "adp") for r in (1, 2, 3))
    n_rows = len(active_sys) + (1 if has_linker else 0)
    height_ratios = [1.0] * len(active_sys) + ([0.75] if has_linker else [])
    yl = (-10, 16)

    fig = plt.figure(figsize=(6.3, 1.85 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[2.15, 1], height_ratios=height_ratios, hspace=0.55, wspace=0.42)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]
    for ax in axs[1:]:
        ax.sharex(axs[0])

    panel_letters = "abcdef"
    max_time = 0.0
    for i, (pre, name, _, start, skip, length) in enumerate(active_sys):
        ax = axs[i]
        for s in ("atp", "adp"):
            for r in (1, 2, 3):
                d = load(pre + s, r)
                if d is None:
                    continue
                max_time = max(max_time, float(d.time_ns.max()))
                ax.plot(d.time_ns, d.rot_docked.rolling(100, center=True, min_periods=20).mean(), color=col[s], lw=0.9,
                        alpha=0.9, solid_capstyle="round")
        w0 = skip if skip is not None else length - 20
        ax.axvspan(w0, length, color=grid, alpha=0.55, lw=0)
        for v, lb in ((0, "1KAX"), (ref["7RAX"], "7RAX")):
            ax.axhline(v, color=muted, lw=0.7, ls=(0, (3, 2)))
            ax.text(1.0, v, f" {lb}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5, color=ink2)
        ax.set_ylim(*yl)
        ax.set_yticks([-8, 0, 8, 16])
        ax.set_title(f"{panel_letters[i]}   {name} ({start})", loc="left", fontsize=7.5, fontweight="bold")
        ax.set_ylabel("Lobe rotation (°)")

    axs[0].legend(handles=[Line2D([], [], color=col["atp"], label="ATP"), Line2D([], [], color=col["adp"], label="ADP")],
                  loc="center right", bbox_to_anchor=(1.0, 0.72), ncol=2, fontsize=7, handlelength=1.5)
    axs[0].text(21, yl[1] - 1, "analysis window", fontsize=6.5, color=ink2, va="top")

    if has_linker:
        ax = axs[-1]
        for s in ("atp", "adp"):
            for r in (1, 2, 3):
                d = load("lk_" + s, r, "linker.csv")
                if d is not None:
                    ax.plot(d.time_ns, d.q_linker.rolling(10, center=True, min_periods=2).mean(), color=col[s], lw=0.9, alpha=0.9)
        ax.axhline(0.3, color=muted, lw=0.7, ls=(0, (1, 1.5)))
        ax.text(1.0, 0.3, " release\n threshold", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.5, color=ink2)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("Contacts kept")
        ax.set_title(f"{panel_letters[len(active_sys)]}   Linker–NBD contacts, NBD + linker", loc="left", fontsize=7.5, fontweight="bold")

    axs[-1].set_xlabel("Time (ns)")
    axs[-1].set_xlim(0, max(100.0, max_time))
    for a in axs[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)

    # Summary panel on right
    ax = fig.add_subplot(gs[:, 1])
    st = pd.read_csv(f"{out_dir}/rotation_stats.csv")
    for x, (pre, *_) in enumerate(active_sys):
        for dx, s in ((-0.13, "atp"), (0.13, "adp")):
            v = st[st.state == pre + s]
            jit = np.linspace(-0.04, 0.04, len(v)) if len(v) > 1 else np.zeros(len(v))
            ax.errorbar(x + dx + jit, v.rot_mean, yerr=v.rot_sd, fmt="o", ms=4.5, color=col[s], mec="white", mew=0.6,
                        elinewidth=0.8, capsize=0, zorder=3)
    for v, lb in [(ref["3HSC"], "3HSC"), (0.0, "1KAX"), (ref["1YUW"], "1YUW"), (ref["7RAX"], "7RAX")]:
        ax.axhline(v, color=muted, lw=0.7, ls=(0, (3, 2)), zorder=1)
        off = {"3HSC": -0.45, "1KAX": 0.45}.get(lb, 0)
        ax.text(1.02, v + off, lb, transform=ax.get_yaxis_transform(), fontsize=6.5, color=ink2, ha="left", va="center")
    ax.text(1.02, yl[1], f"\u2191 SBD-\ndocked\n+{min(docked):.0f}–{max(docked):.0f}°", transform=ax.get_yaxis_transform(),
            fontsize=6.5, color=ink2, ha="left", va="top")
    ax.set_xlim(-0.45, len(active_sys) - 0.55)
    ax.set_xticks(range(len(active_sys)))
    ax.set_xticklabels([row[2] for row in active_sys])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(*yl)
    ax.set_yticks([-8, -4, 0, 4, 8, 12, 16])
    ax.set_ylabel("Window mean ± SD (°)")
    ax.set_title(f"{panel_letters[n_rows]}   Per replica", loc="left", fontsize=7.5, fontweight="bold")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=col["atp"], mec="white", label="ATP"),
                       Line2D([], [], marker="o", ls="", color=col["adp"], mec="white", label="ADP")],
              loc="lower right", fontsize=7, handletextpad=0.2)
    fig.align_ylabels(axs)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_dir}/fig_md_rotation.{ext}", bbox_inches="tight")
    print(f"[OK] Wrote {out_dir}/fig_md_rotation.png")


if __name__ == "__main__":
    main()
