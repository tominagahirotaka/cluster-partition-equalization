#!/usr/bin/env python3
"""Figures of paper 4 from stored outputs only (no integration), 2026-11-17 (author's instruction "図2と図4を描いて
ください").

fig_sweep: the sweep of (0.3, 1.2, 0.45), 50:25:25, realization s = 0 (analysis/universality/rebalance/), N = 100.
  (a) sizes of the three largest clusters at the end of each step against f; steps with more than three clusters
      (or rest) are marked;
  (b) for each step with three clusters, the weak-side limits f_c of the two ordered partitions n1:n2:n3 and n1:n3:n2
      compatible with the sorted sizes (analysis/universality/partition_flow/, scripts/partition_flow_summary.limits),
      against f; the line f_c = f. A partition is internally stable for an ordering while f > f_c.
fig_rest: f at which synchronized rest is reached, every sweep at N = 100 (rebalance/ and rebalance_var/ _s<seed>_N100),
  N = 200 and N = 400, one row per parameter point; the point without rest is marked at the end of its sweeps.
  -> analysis/paper4_figs/fig_sweep.{pdf,png}, fig_rest.{pdf,png}
  python3 scripts/plot_paper4_figs.py
"""
from pathlib import Path
import glob
import json
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partition_flow_summary import limits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
U = ROOT / "analysis/universality"
FIG = ROOT / "analysis/paper4_figs"
plt.rcParams.update({"font.size": 8, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
                     "lines.linewidth": 1.})
INK, MUTED = "0.15", "0.55"
C = ("#2a6f97", "#c8553d", "#6a994e")  # cluster rank 1, 2, 3


def fig_sweep():
    name = "e0.3_b1.2_g0.45_50-25-25"
    r = json.loads((U / "rebalance" / (name + ".json")).read_text())
    S = [s for s in r["steps"] if "sizes" in s or s.get("rest")]
    lim = limits(name)
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
    f = np.array([s["f"] for s in S])
    three = np.array([bool(s.get("sizes")) and len(s["sizes"]) == 3 for s in S])
    for k in range(3):
        y = np.array([sorted(s["sizes"], reverse=True)[k] if t else np.nan for s, t in zip(S, three)])
        ax[0].plot(f, y, drawstyle="steps-post", color=C[k], lw=1.1,
                   label=("largest", "second", "third")[k] + " cluster")
    other = [s for s, t in zip(S, three) if not t]
    for s in other:
        ax[0].axvline(s["f"], color=MUTED, lw=.6, ls=":")
    ax[0].set_ylabel("elements in cluster")
    ax[0].legend(frameon=False, fontsize=7, loc="lower left", ncol=1)
    ax[0].set_ylim(12, 52)
    ax[0].text(.98, .96, "dotted: more than three clusters or rest", transform=ax[0].transAxes, ha="right", va="top",
               fontsize=6.5, color=MUTED)
    ax[0].set_title("(a)", loc="left", fontsize=8)
    for s, t in zip(S, three):
        if not t:
            continue
        a, b, c = sorted(s["sizes"], reverse=True)
        for o, mk, fc in (((a, b, c), "o", INK), ((a, c, b), "o", "white")):
            v = lim.get(o, {}).get("f_c")
            if v is not None:
                ax[1].plot(s["f"], v, mk, ms=3.2, mfc=fc, mec=INK, mew=.6)
    lo, hi = f.min() - .01, f.max() + .01
    ax[1].plot([lo, hi], [lo, hi], color=MUTED, lw=.8)
    ax[1].text(.49, .425, "$f_c=f$", color=MUTED, fontsize=7)
    ax[1].set_xlabel("$f=K/K_1$")
    ax[1].set_ylabel("$f_c$ of the current partition")
    ax[1].set_title("(b)", loc="left", fontsize=8)
    ax[1].plot([], [], "o", ms=3.2, mfc=INK, mec=INK, label="$n_1{:}n_2{:}n_3$")
    ax[1].plot([], [], "o", ms=3.2, mfc="white", mec=INK, label="$n_1{:}n_3{:}n_2$")
    ax[1].legend(frameon=False, fontsize=7, loc="lower left")
    ax[1].set_xlim(hi, lo)  # weaker coupling to the right
    fig.tight_layout(h_pad=.4)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("fig_sweep." + ext), dpi=200)
    plt.close(fig)


def fig_rest():
    pat = re.compile(r"^(.*_\d+-\d+-\d+)_s(\d+)_N(\d+)\.json$")
    data = {}
    for f in glob.glob(str(U / "rebalance/*.json")):
        r = json.loads(Path(f).read_text())
        if r.get("steps"):
            data.setdefault(Path(f).stem, {}).setdefault(100, []).append((r.get("stop"), r["steps"][-1]["f"]))
    for f in glob.glob(str(U / "rebalance_var/*.json")):
        m = pat.match(Path(f).name)
        if not m or (m[3] == "100" and m[2] == "0"):
            continue  # T and D variants excluded; s0_N100 repeats the original sweep
        r = json.loads(Path(f).read_text())
        if r.get("steps"):
            data.setdefault(m[1], {}).setdefault(int(m[3]), []).append((r.get("stop"), r["steps"][-1]["f"]))

    def key(p):
        e, b, g = (float(q[1:]) for q in p.split("_")[:3])
        return (e, b, g)
    pts = sorted(data, key=key, reverse=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    rng = np.random.default_rng(0)
    for i, p in enumerate(pts):
        for N, mk, off, col in ((100, "o", .12, C[0]), (200, "s", -.18, C[1]), (400, "^", -.34, C[2])):
            v = data[p].get(N, [])
            fr = [x for s, x in v if s == "rest"]
            nr = [x for s, x in v if s != "rest"]
            y = i + off + (rng.uniform(-.1, .1, len(fr)) if N == 100 else 0)
            ax.plot(fr, np.full(len(fr), i + off) if N != 100 else y, mk, ms=3 if N == 100 else 3.2, mfc="none",
                    mec=col, mew=.7, alpha=.9 if N == 100 else 1)
            if nr and N == 100:
                ax.text(.37, i, "no rest in 300 steps (end at $f\\approx%.2f$)" % np.median(nr),
                        fontsize=6.5, color=INK, va="center")
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels(["(%s, %s, %s)" % tuple(q[1:] for q in p.split("_")[:3]) for p in pts], fontsize=6.5)
    ax.set_xlabel("$f$ at synchronized rest")
    for N, mk, col in ((100, "o", C[0]), (200, "s", C[1]), (400, "^", C[2])):
        ax.plot([], [], mk, mfc="none", mec=col, ms=3.2, label="$N=%d$" % N)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_xlim(.35, .87)
    ax.grid(axis="x", color="0.9", lw=.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("fig_rest." + ext), dpi=200)
    plt.close(fig)


def fig_map():
    """Fig. 1: first ends of the reduced branches (redrawn from universality_map_plot.py for the paper, 2026-11-17).
    Categories as universality_map_plot.cat; colors checked with the dataviz palette validator (all pairs, light):
    fold, internal-other, internal-subcritical, internal-supercritical pass; the single cell of paper 1's sequence is
    near-black with a white mark."""
    import mcluster_reduced as mr
    from universality_map_plot import cat
    from matplotlib.patches import Patch
    cols = {1: "#e3e3e3", 2: "#1b9e77", 3: "#5ba3dd", 4: "#1d4f9c", 5: "#e6550d", 6: "#1a1a1a", 7: "#ffffff"}
    labels = {2: "fold (clusters internally stable)", 3: "internal first; then other or undetermined",
              4: "internal first; then subcritical PD", 5: "internal first; then supercritical PD",
              6: "PD with clusters internally stable", 1: "no oscillation found"}
    B = np.round(np.arange(.6, 1.4001, .04), 2)
    G = np.round(np.arange(.1, .9501, .05), 2)
    E = (0.2, 0.25, 0.3, 0.35, 0.4)
    parts = ("34-33-33", "40-30-30", "50-25-25")
    data = {s: {(r["eps"], round(r["beta"], 2), round(r["gamma"], 2)): r
                for r in (json.loads(Path(f).read_text()) for f in glob.glob(str(U / "map" / f"e*_{s}.json")))}
            for s in parts}
    fig, axes = plt.subplots(3, 5, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for i, s in enumerate(parts):
        for j, e in enumerate(E):
            ax = axes[i, j]
            for b in B:
                for g in G:
                    mr.E, mr.B, mr.G = e, b, g
                    if not mr.excitable():
                        continue
                    r = data[s].get((e, b, g))
                    c = cat(r)
                    if s != "34-33-33" and r is None:
                        c = 7 if data["34-33-33"].get((e, b, g), {}).get("status") != "ok" else 7
                        if (e, b, g) not in data["34-33-33"]:
                            c = 7
                    ax.add_patch(plt.Rectangle((b - .02, g - .025), .04, .05, fc=cols[c],
                                               ec="0.75" if c == 7 else "white", lw=.3))
                    if c == 6:
                        ax.plot(b, g, marker="o", ms=1.6, color="white")
            ax.set_xlim(.58, 1.42)
            ax.set_ylim(.075, .975)
            ax.tick_params(labelsize=6.5, length=2)
            if i == 0:
                ax.set_title(r"$\varepsilon=%g$" % e, fontsize=8)
            if j == 0:
                ax.set_ylabel(s.replace("-", ":") + "\n" + r"$\gamma$", fontsize=7.5)
            if i == 2:
                ax.set_xlabel(r"$\beta$", fontsize=7.5)
    hd = [Patch(fc=cols[k], ec="0.6", lw=.4, label=labels[k]) for k in (2, 3, 4, 5, 6, 1)]
    hd.append(Patch(fc="white", ec="0.6", lw=.4, label="not computed (no start orbit for 34:33:33)"))
    fig.legend(handles=hd, loc="lower center", ncol=3, fontsize=6.5, frameon=False, handlelength=1.2,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, .1, 1, 1), h_pad=.3, w_pad=.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("fig_map." + ext), dpi=250)
    plt.close(fig)


def fig_flow():
    """Fig. 3: (a)-(c) weak-side limits f_c of the ordered partitions at the three points; (d)-(f) every exit of the
    full system from a three-cluster partition, drawn as the step [f_next, f_last] minus f_c of the ordered partition
    actually present (firing order recorded in the canonical sweeps of rebalance_order_canonical/, same trajectories as
    rebalance/ and rebalance_var/), against that f_c. A bar crossing 0 means the limit lies inside the step of the exit;
    a bar below 0 means the partition was kept past its limit. 2026-11-19: the nearer of the two orderings (used up to
    2026-11-18, when the order was not recorded) was replaced by the recorded order (partition_flow_order_summary)."""
    import partition_flow_order_summary as O
    man = json.loads((O.CAN / "manifest.json").read_text())
    pts = ["e0.3_b1.2_g0.45_50-25-25", "e0.2_b1.28_g0.45_50-25-25", "e0.25_b1_g0.35_50-25-25"]
    lims = {p: limits(p) for p in pts}
    allf = [v["f_c"] for p in pts for v in lims[p].values() if v.get("f_c") is not None]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), gridspec_kw=dict(height_ratios=[1.15, 1], wspace=.12))
    for i, p in enumerate(pts):
        lab = "(%s, %s, %s)" % tuple(q[1:] for q in p.split("_")[:3])
        ax = axes[0, i]
        P = [(k, v) for k, v in lims[p].items() if v.get("f_c") is not None]
        b = np.array([k[1] for k, _ in P]); c = np.array([k[2] for k, _ in P])
        fc = np.array([v["f_c"] for _, v in P]); cr = np.array([v["crossing"] for _, v in P])
        for m, mk in ((0, "s"), (2, "o")):
            sel = cr == m
            sc = ax.scatter(b[sel], c[sel], c=fc[sel], cmap="Blues", vmin=min(allf), vmax=max(allf), marker=mk, s=11,
                            edgecolors="0.35", linewidths=.25)
        ax.set_aspect("equal")
        if i > 0:
            ax.set_yticklabels([])
        ax.set_title("(%s) %s" % ("abc"[i], lab), fontsize=8)
        ax.set_xlabel("$n_2$", fontsize=7.5)
        ax.tick_params(labelsize=6.5, length=2)
        if i == 0:
            ax.set_ylabel("$n_3$", fontsize=7.5)
        ax = axes[1, i]
        xs, lo, hi = [], [], []
        lim = O.load_rows(p)
        for base in sorted(man):
            d = json.loads((O.CAN / base).read_text())
            if d["N"] != 100 or not base.startswith(p.replace("_50-25-25", "") + "_"):
                continue
            S = [st for st in d["steps"] if "sizes" in st]
            for e in O.exits(S, lim, lims[p]):
                if "offset_exact" not in e:
                    continue
                fl, fn = e["f_last"], e["f_next"]
                x = min(e["f_c"], key=lambda v: abs(O.offset(v, fl, fn)))
                xs.append(x); lo.append(fn - x); hi.append(fl - x)
        xs, lo, hi = map(np.array, (xs, lo, hi))
        ax.vlines(xs, lo, hi, color=C[0], lw=.5, alpha=.6)
        ax.axhline(0, color=INK, lw=.6)
        held = int((hi < -.002).sum())
        ax.text(.03, .04, "%d exits; %d kept > 0.002 past $f_c$" % (len(xs), held), transform=ax.transAxes,
                fontsize=6.5, color=INK)
        ax.set_ylim(-.016, .012)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_xlabel("$f_c$ of the partition left", fontsize=7.5)
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_title("(%s)" % "def"[i], fontsize=8, loc="left")
        if i == 0:
            ax.set_ylabel("$f-f_c$ over the step of the exit", fontsize=7.5)
        else:
            ax.set_yticklabels([])
    cb = fig.colorbar(sc, ax=list(axes[0]), shrink=.8, pad=.02)
    cb.set_label("$f_c$", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], ls="", marker="s", ms=3.5, mfc="white", mec="0.35", mew=.4, label="cluster 1 reaches +1"),
          Line2D([], [], ls="", marker="o", ms=3.5, mfc="white", mec="0.35", mew=.4, label="cluster 3 reaches +1")]
    fig.legend(handles=hd, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(.45, .0), ncol=2,
               handletextpad=.2, title="(a)-(c):", title_fontsize=6.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("fig_flow." + ext), dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_sweep()
    fig_rest()
    fig_map()
    fig_flow()
