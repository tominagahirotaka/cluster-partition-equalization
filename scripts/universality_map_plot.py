#!/usr/bin/env python3
"""Maps of the universality runs (2026-11-06): for each partition (rows) and eps (columns), how the branch started at
strong coupling ends when followed towards weak coupling (universality_map.py outputs). Points of 40:30:30 and
50:25:25 were run only where 34:33:33 had a start orbit.
  -> analysis/universality/map_summary.{png,pdf}
"""
from pathlib import Path
import glob
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
CATS = [("not excitable", "#ffffff"), ("no oscillation found", "#d9d9d9"), ("fold (clusters stable)", "#8c6d31"),
        ("internal first, then other", "#9ecae1"), ("internal first, then subcritical PD", "#3182bd"),
        ("internal first, then supercritical PD", "#e6550d"), ("PD with clusters stable (paper 1)", "#d62728"),
        ("not computed", "#f7f7f7")]


def cat(r):
    if r is None:
        return 7
    if r.get("status") != "ok":
        return 1
    if "seconds" not in r:
        return 7
    k = r["end"]["kind"]
    if k == "internal":
        ce = r["collective_end"]
        t = ce.get("pd_type", {}).get("type") if ce["kind"] == "PD" else None
        return 4 if t == "subcritical" else 5 if t == "supercritical" else 3
    if k == "PD":
        return 6 if r["paper1_scenario"] else 3
    return 2


def main():
    B = np.round(np.arange(.6, 1.4001, .04), 2)
    G = np.round(np.arange(.1, .9501, .05), 2)
    E = (0.2, 0.25, 0.3, 0.35, 0.4)
    parts = ("34-33-33", "40-30-30", "50-25-25")
    data = {s: {(r["eps"], round(r["beta"], 2), round(r["gamma"], 2)): r
                for r in (json.loads(Path(f).read_text()) for f in glob.glob(str(MAP / f"e*_{s}.json")))} for s in parts}
    import mcluster_reduced as mr
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.6), sharex=True, sharey=True)
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
                    if s != "34-33-33" and r is None and data["34-33-33"].get((e, b, g), {}).get("status") != "ok":
                        c = 1 if (e, b, g) in data["34-33-33"] else 7
                    ax.add_patch(plt.Rectangle((b - .02, g - .025), .04, .05, fc=CATS[c][1], ec="white", lw=.4))
            ax.plot(0.7, 0.5, "k+", ms=7) if (s == "34-33-33" and e == 0.3) else None
            ax.set_xlim(.58, 1.42); ax.set_ylim(.075, .975)
            if i == 0:
                ax.set_title(rf"$\varepsilon={e:g}$", fontsize=10)
            if j == 0:
                ax.set_ylabel(s.replace("-", ":") + "\n" + r"$\gamma$")
            if i == 2:
                ax.set_xlabel(r"$\beta$")
    fig.legend(handles=[Patch(fc=c, ec="0.6", label=l) for l, c in CATS[1:7]], loc="lower center", ncol=3, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"analysis/universality/map_summary.{ext}", dpi=150)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
