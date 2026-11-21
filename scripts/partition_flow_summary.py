#!/usr/bin/env python3
"""Summary and draft figure of the partition-flow computation of paper 4 (2026-11-17). No integration: reads the
outputs of scripts/partition_flow.py and of the full-system sweeps (rebalance/, rebalance_var/).

For each parameter point (map name) with partition-flow outputs:
1. weak-side internal limit f_c of every ordered partition (a, b, c) (firing order cluster 1 -> 2 -> 3): the f at which
   the largest internal multiplier modulus reaches 1, by linear interpolation between the last stable row and the next
   weaker row; the cluster that reaches 1; whether the collective multipliers stay below 1 over the stable range;
2. comparison with the full-system sweeps at N = 100: each time a three-cluster partition is left (the sorted sizes
   change between two steps f_last > f_next), the signed offset of the nearer f_c of the two orderings (a, b, c),
   (a, c, b) of the sorted sizes from the interval [f_next, f_last] (0 inside, > 0 if the partition was held past
   f_c, < 0 if it was left before f_c); the full-system clustering gives sorted sizes only, so the ordering is not
   known. Counts within 0.002, 0.005, 0.01;
3. a combined figure of the three points (analysis/paper4_figs/partition_flow_all.{png,pdf}).
  -> analysis/universality/partition_flow_summary.json, analysis/paper4_figs/partition_flow_<point>.{png,pdf}
  python3 scripts/partition_flow_summary.py
"""
from pathlib import Path
from collections import Counter, defaultdict
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
U = ROOT / "analysis/universality"
FIG = ROOT / "analysis/paper4_figs"
TOL = .002


def limits(point):
    out = {}
    for f in sorted(glob.glob(str(U / "partition_flow" / (point + "_p*.json")))):
        d = json.loads(Path(f).read_text())
        if d["status"] != "ok":
            out[tuple(d["sizes"])] = dict(status=d["status"])
            continue
        R = sorted(d["rows"], key=lambda q: q["f"])
        fv = np.array([q["f"] for q in R])
        im = np.array([q["internal_moduli"] for q in R])
        cm = np.array([q["collective_max"] for q in R])
        st = im.max(1) < 1
        if not st.any():
            out[tuple(d["sizes"])] = dict(status="no stable row")
            continue
        i = int(np.argmax(st))
        e = dict(status="ok", f_strong_scanned=float(fv[-1]), end_weak=d["end_weak"], end_strong=d["end_strong"],
                 collective_below_1=bool((cm[i:] < 1).all()), internal_stable_to_strong_end=bool(st[i:].all()))
        if i == 0:
            e.update(f_c=None, f_last_stable=float(fv[0]), crossing=None)
        else:
            m = im[i - 1].max()
            e.update(f_c=float(np.interp(1, [im[i].max(), m], [fv[i], fv[i - 1]])), f_last_stable=float(fv[i]),
                     crossing=int(np.argmax(im[i - 1])))
        out[tuple(d["sizes"])] = e
    return out


def sweeps(point):
    fs = [U / "rebalance" / (point + ".json")] + [Path(f) for f in sorted(glob.glob(str(U / "rebalance_var" /
                                                                                          (point + "_s*.json"))))]
    res = []
    for f in fs:
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        S = [s for s in d["steps"] if "sizes" in s]
        if not S or sum(S[0]["sizes"]) != 100:
            continue
        res.append((f.name, S, d.get("stop")))
    return res


def compare(lim, sw):
    """Signed offset of each exit from the nearer f_c of the two orderings: 0 if f_c lies in [f_next, f_last],
    f_c - f_last > 0 if the partition was still held past its limit, f_c - f_next < 0 if it was left before it
    (changed 2026-11-17 after the third point; the first version counted only exits within a tolerance of 0.002)."""
    cnt, cases, lags = Counter(), defaultdict(list), []
    for name, S, _ in sw:
        k = 0
        while k < len(S):
            s = S[k]["sizes"]
            j = k
            while j + 1 < len(S) and S[j + 1].get("sizes") == s:
                j += 1
            if len(s) == 3 and j + 1 < len(S) and S[j + 1]["f"] < S[j]["f"]:
                a, b, c = sorted(s, reverse=True)
                fl, fn = S[j]["f"], S[j + 1]["f"]
                L = [lim[o]["f_c"] for o in ((a, b, c), (a, c, b)) if o in lim and lim[o].get("f_c") is not None]
                if not L:
                    cnt["outside grid"] += 1
                else:
                    d = min((x - fl if x > fl else x - fn if x < fn else 0. for x in L), key=abs)
                    lags.append(d)
                    kind = "within %g" % TOL if abs(d) <= TOL else "held past f_c" if d > 0 else "left before f_c"
                    cnt[kind] += 1
                    if abs(d) > TOL:
                        cases[kind].append(dict(run=name, sizes=s, f_last=fl, f_next=fn, f_c=L, offset=d))
            k = j + 1
    lags = np.array(lags)
    stats = dict(n=len(lags), within={str(t): int((abs(lags) <= t).sum()) for t in (.002, .005, .01)},
                 max_held=float(lags.max()), max_early=float(-lags.min())) if len(lags) else {}
    return cnt, cases, stats


def plot(point, lim, sw):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    P = [(k, v) for k, v in lim.items() if v.get("f_c") is not None]
    b = np.array([k[1] for k, _ in P])
    c = np.array([k[2] for k, _ in P])
    fc = np.array([v["f_c"] for _, v in P])
    cr = np.array([v["crossing"] for _, v in P])
    for m, mk, lab in ((0, "s", "cluster 1 (a) reaches +1"), (1, "^", "cluster 2 (b) reaches +1"),
                       (2, "o", "cluster 3 (c) reaches +1")):
        sel = cr == m
        if sel.any():
            sc = ax[0].scatter(b[sel], c[sel], c=fc[sel], cmap="Blues", vmin=fc.min(), vmax=fc.max(), marker=mk,
                               s=34, edgecolors="0.3", linewidths=.4, label=lab)
    fig.colorbar(sc, ax=ax[0], label="weak-side internal limit $f_c$")
    ax[0].set_xlabel("size $b$ of cluster 2 (fires after cluster 1)")
    ax[0].set_ylabel("size $c$ of cluster 3")
    ax[0].set_title("(a) reduced system, $a = 100 - b - c$")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_aspect("equal")
    xs, ys = [], []
    for name, S, _ in sw:
        k = 0
        while k < len(S):
            s = S[k]["sizes"]
            j = k
            while j + 1 < len(S) and S[j + 1].get("sizes") == s:
                j += 1
            if len(s) == 3 and j + 1 < len(S) and S[j + 1]["f"] < S[j]["f"]:
                a_, b_, c_ = sorted(s, reverse=True)
                L = [lim[o]["f_c"] for o in ((a_, b_, c_), (a_, c_, b_)) if o in lim and lim[o].get("f_c") is not None]
                if L:
                    fm = .5 * (S[j]["f"] + S[j + 1]["f"])
                    xs.append(min(L, key=lambda x: abs(x - fm)))
                    ys.append(fm)
            k = j + 1
    xs, ys = np.array(xs), np.array(ys)
    lo, hi = min(xs.min(), ys.min()) - .01, max(xs.max(), ys.max()) + .01
    ax[1].plot([lo, hi], [lo, hi], "-", color="0.75", lw=.8, zorder=0)
    ax[1].scatter(xs, ys, s=14, color="C0", edgecolors="white", linewidths=.4)
    ax[1].set_xlim(lo, hi)
    ax[1].set_ylim(lo, hi)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("reduced-system limit $f_c$ (nearer of the two orderings)")
    ax[1].set_ylabel("full system: midpoint of the exit step")
    ax[1].set_title("(b) exits from three-cluster partitions, $N = 100$")
    fig.suptitle(point, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("partition_flow_%s.%s" % (point, ext)), dpi=150)
    plt.close(fig)


def exit_pairs(lim, sw):
    xs, ys = [], []
    for name, S, _ in sw:
        k = 0
        while k < len(S):
            s = S[k]["sizes"]
            j = k
            while j + 1 < len(S) and S[j + 1].get("sizes") == s:
                j += 1
            if len(s) == 3 and j + 1 < len(S) and S[j + 1]["f"] < S[j]["f"]:
                a_, b_, c_ = sorted(s, reverse=True)
                L = [lim[o]["f_c"] for o in ((a_, b_, c_), (a_, c_, b_)) if o in lim and lim[o].get("f_c") is not None]
                if L:
                    fm = .5 * (S[j]["f"] + S[j + 1]["f"])
                    xs.append(min(L, key=lambda x: abs(x - fm)))
                    ys.append(fm)
            k = j + 1
    return np.array(xs), np.array(ys)


def plot_all(pts):
    lab = {p: "(%s, %s, %s)" % tuple(q[1:] for q in p.split("_")[:3]) for p in pts}
    fig = plt.figure(figsize=(10, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    lims = {p: limits(p) for p in pts}
    allf = [v["f_c"] for p in pts for v in lims[p].values() if v.get("f_c") is not None]
    for i, p in enumerate(pts):
        ax = fig.add_subplot(gs[0, i])
        P = [(k, v) for k, v in lims[p].items() if v.get("f_c") is not None]
        b = np.array([k[1] for k, _ in P]); c = np.array([k[2] for k, _ in P])
        fc = np.array([v["f_c"] for _, v in P]); cr = np.array([v["crossing"] for _, v in P])
        for m, mk in ((0, "s"), (2, "o")):
            sel = cr == m
            sc = ax.scatter(b[sel], c[sel], c=fc[sel], cmap="Blues", vmin=min(allf), vmax=max(allf), marker=mk, s=16,
                            edgecolors="0.3", linewidths=.3)
        ax.set_aspect("equal")
        ax.set_title("(%s) %s" % ("abc"[i], lab[p]), fontsize=9)
        ax.set_xlabel("$n_2$")
        if i == 0:
            ax.set_ylabel("$n_3$")
    cb = fig.colorbar(sc, ax=fig.axes[:3], shrink=.85, pad=.02)
    cb.set_label("$f_c$")
    ax = fig.add_subplot(gs[1, :2])
    for i, (p, mk) in enumerate(zip(pts, ("o", "s", "^"))):
        xs, ys = exit_pairs(lims[p], sweeps(p))
        ax.scatter(xs, ys - xs, s=12, marker=mk, color="C%d" % i, label=lab[p] + ", %d exits" % len(xs))
    ax.axhline(0, color="0.6", lw=.8)
    ax.set_xlabel("$f_c$ of the partition left (nearer of the two orderings)")
    ax.set_ylabel("$f_c$ minus... ".replace("$f_c$ minus... ", "exit $f$ (step midpoint) $-\\ f_c$"))
    ax.set_title("(d) exits of the full system, $N = 100$", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(FIG / "partition_flow_all.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "partition_flow_all.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    # The three points of Fig. 3 (2026-11-17). The three points added on 2026-11-18 are compared with the recorded
    # firing order in partition_flow_order_summary.py; this script (nearer of two orderings) is kept for those three.
    pts = ["e0.25_b1_g0.35_50-25-25", "e0.2_b1.28_g0.45_50-25-25", "e0.3_b1.2_g0.45_50-25-25"]
    summary = {}
    for p in pts:
        lim = limits(p)
        sw = sweeps(p)
        cnt, cases, stats = compare(lim, sw)
        ok = [v for v in lim.values() if v["status"] == "ok"]
        summary[p] = dict(
            partitions=len(lim), status=dict(Counter(v["status"] for v in lim.values())),
            crossing_cluster=dict(Counter(str(v.get("crossing")) for v in ok)),
            crossing_not_largest=sum(1 for k, v in lim.items() if v.get("crossing") is not None
                                     and k[v["crossing"]] < max(k)),
            collective_below_1_over_stable_range=sum(v["collective_below_1"] for v in ok),
            internal_stable_to_strong_end=sum(v["internal_stable_to_strong_end"] for v in ok),
            f_c={"%d-%d-%d" % k: v.get("f_c") for k, v in lim.items()},
            sweeps_N100=len(sw), exits=dict(cnt), exit_offsets=stats, exceptions=cases, tol=TOL)
        plot(p, lim, sw)
        print(p, summary[p]["status"], summary[p]["crossing_cluster"], "not largest", summary[p]["crossing_not_largest"],
              "sweeps", len(sw), dict(cnt), stats)
    (U / "partition_flow_summary.json").write_text(json.dumps(summary, indent=1))
    plot_all(pts)


if __name__ == "__main__":
    main()
