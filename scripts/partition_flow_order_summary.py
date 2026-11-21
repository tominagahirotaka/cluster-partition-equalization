#!/usr/bin/env python3
"""Exits of the full system compared with the internal limit f_c of the ordered partition actually present
(paper 4, 2026-11-18). No integration.

partition_flow_summary.py compared each exit from a three-cluster partition with the nearer f_c of the two orderings
(a, b, c), (a, c, b) of the sorted sizes, because the sweeps did not record the firing order. The canonical sweeps of
scripts/rebalance_order.py (collected by scripts/collect_order_canonical.py; same trajectories as rebalance/ and
rebalance_var/) record the sizes in firing order from the largest cluster at the end of every step. Here each exit
is compared with the f_c of that ordered partition; the nearer-of-two offset is kept for comparison.

Offset of an exit (as in partition_flow_summary.py): 0 if f_c lies in [f_next, f_last], f_c - f_last > 0 if the
partition was still held past f_c, f_c - f_next < 0 if it was left before f_c. Cyclic rotations of an ordering with
two largest clusters (e.g. 34:32:34 and 34:34:32) are the same state; their f_c agree to 1e-13.

f_c here: every f at which the largest internal multiplier modulus of the ordered partition reaches 1 when f is
lowered (linear interpolation between rows); the offset uses the crossing nearest to the exit step. limits() of
partition_flow_summary.py takes the weak end of the weakest stable row instead. The two agree for every partition of
the first three points (checked 2026-11-18), but at (0.4, 1.04, 0.15) the multipliers fall below 1 again near the
weak end of the continuation (where the continuation fails), and limits() then returns that end or None. The same
crossings are listed for the largest collective multiplier modulus.

Range (2026-11-19): an exit is compared only if the continued branch of its ordered partition reaches the exit step,
i.e. f_next >= (weakest continued f) - TOL and f_last <= (strongest continued f) + TOL, TOL = one sweep step (0.005).
At (0.35, 0.92, 0.65) most continued branches end at a fold (shoot_failed, collective multiplier growing) above the f
of the exit, and the full system is then on another periodic orbit; such exits are counted as "outside continued
range" and not compared.

  -> analysis/universality/partition_flow_order_summary.json
  python3 scripts/partition_flow_order_summary.py
"""
from pathlib import Path
from collections import Counter
import glob
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from partition_flow_summary import limits  # noqa: E402

U = ROOT / "analysis/universality"
TOL = .005
CAN = U / "rebalance_order_canonical"


def offset(x, fl, fn):
    return x - fl if x > fl else x - fn if x < fn else 0.


def crossings(d, key):
    """f values at which the multiplier modulus (internal max or collective max) reaches 1 as f is lowered."""
    R = sorted(d["rows"], key=lambda q: -q["f"])
    fv = np.array([q["f"] for q in R])
    m = np.array([max(q["internal_moduli"]) if key == "internal" else q["collective_max"] for q in R])
    return [float(np.interp(1, [m[j - 1], m[j]], [fv[j - 1], fv[j]])) for j in range(1, len(R))
            if m[j - 1] < 1 <= m[j]]


def load_rows(lp):
    out = {}
    for f in glob.glob(str(U / "partition_flow" / (lp + "_p*.json"))):
        d = json.loads(Path(f).read_text())
        if d["status"] == "ok":
            fs = [q["f"] for q in d["rows"]]
            out[tuple(d["sizes"])] = dict(internal=crossings(d, "internal"), collective=crossings(d, "collective"),
                                          fmin=min(fs), fmax=max(fs))
    return out


def nearest(xs, fl, fn):
    return min((offset(x, fl, fn) for x in xs), key=abs) if xs else None


def exits(S, lim, old):
    out = []
    k = 0
    while k < len(S):
        s = S[k]["sizes"]
        j = k
        while j + 1 < len(S) and S[j + 1].get("sizes") == s:
            j += 1
        if len(s) == 3 and j + 1 < len(S) and S[j + 1]["f"] < S[j]["f"]:
            a, b, c = sorted(s, reverse=True)
            fl, fn = S[j]["f"], S[j + 1]["f"]
            orders = {tuple(S[i]["order"]) if S[i].get("order") else None for i in range(k, j + 1)}
            o = tuple(S[j]["order"]) if S[j].get("order") else None
            e = dict(sizes=[a, b, c], f_last=fl, f_next=fn, order=o, order_changed=len(orders) > 1,
                     settled=S[j].get("settled"))
            two = [x for q in ((a, b, c), (a, c, b)) if q in lim for x in lim[q]["internal"]]
            if two:
                e["offset_nearer"] = nearest(two, fl, fn)
            if o is not None and o in lim:
                e["in_range"] = bool(fn >= lim[o]["fmin"] - TOL and fl <= lim[o]["fmax"] + TOL)
                e["continued_range"] = [lim[o]["fmin"], lim[o]["fmax"]]
            if o is not None and o in lim and lim[o]["internal"] and e["in_range"]:
                e["f_c"] = lim[o]["internal"]
                e["offset_exact"] = nearest(lim[o]["internal"], fl, fn)
                e["exact_is_nearer"] = abs(e["offset_exact"] - e.get("offset_nearer", np.nan)) < 1e-12
                e["offset_collective"] = nearest(lim[o]["collective"], fl, fn)
                e["offset_old_limits"] = (offset(old[o]["f_c"], fl, fn) if o in old and old[o].get("f_c")
                                          is not None else None)
            out.append(e)
        k = j + 1
    return out


def stats(v):
    v = np.array(v)
    if not len(v):
        return {}
    return dict(n=len(v), within={str(t): int((abs(v) <= t).sum()) for t in (.002, .005, .01)},
                max_held=float(v.max()), max_early=float(-v.min()),
                early_beyond_0_001=int((v < -.001).sum()), held_beyond_0_005=int((v > .005).sum()))


def main():
    man = json.loads((CAN / "manifest.json").read_text())
    by_point = {}
    for base, m in man.items():
        if m["status"] != "canonical":
            continue
        d = json.loads((CAN / base).read_text())
        if d["N"] != 100:
            continue
        lp = "%s_50-25-25" % "_".join(d["name"].split("_")[:3])
        by_point.setdefault(lp, []).append((base, d))
    summary = {}
    for lp, runs in sorted(by_point.items()):
        nf = len(glob.glob(str(U / "partition_flow" / (lp + "_p*.json"))))
        if nf < 239:
            summary[lp] = dict(status="partition-flow outputs incomplete (%d of 239)" % nf, sweeps=len(runs))
            print(lp, summary[lp])
            continue
        lim, old = load_rows(lp), limits(lp)
        E = []
        for base, d in sorted(runs):
            S = [s for s in d["steps"] if "sizes" in s]
            for e in exits(S, lim, old):
                e["run"] = base
                E.append(e)
        ex = [e for e in E if "offset_exact" in e]
        cnt = Counter("no order" if e["order"] is None else "order not in grid" if e["order"] not in lim
                      else "outside continued range" if not e["in_range"]
                      else "no internal crossing" if "offset_exact" not in e else "compared" for e in E)
        summary[lp] = dict(
            sweeps=len(runs), exits=len(E), counts=dict(cnt),
            exact=stats([e["offset_exact"] for e in ex]),
            nearer_same_exits=stats([e["offset_nearer"] for e in ex if "offset_nearer" in e]),
            exact_differs_from_nearer=sum(1 for e in ex if not e["exact_is_nearer"]),
            order_changed_while_held=sum(e["order_changed"] for e in E),
            multiple_internal_crossings=sum(len(e["f_c"]) > 1 for e in ex),
            old_limits_differs=sum(e["offset_old_limits"] is None or abs(e["offset_old_limits"] - e["offset_exact"])
                                   > 1e-12 for e in ex),
            exceptions=[e for e in ex if abs(e["offset_exact"]) > .002],
            uncompared=[e for e in E if "offset_exact" not in e])
        s = summary[lp]
        print(lp, "sweeps", s["sweeps"], "exits", s["exits"], s["counts"])
        print("   exact ", s["exact"])
        print("   nearer", s["nearer_same_exits"], "exact differs from nearer:", s["exact_differs_from_nearer"],
              "order changed while held:", s["order_changed_while_held"],
              "multiple crossings:", s["multiple_internal_crossings"], "old limits() differs:", s["old_limits_differs"])
    (U / "partition_flow_order_summary.json").write_text(json.dumps(summary, indent=1, default=list))


if __name__ == "__main__":
    main()
