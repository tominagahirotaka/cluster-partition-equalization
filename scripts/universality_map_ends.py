#!/usr/bin/env python3
"""Step 1 after the universality map (2026-11-05, author's approval "どれも，やってほしい"): how do the branches end
where the continuation stopped because shooting failed (end kind shoot_failed of universality_map.py, either the
first end or the collective end past an internal instability)?

From the stored last good orbit (w, K) of that end: audit (collective and internal multipliers) at K, and the orbit
shot at K - 0.002 |K1| (back towards stronger coupling) for the trend. Recorded: the collective multipliers of largest
modulus at both, the period at both and dT/dK, the period relative to the start orbit, the smallest cluster gap.
Label (a guide only, not a proof):
  fold-like     largest collective multiplier real, positive, > 0.9 at K and larger than at the back point
  period-growth T(K) / T(start) > 2 or |dT/dK| > 50 / |K1|  (approach to a global bifurcation)
  merge-like    smallest cluster gap < 0.05
  unclassified  otherwise
  python3 scripts/universality_map_ends.py e0.3_b0.84_g0.4_34-33-33.json
  -> analysis/universality/map_ends/<same name>
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
OUT = ROOT / "analysis/universality/map_ends"


def top(r):
    c = sorted((complex(*v) for v in r["collective"]), key=abs, reverse=True)
    return [[z.real, z.imag] for z in c[:3]]


def main():
    name = sys.argv[1]
    r = json.loads((MAP / name).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    p = np.array(r["sizes"], float) / sum(r["sizes"])
    kk1 = r["K1"]
    if r["end"]["kind"] == "shoot_failed":
        which, e = "first", r["end"]
    else:
        which, e = "collective", r["collective_end"]
    w, k = np.array(e["last_good_orbit"]), e["K_last_good"]
    out = dict(name=name, eps=r["eps"], beta=r["beta"], gamma=r["gamma"], which=which, K=k, f=k / kk1, K1=kk1)
    a = mr.audit(w, k, p)
    kb = k - .002 * abs(kk1)
    wb = mr.shoot(w, kb, p)
    out.update(period=float(w[-1]), period_start=float(r["start_audit"]["period"]), top=top(a),
               internal_moduli=a["internal_moduli"], min_cluster_gap=a["min_cluster_gap"])
    if wb is not None:
        b = mr.audit(wb, kb, p)
        out.update(back_K=kb, back_period=float(wb[-1]), back_top=top(b),
                   dT_dK=float((w[-1] - wb[-1]) / (k - kb)))
    c0 = complex(*out["top"][0])
    cb = complex(*out["back_top"][0]) if wb is not None else None
    if c0.imag == 0 and c0.real > .9 and (cb is None or c0.real > cb.real):
        lab = "fold-like"
    elif out["period"] / out["period_start"] > 2 or abs(out.get("dT_dK", 0)) > 50 / abs(kk1):
        lab = "period-growth"
    elif out["min_cluster_gap"] < .05:
        lab = "merge-like"
    else:
        lab = "unclassified"
    out["label"] = lab
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(out, indent=1))
    print(name, which, lab, "top %.4f%+.4fi" % (c0.real, c0.imag), "T %.2f/%.2f" % (out["period"], out["period_start"]),
          flush=True)


if __name__ == "__main__":
    main()
