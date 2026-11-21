#!/usr/bin/env python3
"""Stable ranges of three-cluster partitions in the reduced system, for the figure of the "flow of equalization" of
paper 4, 2026-11-17 (author's approval "3.の図はぜひ作ってください"; item "縮約系で、各分割の安定な範囲（内部と集団）を分割
の関数として計算" of paper4/claims_scope.md).

For one parameter point (a map file of the 50:25:25 branch) and one partition (a, b, c) of N = 100 elements with
a = 100 - b - c on cluster 1 (the cluster of 50 of the start orbit), b on cluster 2 and c on cluster 3:
1. start: the map orbit at K_last_good, shot at K0 = K_last_good - 0.05 |K1| (stronger coupling, f0 = K0 / K1);
2. homotopy at K0: the fractions are moved on a straight line from (0.5, 0.25, 0.25) to (a, b, c) / 100 in steps of at
   most 0.005 in any fraction (halved on shooting failure, down to 1/64 of the step);
3. scan in K with the partition fixed, in steps of D |K1| (D = 0.01 by default, optional fourth argument), without
   stopping at instabilities (only the orbit is followed): towards weaker coupling until shooting fails (the step is
   then halved down to 1/16) or the clusters merge, and towards stronger coupling up to f0 + 0.4. Every point is audited (mcluster_reduced.audit): period,
   internal multiplier modulus of each cluster, collective multipliers, firing offsets.
  -> analysis/universality/partition_flow/<map name minus .json>_p<a>-<b>-<c>.json
  python3 scripts/partition_flow.py e0.3_b1.2_g0.45_50-25-25.json 30 25 [D]
"""
from pathlib import Path
import json
import socket
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
OUT = ROOT / "analysis/universality/partition_flow"
KEYS = ("period", "internal_moduli", "collective", "collective_max", "firing_offsets", "min_cluster_gap",
        "return_error")


def row(w, k, p, kk1):
    r = mr.audit(w, k, p)
    return dict(f=k / kk1, K=float(k), **{q: r[q] for q in KEYS})


def main():
    name, b, c = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    D = float(sys.argv[4]) if len(sys.argv) > 4 else .01
    r = json.loads((MAP / name).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    kk1 = r["K1"]
    p0 = np.array(r["sizes"]) / sum(r["sizes"])
    p1 = np.array([100 - b - c, b, c]) / 100
    k0 = r["end"]["K_last_good"] - .05 * abs(kk1)
    dk = D * abs(kk1)
    out = dict(name=name, eps=r["eps"], beta=r["beta"], gamma=r["gamma"], K1=kk1, sizes=[100 - b - c, b, c], K0=k0,
               f0=k0 / kk1, D=D, host=socket.gethostname(), rows=[])
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / ("%s_p%d-%d-%d.json" % (name[:-5], 100 - b - c, b, c))
    t0 = time.time()

    def save(status):
        out.update(status=status, seconds=time.time() - t0)
        out["rows"].sort(key=lambda q: -q["f"])
        fn.write_text(json.dumps(out))
        print(name, out["sizes"], status, "rows", len(out["rows"]),
              "f %.3f..%.3f" % (out["rows"][-1]["f"], out["rows"][0]["f"]) if out["rows"] else "", "%.0f s" % out["seconds"],
              flush=True)

    w = mr.shoot(np.array(r["end"]["last_good_orbit"]), k0, p0)
    if w is None:
        return save("start_failed")
    # 2. homotopy in the fractions at K0
    n = max(1, int(np.ceil(max(abs(p1 - p0)) / .005)))
    t, dt = 0., 1. / n
    while t < 1 - 1e-12:
        tn = min(1., t + dt)
        w2 = mr.shoot(w, k0, p0 + tn * (p1 - p0))
        if w2 is None:
            dt /= 2
            if dt < 1. / n / 64:
                out["homotopy_t"] = t
                return save("homotopy_failed")
            continue
        w, t = w2, tn
        dt = min(1. / n, dt * 2)
    out["rows"].append(row(w, k0, p1, kk1))
    # 3. scan in K, partition fixed
    for direction, nmax in ((1, 10 ** 6), (-1, int(round(.4 / D)))):
        wk, k, d, steps = w, k0, dk, 0
        end = "limit"
        while steps < nmax:
            kn = k + direction * d
            if kn > -.002 * abs(kk1):
                end = "reached weak coupling"
                break
            w2 = mr.shoot(wk, kn, p1)
            if w2 is None:
                d /= 2
                if d < dk / 16:
                    end = "shoot_failed"
                    break
                continue
            q = row(w2, kn, p1, kk1)
            if q["min_cluster_gap"] < 1e-4:
                end = "clusters_merge"
                break
            out["rows"].append(q)
            wk, k, steps = w2, kn, steps + 1
        out["end_weak" if direction == 1 else "end_strong"] = dict(kind=end, f_last=k / kk1)
    save("ok")


if __name__ == "__main__":
    main()
