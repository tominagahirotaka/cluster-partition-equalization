#!/usr/bin/env python3
"""Where does the 33:33:3x:x state of (0.25, 1.32, 0.2) go at f = 0.803? 2026-11-14 (author's approval
"両方進めてください").

From the state at step 5 (f = 0.808) saved by scripts/escape_prepare.py, one step of the sweep is repeated at
K = K + 0.005 |K1| (f = 0.803) for realisations r = R0 .. R1 - 1: noise of sd SD from default_rng(700000 + r) is added
and the full system N = 100 is integrated for 10000 time units. Every 50 time units: the spread of each original
cluster and the distance of the vector of cluster means from the reduced orbit at f = 0.808 (how far the state as a
whole has left it). Outcome: rest or the cluster sizes at t = 8000 and 10000; the time at which the distance first
exceeds 0.1 and the time at which each cluster's spread first exceeds 1e-4.
  -> analysis/universality/escape/<name minus .json>_s<seed>_sd<SD>_r<R0>-<R1>.json
  python3 scripts/escape_runs.py e0.25_b1.32_g0.2_34-33-33.json 15 1e-8 0 25
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from rebalance_sweep import clusters  # noqa: E402
from balanced_strong_end import orbit_from_state  # noqa: E402
from break_membership import labels, spread  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
OUT = ROOT / "analysis/universality/escape"


def main():
    name, seed, sd, r0, r1 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    r = json.loads((MAP / name).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    kk1, N = r["K1"], 100
    stem = OUT / ("%s_s%d" % (name[:-5], seed))
    d = np.load(str(stem) + "_prep.npz")
    z5, k5 = d["z"], float(d["K"])
    lab0 = labels(z5, N)
    order = np.argsort(-np.bincount(lab0), kind="stable")
    lab = np.argsort(order)[lab0]
    m = len(order)
    sz, p, w4 = orbit_from_state(z5, k5, N)
    ref = solve_ivp(lambda t, v: mr.rhs(v, k5, p), (0, w4[-1]), w4[:-1], method="DOP853", rtol=1e-11, atol=1e-13,
                    max_step=.02, dense_output=True).sol(np.linspace(0, w4[-1], 20001))
    k = k5 + .005 * abs(kk1)
    xr, yr = mr.rest_point()
    tt = np.arange(0., 10001., 50.)

    def rhs(t, u):
        x, y = u[:N], u[N:]
        return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
    out = dict(name=name, seed=seed, sd=sd, sizes=sz.tolist(), f_start=k5 / kk1, f=k / kk1, runs=[])
    fn = OUT / ("%s_s%d_sd%g_r%d-%d.json" % (name[:-5], seed, sd, r0, r1))
    for rr in range(r0, r1):
        z = z5 + sd * np.random.default_rng(700000 + rr).normal(size=2 * N)
        s = solve_ivp(rhs, (0, 10000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=tt)
        sp = np.array([spread(s.y[:, i], N, lab, m) for i in range(len(tt))])
        dist = []
        for i in range(len(tt)):
            u = s.y[:, i]
            mv = np.r_[[u[:N][lab == a].mean() for a in range(m)], [u[N:][lab == a].mean() for a in range(m)]]
            dist.append(float(np.min(np.max(abs(ref - mv[:, None]), axis=0))))
        zf = s.y[:, -1]
        rest = bool(np.max(abs(zf[:N] - xr)) < 1e-8 and np.max(abs(zf[N:] - yr)) < 1e-8)
        row = dict(r=rr, rest=rest, t_dist_01=next((float(tt[i]) for i in range(len(tt)) if dist[i] > .1), None),
                   t_spread=[next((float(tt[i]) for i in range(len(tt)) if sp[i, a] > 1e-4), None) for a in range(m)],
                   dist=np.round(dist, 4).tolist()[:60], log10_spread=np.log10(sp[:60] + 1e-300).round(2).tolist())
        if not rest:
            row.update(sizes=clusters(zf, N), sizes_t8000=clusters(s.y[:, -41], N))
        out["runs"].append(row)
        fn.write_text(json.dumps(out, indent=1))
        print(name, "s", seed, "sd", sd, "r", rr, "rest" if rest else row["sizes"], "t_dist", row["t_dist_01"],
              "t_spread", row["t_spread"], flush=True)


if __name__ == "__main__":
    main()
