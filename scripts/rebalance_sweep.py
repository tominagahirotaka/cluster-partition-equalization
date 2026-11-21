#!/usr/bin/env python3
"""Follow-up of (a), 2026-11-07 (author's approval "進めてください"): does the shedding of elements by the largest
cluster repeat as the coupling is weakened, and where does it stop?

Full system with N = 100 elements of one universality-map point (internal end). Start: the map's last good
three-cluster orbit, shot at K_end - 0.02 |K1| (stronger coupling, still stable), lifted to the partition. Then
K is increased (weaker coupling) in steps of 0.005 |K1|; at each step Gaussian noise of sd 1e-8 is added to all 200
coordinates (default_rng(step)) and the system is integrated for 10000 time units (DOP853, rtol 1e-10, atol 1e-12,
max step 0.15), carrying the state. At the end of each step: rest (within 1e-8 of the synchronized equilibrium) or
the cluster sizes (max-norm distance < 1e-4) and whether they changed in the last 2000 time units. Stops at rest, at
K = -0.002 |K1|, or after 300 steps.
  -> analysis/universality/rebalance/<map name>
  python3 scripts/rebalance_sweep.py e0.3_b1.12_g0.5_40-30-30.json
"""
from pathlib import Path
import json
import socket
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
OUT = ROOT / "analysis/universality/rebalance"


def clusters(z, N):
    x, y = z[:N], z[N:]
    dist = np.maximum(abs(x[:, None] - x[None, :]), abs(y[:, None] - y[None, :]))
    nc, labs = connected_components(dist < 1e-4, directed=False)
    return sorted(np.bincount(labs).tolist(), reverse=True)


def main():
    name = sys.argv[1]
    r = json.loads((MAP / name).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    sizes = np.array(r["sizes"])
    p = sizes / sizes.sum()
    m, N = len(p), int(sizes.sum())
    kk1 = r["K1"]
    k_end = r["end"]["K_last_good"]
    k = k_end - .02 * abs(kk1)
    w = mr.shoot(np.array(r["end"]["last_good_orbit"]), k, p)
    out = dict(name=name, eps=r["eps"], beta=r["beta"], gamma=r["gamma"], sizes=sizes.tolist(), K1=kk1, K_end=k_end,
               host=socket.gethostname(), steps=[])
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / name
    if w is None:
        out["stop"] = "start orbit not found"
        fn.write_text(json.dumps(out, indent=1))
        return
    lab = np.repeat(np.arange(m), sizes)
    z = np.r_[w[:m][lab], w[m:2 * m][lab]]
    xr, yr = mr.rest_point()
    t0 = time.time()
    for step in range(300):
        if k > -.002 * abs(kk1):
            out["stop"] = "reached weak coupling"
            break
        z = z + 1e-8 * np.random.default_rng(step).normal(size=2 * N)

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        s = solve_ivp(rhs, (0, 10000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=[8000., 10000.])
        z = s.y[:, -1]
        rest = bool(np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8)
        row = dict(step=step, K=float(k), f=float(k / kk1), rest=rest)
        if not rest:
            c1, c2 = clusters(s.y[:, 0], N), clusters(z, N)
            row.update(sizes=c2, settled=c1 == c2)
        out["steps"].append(row)
        out["seconds"] = time.time() - t0
        fn.write_text(json.dumps(out, indent=1))
        if rest:
            out["stop"] = "rest"
            break
        k += .005 * abs(kk1)
    fn.write_text(json.dumps(out, indent=1))
    ch = []
    for q in out["steps"]:
        lab_ = "rest" if q["rest"] else ":".join(map(str, q["sizes"]))
        if not ch or ch[-1][1] != lab_:
            ch.append(("%.3f" % q["f"], lab_))
    print(name, ch, out.get("stop"), flush=True)


if __name__ == "__main__":
    main()
