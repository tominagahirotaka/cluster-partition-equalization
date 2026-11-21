#!/usr/bin/env python3
"""Dependence of the rebalancing sweep on the noise realisation and on N, 2026-11-09 (author's request "雑音の実現や N
を変えたときの依存を調べてください").

The procedure of scripts/rebalance_sweep.py (start orbit shot at K_end - 0.02 |K1|, steps of +0.005 |K1| to weaker
coupling, noise sd 1e-8 on every coordinate at every step, 10000 time units per step, stop at rest, at
K = -0.002 |K1| or after 300 steps), with two changes:
  seed s: the noise of step j is drawn from default_rng(100000 * s + j) (s = 0 gives default_rng(j), the earlier run);
  N: the partition of the map point scaled to N elements (40:30:30 -> 80:60:60 for N = 200), so that the cluster
     fractions and the reduced start orbit are the same.
  T (optional, 2026-11-14, author's "続けてください" for "段の時間を 2 倍と 1/2 倍に"): time units per step
     (default 10000); the sizes are compared at 0.8 T and T. For T != 10000 the output name ends in _T<T>.
  D (optional, 2026-11-15, author's "続けてください" for the dependence on the step): the step in K in units of |K1|
     (default 0.005); the step limit is 300 * 0.005 / D, so that the same range of f is covered. For D != 0.005 the
     output name ends in _D<D>.
  -> analysis/universality/rebalance_var/<map name minus .json>_s<seed>_N<N>[_T<T>][_D<D>].json
  python3 scripts/rebalance_variants.py e0.3_b1.12_g0.5_40-30-30.json 3 100 [T [D]]
"""
from pathlib import Path
import json
import socket
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from rebalance_sweep import clusters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
OUT = ROOT / "analysis/universality/rebalance_var"


def main():
    name, seed, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    T = float(sys.argv[4]) if len(sys.argv) > 4 else 10000.
    D = float(sys.argv[5]) if len(sys.argv) > 5 else .005
    nmax = int(round(300 * .005 / D))
    r = json.loads((MAP / name).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    sizes0 = np.array(r["sizes"])
    sizes = sizes0 * N // int(sizes0.sum())
    assert sizes.sum() == N and np.allclose(sizes / N, sizes0 / sizes0.sum()), sizes
    p = sizes / N
    m = len(p)
    kk1 = r["K1"]
    k_end = r["end"]["K_last_good"]
    k = k_end - .02 * abs(kk1)
    w = mr.shoot(np.array(r["end"]["last_good_orbit"]), k, p)
    out = dict(name=name, seed=seed, N=N, eps=r["eps"], beta=r["beta"], gamma=r["gamma"], sizes=sizes.tolist(), K1=kk1,
               K_end=k_end, T=T, D=D, host=socket.gethostname(), steps=[])
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / ("%s_s%d_N%d%s%s.json" % (name[:-5], seed, N, "" if T == 10000. else "_T%d" % T, "" if D == .005 else "_D%g" % D))
    if w is None:
        out["stop"] = "start orbit not found"
        fn.write_text(json.dumps(out, indent=1))
        return
    lab = np.repeat(np.arange(m), sizes)
    z = np.r_[w[:m][lab], w[m:2 * m][lab]]
    xr, yr = mr.rest_point()
    t0 = time.time()
    for step in range(nmax):
        if k > -.002 * abs(kk1):
            out["stop"] = "reached weak coupling"
            break
        z = z + 1e-8 * np.random.default_rng(100000 * seed + step).normal(size=2 * N)

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        s = solve_ivp(rhs, (0, T), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=[.8 * T, T])
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
        k += D * abs(kk1)
    else:
        out["stop"] = "%d steps" % nmax
    fn.write_text(json.dumps(out, indent=1))
    ch = []
    for q in out["steps"]:
        lab_ = "rest" if q["rest"] else ":".join(map(str, q["sizes"]))
        if not ch or ch[-1][1] != lab_:
            ch.append(("%.3f" % q["f"], lab_))
    print(name, "seed", seed, "N", N, "T", T, "D", D, ch, out.get("stop"), flush=True)


if __name__ == "__main__":
    main()
