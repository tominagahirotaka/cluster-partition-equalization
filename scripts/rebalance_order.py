#!/usr/bin/env python3
"""Order of firing along the weakening sweeps, 2026-11-18 (author's instruction "もう少し計算すれば改善するような項目
があれば計算を進めてもらった方がいいです").

The sweeps of scripts/rebalance_variants.py record only the cluster sizes, so each exit could be compared with the
internal limit f_c of either of two ordered partitions (paper 4, Sec. IV B, Fig. 3). This script repeats a sweep with
exactly the same procedure, seeds and integrator settings, and in addition records at the end of every step the order
in which the clusters fire. The only change is the list t_eval, which adds output times in the last 300 time units of
each step; in solve_ivp these values are interpolated from the dense output and do not change the steps taken, so the
trajectory is the same as in the earlier run on the same software.

Per step, besides the fields of rebalance_variants.py:
  order: cluster sizes in firing order, starting with the largest cluster (ties: the one that fires first in the
         window), i.e. (n1, n2, n3) in the notation of scripts/partition_flow.py, where cluster 2 fires just after
         cluster 1; None if a cluster has no upward crossing of x = 0 in the window;
  offsets: times from the firing of the largest cluster to the firing of the others, in the same order.
  -> analysis/universality/rebalance_order/<map name minus .json>_s<seed>_N<N>[_T<T>][_D<D>].json
  python3 scripts/rebalance_order.py e0.3_b1.2_g0.45_50-25-25.json 3 100 [T [D]]
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
from rebalance_sweep import clusters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
OUT = ROOT / "analysis/universality/rebalance_order"
WIN, DT = 300., .05


def firing_order(z, xs, ts, N):
    """Sizes in firing order from the largest cluster, and the offsets; xs: x of all elements on the times ts."""
    x, y = z[:N], z[N:]
    dist = np.maximum(abs(x[:, None] - x[None, :]), abs(y[:, None] - y[None, :]))
    nc, labs = connected_components(dist < 1e-4, directed=False)
    size = np.bincount(labs)
    up = {}
    for c in range(nc):
        xc = xs[labs == c].mean(axis=0)
        i = np.nonzero((xc[:-1] < 0) & (xc[1:] >= 0))[0]
        if len(i) == 0:
            return None, None
        up[c] = ts[i] - xc[i] * (ts[i + 1] - ts[i]) / (xc[i + 1] - xc[i])
    big = [c for c in range(nc) if size[c] == size.max()]
    c1 = min(big, key=lambda c: up[c][0])
    t1 = up[c1][-1]
    # the last firing of cluster 1 that is followed by a firing of every other cluster in the window
    cand = [t for t in up[c1] if all((up[c] > t).any() for c in up if c != c1)]
    if not cand:
        return None, None
    t1 = cand[-1]
    nxt = {c: up[c][up[c] > t1][0] - t1 for c in up if c != c1}
    seq = sorted(nxt, key=nxt.get)
    return [int(size[c1])] + [int(size[c]) for c in seq], [float(nxt[c]) for c in seq]


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
               K_end=k_end, T=T, D=D, host=socket.gethostname(), numpy=np.__version__,
               scipy=__import__("scipy").__version__, steps=[])
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / ("%s_s%d_N%d%s%s.json" % (name[:-5], seed, N, "" if T == 10000. else "_T%d" % T, "" if D == .005 else "_D%g" % D))
    if w is None:
        out["stop"] = "start orbit not found"
        fn.write_text(json.dumps(out, indent=1))
        return
    lab = np.repeat(np.arange(m), sizes)
    z = np.r_[w[:m][lab], w[m:2 * m][lab]]
    xr, yr = mr.rest_point()
    ts = np.linspace(T - WIN, T, int(round(WIN / DT)) + 1)
    t_eval = np.unique(np.r_[.8 * T, ts])
    i8, iw = np.searchsorted(t_eval, .8 * T), np.searchsorted(t_eval, ts)
    t0 = time.time()
    for step in range(nmax):
        if k > -.002 * abs(kk1):
            out["stop"] = "reached weak coupling"
            break
        z = z + 1e-8 * np.random.default_rng(100000 * seed + step).normal(size=2 * N)

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        s = solve_ivp(rhs, (0, T), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=t_eval)
        z = s.y[:, -1]
        rest = bool(np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8)
        row = dict(step=step, K=float(k), f=float(k / kk1), rest=rest)
        if not rest:
            c1, c2 = clusters(s.y[:, i8], N), clusters(z, N)
            order, offs = firing_order(z, s.y[:N, iw], ts, N)
            row.update(sizes=c2, settled=c1 == c2, order=order, offsets=offs)
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
    print(name, "seed", seed, "N", N, "T", T, "D", D, out.get("stop"), flush=True)


if __name__ == "__main__":
    main()
