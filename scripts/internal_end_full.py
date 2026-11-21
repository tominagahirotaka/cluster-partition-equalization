#!/usr/bin/env python3
"""(a) of 2026-11-06 (author's request "(b)を先に，(a)を後で確かめてください"): where does the full system go after the
three-cluster branch loses stability within a cluster (end kind internal in universality_map.py)?

At one map point (any partition with N = 100 elements):
 1. the cluster that becomes unstable (largest internal modulus in the map's last bad audit) and the eigenvalues of
    its 2x2 internal monodromy at the last good orbit (sign of the critical multiplier);
 2. for d = 0.002 and 0.01 (times |K1|) past the end (weaker coupling): the reduced orbit shot there (internal
    multipliers ignored), lifted to the N = 100 elements of the partition, plus Gaussian noise of sd 1e-8 on all 200
    coordinates (default_rng(1)); full system, DOP853 (rtol 1e-10, atol 1e-12, max step 0.15) to t = 20000. At
    t = 10000 and 20000: rest (within 1e-8 of the synchronized equilibrium) or the clusters (connected components of
    the max-norm distance < 1e-4) with their sizes; the time when the critical cluster first spreads beyond 1e-3 (samples every time unit).
  -> analysis/universality/internal_end_full/<map name>
  python3 scripts/internal_end_full.py e0.3_b0.72_g0.5_34-33-33.json
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
OUT = ROOT / "analysis/universality/internal_end_full"


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
    w, k = np.array(r["end"]["last_good_orbit"]), r["end"]["K_last_good"]
    im = r["end"]["last_bad_audit"]["internal_moduli"]
    c = int(np.argmax(im))
    out = dict(name=name, eps=r["eps"], beta=r["beta"], gamma=r["gamma"], sizes=sizes.tolist(), K1=kk1, K_end=k,
               f_end=k / kk1, critical_cluster=c, critical_size=int(sizes[c]), host=socket.gethostname(), runs=[])

    def f2(t, v):
        z = v[:2 * m]
        a = np.array([[(1 - z[c] ** 2) / mr.E - k, -1 / mr.E], [mr.E, -mr.E * mr.G]])
        return np.r_[mr.rhs(z, k, p), (a @ v[2 * m:].reshape(2, 2)).ravel()]
    s = solve_ivp(f2, (0, w[-1]), np.r_[w[:2 * m], np.eye(2).ravel()], method="DOP853", rtol=1e-12, atol=1e-14,
                  max_step=.04)
    mu = np.linalg.eigvals(s.y[2 * m:, -1].reshape(2, 2))
    out["internal_multipliers"] = [[float(z.real), float(z.imag)] for z in mu]
    xr, yr = mr.rest_point()
    lab = np.repeat(np.arange(m), sizes)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (.002, .01):
        kp = k + d * abs(kk1)
        wq = mr.shoot(w, kp, p)
        run = dict(d=d, K=kp)
        if wq is None:
            run["stop"] = "reduced orbit not found past the end"
            out["runs"].append(run)
            continue
        z0 = np.r_[wq[:m][lab], wq[m:2 * m][lab]] + 1e-8 * np.random.default_rng(1).normal(size=2 * N)

        def rhs(t, z):
            x, y = z[:N], z[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + kp * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        tt = np.arange(0., 20000.5, 1.)
        sol = solve_ivp(rhs, (0, 20000), z0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=tt)
        idx = np.where(lab == c)[0]
        sp = np.maximum(np.ptp(sol.y[idx], axis=0), np.ptp(sol.y[N + idx], axis=0))
        j = np.where(sp > 1e-3)[0]
        spread = float(tt[j[0]]) if len(j) else None
        for tc in (10000, 20000):
            z = sol.y[:, int(tc)]
            rest = bool(np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8)
            run[f"t{tc}"] = dict(rest=rest, sizes=None if rest else clusters(z, N),
                                 oscillating=bool(np.max(abs(rhs(0, z))) > 1e-6))
        run["critical_spread_time"] = spread
        out["runs"].append(run)
        out["seconds"] = time.time() - t0
        (OUT / name).write_text(json.dumps(out, indent=1))
    (OUT / name).write_text(json.dumps(out, indent=1))
    print(name, "critical", out["critical_size"], "mu", np.round(mu.real, 3),
          [(q["d"], q.get("t20000", {}).get("sizes"), q.get("t20000", {}).get("rest")) for q in out["runs"]], flush=True)


if __name__ == "__main__":
    main()
