#!/usr/bin/env python3
"""Why did the 39 cluster of (0.2, 1.28, 0.45) spread and come together again while the 36 cluster broke? 2026-11-08
(author's approval "３つとも進めてください").

The runs of scripts/break_membership.py repeated for the given seeds (same start: the reduced orbit at K_last_good,
K = K_last_good - 2 * 0.005 |K1|, noise sd 1e-8 with default_rng(seed) at t = 0), for TMAX time units with output every
DT: the spread of each original cluster, the means of each original cluster, and the distance of the vector of
cluster means from the reduced orbit at K_last_good (how far the cluster state as a whole has left that orbit).
  -> analysis/universality/hysteresis_ext/<name minus .json>_transient.json
  python3 scripts/transient_detail.py e0.2_b1.28_g0.45_50-25-25.json 1,9
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from balanced_strong_end import orbit_from_state  # noqa: E402
from break_membership import spread  # noqa: E402
from break_statistics import J  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"
TMAX, DT = 4000., 5.


def main():
    name = sys.argv[1]
    seeds = [int(v) for v in sys.argv[2].split(",")]
    e = json.loads((EXT / name).read_text())
    se = json.loads((EXT / (name[:-5] + "_strong_end.json")).read_text())
    d = np.load(EXT / (name + ".npz"))
    mr.E, mr.B, mr.G = e["eps"], e["beta"], e["gamma"]
    kk1, N = e["K1"], int(sum(e["sizes"]))
    step = se.get("step", len(d["K"]))
    sz, p, w = orbit_from_state(d["Z"][step - 1], float(d["K"][step - 1]), N)
    m = len(p)
    dk = .005 * abs(kk1)
    kg = se["K_last_good"]
    (w, kc), _, _ = mr.continue_branch(w, float(d["K"][step - 1]), p, dk, kg, -1)
    ref = solve_ivp(lambda t, v: mr.rhs(v, kg, p), (0, w[-1]), w[:-1], method="DOP853", rtol=1e-11, atol=1e-13,
                    max_step=.02, dense_output=True).sol(np.linspace(0, w[-1], 20001))
    k = kg - J * dk
    lab = np.repeat(np.arange(m), sz)
    z0 = np.r_[w[:m][lab], w[m:2 * m][lab]]
    tt = np.arange(0., TMAX + 1, DT)

    def rhs(t, u):
        x, y = u[:N], u[N:]
        return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
    out = dict(name=name, sizes=sz.tolist(), K1=kk1, K_last_good=kg, K=k, f=k / kk1, period_ref=float(w[-1]), runs=[])
    for seed in seeds:
        z = z0 + 1e-8 * np.random.default_rng(seed).normal(size=2 * N)
        s = solve_ivp(rhs, (0, TMAX), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=tt)
        sp, mean, dist = [], [], []
        for i in range(len(tt)):
            u = s.y[:, i]
            sp.append(spread(u, N, lab, m))
            mv = np.r_[[u[:N][lab == a].mean() for a in range(m)], [u[N:][lab == a].mean() for a in range(m)]]
            mean.append(mv.round(6).tolist())
            dist.append(float(np.min(np.max(abs(ref - mv[:, None]), axis=0))))
        sp = np.array(sp)
        first = {a: next((float(tt[i]) for i in range(len(tt)) if sp[i, a] > 1e-4), None) for a in range(m)}
        out["runs"].append(dict(seed=seed, t=tt.tolist(), log10_spread=np.log10(sp + 1e-300).round(2).tolist(),
                                means=mean, dist_to_orbit=dist, first_over_1em4=first))
        (EXT / (name[:-5] + "_transient.json")).write_text(json.dumps(out, indent=1))
        print(name, "seed", seed, "first spread > 1e-4 per cluster", first,
              "dist to orbit at t=0,50,100,200,500,1000:", [round(dist[int(t / DT)], 5) for t in (0, 50, 100, 200, 500, 1000)],
              flush=True)
        for t in (0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 4000):
            i = int(t / DT)
            print("   t %5d spread %s dist %.2e" % (t, sp[i].round(8).tolist() if sp[i].max() < 1e-3 else np.log10(sp[i] + 1e-300).round(2).tolist(),
                                                   dist[i]), flush=True)


if __name__ == "__main__":
    main()
