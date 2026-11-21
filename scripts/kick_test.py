#!/usr/bin/env python3
"""Past the end with repeated noise, 2026-11-08 (author's approval "両方進めてください").

In scripts/beyond_end_detail.py and break_membership.py the noise is added once at t = 0. When the elements of a cluster
come together to the last bit (spread exactly 0, as for (0.3, 1.12, 0.5)), an internally unstable cluster can no longer
break, because nothing is left to grow. Here the noise (sd 1e-8, default_rng(1000 * seed + i)) is added again every
KICK time units, as in the sweeps (where it is added at every step). Same start and K as beyond_end_detail.py. Before
every kick the spread of each original cluster and the cluster sizes are recorded; at the end the motion of the final
cluster means (beyond_end_detail.reduced_motion).
  -> analysis/universality/hysteresis_ext/<name minus .json>_kick.json
  python3 scripts/kick_test.py e0.3_b1.12_g0.5_40-30-30.json 1,2 1,2,3 60000
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
from break_membership import spread  # noqa: E402
from beyond_end_detail import reduced_motion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"
KICK = 1000.


def main():
    name = sys.argv[1]
    js = [int(v) for v in sys.argv[2].split(",")]
    seeds = [int(v) for v in sys.argv[3].split(",")]
    tmax = float(sys.argv[4])
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
    assert abs(kc - kg) < 1e-12 * abs(kk1), (kc, kg)
    lab = np.repeat(np.arange(m), sz)
    z0 = np.r_[w[:m][lab], w[m:2 * m][lab]]
    out = dict(name=name, sizes=sz.tolist(), K1=kk1, K_last_good=kg, f_last_good=kg / kk1, kick=KICK, tmax=tmax,
               runs=[])
    fn = EXT / (name[:-5] + "_kick.json")
    for j in js:
        k = kg - j * dk

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        for seed in seeds:
            z = z0.copy()
            log = []
            for i in range(int(tmax / KICK)):
                z = z + 1e-8 * np.random.default_rng(1000 * seed + i).normal(size=2 * N)
                s = solve_ivp(rhs, (0, KICK), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15,
                              t_eval=[KICK])
                z = s.y[:, -1]
                log.append(dict(t=(i + 1) * KICK, log10_spread=np.log10(np.array(spread(z, N, lab, m)) + 1e-300)
                                .round(2).tolist(), sizes=clusters(z, N)))
            row = dict(J=j, K=k, f=k / kk1, seed=seed, log=log, final=reduced_motion(z, k, N, None))
            first = next((q["t"] for q in log if q["sizes"] != sorted(sz.tolist(), reverse=True)), None)
            row["first_change_t"] = first
            out["runs"].append(row)
            fn.write_text(json.dumps(out, indent=1))
            ch = []
            for q in log:
                lb = ":".join(map(str, q["sizes"]))
                if not ch or ch[-1][1] != lb:
                    ch.append((q["t"], lb))
            fm = row["final"]
            print(name, "J", j, "f %.4f" % (k / kk1), "seed", seed, "chain", ch, "final", fm.get("motion"),
                  "int", fm.get("internal_moduli"), flush=True)


if __name__ == "__main__":
    main()
