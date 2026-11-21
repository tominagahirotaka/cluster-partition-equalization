#!/usr/bin/env python3
"""Where does a cluster state go when it breaks past its end, over noise realisations? 2026-11-08 (author's approval
"続けてください").

Uses the end found by scripts/balanced_strong_end.py (<name minus .json>_strong_end[_s<step>].json). The reduced orbit is
rebuilt from the same hysteresis_extend state and carried to the last good K with mcluster_reduced.continue_branch.
At K = K_last_good - J * 0.005 |K1| (J steps of the sweeps past the end) the elements are put on their clusters, noise
sd 1e-8 with default_rng(seed) is added (seeds 1..NSEED), and the full system N = 100 is integrated for 20000 time
units; the cluster sizes at t = 18000 and 20000, or rest.
  -> analysis/universality/hysteresis_ext/<name minus .json>_breaks[_s<step>].json
  python3 scripts/break_statistics.py e0.3_b1.12_g0.5_40-30-30.json [step]
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

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"
J, NSEED = 2, 10


def main():
    name = sys.argv[1]
    suf = "" if len(sys.argv) <= 2 else "_s%d" % int(sys.argv[2])
    e = json.loads((EXT / name).read_text())
    se = json.loads((EXT / (name[:-5] + "_strong_end" + suf + ".json")).read_text())
    d = np.load(EXT / (name + ".npz"))
    mr.E, mr.B, mr.G = e["eps"], e["beta"], e["gamma"]
    kk1, N = e["K1"], int(sum(e["sizes"]))
    step = se.get("step", len(d["K"]))  # the first strong_end runs used the last step and did not record it
    sz, p, w = orbit_from_state(d["Z"][step - 1], float(d["K"][step - 1]), N)
    m = len(p)
    dk = .005 * abs(kk1)
    kg = se["K_last_good"]
    (w, kc), _, _ = mr.continue_branch(w, float(d["K"][step - 1]), p, dk, kg, -1)
    assert abs(kc - kg) < 1e-12 * abs(kk1), (kc, kg)
    k = kg - J * dk
    lab = np.repeat(np.arange(m), sz)
    z0 = np.r_[w[:m][lab], w[m:2 * m][lab]]
    xr, yr = mr.rest_point()

    def rhs(t, u):
        x, y = u[:N], u[N:]
        return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
    out = dict(name=name, step=step, sizes=sz.tolist(), K1=kk1, end=se["end"], K_last_good=kg, f_last_good=kg / kk1,
               J=J, K=k, f=k / kk1, runs=[])
    fn = EXT / (name[:-5] + "_breaks" + suf + ".json")
    for seed in range(1, NSEED + 1):
        z = z0 + 1e-8 * np.random.default_rng(seed).normal(size=2 * N)
        s = solve_ivp(rhs, (0, 20000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15,
                      t_eval=[18000., 20000.])
        z = s.y[:, -1]
        row = dict(seed=seed, rest=bool(np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8))
        if not row["rest"]:
            c1, c2 = clusters(s.y[:, 0], N), clusters(z, N)
            row.update(sizes=c2, settled=c1 == c2)
        out["runs"].append(row)
        fn.write_text(json.dumps(out, indent=1))
    labs = ["rest" if r["rest"] else ":".join(map(str, r["sizes"])) + ("" if r["settled"] else "*") for r in out["runs"]]
    print(name, suf, sz.tolist(), "end f %.4f" % out["f_last_good"], "at f %.4f" % out["f"],
          {v: labs.count(v) for v in dict.fromkeys(labs)}, flush=True)


if __name__ == "__main__":
    main()
