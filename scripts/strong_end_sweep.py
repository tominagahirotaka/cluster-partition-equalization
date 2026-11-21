#!/usr/bin/env python3
"""Supplement to scripts/balanced_strong_end.py, 2026-11-08 (author's approval "続けてください"): at the points where
the full system one step (0.005 |K1|) past the reduced end still kept the balanced partition, the full system is swept on
past the end as in the sweeps (from the reduced orbit at the last good K, steps of 0.005 |K1| to stronger coupling,
noise sd 1e-8 with default_rng(20000 + j), 10000 time units per step), up to NMAX steps or until the partition changes
or rest. The largest internal multiplier of the reduced orbit at each K is recorded with it.
  -> analysis/universality/hysteresis_ext/<name minus .json>_strong_sweep.json
  python3 scripts/strong_end_sweep.py e0.4_b1.2_g0.2_40-30-30.json
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
NMAX = 40


def main():
    name = sys.argv[1]
    e = json.loads((EXT / name).read_text())
    se = json.loads((EXT / (name[:-5] + "_strong_end.json")).read_text())
    d = np.load(EXT / (name + ".npz"))
    mr.E, mr.B, mr.G = e["eps"], e["beta"], e["gamma"]
    kk1, N = e["K1"], int(sum(e["sizes"]))
    sz, p, w = orbit_from_state(d["Z"][-1], float(d["K"][-1]), N)
    m = len(p)
    kg = se["K_last_good"]
    dk = .005 * abs(kk1)
    (w, kc), _, _ = mr.continue_branch(w, float(d["K"][-1]), p, dk, kg, -1)
    assert abs(kc - kg) < 1e-12 * abs(kk1), (kc, kg)
    lab = np.repeat(np.arange(m), sz)
    z = np.r_[w[:m][lab], w[m:2 * m][lab]]
    xr, yr = mr.rest_point()
    out = dict(name=name, sizes=sz.tolist(), K1=kk1, K_last_good=kg, f_last_good=kg / kk1, steps=[])
    fn = EXT / (name[:-5] + "_strong_sweep.json")
    wr = w
    for j in range(1, NMAX + 1):
        k = kg - j * dk
        z = z + 1e-8 * np.random.default_rng(20000 + j).normal(size=2 * N)

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        s = solve_ivp(rhs, (0, 10000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=[8000., 10000.])
        z = s.y[:, -1]
        row = dict(step=j, K=float(k), f=float(k / kk1))
        if wr is not None:
            wr = mr.shoot(wr, k, p)
        if wr is not None:
            r = mr.audit(wr, k, p)
            row.update(reduced_internal=r["internal_moduli"], reduced_collective_max=r["collective_max"],
                       period=r["period"])
        row["rest"] = bool(np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8)
        if not row["rest"]:
            c1, c2 = clusters(s.y[:, 0], N), clusters(z, N)
            row.update(sizes=c2, settled=c1 == c2)
        out["steps"].append(row)
        fn.write_text(json.dumps(out, indent=1))
        if row["rest"] or row["sizes"] != sorted(sz.tolist(), reverse=True):
            break
    q = out["steps"][-1]
    print(name, sz.tolist(), "end f %.4f" % out["f_last_good"], "steps", len(out["steps"]), "f %.4f" % q["f"],
          "rest" if q["rest"] else q["sizes"], "int", [round(v, 4) for v in q.get("reduced_internal", [])], flush=True)


if __name__ == "__main__":
    main()
