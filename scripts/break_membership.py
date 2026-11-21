#!/usr/bin/env python3
"""Which cluster breaks, and where do its elements go? 2026-11-08 (author's approval "続けてください").

The runs of scripts/break_statistics.py repeated (same K = K_last_good - 2 * 0.005 |K1|, same seeds 1..10, 20000 time
units), with the elements followed: every 100 time units the spread of each original cluster (max-norm diameter of its
elements in (x, y)) is recorded; the first original cluster whose spread exceeds 1e-4 is the one that breaks first. At
t = 20000 the elements are grouped into clusters (max-norm distance < 1e-4) and the table original cluster -> final
cluster (counts of elements) is recorded. Original clusters are numbered by size, largest first (as the sizes of the
reduced orbit; equal sizes in the order of the cluster means).
  -> analysis/universality/hysteresis_ext/<name minus .json>_members[_s<step>].json
  python3 scripts/break_membership.py e0.3_b1.12_g0.5_40-30-30.json [step]
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from rebalance_sweep import clusters  # noqa: E402
from balanced_strong_end import orbit_from_state  # noqa: E402
from break_statistics import J, NSEED  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"


def labels(z, N):
    x, y = z[:N], z[N:]
    dist = np.maximum(abs(x[:, None] - x[None, :]), abs(y[:, None] - y[None, :]))
    return connected_components(dist < 1e-4, directed=False)[1]


def spread(z, N, lab, m):
    x, y = z[:N], z[N:]
    return [float(max(np.ptp(x[lab == a]), np.ptp(y[lab == a]))) for a in range(m)]


def main():
    name = sys.argv[1]
    suf = "" if len(sys.argv) <= 2 else "_s%d" % int(sys.argv[2])
    e = json.loads((EXT / name).read_text())
    se = json.loads((EXT / (name[:-5] + "_strong_end" + suf + ".json")).read_text())
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
    k = kg - J * dk
    lab = np.repeat(np.arange(m), sz)
    z0 = np.r_[w[:m][lab], w[m:2 * m][lab]]
    # internal multipliers of the reduced orbit at K (which cluster the reduced system says is unstable)
    wk = mr.shoot(w, k, p)
    internal = mr.audit(wk, k, p)["internal_moduli"] if wk is not None else None
    xr, yr = mr.rest_point()
    tt = np.arange(0., 20001., 100.)

    def rhs(t, u):
        x, y = u[:N], u[N:]
        return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
    out = dict(name=name, step=step, sizes=sz.tolist(), K1=kk1, K_last_good=kg, f_last_good=kg / kk1, J=J, K=k,
               f=k / kk1, internal_moduli_at_K=internal, runs=[])
    fn = EXT / (name[:-5] + "_members" + suf + ".json")
    for seed in range(1, NSEED + 1):
        z = z0 + 1e-8 * np.random.default_rng(seed).normal(size=2 * N)
        s = solve_ivp(rhs, (0, 20000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=tt)
        sp = np.array([spread(s.y[:, i], N, lab, m) for i in range(len(tt))])
        first = None
        for i in range(len(tt)):
            over = np.where(sp[i] > 1e-4)[0]
            if len(over):
                first = dict(t=float(tt[i]), clusters=over.tolist(), spreads=sp[i, over].tolist())
                break
        zf = s.y[:, -1]
        row = dict(seed=seed, first_break=first, log10_spread_every_1000=np.log10(sp[::10] + 1e-300).round(2).tolist())
        if np.max(abs(zf[:N] - xr)) < 1e-8 and np.max(abs(zf[N:] - yr)) < 1e-8:
            row["rest"] = True
        else:
            fl = labels(zf, N)
            fsz = np.bincount(fl)
            order = np.argsort(-fsz, kind="stable")
            table = [[int(np.sum((lab == a) & (fl == b))) for b in order] for a in range(m)]
            row.update(rest=False, sizes=fsz[order].tolist(), sizes_t18000=clusters(s.y[:, -21], N),
                       table=table)
        out["runs"].append(row)
        fn.write_text(json.dumps(out, indent=1))
        fb = first["clusters"] if first else None
        tb = first["t"] if first else None
        print(name, suf, "seed", seed, "first break", fb, "t", tb, "final", row.get("sizes", "rest"),
              "table", row.get("table"), flush=True)


if __name__ == "__main__":
    main()
