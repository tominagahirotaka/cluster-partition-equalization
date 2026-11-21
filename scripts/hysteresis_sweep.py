#!/usr/bin/env python3
"""Hysteresis test of the rebalancing, 2026-11-07 (author's approval "１，２とも続けてください"): after the coupling is
weakened until the partition has become more balanced, does it go back to the unbalanced partition when the coupling
is made stronger again?

Forward: the same procedure as scripts/rebalance_sweep.py (start orbit shot at K_end - 0.02 |K1|, steps of
+0.005 |K1|, noise sd 1e-8 with default_rng(step), 10000 time units per step), replayed up to the turning step.
The turning step is taken from analysis/universality/rebalance/<name>: the earliest step whose state there has
exactly three clusters with the smallest largest cluster (the most balanced three-cluster state of that run).
The replay is a new integration and may differ from the earlier run; its own sizes are recorded.
Backward: from the state at the turning step, K is decreased (stronger coupling) in steps of 0.005 |K1|, noise sd
1e-8 with default_rng(10000 + j), 10000 time units per step, down to 10 steps beyond the starting K
(K_end - 0.07 |K1|). Stops early at rest. Each step records rest or the cluster sizes (max-norm distance < 1e-4) and
whether they are the same at t = 8000 and 10000 of the step.
  -> analysis/universality/hysteresis/<name>  (json; the state at the turning step in <name>.npz)
  python3 scripts/hysteresis_sweep.py e0.3_b1.12_g0.5_40-30-30.json
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
REB = ROOT / "analysis/universality/rebalance"
OUT = ROOT / "analysis/universality/hysteresis"


def turning_step(reb):
    three = [q for q in reb["steps"] if not q["rest"] and len(q["sizes"]) == 3]
    best = min(max(q["sizes"]) for q in three)
    return min(q["step"] for q in three if max(q["sizes"]) == best)


def main():
    name = sys.argv[1]
    r = json.loads((MAP / name).read_text())
    reb = json.loads((REB / name).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    sizes = np.array(r["sizes"])
    p = sizes / sizes.sum()
    m, N = len(p), int(sizes.sum())
    kk1 = r["K1"]
    k0 = r["end"]["K_last_good"] - .02 * abs(kk1)
    dk = .005 * abs(kk1)
    st = turning_step(reb)
    out = dict(name=name, eps=r["eps"], beta=r["beta"], gamma=r["gamma"], sizes=sizes.tolist(), K1=kk1, K0=k0,
               turn_step=st, turn_sizes_earlier_run=[q for q in reb["steps"] if q["step"] == st][0]["sizes"],
               host=socket.gethostname(), forward=[], backward=[])
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / name
    w = mr.shoot(np.array(r["end"]["last_good_orbit"]), k0, p)
    lab = np.repeat(np.arange(m), sizes)
    z = np.r_[w[:m][lab], w[m:2 * m][lab]]
    xr, yr = mr.rest_point()
    t0 = time.time()

    def step(z, k, seed, rows, idx):
        z = z + 1e-8 * np.random.default_rng(seed).normal(size=2 * N)

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        s = solve_ivp(rhs, (0, 10000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=[8000., 10000.])
        z = s.y[:, -1]
        rest = bool(np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8)
        row = dict(step=idx, K=float(k), f=float(k / kk1), rest=rest)
        if not rest:
            c1, c2 = clusters(s.y[:, 0], N), clusters(z, N)
            row.update(sizes=c2, settled=c1 == c2)
        rows.append(row)
        out["seconds"] = time.time() - t0
        fn.write_text(json.dumps(out, indent=1))
        return z, rest

    k = k0
    for i in range(st + 1):
        z, rest = step(z, k, i, out["forward"], i)
        if rest:
            out["stop"] = "rest in the forward replay"
            fn.write_text(json.dumps(out, indent=1))
            return
        if i < st:
            k += dk
    np.savez(OUT / (name + ".npz"), z=z, K=k)
    for j in range(1, st + 11):
        z, rest = step(z, k - j * dk, 10000 + j, out["backward"], j)
        if rest:
            out["stop"] = "rest in the backward sweep"
            break
    else:
        out["stop"] = "done"
    fn.write_text(json.dumps(out, indent=1))

    def chain(rows):
        ch = []
        for q in rows:
            lab_ = "rest" if q["rest"] else ":".join(map(str, q["sizes"]))
            if not ch or ch[-1][1] != lab_:
                ch.append(("%.3f" % q["f"], lab_))
        return ch
    print(name, "turn", st, "fwd", chain(out["forward"]), "bwd", chain(out["backward"]), out["stop"], flush=True)


if __name__ == "__main__":
    main()
