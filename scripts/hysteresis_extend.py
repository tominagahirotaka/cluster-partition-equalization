#!/usr/bin/env python3
"""Continuation of scripts/hysteresis_sweep.py, 2026-11-07 (author's approval "１，２とも進めてください"): how far to
the strong-coupling side does the balanced partition persist, and which states appear there?

From the state at the turning step (analysis/universality/hysteresis/<name>.npz) the backward sweep is replayed with the
same seeds (default_rng(10000 + j), noise sd 1e-8, 10000 time units per step, steps of 0.005 |K1| to stronger coupling)
and checked against the earlier sizes; it is then continued up to EXTRA further steps. Stops at rest. The state at the
end of every step is kept, so that the states which appear can be examined (scripts/hysteresis_new_states.py).
  -> analysis/universality/hysteresis_ext/<name>  (json; states in <name>.npz: Z[j-1] = state after step j, K[j-1])
  python3 scripts/hysteresis_extend.py e0.3_b1.12_g0.5_40-30-30.json
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
HYS = ROOT / "analysis/universality/hysteresis"
OUT = ROOT / "analysis/universality/hysteresis_ext"
EXTRA = 200


def main():
    name = sys.argv[1]
    h = json.loads((HYS / name).read_text())
    d = np.load(HYS / (name + ".npz"))
    z, k_turn = d["z"], float(d["K"])
    mr.E, mr.B, mr.G = h["eps"], h["beta"], h["gamma"]
    kk1 = h["K1"]
    N = int(sum(h["sizes"]))
    dk = .005 * abs(kk1)
    prev = h["backward"]
    xr, yr = mr.rest_point()
    out = dict(name=name, eps=h["eps"], beta=h["beta"], gamma=h["gamma"], sizes=h["sizes"], K1=kk1, K_turn=k_turn,
               n_replay=len(prev), host=socket.gethostname(), replay_mismatch=[], steps=[])
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / name
    Z, KK = [], []
    t0 = time.time()
    for j in range(1, len(prev) + EXTRA + 1):
        k = k_turn - j * dk
        z = z + 1e-8 * np.random.default_rng(10000 + j).normal(size=2 * N)

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        s = solve_ivp(rhs, (0, 10000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=[8000., 10000.])
        z = s.y[:, -1]
        Z.append(z)
        KK.append(k)
        rest = bool(np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8)
        row = dict(step=j, K=float(k), f=float(k / kk1), rest=rest)
        if not rest:
            c1, c2 = clusters(s.y[:, 0], N), clusters(z, N)
            row.update(sizes=c2, settled=c1 == c2)
        if j <= len(prev) and (row["rest"] != prev[j - 1]["rest"] or row.get("sizes") != prev[j - 1].get("sizes")):
            out["replay_mismatch"].append(j)
        out["steps"].append(row)
        out["seconds"] = time.time() - t0
        fn.write_text(json.dumps(out, indent=1))
        np.savez(OUT / (name + ".npz"), Z=np.array(Z), K=np.array(KK))
        if rest:
            out["stop"] = "rest"
            break
    else:
        out["stop"] = "done"
    fn.write_text(json.dumps(out, indent=1))
    ch = []
    for q in out["steps"]:
        lab = "rest" if q["rest"] else ":".join(map(str, q["sizes"]))
        if not ch or ch[-1][1] != lab:
            ch.append(("%.3f" % q["f"], lab))
    print(name, "mismatch", out["replay_mismatch"], "chain", ch, out["stop"], flush=True)


if __name__ == "__main__":
    main()
