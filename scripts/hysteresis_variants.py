#!/usr/bin/env python3
"""Hysteresis and the strong-coupling end over noise realisations, 2026-11-10 (author's request "新しい計算を始めてく
ださい", for "戻す向き（ヒステリシス）と強い結合の側の端を、雑音の実現を変えて調べる").

Forward: the run of scripts/rebalance_variants.py with the same seed and N is replayed (same noise
default_rng(100000 * seed + j)) up to its turning step: the earliest step whose state has exactly three clusters with
the smallest largest cluster in analysis/universality/rebalance_var/<name>_s<seed>_N<N>.json. Its sizes are checked
step by step against that run. Backward: from there K is decreased (stronger coupling) in steps of 0.005 |K1|, noise
sd 1e-8 with default_rng(100000 * seed + 50000 + j), 10000 time units per step, up to f0 + 0.05 (turning step + 10
steps, as scripts/hysteresis_sweep.py; the strong-coupling end itself is set by the reduced orbit, see
balanced_strong_end.py, so it is not followed here). Stops at rest.
  -> analysis/universality/hysteresis_var/<name minus .json>_s<seed>_N<N>.json
  python3 scripts/hysteresis_variants.py e0.3_b1.12_g0.5_40-30-30.json 3 100
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
from hysteresis_sweep import turning_step  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
VAR = ROOT / "analysis/universality/rebalance_var"
OUT = ROOT / "analysis/universality/hysteresis_var"


def main():
    name, seed, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    r = json.loads((MAP / name).read_text())
    reb = json.loads((VAR / ("%s_s%d_N%d.json" % (name[:-5], seed, N))).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    sizes = np.array(reb["sizes"])
    p = sizes / N
    m = len(p)
    kk1 = r["K1"]
    k0 = r["end"]["K_last_good"] - .02 * abs(kk1)
    dk = .005 * abs(kk1)
    st = turning_step(reb)
    out = dict(name=name, seed=seed, N=N, sizes=sizes.tolist(), K1=kk1, K0=k0, f0=k0 / kk1, turn_step=st,
               turn_sizes_earlier_run=[q for q in reb["steps"] if q["step"] == st][0]["sizes"],
               host=socket.gethostname(), forward_mismatch=[], forward=[], backward=[])
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / ("%s_s%d_N%d.json" % (name[:-5], seed, N))
    w = mr.shoot(np.array(r["end"]["last_good_orbit"]), k0, p)
    lab = np.repeat(np.arange(m), sizes)
    z = np.r_[w[:m][lab], w[m:2 * m][lab]]
    xr, yr = mr.rest_point()
    t0 = time.time()

    def step(z, k, rng_seed, rows, idx):
        z = z + 1e-8 * np.random.default_rng(rng_seed).normal(size=2 * N)

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
        z, rest = step(z, k, 100000 * seed + i, out["forward"], i)
        q = reb["steps"][i]
        if (rest, out["forward"][-1].get("sizes")) != (q["rest"], q.get("sizes")):
            out["forward_mismatch"].append(i)
        if rest:
            out["stop"] = "rest in the forward replay"
            fn.write_text(json.dumps(out, indent=1))
            return
        if i < st:
            k += dk
    for j in range(1, st + 11):
        z, rest = step(z, k - j * dk, 100000 * seed + 50000 + j, out["backward"], j)
        if rest:
            out["stop"] = "rest in the backward sweep"
            break
    else:
        out["stop"] = "done"
    fn.write_text(json.dumps(out, indent=1))
    ch = []
    for q in out["backward"]:
        lb = "rest" if q["rest"] else ":".join(map(str, q["sizes"]))
        if not ch or ch[-1][1] != lb:
            ch.append(("%.3f" % q["f"], lb))
    print(name, "seed", seed, "N", N, "turn", st, out["forward"][-1].get("sizes"), "mismatch", out["forward_mismatch"],
          "bwd", ch[:12], out["stop"], flush=True)


if __name__ == "__main__":
    main()
