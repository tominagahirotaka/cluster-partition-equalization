#!/usr/bin/env python3
"""Why does (0.25, 1.32, 0.2) sometimes go to rest early (f = 0.803)? Preparation, 2026-11-14 (author's approval
"両方進めてください").

In all 41 runs of scripts/rebalance_variants.py at this point the state at step 5 (f = 0.808) is 33:33:3x:x (a fourth
cluster of 2-4 elements); at step 6 (f = 0.803) it is rest in 6 runs and a four-cluster state with a fourth cluster of
9-17 elements in the others. Here: the run with the given seed is replayed (same noise) up to step 5 and its state saved;
the reduced orbit of that partition is rebuilt from the cluster means and continued towards weaker coupling
(continue_branch, steps of 0.0005 |K1|, up to f = 0.79), and past the stop by pseudo-arclength (fold_saddle_test
.arclength, sign +1) to see how it ends.
  -> analysis/universality/escape/<name minus .json>_s<seed>_prep.json and .npz (state at step 5)
  python3 scripts/escape_prepare.py e0.25_b1.32_g0.2_34-33-33.json 15
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
from fold_saddle_test import arclength  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
VAR = ROOT / "analysis/universality/rebalance_var"
OUT = ROOT / "analysis/universality/escape"
STEP = 5


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    r = json.loads((MAP / name).read_text())
    reb = json.loads((VAR / ("%s_s%d_N100.json" % (name[:-5], seed))).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    sizes = np.array(reb["sizes"])
    N, m = 100, len(sizes)
    kk1 = r["K1"]
    k = r["end"]["K_last_good"] - .02 * abs(kk1)
    dk = .005 * abs(kk1)
    w = mr.shoot(np.array(r["end"]["last_good_orbit"]), k, sizes / N)
    lab = np.repeat(np.arange(m), sizes)
    z = np.r_[w[:m][lab], w[m:2 * m][lab]]
    rows = []
    for step in range(STEP + 1):
        z = z + 1e-8 * np.random.default_rng(100000 * seed + step).normal(size=2 * N)

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        s = solve_ivp(rhs, (0, 10000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=[8000., 10000.])
        z = s.y[:, -1]
        rows.append(dict(step=step, f=k / kk1, sizes=clusters(z, N), earlier=reb["steps"][step].get("sizes")))
        if step < STEP:
            k += dk
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / ("%s_s%d" % (name[:-5], seed))
    np.savez(str(stem) + "_prep.npz", z=z, K=k)
    out = dict(name=name, seed=seed, K1=kk1, K=k, f=k / kk1, replay=rows, match=all(q["sizes"] == q["earlier"] for q in rows))
    sz, p, w4 = orbit_from_state(z, k, N)
    out["sizes"] = sz.tolist()
    if w4 is None:
        out["reduced"] = "shooting failed at the start"
    else:
        a = mr.audit(w4, k, p)
        out["start_orbit"] = dict(period=a["period"], collective=a["collective"][:3], internal_moduli=a["internal_moduli"])
        (wg, kg), crow, bad = mr.continue_branch(w4, k, p, .0005 * abs(kk1), .79 * kk1, +1)
        out["continued"] = dict(f_last_good=kg / kk1, end=mr.end_kind(bad), f_first_bad=bad[0] / kk1 if bad else None,
                                last=crow[-1] if crow else None, first_bad=bad[1] if bad else None)
        (wi, ki), crow2, bad2 = mr.continue_branch(w4, k, p, .0005 * abs(kk1), .79 * kk1, +1, ignore_internal=True)
        out["continued_collective_only"] = dict(f_last_good=ki / kk1, end=mr.end_kind(bad2, True),
                                                last=crow2[-1] if crow2 else None)
        path = arclength(wi, ki, p, kk1, sign=+1)
        ks = np.array([q["K"] for q in path])
        turns = [i for i in range(1, len(ks) - 1) if (ks[i] - ks[i - 1]) * (ks[i + 1] - ks[i]) < 0]
        out["arclength"] = dict(n=len(path), f_range=[float(ks.min() / kk1), float(ks.max() / kk1)] if len(ks) else None,
                                turns=[dict(f=float(ks[i] / kk1), top=path[i]["top"], internal=path[i]["internal_moduli"],
                                            period=path[i]["period"]) for i in turns],
                                path=[dict(f=q["K"] / kk1, period=q["period"], top=q["top"], internal=q["internal_moduli"])
                                      for q in path[::max(1, len(path) // 40)]])
    (Path(str(stem) + "_prep.json")).write_text(json.dumps(out, indent=1))
    c = out.get("continued", {})
    print(name, "seed", seed, "match", out["match"], "sizes", out["sizes"], "start int",
          [round(v, 4) for v in out.get("start_orbit", {}).get("internal_moduli", [])],
          "continued to f %.5f" % c.get("f_last_good", np.nan), c.get("end"),
          "collective-only to f %.5f" % out.get("continued_collective_only", {}).get("f_last_good", np.nan),
          out.get("continued_collective_only", {}).get("end"),
          "arclength turns", [round(t["f"], 5) for t in out.get("arclength", {}).get("turns", [])], flush=True)


if __name__ == "__main__":
    main()
