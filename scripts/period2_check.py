#!/usr/bin/env python3
"""Is the 34:33:33 motion two steps past the end of (0.3, 1.12, 0.5) a period-2 orbit? 2026-11-08 (author's approval
"３つとも進めてください").

From the final full-system state of scripts/beyond_end_detail.py (_beyond.npz, J = 2, the given seed): the cluster means
are integrated in the reduced system, the orbit with T = two returns to x_1 = 0 is solved by shooting and audited
(collective and internal multipliers). It is then continued in both directions (continue_branch, steps of
0.0005 |K1|, up to 0.05 |K1|) to see where it exists and how it ends.
  -> analysis/universality/hysteresis_ext/<name minus .json>_period2.json
  python3 scripts/period2_check.py e0.3_b1.12_g0.5_40-30-30.json 1
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from break_membership import labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"


def main():
    name, seed = sys.argv[1], int(sys.argv[2])
    e = json.loads((EXT / name).read_text())
    b = json.loads((EXT / (name[:-5] + "_beyond.json")).read_text())
    z = np.load(EXT / (name[:-5] + "_beyond.npz"))["J2_seed%d" % seed]
    mr.E, mr.B, mr.G = e["eps"], e["beta"], e["gamma"]
    kk1, N = e["K1"], int(sum(e["sizes"]))
    k = next(r["K"] for r in b["runs"] if r["J"] == 2 and r["seed"] == seed)
    fl = labels(z, N)
    fsz = np.bincount(fl)
    order = np.argsort(-fsz, kind="stable")
    w0 = np.r_[[z[:N][fl == c].mean() for c in order], [z[N:][fl == c].mean() for c in order]]
    p = fsz[order] / N
    s = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, 1500), w0, method="DOP853", rtol=1e-11, atol=1e-13, max_step=.08,
                  events=mr.sec, dense_output=True)
    te = s.t_events[0][s.t_events[0] > 500]
    tau = np.diff(te)
    w = np.r_[s.sol(te[-3]), te[-1] - te[-3]]
    w[0] = 0.
    out = dict(name=name, seed=seed, K=k, f=k / kk1, sizes=fsz[order].tolist(), tau_last=tau[-6:].tolist())
    ws = mr.shoot(w, k, p)
    if ws is None:
        out["result"] = "shooting failed"
    else:
        r = mr.audit(ws, k, p)
        out.update(result="orbit", period=r["period"], kind=mr.kind_of(r), collective=r["collective"][:4],
                   collective_max=r["collective_max"], internal_moduli=r["internal_moduli"],
                   min_cluster_gap=r["min_cluster_gap"])
        # is it really period two: the first return differs from the start
        s1 = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, ws[-1]), ws[:-1], method="DOP853", rtol=1e-11, atol=1e-13,
                       max_step=.08, events=mr.sec)
        tr = s1.t_events[0][(s1.t_events[0] > 1e-6) & (s1.t_events[0] < ws[-1] - 1e-6)]
        s2 = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, ws[-1]), ws[:-1], method="DOP853", rtol=1e-11, atol=1e-13,
                       max_step=.08, dense_output=True)
        out["first_return_times"] = tr.tolist()
        out["first_return_distance"] = float(max(abs(s2.sol(tr[0]) - ws[:-1]))) if len(tr) else None
        dk0 = .0005 * abs(kk1)
        for direction, lab in ((+1, "weaker"), (-1, "stronger")):
            (wg, kg_), rows, bad = mr.continue_branch(ws, k, p, dk0, k + direction * .05 * abs(kk1), direction)
            out[lab] = dict(K_last_good=kg_, f_last_good=kg_ / kk1, end=mr.end_kind(bad), n=len(rows),
                            last=rows[-1] if rows else None, first_bad=bad[1] if bad else None)
    (EXT / (name[:-5] + "_period2.json")).write_text(json.dumps(out, indent=1))
    print(name, "seed", seed, "f %.4f" % out["f"], out["sizes"], out["result"], "T", out.get("period"),
          "kind", out.get("kind"), "coll", out.get("collective_max"), "int", out.get("internal_moduli"),
          "first returns", out.get("first_return_times"), "dist", out.get("first_return_distance"), flush=True)
    for lab in ("weaker", "stronger"):
        if lab in out:
            q = out[lab]
            print("  ", lab, "f_last %.5f" % q["f_last_good"], q["end"], "n", q["n"],
                  "last int", [round(v, 4) for v in (q["last"] or {}).get("internal_moduli", [])],
                  "coll", (q["last"] or {}).get("collective", [None])[:2], flush=True)


if __name__ == "__main__":
    main()
