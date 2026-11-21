#!/usr/bin/env python3
"""Why does the largest cluster lose internal stability first? 2026-11-15 (author's "続けてください", for the open item
"なぜ最大のクラスターの内部が先に不安定になるか" of paper4/claims_scope.md, claim B).

The internal 2x2 block of cluster a is A_a(t) = [[(1 - x_a^2)/eps - K, -1/eps], [eps, -eps gamma]]; all clusters share
K and the period T, so det M_a = exp(int_0^T tr A_a dt) differs between clusters only through I_a = int_0^T (1 - x_a^2) dt.
For the end orbits of the balanced-partition ranges (weak-coupling side: the map orbit at K_last_good; strong-coupling
side: the orbit of _strong_end[_s<step>].json, recovered as in kick_test.py) this script records for each cluster the
two internal multipliers, ln det M_a (checked against the trace integral), I_a / T, the time with |x_a| < 1 and the
number of passages through |x_a| < 1 per period.
  -> analysis/universality/internal_trace.json
  python3 scripts/internal_trace.py
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from balanced_strong_end import orbit_from_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
U = ROOT / "analysis/universality"
EXT = U / "hysteresis_ext"


def analyse(w, k, p):
    m = len(p)
    n = 2 * m

    def f(t, v):
        z = v[:n]
        a = np.zeros((m, 2, 2))
        a[:, 0, 0] = (1 - z[:m] ** 2) / mr.E - k
        a[:, 0, 1] = -1 / mr.E
        a[:, 1, 0] = mr.E
        a[:, 1, 1] = -mr.E * mr.G
        return np.r_[mr.rhs(z, k, p), (a @ v[n:n + 4 * m].reshape(m, 2, 2)).ravel(), 1 - z[:m] ** 2,
                     (abs(z[:m]) < 1).astype(float)]
    T = w[-1]
    s = solve_ivp(f, (0, T), np.r_[w[:n], np.tile(np.eye(2), (m, 1, 1)).ravel(), np.zeros(2 * m)], method="DOP853",
                  rtol=1e-11, atol=1e-13, max_step=.01, dense_output=True)
    v = s.y[:, -1]
    mono = v[n:n + 4 * m].reshape(m, 2, 2)
    integ = v[n + 4 * m:n + 5 * m]
    tin = v[n + 5 * m:]
    xx = s.sol(np.linspace(0, T, 200001))[:m]
    rows = []
    for a in range(m):
        ev = np.linalg.eigvals(mono[a])
        inside = abs(xx[a]) < 1
        rows.append(dict(p=float(p[a]), multipliers=[[float(e.real), float(e.imag)] for e in ev],
                         max_modulus=float(max(abs(ev))), ln_det=float(np.log(abs(np.linalg.det(mono[a])))),
                         ln_det_from_trace=float(integ[a] / mr.E - k * T - mr.E * mr.G * T),
                         I_over_T=float(integ[a] / T), time_inside=float(tin[a]),
                         passages=int(np.sum(inside[1:] & ~inside[:-1]))))
    return dict(K=float(k), period=float(T), return_error=float(max(abs(v[:n] - w[:n]))), clusters=rows)


def main():
    names = sorted(q.name.replace("_s1_N100.json", ".json") for q in (U / "rebalance_var").glob("*_s1_N100.json"))
    out = []
    for name in names:
        r = json.loads((U / "map" / name).read_text())
        mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
        kk1 = r["K1"]
        p = np.array(r["sizes"]) / sum(r["sizes"])
        row = dict(name=name, K1=kk1)
        kg = r["end"]["K_last_good"]
        row["weak_end"] = dict(f=kg / kk1, sizes=r["sizes"], kind=r["end"]["kind"],
                               **analyse(np.array(r["end"]["last_good_orbit"]), kg, p))
        se_f = sorted(EXT.glob(name[:-5] + "_strong_end*.json"))
        if se_f:
            se = json.loads(se_f[0].read_text())
            d = np.load(EXT / (name + ".npz"))
            N = int(sum(se["sizes"]))
            step = se.get("step", len(d["K"]))
            sz, pp, w = orbit_from_state(d["Z"][step - 1], float(d["K"][step - 1]), N)
            ks = se["K_last_good"]
            (w, kc), _, _ = mr.continue_branch(w, float(d["K"][step - 1]), pp, .005 * abs(kk1), ks, -1)
            if abs(kc - ks) < 1e-12 * abs(kk1):
                row["strong_end"] = dict(file=se_f[0].name, f=ks / kk1, sizes=sz.tolist(), kind=se["end"],
                                         **analyse(w, ks, pp))
            else:
                row["strong_end"] = dict(file=se_f[0].name, error="continuation stopped at f %.5f" % (kc / kk1))
        out.append(row)
        (U / "internal_trace.json").write_text(json.dumps(out, indent=1))
        for side in ("weak_end", "strong_end"):
            q = row.get(side)
            if q is None or "clusters" not in q:
                print(name, side, q, flush=True)
                continue
            print(name, side, "f %.4f" % q["f"], q["sizes"], "max|mu|", [round(c["max_modulus"], 3) for c in q["clusters"]],
                  "lndet", [round(c["ln_det"], 2) for c in q["clusters"]], "I/T", [round(c["I_over_T"], 3) for c in q["clusters"]],
                  "t_in", [round(c["time_inside"], 2) for c in q["clusters"]], "pass", [c["passages"] for c in q["clusters"]],
                  flush=True)


if __name__ == "__main__":
    main()
