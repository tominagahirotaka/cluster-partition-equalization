#!/usr/bin/env python3
"""Are the states met in scripts/hysteresis_extend.py stable periodic orbits? 2026-11-07 (author's approval
"１，２とも進めてください").

For every run of steps with the same cluster sizes in analysis/universality/hysteresis_ext/<name>, the state at the
last step of the run is taken: elements are grouped into clusters (max-norm distance < 1e-4), the reduced M-cluster
system is started from the cluster means (largest cluster first), the section x_1 = 0 (upward) is found after 300 time
units, the periodic orbit is solved by shooting (mcluster_reduced.shoot) and audited (collective multipliers and the
2x2 internal multipliers of every cluster, mcluster_reduced.audit). The distance between the full-system cluster
means and the reduced orbit shows whether the orbit found is the one the full system is on.
  -> analysis/universality/hysteresis_ext/<name minus .json>_states.json
  python3 scripts/hysteresis_new_states.py e0.35_b0.92_g0.65_50-25-25.json
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"


def means(z, N):
    x, y = z[:N], z[N:]
    dist = np.maximum(abs(x[:, None] - x[None, :]), abs(y[:, None] - y[None, :]))
    _, labs = connected_components(dist < 1e-4, directed=False)
    sz = np.bincount(labs)
    order = np.argsort(-sz, kind="stable")
    return sz[order], np.r_[[x[labs == c].mean() for c in order], [y[labs == c].mean() for c in order]]


def examine(z, k, N):
    sz, w0 = means(z, N)
    p = sz / N
    m = len(p)
    row = dict(sizes=sz.tolist(), K=float(k))

    def sec(t, v):
        return v[0]
    sec.direction = 1
    s = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, 600), w0, method="DOP853", rtol=1e-11, atol=1e-13, max_step=.08,
                  events=sec, dense_output=True)
    te = s.t_events[0][s.t_events[0] > 300]
    zz = s.sol(np.linspace(300, 600, 3001))
    if np.max(zz.max(1) - zz.min(1)) < 1e-6:
        # a stationary cluster state (added 2026-11-08: the 75:25 state of (0.35, 0.92, 0.65) does not oscillate)
        ze = fsolve(lambda v: mr.rhs(v, k, p), zz[:, -1], xtol=1e-14)
        ev = np.linalg.eigvals(mr.jac(ze, k, p))
        inter = [float(max(np.linalg.eigvals(np.array([[(1 - ze[a] ** 2) / mr.E - k, -1 / mr.E], [mr.E, -mr.E * mr.G]])).real))
                 for a in range(m)]
        row.update(result="equilibrium", x=ze[:m].tolist(), y=ze[m:].tolist(), residual=float(max(abs(mr.rhs(ze, k, p)))),
                   collective_max_re=float(ev.real.max()), internal_max_re=inter,
                   kind=None if ev.real.max() < 0 and max(inter) < 0 else "unstable",
                   dist_full_to_orbit=float(max(abs(ze - w0))))
        return row
    if len(te) < 3:
        row["result"] = "no section crossings"
        return row
    per = np.diff(te)
    row["crossing_periods_spread"] = float(per.max() - per.min())
    w = np.r_[s.sol(te[-2]), per[-1]]
    w[0] = 0.
    ws = mr.shoot(w, k, p)
    if ws is None:
        row["result"] = "shooting failed"
        row["crossing_period_last"] = float(per[-1])
        return row
    r = mr.audit(ws, k, p)
    orb = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, ws[-1]), ws[:2 * m], method="DOP853", rtol=1e-11, atol=1e-13,
                    max_step=.02, dense_output=True).sol(np.linspace(0, ws[-1], 20001))
    row.update(result="periodic orbit", kind=mr.kind_of(r), period=r["period"], collective_max=r["collective_max"],
               collective=r["collective"][:3], internal_moduli=r["internal_moduli"], firing_offsets=r["firing_offsets"],
               min_cluster_gap=r["min_cluster_gap"], return_error=r["return_error"],
               dist_full_to_orbit=float(np.min(np.max(abs(orb - w0[:, None]), axis=0))))
    return row


def main():
    name = sys.argv[1]
    e = json.loads((EXT / name).read_text())
    d = np.load(EXT / (name + ".npz"))
    mr.E, mr.B, mr.G = e["eps"], e["beta"], e["gamma"]
    N = int(sum(e["sizes"]))
    st = e["steps"]
    lab = ["rest" if q["rest"] else ":".join(map(str, q["sizes"])) for q in st]
    out = dict(name=name, eps=e["eps"], beta=e["beta"], gamma=e["gamma"], K1=e["K1"], states=[])
    for i in range(len(st)):
        if lab[i] == "rest" or (i + 1 < len(st) and lab[i + 1] == lab[i]):
            continue
        first = i
        while first > 0 and lab[first - 1] == lab[i]:
            first -= 1
        row = examine(d["Z"][i], float(d["K"][i]), N)
        row.update(step_first=st[first]["step"], step_last=st[i]["step"], f_first=st[first]["f"], f_last=st[i]["f"])
        out["states"].append(row)
        (EXT / (name[:-5] + "_states.json")).write_text(json.dumps(out, indent=1))
        print(name, lab[i], "f %.3f-%.3f" % (row["f_first"], row["f_last"]), row["result"], row.get("kind"),
              "coll %.4f" % row.get("collective_max", row.get("collective_max_re", np.nan)),
              "int", [round(v, 4) for v in row.get("internal_moduli", row.get("internal_max_re", []))],
              "dist %.1e" % row.get("dist_full_to_orbit", np.nan), flush=True)


if __name__ == "__main__":
    main()
