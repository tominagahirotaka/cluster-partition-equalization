#!/usr/bin/env python3
"""What kind of motion is a cluster state that is not a period-1 orbit? 2026-11-08 (author's approval "続けてください");
written for the 36:34:30 state of (0.35, 0.84, 0.55) in scripts/hysteresis_extend.py (steps 367-418, shooting for a
period-1 orbit failed).

For the first, middle and last step of the run with the given sizes: the reduced system is started from the full-system
cluster means and integrated for 1000 time units (transient) and then TMAX more; the return times tau_i between upward
crossings of x_1 = 0 are recorded. If tau_{i+n} = tau_i (to 1e-6) for the smallest n <= 16, the period-n orbit is solved
by shooting (T = sum of n returns) and audited; otherwise the largest Lyapunov exponent of the reduced system is
estimated (tangent vector renormalised every unit time). The full system N = 100 is integrated from the same state for
the same time: its return times of the cluster-1 mean and its cluster sizes at the end are compared.
  -> analysis/universality/hysteresis_ext/<name minus .json>_motion_<sizes>.json
  python3 scripts/nonperiodic_state.py e0.35_b0.84_g0.55_40-30-30.json 36:34:30
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from rebalance_sweep import clusters  # noqa: E402
from hysteresis_new_states import means  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"
TMAX = 4000.


def lyapunov(z, k, p, T):
    n = len(z)
    v = np.random.default_rng(0).normal(size=n)
    v /= np.linalg.norm(v)
    acc = 0.
    for _ in range(int(T)):
        s = solve_ivp(lambda t, u: np.r_[mr.rhs(u[:n], k, p), mr.jac(u[:n], k, p) @ u[n:]], (0, 1), np.r_[z, v],
                      method="DOP853", rtol=1e-10, atol=1e-12, max_step=.05)
        z, v = s.y[:n, -1], s.y[n:, -1]
        g = np.linalg.norm(v)
        acc += np.log(g)
        v /= g
    return acc / int(T)


def examine(z, k, N):
    sz, w0 = means(z, N)
    p = sz / N
    m = len(p)
    row = dict(sizes=sz.tolist(), K=float(k))
    s = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, 1000 + TMAX), w0, method="DOP853", rtol=1e-11, atol=1e-13,
                  max_step=.08, events=mr.sec, dense_output=True)
    te = s.t_events[0][s.t_events[0] > 1000]
    tau = np.diff(te)
    row.update(n_returns=len(tau), tau_min=float(tau.min()), tau_max=float(tau.max()),
               tau_distinct_1e3=len(np.unique(tau.round(3))), tau_first=tau[:40].round(4).tolist())
    per = None
    for n in range(1, 17):
        if len(tau) > 3 * n and np.max(abs(tau[n:] - tau[:-n])) < 1e-6:
            per = n
            break
    row["period_n"] = per
    if per is not None:
        w = np.r_[s.sol(te[0]), te[per] - te[0]]
        w[0] = 0.
        ws = mr.shoot(w, k, p)
        if ws is None:
            row["result"] = "period-%d returns, shooting failed" % per
        else:
            r = mr.audit(ws, k, p)
            row.update(result="period-%d orbit" % per, kind=mr.kind_of(r), period=r["period"],
                       collective_max=r["collective_max"], collective=r["collective"][:3],
                       internal_moduli=r["internal_moduli"], min_cluster_gap=r["min_cluster_gap"])
    else:
        row["result"] = "no period up to 16"
        row["lyapunov_max_reduced"] = lyapunov(s.sol(1000 + TMAX), k, p, 2000)
    # full system from the same state (crossings as events; the first run with dense output of the full system ended without output, presumably out of memory)
    labs = _labels(z, N)
    members = np.where(labs == np.argsort(-np.bincount(labs), kind="stable")[0])[0]

    def rhs(t, u):
        x, y = u[:N], u[N:]
        return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))

    def cross(t, u):
        return u[members].mean()
    cross.direction = 1
    f = solve_ivp(rhs, (0, 1000 + TMAX), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, events=cross,
                  t_eval=[1000 + TMAX])
    tf = f.t_events[0][f.t_events[0] > 1000]
    tauf = np.diff(tf)
    row.update(full_n_returns=len(tauf), full_tau_min=float(tauf.min()), full_tau_max=float(tauf.max()),
               full_tau_first=tauf[:40].round(4).tolist(), full_sizes_end=clusters(f.y[:, -1], N))
    return row


def _labels(z, N):
    from scipy.sparse.csgraph import connected_components
    x, y = z[:N], z[N:]
    dist = np.maximum(abs(x[:, None] - x[None, :]), abs(y[:, None] - y[None, :]))
    return connected_components(dist < 1e-4, directed=False)[1]


def main():
    name, want = sys.argv[1], [int(v) for v in sys.argv[2].split(":")]
    e = json.loads((EXT / name).read_text())
    d = np.load(EXT / (name + ".npz"))
    mr.E, mr.B, mr.G = e["eps"], e["beta"], e["gamma"]
    N = int(sum(e["sizes"]))
    idx = [i for i, q in enumerate(e["steps"]) if q.get("sizes") == want]
    out = dict(name=name, sizes=want, K1=e["K1"], steps=[e["steps"][i]["step"] for i in idx], rows=[])
    fn = EXT / (name[:-5] + "_motion_" + "-".join(map(str, want)) + ".json")
    for i in (idx[0], idx[len(idx) // 2], idx[-1]):
        row = examine(d["Z"][i], float(d["K"][i]), N)
        row.update(step=e["steps"][i]["step"], f=e["steps"][i]["f"])
        out["rows"].append(row)
        fn.write_text(json.dumps(out, indent=1))
        print(name, want, "step", row["step"], "f %.3f" % row["f"], row["result"],
              "tau %.3f-%.3f distinct %d" % (row["tau_min"], row["tau_max"], row["tau_distinct_1e3"]),
              "lyap", row.get("lyapunov_max_reduced"), "coll", row.get("collective_max"),
              "int", row.get("internal_moduli"),
              "full tau %.3f-%.3f end %s" % (row["full_tau_min"], row["full_tau_max"], row["full_sizes_end"]), flush=True)


if __name__ == "__main__":
    main()
