#!/usr/bin/env python3
"""What happens just past the strong-coupling end of a balanced partition? 2026-11-08 (author's approval
"両方進めてください"); written for (0.3, 1.12, 0.5) 34:33:33 (no break two steps past the end, all spreads exactly 0)
and (0.25, 1, 0.35) 34:34:32 (slow growth of the spread of one 34 cluster).

1. Reduced system: the orbit is carried to the end K_last_good of _strong_end.json and continued further towards
   stronger coupling with only the collective multipliers counted (continue_branch, ignore_internal, steps of
   0.0005 |K1|, up to 3 steps of the sweeps): the internal and leading collective multipliers along the way and how
   the period-1 orbit ends.
2. Full system N = 100 at K = K_last_good - J * 0.005 |K1| for each J: from the orbit at K_last_good (as in
   break_statistics.py), noise sd 1e-8 with default_rng(seed), TMAX time units. Every 100 time units the spread of each
   original cluster; at the end the clusters (max-norm distance < 1e-4, single linkage) with the spread inside each,
   and the motion of the final cluster means in the reduced system (return times on x_1 = 0; period-1 orbit by
   shooting and its multipliers, and its distance to the continued period-1 orbit at K if that exists).
  -> analysis/universality/hysteresis_ext/<name minus .json>_beyond.json (final states in _beyond.npz)
  python3 scripts/beyond_end_detail.py e0.3_b1.12_g0.5_40-30-30.json 1,2 1,2,3 60000
"""
from pathlib import Path
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from balanced_strong_end import orbit_from_state  # noqa: E402
from break_membership import labels, spread  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"


def reduced_motion(z, k, N, w_ref):
    fl = labels(z, N)
    fsz = np.bincount(fl)
    order = np.argsort(-fsz, kind="stable")
    x, y = z[:N], z[N:]
    w0 = np.r_[[x[fl == c].mean() for c in order], [y[fl == c].mean() for c in order]]
    p = fsz[order] / N
    m = len(p)
    s = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, 1500), w0, method="DOP853", rtol=1e-11, atol=1e-13, max_step=.08,
                  events=mr.sec, dense_output=True)
    te = s.t_events[0][s.t_events[0] > 500]
    row = dict(sizes=fsz[order].tolist(), inner_spread=[float(v) for v in spread(z, N, np.argsort(order)[fl], m)])
    if len(te) < 3:
        zz = s.sol(np.linspace(500, 1500, 1001))
        row.update(motion="no section crossings", range=float(np.max(zz.max(1) - zz.min(1))))
        return row
    tau = np.diff(te)
    row.update(tau_distinct=np.unique(tau.round(3)).tolist())
    if np.ptp(tau) < 1e-6:
        w = np.r_[s.sol(te[-2]), tau[-1]]
        w[0] = 0.
        ws = mr.shoot(w, k, p)
        if ws is not None:
            r = mr.audit(ws, k, p)
            row.update(motion="period-1 orbit", period=r["period"], kind=mr.kind_of(r),
                       collective_max=r["collective_max"], internal_moduli=r["internal_moduli"])
            if w_ref is not None and len(w_ref) == len(ws):
                o = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, w_ref[-1]), w_ref[:2 * m], method="DOP853", rtol=1e-11,
                              atol=1e-13, max_step=.02, dense_output=True).sol(np.linspace(0, w_ref[-1], 20001))
                row["dist_to_continued_orbit"] = float(np.min(np.max(abs(o - ws[:2 * m, None]), axis=0)))
                row["period_continued_orbit"] = float(w_ref[-1])
        else:
            row["motion"] = "period-1 returns, shooting failed"
    else:
        row["motion"] = "not period-1"
    return row


def main():
    name = sys.argv[1]
    js = [int(v) for v in sys.argv[2].split(",")]
    seeds = [int(v) for v in sys.argv[3].split(",")]
    tmax = float(sys.argv[4])
    e = json.loads((EXT / name).read_text())
    se = json.loads((EXT / (name[:-5] + "_strong_end.json")).read_text())
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
    out = dict(name=name, sizes=sz.tolist(), K1=kk1, K_last_good=kg, f_last_good=kg / kk1, tmax=tmax)
    fn = EXT / (name[:-5] + "_beyond.json")
    # 1. period-1 orbit past the end, collective multipliers only
    rows, wr, kr, wlist = [], w, kg, []
    for i in range(1, 31):
        kn = kg - i * dk / 10
        wn = mr.shoot(wr, kn, p)
        if wn is None:
            break
        r = mr.audit(wn, kn, p)
        rows.append(dict(f=kn / kk1, K=kn, period=r["period"], internal=r["internal_moduli"],
                         collective_lead=r["collective"][0], collective_max=r["collective_max"]))
        wlist.append(wn)
        wr, kr = wn, kn
        if r["collective_max"] >= 1:
            break
    stop = ("collective >= 1" if rows and rows[-1]["collective_max"] >= 1 else
            "reached 3 steps" if len(rows) == 30 else "shooting failed")
    out["reduced_past_end"] = dict(rows=rows, f_last=kr / kk1, stop=stop)
    orbits = {j: (wlist[10 * j - 1] if len(wlist) >= 10 * j else None) for j in js}
    fn.write_text(json.dumps(out, indent=1))
    print(name, "reduced past end:", out["reduced_past_end"]["stop"], "f_last %.5f" % (kr / kk1), flush=True)
    # 2. full system
    lab = np.repeat(np.arange(m), sz)
    z0 = np.r_[w[:m][lab], w[m:2 * m][lab]]
    tt = np.arange(0., tmax + 1, 100.)
    out["runs"] = []
    finals = {}
    for j in js:
        k = kg - j * dk

        def rhs(t, u):
            x, y = u[:N], u[N:]
            return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
        for seed in seeds:
            z = z0 + 1e-8 * np.random.default_rng(seed).normal(size=2 * N)
            s = solve_ivp(rhs, (0, tmax), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=tt)
            sp = np.array([spread(s.y[:, i], N, lab, m) for i in range(len(tt))])
            zf = s.y[:, -1]
            finals["J%d_seed%d" % (j, seed)] = zf
            row = dict(J=j, K=k, f=k / kk1, seed=seed, log10_spread_every_1000=np.log10(sp[::10] + 1e-300).round(2).tolist(),
                       final=reduced_motion(zf, k, N, orbits.get(j)))
            out["runs"].append(row)
            fn.write_text(json.dumps(out, indent=1))
            np.savez(EXT / (name[:-5] + "_beyond.npz"), **finals)
            fm = row["final"]
            print(name, "J", j, "f %.4f" % (k / kk1), "seed", seed, "final", fm["sizes"], fm.get("motion"),
                  "T", fm.get("period"), "int", fm.get("internal_moduli"), "dist", fm.get("dist_to_continued_orbit"),
                  "inner", [float("%.1e" % v) for v in fm["inner_spread"]], flush=True)


if __name__ == "__main__":
    main()
