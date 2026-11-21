#!/usr/bin/env python3
"""Where does the balanced partition reached by the rebalancing end on the strong-coupling side? 2026-11-08 (author's
approval "続けてください").

For the points whose balanced partition persisted to the end of scripts/hysteresis_extend.py: the reduced orbit is
rebuilt from the full-system state at the last step (as in scripts/hysteresis_new_states.py) and continued with
mcluster_reduced.continue_branch towards stronger coupling (steps of 0.005 |K1|, down to f = F_STOP) to the first
instability. The end is checked in the full system N = 100: from the reduced orbit at the last good K (inside) and
from the same orbit at the first bad K (beyond), elements are put on their clusters, noise sd 1e-8 is added and the
system is integrated for 10000 time units (as one step of the sweeps); the cluster sizes at t = 8000 and 10000.
  -> analysis/universality/hysteresis_ext/<name minus .json>_strong_end.json
  python3 scripts/balanced_strong_end.py e0.3_b1.12_g0.5_40-30-30.json
Optional second argument (added 2026-11-08): the step of hysteresis_extend whose state is used instead of the last one;
the output is then <name minus .json>_strong_end_s<step>.json.
  python3 scripts/balanced_strong_end.py e0.35_b0.84_g0.55_40-30-30.json 393
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
F_STOP = 20.


def orbit_from_state(z, k, N):
    sz, w0 = means(z, N)
    p = sz / N
    s = solve_ivp(lambda t, v: mr.rhs(v, k, p), (0, 600), w0, method="DOP853", rtol=1e-11, atol=1e-13, max_step=.08,
                  events=mr.sec, dense_output=True)
    te = s.t_events[0][s.t_events[0] > 300]
    w = np.r_[s.sol(te[-2]), te[-1] - te[-2]]
    w[0] = 0.
    return sz, p, mr.shoot(w, k, p)


def full_step(w, k, sizes, seed):
    m, N = len(sizes), int(sum(sizes))
    lab = np.repeat(np.arange(m), sizes)
    z = np.r_[w[:m][lab], w[m:2 * m][lab]] + 1e-8 * np.random.default_rng(seed).normal(size=2 * N)

    def rhs(t, u):
        x, y = u[:N], u[N:]
        return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
    s = solve_ivp(rhs, (0, 10000), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15, t_eval=[8000., 10000.])
    xr, yr = mr.rest_point()
    z = s.y[:, -1]
    if np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8:
        return "rest"
    c1, c2 = clusters(s.y[:, 0], N), clusters(z, N)
    return dict(sizes=c2, settled=c1 == c2)


def main():
    name = sys.argv[1]
    e = json.loads((EXT / name).read_text())
    d = np.load(EXT / (name + ".npz"))
    mr.E, mr.B, mr.G = e["eps"], e["beta"], e["gamma"]
    kk1, N = e["K1"], int(sum(e["sizes"]))
    step = int(sys.argv[2]) if len(sys.argv) > 2 else len(d["K"])
    k = float(d["K"][step - 1])
    sz, p, w = orbit_from_state(d["Z"][step - 1], k, N)
    out = dict(name=name, eps=e["eps"], beta=e["beta"], gamma=e["gamma"], K1=kk1, sizes=sz.tolist(), K_start=k,
               f_start=k / kk1, step=step)
    fn = EXT / (name[:-5] + ("_strong_end.json" if len(sys.argv) <= 2 else "_strong_end_s%d.json" % step))
    if w is None:
        out["result"] = "shooting failed at the start"
        fn.write_text(json.dumps(out, indent=1))
        print(name, out["result"], flush=True)
        return
    (wg, kg), rows, bad = mr.continue_branch(w, k, p, .005 * abs(kk1), F_STOP * kk1, -1)
    out.update(end=mr.end_kind(bad), K_last_good=kg, f_last_good=kg / kk1, last_good=rows[-1] if rows else None,
               K_first_bad=bad[0] if bad else None, f_first_bad=bad[0] / kk1 if bad else None,
               first_bad=bad[1] if bad else None, n_orbits=len(rows))
    fn.write_text(json.dumps(out, indent=1))
    if bad is not None:
        # beyond: the step 0.005 |K1| past the last good K (the size of a step of the sweeps)
        k_in, k_out = kg, kg - .005 * abs(kk1)
        out["full_inside"] = dict(K=k_in, f=k_in / kk1, **{"r": full_step(wg, k_in, sz, 1)})
        fn.write_text(json.dumps(out, indent=1))
        out["full_beyond"] = dict(K=k_out, f=k_out / kk1, **{"r": full_step(wg, k_out, sz, 2)})
        fn.write_text(json.dumps(out, indent=1))
    lg = out["last_good"] or {}
    print(name, sz.tolist(), "f %.3f -> %.4f" % (out["f_start"], out["f_last_good"]), out["end"],
          "int", [round(v, 4) for v in lg.get("internal_moduli", [])], "coll %.4f" % lg.get("collective_max", np.nan),
          "full in", out.get("full_inside", {}).get("r"), "beyond", out.get("full_beyond", {}).get("r"), flush=True)


if __name__ == "__main__":
    main()
