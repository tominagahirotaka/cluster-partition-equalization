#!/usr/bin/env python3
"""Step 3 of the route-to-chaos check (2026-11-04, author's approval "1〜3を進めてください"): follow the period-two orbit
born at a supercritical collective period doubling of a five-cluster branch, and, if it doubles again, the period-four
orbit, and so on (up to period 16), by shooting (mcluster_reduced; the period of w is the full period 2^m T).

Seed of period 2^m: at K = K_prev_end + direction * 0.01 |K1| past the previous doubling (for m = 1: K_PD of
universality_branches.py), integrate (pd_route_scan.run, RK4) from the parent orbit + 1e-4 along its -1 eigenvector,
transient 20000, and take the last section point with the sum of the last 2^m return times. Shoot; require that the
orbit really has period 2^m (the section point after 2^(m-1) returns differs by > 1e-6). Then continue towards the
doubling (step 0.002 |K1|, back to where the amplitude falls below 1e-6 or shooting fails) and away from it (step
|K1|/2000, halved at failure or instability, as mcluster_reduced.continue_branch) and record how it ends. Feigenbaum
ratio from three successive doublings if found.
  -> analysis/universality/pd_cascade_<tag>_d<dir>.json

  python3 scripts/pd_cascade_shoot.py plan2_b0.8_g0.5_23-21-20-19-17 -1
"""
from pathlib import Path
import json
import socket
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
import pd_route_scan as ps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/universality"


def monodromy(w, k, p):
    n = 2 * len(p)
    s = solve_ivp(lambda t, v: np.r_[mr.rhs(v[:n], k, p), (mr.jac(v[:n], k, p) @ v[n:].reshape(n, n)).ravel()],
                  (0, w[-1]), np.r_[w[:n], np.eye(n).ravel()], method="DOP853", rtol=1e-12, atol=1e-14, max_step=.04)
    return s.y[n:, -1].reshape(n, n)


def half_amplitude(w, k, p, m):
    """max-norm difference between the start and the section point after 2^(m-1) returns (0 for m = 0)."""
    if m == 0:
        return 0.
    n = 2 * len(p)
    s = solve_ivp(lambda t, z: mr.rhs(z, k, p), (0, w[-1] * .75), w[:n], method="DOP853", rtol=1e-12, atol=1e-14,
                  max_step=.04, events=mr.sec)
    pts = s.y_events[0][s.t_events[0] > 1e-6]  # the start lies on the section
    j = 2 ** (m - 1)
    return float(np.max(abs(pts[j - 1] - w[:n]))) if len(pts) >= j else float("nan")


def main():
    tag, direction = sys.argv[1], int(sys.argv[2])
    r = json.loads((OUT / f"{tag}.json").read_text())
    mr.B, mr.G = r["beta"], r["gamma"]
    sz = r.get("start_order_sizes", r["sizes"])
    p = np.array(sz, float) / sum(sz)
    n = 2 * len(p)
    d = [x for x in r["directions"] if x["direction"] == direction][0]
    w, k_end, kk1 = np.array(d["last_good_orbit"]), d["K_last_good"], r["K1"]
    out = dict(tag=tag, direction=direction, K1=kk1, sizes=sz, host=socket.gethostname(),
               doublings=[dict(period_index=0, K_PD=k_end, period=float(w[-1]))], branches=[])
    for m in range(1, 5):
        k = k_end + direction * .01 * abs(kk1)
        wp = mr.shoot(w, k, p)  # parent orbit (period 2^(m-1)) carried past its doubling
        if wp is None:
            out["stop"] = f"parent of period {2 ** m} not found past the doubling"
            break
        ev, V = np.linalg.eig(monodromy(wp, k, p))
        j = int(np.argmin(abs(ev.real + 1) + abs(ev.imag)))
        v = np.real(V[:, j]); v /= np.linalg.norm(v)
        sec, ts, le, il, z = ps.run(wp[:n] + 1e-4 * v, k, p, mr.E, mr.B, mr.G, ps.DT, 20000., 4 * 2 ** m)
        T = float(ts[-1] - ts[-1 - 2 ** m])
        wn = mr.shoot(np.r_[sec[-1], T], k, p)
        amp = half_amplitude(wn, k, p, m) if wn is not None else float("nan")
        br = dict(period_index=m, seed_K=k, seed_lyap=le, seed_period_estimate=ps.distinct(sec),
                  shoot_ok=wn is not None, half_amplitude=amp)
        out["branches"].append(br)
        if wn is None or not amp > 1e-6:
            out["stop"] = f"no period-{2 ** m} orbit at the seed (shoot_ok={wn is not None}, amplitude={amp})"
            break
        br["seed_audit"] = mr.audit(wn, k, p)
        # towards the doubling: amplitude should fall to 0 at the parent's K_PD
        back, wb, kb = [], wn, k
        for _ in range(10):
            kb2 = kb - direction * .001 * abs(kk1)
            w2 = mr.shoot(wb, kb2, p)
            if w2 is None:
                break
            a2 = half_amplitude(w2, kb2, p, m)
            back.append(dict(K=kb2, half_amplitude=a2, period=float(w2[-1])))
            if not a2 > 1e-6:
                break
            wb, kb = w2, kb2
        br["towards_doubling"] = back
        # away from the doubling, until it ends
        good, rows, last_bad = mr.continue_branch(wn, k, p, abs(kk1) / 2000, k + direction * .3 * abs(kk1),
                                                  direction)
        kind = mr.end_kind(last_bad)
        br.update(end_kind=kind, K_last_good=float(good[1]), npts=len(rows),
                  tail=[dict(K=q["K"], period=q["period"], collective=q["collective"][:2],
                             internal_moduli=q["internal_moduli"]) for q in rows[-5:]],
                  last_bad_audit=None if last_bad is None else last_bad[1],
                  half_amplitude_end=half_amplitude(good[0], good[1], p, m))
        print(tag, direction, "period", 2 ** m, kind, "K_last_good %.8f" % good[1], flush=True)
        (OUT / f"pd_cascade_{tag}_d{direction:+d}.json").write_text(json.dumps(out, indent=1))
        if kind != "PD":
            out["stop"] = f"period-{2 ** m} orbit ends by {kind}"
            break
        out["doublings"].append(dict(period_index=m, K_PD=float(good[1]), period=float(good[0][-1])))
        w, k_end = good[0], float(good[1])
    kp = [x["K_PD"] for x in out["doublings"]]
    if len(kp) >= 3:
        out["feigenbaum_ratios"] = [(kp[i + 1] - kp[i]) / (kp[i + 2] - kp[i + 1]) for i in range(len(kp) - 2)]
    (OUT / f"pd_cascade_{tag}_d{direction:+d}.json").write_text(json.dumps(out, indent=1))
    print(tag, direction, "doublings", kp, out.get("stop"), flush=True)


if __name__ == "__main__":
    main()
