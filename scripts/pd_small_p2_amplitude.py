#!/usr/bin/env python3
"""(a) of the follow-up (2026-11-04, author's approval "(a)と(b)の両方を進めてください"): the small period-two orbit
born at a supercritical collective period doubling of a five-cluster branch, continued in its amplitude.

Why not in K (pd_branch2_follow.py --small): near the doubling the period-two amplitude settles on a time scale of
order 1/(mu^2 - 1) returns (about 6e4 time units at f = 0.003), so the scan (transient 5000) is not converged there,
and Newton in K from such a seed goes to the parent orbit (first step 2.5e-3, checked 2026-11-04).

Here: reference orbit zp and the eigenvector v of the multiplier nearest -1 at K_PD. Unknowns (z, T, K); equations
phi_T(z) - z = 0 (two returns), z_1 = 0, (z - zp).v = a, with v made orthogonal to the direction dzp/dK in which
the parent moves (else the parent at a shifted K solves it; added 2026-11-04 after the first (1.0, 0.3) runs). The K column of the Jacobian by central differences
(h = 1e-7). a from 1e-3 upwards (factor 1.15, halved at failure, secant prediction), up to a = 1.5. At each point:
K, T, half amplitude (section point after one return minus start), collective and internal multipliers (audit). A
fold of the period-two branch shows as a turn of K(a).
  -> analysis/universality/pd_small_p2_<tag>_d<dir>.json

  python3 scripts/pd_small_p2_amplitude.py plan2_b0.8_g0.5_23-21-20-19-17 -1
"""
from pathlib import Path
import json
import socket
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from pd_cascade_shoot import half_amplitude, monodromy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/universality"


def end_point(z, T, k, p):
    s = solve_ivp(lambda t, u: mr.rhs(u, k, p), (0, T), z, method="DOP853", rtol=mr.TOL, atol=mr.TOL * .01,
                  max_step=.08)
    return s.y[:, -1]


def solve(u, a, zp, v, p):
    """Newton on (z, T, K); returns u or None."""
    n = len(zp)
    u = u.copy()
    for _ in range(25):
        z, T, k = u[:n], u[n], u[n + 1]
        r = mr.flow(np.r_[z, T], k, p)
        if r is None:
            return None
        res0, a0 = r
        res = np.r_[res0, (z - zp) @ v - a]
        if max(abs(res)) < 1e-11:
            return u
        h = 1e-7
        dk = (end_point(z, T, k + h, p) - end_point(z, T, k - h, p)) / (2 * h)
        J = np.zeros((n + 2, n + 2))
        J[:n + 1, :n + 1] = a0
        J[:n, n + 1] = dk
        J[n + 1, :n] = v
        d = np.linalg.solve(J, -res)
        if max(abs(d)) > .5:
            return None
        u = u + d
    return None


def main():
    tag, direction = sys.argv[1], int(sys.argv[2])
    r = json.loads((OUT / f"{tag}.json").read_text())
    mr.B, mr.G = r["beta"], r["gamma"]
    sz = r.get("start_order_sizes", r["sizes"])
    p = np.array(sz, float) / sum(sz)
    n = 2 * len(p)
    d = [x for x in r["directions"] if x["direction"] == direction][0]
    wp, kpd, kk1 = np.array(d["last_good_orbit"]), d["K_last_good"], r["K1"]
    ev, V = np.linalg.eig(monodromy(wp, kpd, p))
    j = int(np.argmin(abs(ev.real + 1) + abs(ev.imag)))
    v = np.real(V[:, j]); v[0] = 0.; v /= np.linalg.norm(v)
    zp = wp[:n]
    # remove the direction in which the parent orbit moves with K (otherwise the parent at a shifted K satisfies the
    # amplitude condition; this happened for (1.0, 0.3), 2026-11-04)
    hk = 1e-5 * abs(kk1)
    tk = mr.shoot(wp, kpd + hk, p)[:n] - mr.shoot(wp, kpd - hk, p)[:n]
    tk[0] = 0.; tk /= np.linalg.norm(tk)
    v_raw_dot_tk = float(v @ tk)
    v = v - (v @ tk) * tk; v /= np.linalg.norm(v)
    out = dict(tag=tag, direction=direction, K_PD=kpd, K1=kk1, sizes=sz, parent_multiplier=ev[j].real, v_dot_parent_K_tangent=v_raw_dot_tk,
               host=socket.gethostname(), points=[])
    fn = OUT / f"pd_small_p2_{tag}_d{direction:+d}.json"
    a, fac = 1e-3, 1.15
    hist = []
    guess = np.r_[zp + a * v, 2 * wp[-1], kpd + direction * 1e-4 * abs(kk1)]
    while a < 1.5 and fac > 1.0005:
        u = solve(guess, a, zp, v, p)
        if u is None:
            if not hist:
                out["stop"] = "no solution at the first amplitude"
                break
            fac = np.sqrt(fac)
            a = hist[-1][0] * fac
            guess = hist[-1][1] if len(hist) < 2 else hist[-1][1] + (hist[-1][1] - hist[-2][1]) * (
                (a - hist[-1][0]) / (hist[-1][0] - hist[-2][0]))
            continue
        w = np.r_[u[:n], u[n]]
        k = float(u[n + 1])
        au = mr.audit(w, k, p)
        row = dict(a=a, K=k, f=(k - kpd) * direction / abs(kk1), period=float(u[n]),
                   half_amplitude=half_amplitude(w, k, p, 1), collective=au["collective"][:3],
                   internal_moduli=au["internal_moduli"], min_cluster_gap=au["min_cluster_gap"])
        out["points"].append(row)
        print(tag, direction, "a %.4e f %.6f half %.3e mult %s" % (a, row["f"], row["half_amplitude"],
              au["collective"][0]), flush=True)
        fn.write_text(json.dumps(out, indent=1))
        hist.append((a, u))
        a_new = a * fac
        guess = u if len(hist) < 2 else u + (u - hist[-2][1]) * ((a_new - a) / (a - hist[-2][0]))
        a = a_new
    fs = [q["f"] for q in out["points"]]
    if fs:
        i = int(np.argmax(fs))
        out["f_max"] = fs[i]
        out["a_at_f_max"] = out["points"][i]["a"]
        out["turns"] = i not in (0, len(fs) - 1)
    fn.write_text(json.dumps(out, indent=1))
    print(tag, direction, "f_max", out.get("f_max"), "turns", out.get("turns"), out.get("stop"), flush=True)


if __name__ == "__main__":
    main()
