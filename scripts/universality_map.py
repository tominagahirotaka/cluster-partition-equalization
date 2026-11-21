#!/usr/bin/env python3
"""Universality map of paper 1's scenario (2026-11-05, author's request "Universalityが知りたいので、時間がかかっても
いいので、Parameterの範囲を広げて計算を流してください"; reduced system, mcluster_reduced, any eps).

At one (eps, beta, gamma) and partition (default 34:33:33):
 1. excitable? (else stop)
 2. start orbit as universality_branches.py plan 1 (random centroids, K = fac K1, fac = 3, 1.5, 6; seeds 1-5)
 3. continue towards weak coupling (K -> 0), step |K1|/200 (mcluster_reduced.continue_branch): first end kind
 4. if that end is internal: continue the reduced orbit past it ignoring internal multipliers (collective end)
 5. for a collective PD (from 3 or 4): its type from the period-two branch continued in amplitude
    (pd_small_p2_amplitude.solve, a = 1e-3, 2e-3, 4e-3): subcritical if the branch lies before the doubling (f < 0)
    with multiplier > 1, supercritical if after it (f > 0) with multiplier < 1
 6. for a PD from 3 (internally stable): where the orbit goes 0.005 |K1| past it, from the orbit +- 1e-4 along the -1
    eigenvector (RK4, pd_route_scan.run, transient 3e5): rest, or the period in returns and amplitude
Paper 1's scenario = end in 3 is PD, subcritical, and at least one side goes to rest.
  -> analysis/universality/map/e{eps}_b{beta}_g{gamma}_{sizes}.json

  python3 scripts/universality_map.py --eps 0.3 --beta 0.7 --gamma 0.5
"""
from pathlib import Path
import argparse
import json
import socket
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
import pd_route_scan as ps  # noqa: E402
import universality_branches as ub  # noqa: E402
from pd_cascade_shoot import half_amplitude, monodromy  # noqa: E402
from pd_small_p2_amplitude import solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/universality/map"


def minus_one_vector(w, k, p):
    ev, V = np.linalg.eig(monodromy(w, k, p))
    # the most negative real multiplier (nearest -1 fails past the doubling, where it is about -2; fixed 2026-11-05)
    j = int(np.argmin(np.where(abs(ev.imag) < 1e-9, ev.real, np.inf)))
    v = np.real(V[:, j])
    return v / np.linalg.norm(v), float(ev[j].real)


def pd_type(w, kpd, p, kk1, direction):
    n = 2 * len(p)
    v, mu = minus_one_vector(w, kpd, p)
    v[0] = 0.; v /= np.linalg.norm(v)
    hk = 1e-5 * abs(kk1)
    a1, a2 = mr.shoot(w, kpd + hk, p), mr.shoot(w, kpd - hk, p)
    if a1 is None or a2 is None:
        return dict(type="undetermined", why="parent not found next to the doubling")
    tk = a1[:n] - a2[:n]; tk[0] = 0.; tk /= np.linalg.norm(tk)
    v = v - (v @ tk) * tk; v /= np.linalg.norm(v)
    zp, pts = w[:n], []
    guess = np.r_[zp + 1e-3 * v, 2 * w[-1], kpd]
    for a in (1e-3, 2e-3, 4e-3):
        u = solve(guess, a, zp, v, p)
        if u is None:
            break
        ww, k = np.r_[u[:n], u[n]], float(u[n + 1])
        au = mr.audit(ww, k, p)
        c = au["collective"][0]
        pts.append(dict(a=a, f=(k - kpd) * direction / abs(kk1), half_amplitude=half_amplitude(ww, k, p, 1),
                        multiplier=c, internal_max=max(au["internal_moduli"])))
        guess = u
    ok = [q for q in pts if q["half_amplitude"] > 1e-4]
    if not ok:
        return dict(type="undetermined", why="no period-two solution away from the parent", points=pts,
                    parent_multiplier=mu)
    q = ok[-1]
    m = abs(complex(*q["multiplier"]))
    t = ("subcritical" if q["f"] < 0 and m > 1 else "supercritical" if q["f"] > 0 and m < 1 else "unclear")
    return dict(type=t, points=pts, parent_multiplier=mu)


def destination(w, kpd, p, kk1, direction):
    n = 2 * len(p)
    k = kpd + direction * .005 * abs(kk1)
    ww = w
    for kk in np.linspace(kpd, k, 6)[1:]:
        ww = mr.shoot(ww, kk, p)
        if ww is None:
            return dict(status="parent not found past the doubling")
    v, mu = minus_one_vector(ww, k, p)
    xr, yr = mr.rest_point()
    eq = np.r_[np.full(len(p), xr), np.full(len(p), yr)]
    out = dict(K=k, multiplier=mu, runs=[])
    for sign in (1, -1):
        sec, ts, le, il, z = ps.run(ww[:n] + sign * 1e-4 * v, k, p, mr.E, mr.B, mr.G, ps.DT, 3e5, 40)
        rest = bool(np.max(abs(z - eq)) < 1e-6)
        r = dict(sign=sign, rest=rest)
        if not rest and len(sec) > 2:
            r.update(period_returns=ps.distinct(sec), step=float(np.max(abs(sec[-1] - sec[-2]))),
                     dist_from_parent=float(np.max(abs(sec[-1] - ww[:n]))), lyap=float(le),
                     internal_max=float(il.max()))
        out["runs"].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--sizes", default="34,33,33")
    a = ap.parse_args()
    mr.E, mr.B, mr.G = a.eps, a.beta, a.gamma
    sizes = [int(v) for v in a.sizes.split(",")]
    p = np.array(sizes, float) / sum(sizes)
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / f"e{a.eps:g}_b{a.beta:g}_g{a.gamma:g}_{'-'.join(map(str, sizes))}.json"
    t0 = time.time()
    row = dict(eps=a.eps, beta=a.beta, gamma=a.gamma, sizes=sizes, host=socket.gethostname(), excitable=mr.excitable())
    if not row["excitable"]:
        row["status"] = "not excitable"
        fn.write_text(json.dumps(row, indent=1))
        return
    kk1 = mr.k1()
    row["K1"] = kk1
    w, why = None, ""
    for fac in (3., 1.5, 6.):
        k0 = fac * kk1
        for seed in range(1, 6):
            rng = np.random.default_rng(seed)
            m = len(p)
            z = np.r_[rng.uniform(-2, 2, m), rng.uniform(-.6, 1.4, m)]
            w, why = ub.periodic_start(z, k0, p, 3000)
            if w is not None:
                row.update(start_fac=fac, start_seed=seed, start_K=k0)
                break
        if w is not None:
            break
    if w is None:
        row["status"] = f"no start orbit ({why})"
        row["seconds"] = time.time() - t0
        fn.write_text(json.dumps(row, indent=1))
        return
    row["status"] = "ok"
    row["start_audit"] = mr.audit(w, k0, p)
    good, rows, last_bad = mr.continue_branch(w, k0, p, abs(kk1) / 200, -1e-6, 1)
    kind = mr.end_kind(last_bad)
    row["end"] = dict(kind=kind, K_last_good=float(good[1]), f_last_good=float(good[1] / kk1), npts=len(rows),
                      period=float(good[0][-1]), last_bad_audit=None if last_bad is None else last_bad[1],
                      last_good_orbit=good[0].tolist())
    fn.write_text(json.dumps(row, indent=1))
    if kind == "PD":
        row["pd_type"] = pd_type(good[0], good[1], p, kk1, 1)
        fn.write_text(json.dumps(row, indent=1))
        row["destination"] = destination(good[0], good[1], p, kk1, 1)
    elif kind == "internal":
        g2, rows2, lb2 = mr.continue_branch(good[0], good[1], p, abs(kk1) / 200, -1e-6, 1, ignore_internal=True)
        k2 = mr.end_kind(lb2, ignore_internal=True)
        row["collective_end"] = dict(kind=k2, K_last_good=float(g2[1]), f_last_good=float(g2[1] / kk1),
                                     npts=len(rows2), last_good_orbit=g2[0].tolist())
        if k2 == "PD":
            row["collective_end"]["pd_type"] = pd_type(g2[0], g2[1], p, kk1, 1)
    d = row.get("destination", {})
    row["paper1_scenario"] = bool(kind == "PD" and row["pd_type"]["type"] == "subcritical"
                                  and any(r.get("rest") for r in d.get("runs", [])))
    row["seconds"] = time.time() - t0
    fn.write_text(json.dumps(row, indent=1))
    print(fn.name, kind, row.get("pd_type", {}).get("type"), row["paper1_scenario"], "%.0fs" % row["seconds"],
          flush=True)


if __name__ == "__main__":
    main()
