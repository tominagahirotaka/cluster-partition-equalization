#!/usr/bin/env python3
"""(b) of 2026-11-06 (author's request "(b)を先に，(a)を後で確かめてください"): at points of the universality map where
the internally stable 34:33:33 branch ends by a fold (universality_map.py end kind shoot_failed, labelled fold-like by
universality_map_ends.py), is the unstable period-one orbit born at the fold a saddle whose stable manifold separates
the stable cluster oscillation from synchronized rest, as the period-doubled saddle does in paper 1?

Reduced system (mcluster_reduced). Steps:
 1. pseudo-arclength continuation in (z, T, K) from the last good orbit of the map in both directions (joined so that
    the path starts on the stable side; 2026-11-06), each until K is 0.03 |K1| from its start or 300 points (step adaptive, Newton with the K column by central differences,
    h = 1e-7). Along the path: K, period, collective and internal multipliers (every point).
 2. K_f = the first turning point with a local maximum of K (parabola through the three points around it).
 3. At d = 0.05, 0.1, 0.25, 0.5 times the width of the middle segment (from the fold to the next turning point;
    changed 2026-11-06 after the first point showed an S-shaped branch only 3e-5 |K1| wide) below the fold: the stable and the saddle
    orbit (shot from the nearest path points), their section distance (max norm), the unstable multiplier mu_u of the
    saddle, internal moduli of both. Fold scaling: distance and ln mu_u ~ d^(1/2).
 4. Two-sided test at a quarter of the width: start from the saddle +- delta along the eigenvector of mu_u (delta = 1e-4, 1e-6),
    RK4 (pd_route_scan.run, first returns, up to 400): outcome rest (no returns at the end and within 1e-6 of the rest
    point), the stable orbit (last section point within 1e-4 of it) or other (period in returns and distances
    recorded), and the number of returns spent within 0.05 of
    the saddle compared with ln(0.05/delta)/ln mu_u.
  -> analysis/universality/fold_saddle/<map name>
  python3 scripts/fold_saddle_test.py e0.3_b0.84_g0.4_34-33-33.json
"""
from pathlib import Path
import json
import socket
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
import pd_route_scan as ps  # noqa: E402
from pd_cascade_shoot import monodromy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
OUT = ROOT / "analysis/universality/fold_saddle"


def end_point(z, T, k, p):
    s = solve_ivp(lambda t, u: mr.rhs(u, k, p), (0, T), z, method="DOP853", rtol=mr.TOL, atol=mr.TOL * .01,
                  max_step=.08)
    return s.y[:, -1]


def jac_full(u, p):
    """residual (n+1) and Jacobian (n+1) x (n+2) of the periodic-orbit equations in u = (z, T, K)."""
    n = len(u) - 2
    r = mr.flow(u[:n + 1], u[n + 1], p)
    if r is None:
        return None
    res, a = r
    h = 1e-7
    dk = np.r_[(end_point(u[:n], u[n], u[n + 1] + h, p) - end_point(u[:n], u[n], u[n + 1] - h, p)) / (2 * h), 0.]
    return res, np.c_[a, dk]


def corrector(u, u0, t0, ds, p):
    for _ in range(12):
        r = jac_full(u, p)
        if r is None:
            return None
        res, J = r
        g = t0 @ (u - u0) - ds
        if max(abs(res)) < 1e-11 and abs(g) < 1e-11:
            return u
        d = np.linalg.solve(np.r_[J, t0[None, :]], -np.r_[res, g])
        if max(abs(d)) > .3:
            return None
        u = u + d
    return None


def tangent(u, p, t_old):
    res, J = jac_full(u, p)
    _, _, vt = np.linalg.svd(J)
    t = vt[-1]
    return t if t @ t_old > 0 else -t


def multipliers(w, k, p):
    a = mr.audit(w, k, p)
    c = sorted((complex(*v) for v in a["collective"]), key=abs, reverse=True)
    return a, c


def arclength(w, k, p, kk1, sign=1):
    n = 2 * len(p)
    u = np.r_[w, k]
    t = np.zeros(n + 2); t[-1] = 1.
    t = sign * tangent(u, p, t)
    ds, path, kmax = 2e-3 * abs(kk1), [], -np.inf
    while ds > 1e-7:
        u_new = corrector(u + ds * t, u, t, ds, p)
        if u_new is None:
            ds /= 2
            continue
        u, t = u_new, tangent(u_new, p, t)
        a, c = multipliers(u[:n + 1], u[n + 1], p)
        path.append(dict(K=float(u[n + 1]), period=float(u[n]), top=[[z.real, z.imag] for z in c[:2]],
                         internal_moduli=a["internal_moduli"], u=u.tolist()))
        kmax = max(kmax, u[n + 1])
        if abs(u[n + 1] - k) > .03 * abs(kk1) or len(path) >= 300:
            break
        ds = min(ds * 1.3, 5e-3 * abs(kk1))
    return path


def unstable_vector(w, k, p):
    ev, V = np.linalg.eig(monodromy(w, k, p))
    j = int(np.argmax(np.where(abs(ev.imag) < 1e-9, ev.real, -np.inf)))
    v = np.real(V[:, j])
    return v / np.linalg.norm(v), float(ev[j].real)


def main():
    name = sys.argv[1]
    r = json.loads((MAP / name).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    p = np.array(r["sizes"], float) / sum(r["sizes"])
    n = 2 * len(p)
    kk1 = r["K1"]
    t0 = time.time()
    w0, k0 = np.array(r["end"]["last_good_orbit"]), r["end"]["K_last_good"]
    out = dict(name=name, eps=r["eps"], beta=r["beta"], gamma=r["gamma"], K1=kk1, K_map_end=k0,
               host=socket.gethostname())
    # both directions from the last good orbit, joined (the map often stops within 1e-7 |K1| of the fold, where the
    # tangent is nearly perpendicular to K and one direction alone may start on either branch; changed 2026-11-06)
    a0, c0 = multipliers(w0, k0, p)
    start = dict(K=float(k0), period=float(w0[-1]), top=[[z.real, z.imag] for z in c0[:2]],
                 internal_moduli=a0["internal_moduli"], u=np.r_[w0, k0].tolist())
    back = arclength(w0, k0, p, kk1, -1)
    fwd = arclength(w0, k0, p, kk1, 1)
    # order the joined path so that it starts on the stable side (smaller leading multiplier at its far end)
    if abs(complex(*back[-1]["top"][0])) > abs(complex(*fwd[-1]["top"][0])):
        back, fwd = fwd, back
    path = back[::-1] + [start] + fwd
    out["start_index"] = len(back)
    ks = np.array([q["K"] for q in path])
    turns = [j for j in range(1, len(ks) - 1) if (ks[j] - ks[j - 1]) * (ks[j + 1] - ks[j]) < 0]
    out["turning_points"] = [dict(index=j, K=float(ks[j]), kind="max" if ks[j] > ks[j - 1] else "min",
                                  period=path[j]["period"]) for j in turns]
    i = turns[0] if turns and ks[turns[0]] > ks[turns[0] - 1] else 0
    out["path"] = [{a: b for a, b in q.items() if a != "u"} for q in path]
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / name
    if i == 0:
        out["stop"] = "no turning point on the path"
        fn.write_text(json.dumps(out, indent=1))
        print(name, out["stop"], flush=True)
        return
    s = np.array([0., 1., 2.])
    c = np.polyfit(s, ks[i - 1:i + 2], 2)
    kf = float(np.polyval(c, -c[1] / (2 * c[0])))
    out["K_fold"] = kf
    out["f_fold"] = kf / kk1
    # stable side = path points before i (the map branch), saddle side = after i
    xr, yr = mr.rest_point()
    eq = np.r_[np.full(len(p), xr), np.full(len(p), yr)]
    out["coexistence"] = []
    # the middle (saddle) segment runs from the first fold to the next turning point (or to the end of the path);
    # distances d below the fold are fractions of its width (changed 2026-11-06 after the first point, where the width
    # was only 3e-5 |K1|)
    nxt = turns[1] if len(turns) > 1 else len(ks) - 1
    width = (kf - ks[nxt]) / abs(kk1)
    out["middle_segment"] = dict(end_index=nxt, width_over_K1=float(width), end_is_turn=len(turns) > 1)
    for frac in (.05, .1, .25, .5):
        d = frac * width
        k = kf - d * abs(kk1)
        js = int(np.argmin(abs(ks[:i] - k))) if i > 0 else 0
        ju = i + 1 + int(np.argmin(abs(ks[i + 1:nxt + 1] - k)))
        ws = mr.shoot(np.array(path[js]["u"][:n + 1]), k, p)
        wu = mr.shoot(np.array(path[ju]["u"][:n + 1]), k, p)
        row = dict(frac=frac, d=d, K=k)
        if ws is None or wu is None:
            row["stop"] = "shooting failed"
            out["coexistence"].append(row)
            continue
        vu, mu = unstable_vector(wu, k, p)
        a_s, c_s = multipliers(ws, k, p)
        a_u, c_u = multipliers(wu, k, p)
        row.update(distance=float(np.max(abs(ws[:n] - wu[:n]))), mu_u=mu, stable_top=[c_s[0].real, c_s[0].imag],
                   saddle_top=[c_u[0].real, c_u[0].imag], saddle_second=[c_u[1].real, c_u[1].imag],
                   stable_internal=a_s["internal_moduli"], saddle_internal=a_u["internal_moduli"],
                   period_stable=float(ws[-1]), period_saddle=float(wu[-1]))
        if frac == .25:
            vu[0] = 0.
            tests = []
            for sign in (1, -1):
                for delta in (1e-4, 1e-6):
                    sec, ts, _, _, z = ps.run(wu[:n] + sign * delta * vu / np.linalg.norm(vu), k, p, mr.E, mr.B, mr.G,
                                              ps.DT, 1e-9, 400)
                    rest = bool(len(sec) < 400 and np.max(abs(z - eq)) < 1e-6)
                    osc = bool(len(sec) > 5 and np.max(abs(sec[-1] - ws[:n])) < 1e-4)
                    near = int(np.sum(np.max(abs(sec - wu[:n]), axis=1) < .05)) if len(sec) else 0
                    t = dict(sign=sign, delta=delta, outcome="rest" if rest else "stable orbit" if osc else "other",
                             returns=len(sec), returns_near_saddle=near,
                             predicted=float(np.log(.05 / delta) / np.log(mu)) if mu > 1 else None)
                    if not rest and not osc and len(sec) > 2:
                        t.update(period_returns=ps.distinct(sec), last_section=sec[-1].tolist(),
                                 dist_from_stable=float(np.max(abs(sec[-1] - ws[:n]))),
                                 dist_from_saddle=float(np.max(abs(sec[-1] - wu[:n]))))
                    tests.append(t)
            row["two_sided"] = tests
        out["coexistence"].append(row)
        fn.write_text(json.dumps(out, indent=1))
    ok = [q for q in out["coexistence"] if "distance" in q and q["mu_u"] > 1]
    if len(ok) >= 3:
        dd = np.log([q["d"] for q in ok])
        out["slope_distance"] = float(np.polyfit(dd, np.log([q["distance"] for q in ok]), 1)[0])
        out["slope_log_mu"] = float(np.polyfit(dd, np.log([np.log(q["mu_u"]) for q in ok]), 1)[0])
    out["seconds"] = time.time() - t0
    fn.write_text(json.dumps(out, indent=1))
    ts = [t["outcome"] for q in out["coexistence"] for t in q.get("two_sided", [])]
    print(name, "f_fold %.4f" % out["f_fold"], "slopes", out.get("slope_distance"), out.get("slope_log_mu"), ts,
          "%.0fs" % out["seconds"], flush=True)


if __name__ == "__main__":
    main()
