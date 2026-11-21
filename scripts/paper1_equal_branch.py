#!/usr/bin/env python3
"""Paper 1, reviewer preparation: the three-cluster branch C at exactly equal fractions (2026-11-03, author's approval).

Parameters of paper 1 (eps = 0.3, beta = 0.7, gamma = 0.5; coupling K, not eps K). Branch C of paper 1 has the partition
34:33:33 and a collective period doubling at K_PD ~ -0.18123536 (subcritical). This script
  1. shoots the parent orbit of 34:33:33 at K = -0.18140440212409442 from the stored parent
     (analysis/three_doubled_branch/confirmed_sign1_tol2e-12/point02_parameters.npz),
  2. scans K for 34:33:33 and locates the crossing of a collective multiplier through -1 (secant on mu + 1), as a check
     against K_PD of paper 1,
  3. moves the fractions from (0.34, 0.33, 0.33) to (1/3, 1/3, 1/3) in 20 steps at that K (continuation in p),
  4. scans K for the equal fractions and locates its crossings of -1 in the same way.
At every point: period, all collective multipliers (the trivial one removed), the internal multiplier moduli of the three
clusters (2x2 blocks), and the phase offsets of the firings (equal partition: a third of a period apart or not).
Shooting as ThreeOrbit.shoot of paper 1 (DOP853, tol 2e-12, section x_1 = 0).
  -> analysis/three_equal_branch/equal_branch_e0.3.json

  python3 scripts/paper1_equal_branch.py
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
E, B, G, TOL = .3, .7, .5, 2e-12
OUT = ROOT / "analysis/three_equal_branch"


def rhs(z, k, p):
    x, y = z[:3], z[3:]
    return np.r_[(x - x ** 3 / 3 - y) / E + k * (p @ x - x), E * (x + B - G * y)]


def jac(z, k, p):
    a = np.zeros((6, 6))
    a[:3, :3] = np.diag((1 - z[:3] ** 2) / E - k) + k * np.ones((3, 1)) * p
    a[:3, 3:] = -np.eye(3) / E
    a[3:, :3] = E * np.eye(3)
    a[3:, 3:] = -E * G * np.eye(3)
    return a


def flow(w, k, p):
    if not .5 < w[-1] < 400:
        return None
    s = solve_ivp(lambda t, v: np.r_[rhs(v[:6], k, p), (jac(v[:6], k, p) @ v[6:].reshape(6, 6)).ravel()],
                  (0, w[-1]), np.r_[w[:6], np.eye(6).ravel()], method="DOP853", rtol=TOL, atol=TOL * .01, max_step=.08)
    if not s.success:
        return None
    a = np.zeros((7, 7))
    a[:6, :6] = s.y[6:, -1].reshape(6, 6) - np.eye(6)
    a[:6, 6] = rhs(s.y[:6, -1], k, p)
    a[6, 0] = 1
    return np.r_[s.y[:6, -1] - w[:6], w[0]], a


def shoot(w, k, p):
    w = w.copy()
    for _ in range(20):
        r = flow(w, k, p)
        if r is None:
            return None
        res, a = r
        err = max(abs(res))
        if err < 2e-12:
            return w
        d = np.linalg.lstsq(a, -res, rcond=1e-13)[0]
        for sc in (1, .5, .25, .125, .0625, .03125):
            c = flow(w + sc * d, k, p)
            if c is not None and max(abs(c[0])) < err:
                w = w + sc * d
                break
        else:
            return None
    return None


def audit(w, k, p):
    def f(t, v):
        z = v[:6]
        a = np.zeros((3, 2, 2))
        a[:, 0, 0] = (1 - z[:3] ** 2) / E - k
        a[:, 0, 1] = -1 / E
        a[:, 1, 0] = E
        a[:, 1, 1] = -E * G
        return np.r_[rhs(z, k, p), (jac(z, k, p) @ v[6:42].reshape(6, 6)).ravel(), (a @ v[42:].reshape(3, 2, 2)).ravel()]
    T = w[-1]
    s = solve_ivp(f, (0, T), np.r_[w[:6], np.eye(6).ravel(), np.tile(np.eye(2), (3, 1, 1)).ravel()], method="DOP853",
                  rtol=TOL * .1, atol=TOL * .001, max_step=.04, dense_output=True)
    v = s.y[:, -1]
    ev = np.linalg.eigvals(v[6:42].reshape(6, 6))
    i = int(np.argmin(abs(ev - 1)))
    coll = np.delete(ev, i)
    inter = [float(max(abs(np.linalg.eigvals(b)))) for b in v[42:].reshape(3, 2, 2)]
    t = np.linspace(0, T, 40001)
    x = s.sol(t)[:3]
    fire = np.array([t[np.argmax((x[c, :-1] < 0) & (x[c, 1:] >= 0))] for c in range(3)])
    offs = sorted(((fire - fire[0]) % T / T).round(4).tolist())
    order = np.argsort(-abs(coll))
    return dict(K=float(k), p=p.tolist(), period=float(T), trivial=[float(ev[i].real), float(ev[i].imag)],
                collective=[[float(c.real), float(c.imag)] for c in coll[order]],
                collective_max=float(abs(coll).max()), internal_moduli=inter, firing_offsets=offs,
                return_error=float(max(abs(v[:6] - w[:6]))))


def neg_real(row):
    """mu + 1 for the collective multiplier closest to -1 among the (nearly) real ones, or None."""
    re = [c[0] for c in row["collective"] if abs(c[1]) < 1e-9]
    return min(re, key=lambda r: abs(r + 1)) + 1 if re else None


def scan(w, p, ks):
    rows, ws = [], {}
    for k in ks:
        w2 = shoot(w, k, p)
        if w2 is None:
            rows.append(dict(K=float(k), status="failed"))
            print("failed", k, flush=True)
            break
        w = w2
        ws[float(k)] = w.copy()
        r = audit(w, k, p)
        rows.append(r)
        print(f"K {k:.5f} T {r['period']:.6f} coll {np.round(r['collective'][:2], 5).tolist()} int "
              f"{np.round(r['internal_moduli'], 4).tolist()} offs {r['firing_offsets']}", flush=True)
    return rows, ws


def crossing(rows, ws, p):
    """secant refinement of every sign change of mu + 1 between neighbouring scan points"""
    out = []
    good = [r for r in rows if "period" in r]
    for a, b in zip(good, good[1:]):
        fa, fb = neg_real(a), neg_real(b)
        if fa is None or fb is None or fa * fb > 0:
            continue
        ka, kb, w = a["K"], b["K"], ws[a["K"]]
        for _ in range(30):
            kc = ka - fa * (kb - ka) / (fb - fa)
            wc = shoot(w, kc, p)
            rc = audit(wc, kc, p)
            fc = neg_real(rc)
            if fc is None or abs(fc) < 1e-10:
                break
            if fa * fc < 0:
                kb, fb = kc, fc
            else:
                ka, fa, w = kc, fc, wc
        out.append(dict(K=kc, mu_plus_1=fc, row=rc))
        print("crossing", kc, fc, flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    par = np.load(ROOT / "analysis/three_doubled_branch/confirmed_sign1_tol2e-12/point02_parameters.npz")
    k0 = -0.18140440212409442
    p34 = np.array([34, 33, 33]) / 100
    w0 = shoot(np.asarray(par["parent"], float), k0, p34)
    res = dict(script="scripts/paper1_equal_branch.py", sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               eps=E, K0=k0, parent_stored_period=float(par["parent"][-1]), parent_shot_period=float(w0[-1]))
    ks = np.round(np.arange(-0.1840, -0.1779, 0.0002), 6)
    r34, w34 = scan(w0, p34, [k0] + [k for k in ks if k > k0])
    r34b, w34b = scan(w0, p34, [k for k in ks[::-1] if k < k0])
    rows34 = sorted(r34 + r34b, key=lambda r: r["K"])
    res["p34_scan"] = rows34
    res["p34_crossings"] = crossing(rows34, {**w34, **w34b}, p34)
    w = w0
    path = []
    for s in np.linspace(0, 1, 21)[1:]:
        p = (1 - s) * p34 + s * np.ones(3) / 3
        w = shoot(w, k0, p)
        r = audit(w, k0, p)
        path.append(r)
        print("homotopy", np.round(p, 4).tolist(), f"T {r['period']:.6f}", np.round(r["collective"][:2], 5).tolist(),
              np.round(r["internal_moduli"], 4).tolist(), flush=True)
    res["homotopy"] = path
    peq = np.ones(3) / 3
    ks2 = np.round(np.arange(-0.2200, -0.1399, 0.0005), 6)
    re1, we1 = scan(w, peq, [k0] + [k for k in ks2 if k > k0])
    re2, we2 = scan(w, peq, [k for k in ks2[::-1] if k < k0])
    rowsq = sorted(re1 + re2, key=lambda r: r["K"])
    res["equal_scan"] = rowsq
    res["equal_crossings"] = crossing(rowsq, {**we1, **we2}, peq)
    (OUT / "equal_branch_e0.3.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
