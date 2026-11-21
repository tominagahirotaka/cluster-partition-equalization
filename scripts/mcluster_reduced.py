#!/usr/bin/env python3
"""Reduced M-cluster system with general (beta, gamma): the functions of paper1_equal_branch.py (written for M = 3)
for any number of clusters M = len(p) (2026-11-03, universality checks for paper 1, author's approval "案1と案2").

  x_a' = (x_a - x_a^3/3 - y_a)/eps + K (sum_b p_b x_b - x_a),  y_a' = eps (x_a + beta - gamma y_a)

State z = (x_1..x_M, y_1..y_M); a periodic orbit is w = (z(0), T) with the section x_1 = 0 (as in paper 1). The
parameters are module globals E, B, G (set by the caller). For M = 3 the functions agree with paper1_equal_branch.py
(checked by `python3 scripts/mcluster_reduced.py`).
"""
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

E, B, G, TOL = .3, .7, .5, 2e-12


def rhs(z, k, p):
    m = len(p)
    x, y = z[:m], z[m:]
    return np.r_[(x - x ** 3 / 3 - y) / E + k * (p @ x - x), E * (x + B - G * y)]


def jac(z, k, p):
    m = len(p)
    a = np.zeros((2 * m, 2 * m))
    a[:m, :m] = np.diag((1 - z[:m] ** 2) / E - k) + k * np.ones((m, 1)) * p
    a[:m, m:] = -np.eye(m) / E
    a[m:, :m] = E * np.eye(m)
    a[m:, m:] = -E * G * np.eye(m)
    return a


def flow(w, k, p):
    n = 2 * len(p)
    if not .5 < w[-1] < 400:
        return None
    s = solve_ivp(lambda t, v: np.r_[rhs(v[:n], k, p), (jac(v[:n], k, p) @ v[n:].reshape(n, n)).ravel()],
                  (0, w[-1]), np.r_[w[:n], np.eye(n).ravel()], method="DOP853", rtol=TOL, atol=TOL * .01, max_step=.08)
    if not s.success:
        return None
    a = np.zeros((n + 1, n + 1))
    a[:n, :n] = s.y[n:, -1].reshape(n, n) - np.eye(n)
    a[:n, n] = rhs(s.y[:n, -1], k, p)
    a[n, 0] = 1
    return np.r_[s.y[:n, -1] - w[:n], w[0]], a


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
    m = len(p)
    n = 2 * m

    def f(t, v):
        z = v[:n]
        a = np.zeros((m, 2, 2))
        a[:, 0, 0] = (1 - z[:m] ** 2) / E - k
        a[:, 0, 1] = -1 / E
        a[:, 1, 0] = E
        a[:, 1, 1] = -E * G
        return np.r_[rhs(z, k, p), (jac(z, k, p) @ v[n:n + n * n].reshape(n, n)).ravel(),
                     (a @ v[n + n * n:].reshape(m, 2, 2)).ravel()]
    T = w[-1]
    s = solve_ivp(f, (0, T), np.r_[w[:n], np.eye(n).ravel(), np.tile(np.eye(2), (m, 1, 1)).ravel()], method="DOP853",
                  rtol=TOL * .1, atol=TOL * .001, max_step=.04, dense_output=True)
    v = s.y[:, -1]
    ev = np.linalg.eigvals(v[n:n + n * n].reshape(n, n))
    i = int(np.argmin(abs(ev - 1)))
    coll = np.delete(ev, i)
    inter = [float(max(abs(np.linalg.eigvals(b)))) for b in v[n + n * n:].reshape(m, 2, 2)]
    t = np.linspace(0, T, 40001)
    x = s.sol(t)[:m]
    fire = np.array([t[np.argmax((x[c, :-1] < 0) & (x[c, 1:] >= 0))] for c in range(m)])
    offs = sorted(((fire - fire[0]) % T / T).round(4).tolist())
    order = np.argsort(-abs(coll))
    zz = s.sol(np.linspace(0, T, 400))
    gap = min(float(np.max(np.maximum(abs(zz[i_] - zz[j_]), abs(zz[m + i_] - zz[m + j_]))))
              for i_ in range(m) for j_ in range(i_))
    return dict(K=float(k), p=p.tolist(), period=float(T), trivial=[float(ev[i].real), float(ev[i].imag)],
                collective=[[float(c.real), float(c.imag)] for c in coll[order]],
                collective_max=float(abs(coll).max()), internal_moduli=inter, firing_offsets=offs,
                min_cluster_gap=gap, return_error=float(max(abs(v[:n] - w[:n]))))


def kind_of(r, ignore_internal=False):
    """First instability of an audited orbit, as bg_three_branch.kind_of; with ignore_internal only the collective
    multipliers count (the reduced orbit is followed past a loss of synchrony within a cluster)."""
    c = np.array([complex(*v) for v in r["collective"]])
    big = c[np.argmax(abs(c))]
    if not ignore_internal and max(r["internal_moduli"]) >= 1 and max(r["internal_moduli"]) >= abs(big):
        return "internal"
    if abs(big) < 1:
        return None
    if abs(big.imag) > 1e-9:
        return "torus"
    return "PD" if big.real < 0 else "fold"


def rest_point():
    x = brentq(lambda x: x - x ** 3 / 3 - (x + B) / G, -5, 5)
    return x, (x + B) / G


def k1():
    x, _ = rest_point()
    return (1 - x * x) / E - E * G


def excitable():
    """Unique equilibrium (gamma < 1 suffices for the cubic nullcline) and stable single element."""
    if G >= 1:
        return False
    x, _ = rest_point()
    return (1 - x * x) / E - E * G < 0


def continue_branch(w, k, p, dk0, k_stop, direction, ignore_internal=False):
    """Continue the orbit from K = k towards k_stop (direction +1: weaker coupling, -1: stronger); on shooting failure
    or at an instability halve the step and retry from the last good orbit until the step is below 1e-5 dk0.
    Returns the last good (w, K), the audited rows and the last bad (K, audit or None)."""
    dk, rows, good, last_bad = dk0, [], (w, k), None
    while direction * (k_stop - k) > 0 and dk > 1e-5 * dk0:
        kn = k + direction * dk
        if direction * (kn - k_stop) > 0:
            kn = k_stop
        w2 = shoot(good[0], kn, p)
        r = audit(w2, kn, p) if w2 is not None else None
        if r is None or kind_of(r, ignore_internal) is not None or r["min_cluster_gap"] < 1e-4:
            dk /= 2
            last_bad = (kn, r)
            continue
        good, k = (w2, kn), kn
        rows.append(r)
    return good, rows, last_bad


def end_kind(last_bad, ignore_internal=False):
    if last_bad is None:
        return "reached_stop"
    r = last_bad[1]
    if r is None:
        return "shoot_failed"
    if r["min_cluster_gap"] < 1e-4:
        return "clusters_merge"
    return kind_of(r, ignore_internal)


def sec(t, z):
    return z[0]


sec.direction = 1


def pd_side_test(w, k_pd, p, dk_list, direction):
    """As bg_three_branch.side_test, for M clusters: past the PD (K = k_pd + direction dk) perturb the orbit by +-1e-6
    along the eigenvector of the multiplier nearest -1 and integrate to t = 20000: rest, small period-two orbit, other."""
    n = 2 * len(p)
    xr, yr = rest_point()
    eq = np.r_[np.full(len(p), xr), np.full(len(p), yr)]
    out = []
    for dk in dk_list:
        k = k_pd + direction * dk
        ww = w
        for kk in np.linspace(k_pd, k, 6)[1:]:
            ww = shoot(ww, kk, p)
            if ww is None:
                break
        if ww is None:
            out.append(dict(dK=dk, status="shoot_failed"))
            continue
        s = solve_ivp(lambda t, v: np.r_[rhs(v[:n], k, p), (jac(v[:n], k, p) @ v[n:].reshape(n, n)).ravel()],
                      (0, ww[-1]), np.r_[ww[:n], np.eye(n).ravel()], method="DOP853", rtol=1e-12, atol=1e-14,
                      max_step=.04)
        ev, V = np.linalg.eig(s.y[n:, -1].reshape(n, n))
        j = int(np.argmin(abs(ev.real + 1) + abs(ev.imag)))
        v = np.real(V[:, j]); v /= np.linalg.norm(v)
        runs = []
        for sign in (1, -1):
            r = solve_ivp(lambda t, z: rhs(z, k, p), (0, 20000), ww[:n] + sign * 1e-6 * v, method="DOP853",
                          rtol=1e-11, atol=1e-13, max_step=.08, events=sec)
            fin = r.y[:, -1]
            pts, tt = r.y_events[0], r.t_events[0]
            late = pts[tt > 18000] if len(pts) else pts
            dist = float(np.max(abs(late - ww[:n]))) if len(late) else None
            gaps = r.t_events[0][tt > 18000]
            runs.append(dict(sign=sign, rest=bool(max(abs(fin - eq)) < 1e-8), late_max_dist=dist,
                             crossings=int(len(tt)),
                             late_return_times=np.diff(gaps)[-4:].tolist() if len(gaps) > 4 else None))
        out.append(dict(dK=dk, K=k, multiplier=[float(ev[j].real), float(ev[j].imag)], runs=runs))
    return out


if __name__ == "__main__":  # agreement with paper1_equal_branch for M = 3
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paper1_equal_branch as eb
    root = Path(__file__).resolve().parents[1]
    d = np.load(root / "analysis/three_doubled_branch/confirmed_sign1_tol2e-12/point02_parameters.npz")
    k = float(np.load(root / "analysis/three_doubled_branch/confirmed_sign1_tol2e-12/point02.npz")["K"])
    p = np.array([34, 33, 33]) / 100
    w = np.asarray(d["parent"], float)
    a, b = eb.audit(eb.shoot(w, k, p), k, p), audit(shoot(w, k, p), k, p)
    print("period", a["period"], b["period"], "collective", a["collective"][:2], b["collective"][:2],
          "internal", a["internal_moduli"], b["internal_moduli"])
