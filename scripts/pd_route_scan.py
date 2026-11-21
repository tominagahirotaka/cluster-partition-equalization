#!/usr/bin/env python3
"""Past the supercritical collective period doublings of the five-cluster branches: is there a route to chaos?
(2026-11-04, author's approval "1〜3を進めてください"; steps 1 and 2: bifurcation diagram and Lyapunov exponents.
Step 3, the period-two orbit and its next doubling by shooting, is pd_cascade_shoot.py.)

Reduced M-cluster system (as mcluster_reduced.py), integrated by fixed-step RK4 in Numba (dt = 0.005), together with
one collective tangent vector (2M) and one internal tangent vector per cluster (the 2x2 block of each cluster). For
K = K_PD + direction f |K1|, f from -0.005 to 0.15 (300 values), swept forward (increasing f) from the stored orbit
plus 1e-6 and then backward from the final forward state (attractor following, hysteresis): at each K a transient of
5000, then 300 upward crossings of x_1 = 0 (section points by cubic Hermite interpolation). Recorded: the section
points, the largest collective Lyapunov exponent over the recorded window (tangent renormalized at every crossing;
it is 0 on a periodic orbit, along the flow, and positive on a chaotic attractor), the internal exponent of each
cluster, and the period in section returns (smallest j <= 64 with the last two points repeating after j returns within
1e-5; 0 if none). The accuracy of RK4 here is checked
against the DOP853 orbit of mcluster_reduced at the first K (section point difference written to the output).
  -> analysis/universality/pd_route_<tag>_d<dir>.npz and .json

  python3 scripts/pd_route_scan.py plan2_b0.8_g0.5_23-21-20-19-17 -1
"""
from pathlib import Path
import json
import socket
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/universality"
DT, TRANS, NREC, NK = 0.005, 5000., 300, 300


@njit(cache=True)
def deriv(s, k, p, E, B, G, M, out):
    mean = 0.
    vmean = 0.
    for a in range(M):
        mean += p[a] * s[a]
        vmean += p[a] * s[2 * M + a]
    for a in range(M):
        x, y = s[a], s[M + a]
        out[a] = (x - x * x * x / 3 - y) / E + k * (mean - x)
        out[M + a] = E * (x + B - G * y)
        d = (1 - x * x) / E - k
        vx, vy = s[2 * M + a], s[3 * M + a]
        out[2 * M + a] = d * vx + k * vmean - vy / E
        out[3 * M + a] = E * vx - E * G * vy
        ux, uy = s[4 * M + 2 * a], s[4 * M + 2 * a + 1]
        out[4 * M + 2 * a] = d * ux - uy / E
        out[4 * M + 2 * a + 1] = E * ux - E * G * uy


@njit(cache=True)
def rk4(s, k, p, E, B, G, M, dt, k1, k2, k3, k4, tmp):
    deriv(s, k, p, E, B, G, M, k1)
    for i in range(s.size):
        tmp[i] = s[i] + .5 * dt * k1[i]
    deriv(tmp, k, p, E, B, G, M, k2)
    for i in range(s.size):
        tmp[i] = s[i] + .5 * dt * k2[i]
    deriv(tmp, k, p, E, B, G, M, k3)
    for i in range(s.size):
        tmp[i] = s[i] + dt * k3[i]
    deriv(tmp, k, p, E, B, G, M, k4)
    for i in range(s.size):
        s[i] += dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])


@njit(cache=True)
def run(z0, k, p, E, B, G, dt, t_trans, nrec):
    M = p.size
    n = 2 * M
    s = np.zeros(6 * M)
    s[:n] = z0
    for a in range(M):  # unit tangent vectors
        s[2 * M + a] = 1. / np.sqrt(n)
        s[3 * M + a] = 1. / np.sqrt(n)
        s[4 * M + 2 * a] = 1.
    k1 = np.zeros(6 * M); k2 = np.zeros(6 * M); k3 = np.zeros(6 * M); k4 = np.zeros(6 * M); tmp = np.zeros(6 * M)
    fa = np.zeros(6 * M); fb = np.zeros(6 * M)
    sec = np.zeros((nrec, n)); tsec = np.zeros(nrec)
    t, nr = 0., 0
    lsum = 0.; isum = np.zeros(M); t_start = -1.
    old = s.copy()
    tmax = t_trans + nrec * 400.
    while nr < nrec and t < tmax:
        old[:] = s
        rk4(s, k, p, E, B, G, M, dt, k1, k2, k3, k4, tmp)
        t += dt
        if old[0] < 0 <= s[0]:
            deriv(old, k, p, E, B, G, M, fa)
            deriv(s, k, p, E, B, G, M, fb)
            th = -old[0] / (s[0] - old[0])  # Newton on the cubic Hermite interpolant of x_1
            for _ in range(8):
                h00 = 2 * th ** 3 - 3 * th ** 2 + 1; h10 = th ** 3 - 2 * th ** 2 + th
                h01 = -2 * th ** 3 + 3 * th ** 2; h11 = th ** 3 - th ** 2
                f = h00 * old[0] + h10 * dt * fa[0] + h01 * s[0] + h11 * dt * fb[0]
                d00 = 6 * th ** 2 - 6 * th; d10 = 3 * th ** 2 - 4 * th + 1
                d01 = -6 * th ** 2 + 6 * th; d11 = 3 * th ** 2 - 2 * th
                df = d00 * old[0] + d10 * dt * fa[0] + d01 * s[0] + d11 * dt * fb[0]
                th -= f / df
            h00 = 2 * th ** 3 - 3 * th ** 2 + 1; h10 = th ** 3 - 2 * th ** 2 + th
            h01 = -2 * th ** 3 + 3 * th ** 2; h11 = th ** 3 - th ** 2
            # renormalize tangents at every crossing
            nv = 0.
            for i in range(2 * M, 4 * M):
                nv += s[i] ** 2
            nv = np.sqrt(nv)
            for i in range(2 * M, 4 * M):
                s[i] /= nv
            for a in range(M):
                nu = np.sqrt(s[4 * M + 2 * a] ** 2 + s[4 * M + 2 * a + 1] ** 2)
                s[4 * M + 2 * a] /= nu; s[4 * M + 2 * a + 1] /= nu
                if t_start >= 0:
                    isum[a] += np.log(nu)
            if t_start >= 0:
                lsum += np.log(nv)
            if t >= t_trans:
                if t_start < 0:
                    t_start = t - dt + th * dt
                    lsum = 0.; isum[:] = 0.
                else:
                    for i in range(n):
                        sec[nr, i] = h00 * old[i] + h10 * dt * fa[i] + h01 * s[i] + h11 * dt * fb[i]
                    tsec[nr] = t - dt + th * dt
                    nr += 1
    span = tsec[nr - 1] - t_start if nr > 0 else 1.
    return sec[:nr], tsec[:nr], lsum / span, isum / span, s[:n].copy()


def distinct(sec, tol=1e-5, cap=64):
    """Smallest j <= cap with |z_last - z_(last-j)| < tol in all coordinates (the period in section returns), else 0.
    (Changed 2026-11-04 before the production runs: counting distinct points with tolerance 1e-6 failed on slowly
    converging period-two orbits.)"""
    for j in range(1, min(cap, len(sec) - 1) + 1):
        if np.max(abs(sec[-1] - sec[-1 - j])) < tol and np.max(abs(sec[-2] - sec[-2 - j])) < tol:
            return j
    return 0


def main():
    tag, direction = sys.argv[1], int(sys.argv[2])
    r = json.loads((OUT / f"{tag}.json").read_text())
    mr.B, mr.G = r["beta"], r["gamma"]
    sz = r.get("start_order_sizes", r["sizes"])
    p = np.array(sz, float) / sum(sz)
    n = 2 * len(p)
    d = [x for x in r["directions"] if x["direction"] == direction][0]
    w, kpd, kk1 = np.array(d["last_good_orbit"]), d["K_last_good"], r["K1"]
    fs = np.linspace(-0.005, 0.15, NK)
    ks = kpd + direction * fs * abs(kk1)
    t0 = time.time()
    # accuracy check: RK4 section point against the DOP853 orbit at the first K (stable side)
    w0 = mr.shoot(w, ks[0], p)
    sec, _, _, _, _ = run(w0[:n] + 1e-12, ks[0], p, mr.E, mr.B, mr.G, DT, 500., 5)
    rk_err = float(np.max(abs(sec[-1] - w0[:n])))
    out = dict(tag=tag, direction=direction, K_PD=kpd, K1=kk1, sizes=sz, dt=DT, transient=TRANS, nrec=NREC,
               rk4_vs_dop853_section=rk_err, host=socket.gethostname(), sweeps={})
    arrays = dict(K=ks, f=fs)
    z = w[:n] + 1e-6
    for name, order in (("forward", range(NK)), ("backward", range(NK - 1, -1, -1))):
        lyap, intl, ndist, secs, rets = np.zeros(NK), np.zeros((NK, len(p))), np.zeros(NK, int), [], {}
        S = np.full((NK, NREC, n), np.nan)
        for i in order:
            sec, ts, le, il, z = run(z, ks[i], p, mr.E, mr.B, mr.G, DT, TRANS, NREC)
            lyap[i], intl[i], ndist[i] = le, il, distinct(sec)
            S[i, :len(sec)] = sec
            rets[i] = np.diff(ts)[-8:].tolist()
        arrays[f"{name}_section"] = S
        arrays[f"{name}_lyap"] = lyap
        arrays[f"{name}_internal"] = intl
        arrays[f"{name}_ndistinct"] = ndist
        out["sweeps"][name] = dict(max_lyap=float(lyap.max()), K_max_lyap=float(ks[int(np.argmax(lyap))]),
                                   n_lyap_above_1e3=int((lyap > 1e-3).sum()),
                                   ndistinct_values=sorted(set(ndist.tolist())),
                                   max_internal_exponent=float(intl.max()))
        print(tag, direction, name, out["sweeps"][name], "%.0fs" % (time.time() - t0), flush=True)
    out["seconds"] = time.time() - t0
    np.savez_compressed(OUT / f"pd_route_{tag}_d{direction:+d}.npz", **arrays)
    (OUT / f"pd_route_{tag}_d{direction:+d}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
