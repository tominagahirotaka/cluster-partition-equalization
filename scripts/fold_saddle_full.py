#!/usr/bin/env python3
"""Follow-up of (b), 2026-11-07 (author's approval "進めてください"): the two-sided test of fold_saddle_test.py in the
full system with N = 100 (34:33:33), as in paper 1's full-system check.

From analysis/universality/fold_saddle/<name>: at the coexistence point with a quarter of the middle-segment width
below the fold, the stable and saddle orbits are found again (the joined path of fold_saddle_test.py is recomputed,
since its json keeps no orbits; nearest path points on each side of the fold, shot at that K), and the saddle's
unstable Floquet vector v_u (real multiplier > 1) computed. Start: saddle + sign * delta * v_u (reduced, delta = 1e-4 and 1e-6) lifted to the elements,
plus Gaussian noise of sd 1e-8 on all 200 coordinates (default_rng(1)); also the same without the reduced
perturbation (noise only). DOP853 (rtol 1e-10, atol 1e-12, max step 0.15) to t = 20000. Outcome at t = 10000 and
20000: rest, or the cluster sizes and whether the state is the stable three-cluster orbit (the lifted stable orbit's
section point within 1e-4 of a section point of group means near the end).
  -> analysis/universality/fold_saddle_full/<name>
  python3 scripts/fold_saddle_full.py e0.2_b0.6_g0.6_34-33-33.json
"""
from pathlib import Path
import json
import socket
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from pd_cascade_shoot import monodromy  # noqa: E402
import fold_saddle_test as ft  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "analysis/universality/fold_saddle"
OUT = ROOT / "analysis/universality/fold_saddle_full"


def clusters(z, N):
    x, y = z[:N], z[N:]
    dist = np.maximum(abs(x[:, None] - x[None, :]), abs(y[:, None] - y[None, :]))
    nc, labs = connected_components(dist < 1e-4, directed=False)
    return sorted(np.bincount(labs).tolist(), reverse=True), labs


def main():
    name = sys.argv[1]
    r = json.loads((FS / name).read_text())
    mr.E, mr.B, mr.G = r["eps"], r["beta"], r["gamma"]
    sizes = np.array([34, 33, 33])
    p = sizes / 100
    m, n, N = 3, 6, 100
    row = [q for q in r["coexistence"] if q["frac"] == .25][0]
    k = row["K"]
    # the json keeps no orbits: redo the joined path as fold_saddle_test.py does and take the nearest points on the
    # stable side (before the fold) and on the middle segment (after it)
    mp = json.loads((ROOT / "analysis/universality/map" / name).read_text())
    w0, k0 = np.array(mp["end"]["last_good_orbit"]), mp["end"]["K_last_good"]
    back, fwd = ft.arclength(w0, k0, p, mp["K1"], -1), ft.arclength(w0, k0, p, mp["K1"], 1)
    if abs(complex(*back[-1]["top"][0])) > abs(complex(*fwd[-1]["top"][0])):
        back, fwd = fwd, back
    path = back[::-1] + [dict(K=k0, u=np.r_[w0, k0].tolist())] + fwd
    ks = np.array([q["K"] for q in path])
    i = int(np.argmax(ks))
    js, ju = int(np.argmin(abs(ks[:i] - k))), i + 1 + int(np.argmin(abs(ks[i + 1:] - k)))
    ws, wu = mr.shoot(np.array(path[js]["u"][:n + 1]), k, p), mr.shoot(np.array(path[ju]["u"][:n + 1]), k, p)
    out = dict(name=name, eps=r["eps"], beta=r["beta"], gamma=r["gamma"], K=k, host=socket.gethostname(), runs=[])
    ev, V = np.linalg.eig(monodromy(wu, k, p))
    j = int(np.argmax(np.where(abs(ev.imag) < 1e-9, ev.real, -np.inf)))
    v = np.real(V[:, j]); v[0] = 0.; v /= np.linalg.norm(v)
    out["mu_u"] = float(ev[j].real)
    au = mr.audit(wu, k, p)
    out["saddle_internal_moduli"] = au["internal_moduli"]
    lab = np.repeat(np.arange(m), sizes)
    xr, yr = mr.rest_point()

    def rhs(t, u):
        x, y = u[:N], u[N:]
        return np.concatenate(((x - x ** 3 / 3 - y) / mr.E + k * (x.mean() - x), mr.E * (x + mr.B - mr.G * y)))
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for sign, delta in ((1, 1e-4), (-1, 1e-4), (1, 1e-6), (-1, 1e-6), (0, 0.)):
        zr = wu[:n] + sign * delta * v
        z0 = np.r_[zr[:m][lab], zr[m:][lab]] + 1e-8 * np.random.default_rng(1).normal(size=2 * N)
        s = solve_ivp(rhs, (0, 20000), z0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.15,
                      t_eval=np.r_[np.arange(0, 20000.1, 1.)])
        res = dict(sign=sign, delta=delta)
        for tc in (10000, 20000):
            z = s.y[:, tc]
            rest = bool(np.max(abs(z[:N] - xr)) < 1e-8 and np.max(abs(z[N:] - yr)) < 1e-8)
            res[f"t{tc}"] = dict(rest=rest, sizes=None if rest else clusters(z, N)[0])
        # distance of the group means from the saddle and from the stable orbit over time (reduced coordinates)
        g = np.array([s.y[:N][lab == a].mean(axis=0) for a in range(m)] + [s.y[N:][lab == a].mean(axis=0) for a in range(m)])
        sol_s = solve_ivp(lambda t, u: mr.rhs(u, k, p), (0, ws[-1]), ws[:n], method="DOP853", rtol=1e-11, atol=1e-13,
                          dense_output=True)
        last = g[:, -int(ws[-1]) - 2:]
        res["dist_to_stable_orbit_end"] = float(min(np.max(abs(sol_s.sol(t)[:, None] - last), axis=0).min()
                                                for t in np.linspace(0, ws[-1], 400)))
        res["max_internal_spread"] = float(max(np.ptp(s.y[:N][lab == a][:, -1]) for a in range(m)))
        out["runs"].append(res)
        out["seconds"] = time.time() - t0
        (OUT / name).write_text(json.dumps(out, indent=1))
    print(name, "mu_u %.3f" % out["mu_u"], "sad_int %.2f" % max(out["saddle_internal_moduli"]),
          [(q["sign"], q["delta"], "rest" if q["t20000"]["rest"] else q["t20000"]["sizes"],
            "%.1e" % q["dist_to_stable_orbit_end"]) for q in out["runs"]], flush=True)


if __name__ == "__main__":
    main()
