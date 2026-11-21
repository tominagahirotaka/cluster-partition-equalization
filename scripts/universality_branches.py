#!/usr/bin/env python3
"""Does the scenario of paper 1 (collective period doubling of a three-cluster branch, subcritical, towards weak
coupling) occur at other (beta, gamma)? (2026-11-03, author's approval "案1と案2の両方を進めてください"; eps = 0.3,
reduced system only, scripts/mcluster_reduced.py.)

Plan 1 (--plan 1): partition 34:33:33 on a path in (beta, gamma) through paper 1's point (0.7, 0.5): start from random
  centroids (seeds 1-5, transient 3000) at K = fac K1 (fac = 3, else 1.5, else 6), keep the first orbit with three
  distinct clusters, continue towards K = 0 with step |K1|/200, halving at failure or instability.
Plan 2 (--plan 2): other partitions at the other excitable points; the start is a run of the ensemble
  analysis/eps_cluster_ensemble/runs_bg_b{beta}_g{gamma}.json with the given sorted sizes and K (its final centroids, in
  the stored order), transient 2000 in the reduced system; continue towards weak coupling (K -> 0) and towards strong
  coupling (down to 2 K0).
With --ignore-internal (added 2026-11-03 after the first plan 1 results) the reduced orbit is followed past internal
instabilities, to see whether a collective period doubling would follow (output tag _coll).
At each end: the kind (PD, fold, torus, internal, shoot_failed, clusters_merge, reached_stop); for PD the PD side test
(K_PD + 2e-3 and 5e-3 |K1| past it, +-1e-6 along the -1 eigenvector, t = 20000).
  -> analysis/universality/plan{1,2}_b{beta}_g{gamma}_{sizes}.json

  python3 scripts/universality_branches.py --plan 1 --beta 0.74 --gamma 0.5
  python3 scripts/universality_branches.py --plan 2 --beta 0.8 --gamma 0.5 --sizes 47,27,26 --K0 -1.556
"""
from pathlib import Path
import argparse
import json
import socket
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/universality"


def periodic_start(z, k, p, tmax):
    m = len(p)
    s = solve_ivp(lambda t, u: mr.rhs(u, k, p), (0, tmax), z, method="DOP853", rtol=1e-10, atol=1e-12, max_step=.05,
                  events=mr.sec)
    if len(s.t_events[0]) < 3:
        return None, "no oscillation"
    w = mr.shoot(np.r_[s.y_events[0][-1], np.diff(s.t_events[0])[-1]], k, p)
    if w is None:
        return None, "shooting failed (not periodic?)"
    a = mr.audit(w, k, p)
    if a["min_cluster_gap"] < 1e-3:
        return None, "clusters merged"
    if mr.kind_of(a) is not None:
        return None, f"unstable start ({mr.kind_of(a)})"
    return w, "ok"


def summarize(rows):
    keep = rows[::max(1, len(rows) // 40)] + rows[-15:]
    return [dict(K=r["K"], period=r["period"], collective=r["collective"][:3], internal_moduli=r["internal_moduli"],
                 firing_offsets=r["firing_offsets"], min_cluster_gap=r["min_cluster_gap"]) for r in keep]


def run_direction(w, k, p, kk1, direction, k_stop, ignore_internal=False):
    good, rows, last_bad = mr.continue_branch(w, k, p, abs(kk1) / 200, k_stop, direction, ignore_internal)
    kind = mr.end_kind(last_bad, ignore_internal)
    out = dict(direction=direction, K_stop=k_stop, npts=len(rows), end_kind=kind, K_last_good=float(good[1]),
               K_last_bad=None if last_bad is None else float(last_bad[0]),
               last_bad_audit=None if last_bad is None else last_bad[1], last_good_orbit=good[0].tolist(),
               scan=summarize(rows))
    if kind == "PD":
        out["pd_side"] = mr.pd_side_test(good[0], good[1], p, [2e-3 * abs(kk1), 5e-3 * abs(kk1)], direction)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=int, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--sizes", default="34,33,33")
    ap.add_argument("--K0", type=float)
    ap.add_argument("--ignore-internal", action="store_true",
                    help="follow the reduced orbit past internal instabilities (collective ends only); tag _coll")
    a = ap.parse_args()
    mr.B, mr.G = a.beta, a.gamma
    sizes = [int(v) for v in a.sizes.split(",")]
    p = np.array(sizes, float) / sum(sizes)
    tag = f"plan{a.plan}_b{a.beta:g}_g{a.gamma:g}_{'-'.join(map(str, sizes))}" + ("_coll" if a.ignore_internal else "")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    row = dict(plan=a.plan, beta=a.beta, gamma=a.gamma, eps=mr.E, sizes=sizes, host=socket.gethostname(),
               excitable=mr.excitable(), ignore_internal=a.ignore_internal)
    if not row["excitable"]:
        row["status"] = "not excitable"
        (OUT / f"{tag}.json").write_text(json.dumps(row, indent=1))
        print(tag, row["status"], flush=True)
        return
    kk1 = mr.k1()
    row["K1"] = kk1
    w, why = None, "no ensemble run with these sizes"
    if a.plan == 1:
        for fac in (3., 1.5, 6.):
            k0 = fac * kk1
            for seed in range(1, 6):
                rng = np.random.default_rng(seed)
                z = np.r_[rng.uniform(-2, 2, 3), rng.uniform(-.6, 1.4, 3)]
                w, why = periodic_start(z, k0, p, 3000)
                if w is not None:
                    row.update(start_fac=fac, start_seed=seed)
                    break
            if w is not None:
                break
        dirs = [(1, -1e-6)]
    else:
        runs = json.loads((ROOT / f"analysis/eps_cluster_ensemble/runs_bg_b{a.beta:.1f}_g{a.gamma:.1f}.json").read_text())
        cand = [r for r in runs if abs(r["K"] - a.K0) < 1e-3 and sorted(r["sizes"], reverse=True) == sizes]
        for r in cand:
            c = np.array(r["centroids"], float)
            order = [list(c[:, 2].astype(int)).index(s) for s in sizes] if len(set(sizes)) == len(sizes) else None
            if order is None:  # equal sizes: keep the stored order, sizes taken from the centroids
                order = list(range(len(c)))
            c = c[order]
            pp = c[:, 2] / c[:, 2].sum()
            k0 = r["K"]
            w, why = periodic_start(np.r_[c[:, 0], c[:, 1]], k0, pp, 2000)
            if w is not None:
                p = pp
                row.update(start_K=k0, start_seed=r["seed"], start_order_sizes=c[:, 2].astype(int).tolist())
                break
        dirs = [(1, -1e-6), (-1, 2 * a.K0)]
    if w is None:
        row["status"] = f"no start orbit ({why})"
        (OUT / f"{tag}.json").write_text(json.dumps(row, indent=1))
        print(tag, row["status"], flush=True)
        return
    st = mr.audit(w, k0, p)
    row.update(status="ok", start_K=k0, start_audit=st)
    row["directions"] = []
    for d, ks in dirs:
        res = run_direction(w, k0, p, kk1, d, ks, a.ignore_internal)
        row["directions"].append(res)
        (OUT / f"{tag}.json").write_text(json.dumps(row, indent=1))
        print(tag, "dir", d, res["end_kind"], "K_last_good %.8f" % res["K_last_good"],
              "pd_side" if "pd_side" in res else "", flush=True)
    row["seconds"] = time.time() - t0
    (OUT / f"{tag}.json").write_text(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
