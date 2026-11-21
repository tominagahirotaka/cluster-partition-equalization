#!/usr/bin/env python3
"""How does the period-1 orbit of a balanced partition end on the strong-coupling side? 2026-11-08 (author's approval
"３つとも進めてください"); for (0.3, 1.12, 0.5) 34:33:33 and (0.2, 1.28, 0.45) 39:36:25, where shooting stops just past
the end found by scripts/balanced_strong_end.py.

The reduced orbit is carried to K_last_good of _strong_end.json (continue_branch) and followed further towards stronger
coupling by pseudo-arclength continuation in (z, T, K) (fold_saddle_test.arclength, sign -1; up to 0.03 |K1| in K or
300 points). Along the path: K, period, the two largest collective multipliers and the internal moduli. A turning
point of K (K decreasing, then increasing) is a fold.
  -> analysis/universality/hysteresis_ext/<name minus .json>_arclength.json
  python3 scripts/end_arclength.py e0.3_b1.12_g0.5_40-30-30.json
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402
from balanced_strong_end import orbit_from_state  # noqa: E402
from fold_saddle_test import arclength  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "analysis/universality/hysteresis_ext"


def main():
    name = sys.argv[1]
    e = json.loads((EXT / name).read_text())
    se = json.loads((EXT / (name[:-5] + "_strong_end.json")).read_text())
    d = np.load(EXT / (name + ".npz"))
    mr.E, mr.B, mr.G = e["eps"], e["beta"], e["gamma"]
    kk1, N = e["K1"], int(sum(e["sizes"]))
    step = se.get("step", len(d["K"]))
    sz, p, w = orbit_from_state(d["Z"][step - 1], float(d["K"][step - 1]), N)
    dk = .005 * abs(kk1)
    kg = se["K_last_good"]
    (w, kc), _, _ = mr.continue_branch(w, float(d["K"][step - 1]), p, dk, kg, -1)
    path = arclength(w, kg, p, kk1, sign=-1)
    ks = np.array([q["K"] for q in path])
    turns = [i for i in range(1, len(ks) - 1) if (ks[i] - ks[i - 1]) * (ks[i + 1] - ks[i]) < 0]
    out = dict(name=name, sizes=sz.tolist(), K1=kk1, K_last_good=kg, f_last_good=kg / kk1,
               turning_points=[dict(i=i, K=float(ks[i]), f=float(ks[i] / kk1), period=path[i]["period"],
                                    top=path[i]["top"], internal_moduli=path[i]["internal_moduli"]) for i in turns],
               path=[{k: v for k, v in q.items() if k != "u"} | dict(f=q["K"] / kk1) for q in path])
    (EXT / (name[:-5] + "_arclength.json")).write_text(json.dumps(out, indent=1))
    print(name, sz.tolist(), "end f %.5f" % (kg / kk1), "points", len(path), "f range %.5f-%.5f" % (ks.min() / kk1, ks.max() / kk1),
          "turns", [("%.5f" % (t["f"]), [round(abs(complex(*c)), 4) for c in t["top"]],
                     [round(v, 3) for v in t["internal_moduli"]], round(t["period"], 3)) for t in out["turning_points"]],
          flush=True)
    for q in path[::max(1, len(path) // 25)]:
        print("   f %.5f T %.3f top %s int %s" % (q["K"] / kk1, q["period"],
              [(round(c[0], 4), round(c[1], 4)) for c in q["top"]], [round(v, 3) for v in q["internal_moduli"]]),
              flush=True)


if __name__ == "__main__":
    main()
