#!/usr/bin/env python3
"""Collect the canonical firing-order sweeps of paper 4 (2026-11-18). No integration.

scripts/rebalance_order.py repeats the sweeps of rebalance/ and rebalance_var/ and records the firing order. Noisy
sweeps are bit-for-bit reproducible only on the same CPU code path (docs/RESULTS_NOTES.md, 2026-11-18), so the
canonical version of each sweep is the one run on the host of the original sweep and identical to it at every step
(rest, sizes, settled). This script looks for such a version among the local outputs and the fetched remote copies,
copies it (never overwriting) to analysis/universality/rebalance_order_canonical/ and writes a manifest with the
status of every sweep (canonical / pending / no matching version).

  python3 scripts/collect_order_canonical.py <fetch_dir> [<fetch_dir> ...]
  (each <fetch_dir> holds <host>/analysis/universality/rebalance_order/*.json)
"""
from pathlib import Path
import glob
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
U = ROOT / "analysis/universality"
OUT = U / "rebalance_order_canonical"


def original(d, base):
    if d["seed"] == 0 and d["T"] == 10000 and d["D"] == .005:
        return U / "rebalance" / d["name"]
    return U / "rebalance_var" / base


def same_steps(s, e):
    return len(s) == len(e) and all(a["rest"] == c["rest"] and a.get("sizes") == c.get("sizes")
                                    and a.get("settled") == c.get("settled") for a, c in zip(s, e))


def main():
    files = glob.glob(str(U / "rebalance_order/*.json"))
    for fd in sys.argv[1:]:
        files += glob.glob(str(Path(fd) / "*/analysis/universality/rebalance_order/*.json"))
    OUT.mkdir(exist_ok=True)
    runs = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        runs.setdefault(Path(f).name, []).append((f, d))
    manifest = {}
    for base, cand in sorted(runs.items()):
        o = original(cand[0][1], base)
        if not o.exists():
            manifest[base] = dict(status="no original sweep")
            continue
        E = json.loads(o.read_text())
        good = [(f, d) for f, d in cand if d["host"] == E.get("host") and d.get("stop") is not None
                and same_steps(d["steps"], E["steps"])]
        if good:
            f, d = good[0]
            dst = OUT / base
            if dst.exists():
                if json.loads(dst.read_text())["steps"] != d["steps"]:
                    raise SystemExit("canonical copy differs from a new candidate: %s" % base)
            else:
                shutil.copy2(f, dst)
            manifest[base] = dict(status="canonical", host=d["host"], source=str(f), steps=len(d["steps"]))
        else:
            on_host = [d for _, d in cand if d["host"] == E.get("host")]
            running = [d for d in on_host if d.get("stop") is None]
            manifest[base] = dict(status="pending" if running else "no matching version", original_host=E.get("host"),
                                  candidates=[(d["host"], len(d["steps"]), d.get("stop") is not None) for _, d in cand])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    from collections import Counter
    print(Counter(v["status"] for v in manifest.values()))
    for k, v in manifest.items():
        if v["status"] != "canonical":
            print(k, v)


if __name__ == "__main__":
    main()
