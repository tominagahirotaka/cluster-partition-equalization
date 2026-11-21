#!/usr/bin/env python3
"""Launch the universality map (scripts/universality_map.py) (2026-11-05; public version, local runs only).

Grid: eps = 0.3 first, then 0.2, 0.25, 0.35, 0.4; beta = 0.60..1.40 step 0.04; gamma = 0.10..0.95 step 0.05;
partition 34:33:33. Non-excitable points are skipped here. The tasks of each eps are shuffled (seed 1) and dealt to
the hosts in proportion to their slots. Each host runs its list with xargs -P <slots>, every task under timeout
4h, detached (setsid nohup, timeout 72h). The original computation used several computers of the author; this
public version lists only the local computer (edit HOSTS to add others reachable by ssh).

  python3 scripts/universality_map_launch.py            # write the task lists and launch
  python3 scripts/universality_map_launch.py --collect  # copy the remote outputs here

Other partitions (2026-11-06, step 3 of the follow-up, author's approval "どれも，やってほしい"):
  python3 scripts/universality_map_launch.py --sizes 40,30,30 --sizes 50,25,25
runs the points where 34:33:33 found a start orbit (status ok), with the given partitions, in the same way; the
task and runner files get the suffix _parts.
"""
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "analysis/universality/map"
HOSTS = [("local", 8, "1500M")]  # (host, parallel tasks, memory per task); the original run used five computers
PY = "python"
FILES = ["mcluster_reduced.py", "pd_route_scan.py", "universality_branches.py", "pd_cascade_shoot.py",
         "pd_small_p2_amplitude.py", "universality_map.py"]


def tasks():
    out = []
    rng = np.random.default_rng(1)
    for eps in (0.3, 0.2, 0.25, 0.35, 0.4):
        t = []
        for b in np.round(np.arange(0.60, 1.4001, 0.04), 2):
            for g in np.round(np.arange(0.10, 0.9501, 0.05), 2):
                mr.E, mr.B, mr.G = eps, b, g
                if mr.excitable():
                    t.append(f"--eps {eps:g} --beta {b:g} --gamma {g:g}")
        rng.shuffle(t)
        out.append(t)
    return out


def tasks_parts(sizes_list):
    import glob
    import json
    ok = []
    for f in sorted(glob.glob(str(MAP / "e*_34-33-33.json"))):
        r = json.loads(Path(f).read_text())
        if r.get("status") == "ok":
            ok.append((r["eps"], r["beta"], r["gamma"]))
    out = []
    rng = np.random.default_rng(2)
    for eps in (0.3, 0.2, 0.25, 0.35, 0.4):
        t = [f"--eps {e:g} --beta {b:g} --gamma {g:g} --sizes {sz}" for sz in sizes_list for e, b, g in ok if e == eps]
        rng.shuffle(t)
        out.append(t)
    return out


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True)


def main():
    MAP.mkdir(parents=True, exist_ok=True)
    if "--collect" in sys.argv:
        for h, _, _ in HOSTS:
            if h != "local":
                sh(f"rsync -a {h}:ExcitableOscillation_remote/analysis/universality/map/ {MAP}/")
        return
    hosts, suffix, task_lists = HOSTS, "", None
    sizes_list = [a for i, a in enumerate(sys.argv[1:]) if sys.argv[i] == "--sizes"]
    if sizes_list:
        hosts = HOSTS
        suffix, task_lists = "_parts", tasks_parts(sizes_list)
    lists = {h: [] for h, _, _ in hosts}
    tot = sum(s for _, s, _ in hosts)
    for t in (task_lists if task_lists is not None else tasks()):
        credit = {h: 0. for h, _, _ in hosts}
        for task in t:
            for h, s, _ in hosts:
                credit[h] += s / tot
            h = max(credit, key=credit.get)
            credit[h] -= 1
            lists[h].append(task)
    for h, s, mem in hosts:
        name = h.split('.')[0] + suffix
        fn = MAP / f"tasks_{name}.txt"
        fn.write_text("\n".join(lists[h]) + "\n")
        runner = MAP / f"run_{name}.sh"  # a file, to avoid nested quoting over ssh
        runner.write_text(f"""#!/bin/sh
cd "$(dirname "$0")/../../.."
xargs -P {s} -L 1 sh -c 'timeout 4h {PY} scripts/universality_map.py "$@" \\
    >> analysis/universality/map/run.log 2>&1' _ < analysis/universality/map/{fn.name}
""")
        start = f"setsid nohup timeout 72h sh analysis/universality/map/{runner.name} > /dev/null 2>&1 < /dev/null &"
        if h == "local":
            sh(f"cd {ROOT} && {start}")
        else:
            sh(f"ssh -o BatchMode=yes {h} 'mkdir -p ~/ExcitableOscillation_remote/scripts "
               f"~/ExcitableOscillation_remote/analysis/universality/map'")
            sh(f"scp -q {' '.join(str(ROOT / 'scripts' / f) for f in FILES)} {h}:ExcitableOscillation_remote/scripts/")
            sh(f"scp -q {fn} {runner} {h}:ExcitableOscillation_remote/analysis/universality/map/")
            sh(f"timeout 30 ssh -o BatchMode=yes {h} 'cd ~/ExcitableOscillation_remote && {start}' < /dev/null")
        print(h, len(lists[h]), "tasks", flush=True)


if __name__ == "__main__":
    main()
