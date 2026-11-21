#!/usr/bin/env python3
"""Aggregates of paper 4, regenerated from the stored outputs only (no integration), 2026-11-17 (author's instruction
"待つ間に集計スクリプトを進めてください"; item 2 of paper4/skeleton.md section 5, list in paper4/evidence_inventory.md).

Each block recomputes values quoted in paper4/claims_scope.md and stores them next to the quoted ("claimed") value.
Differences are reported on stdout and noted in claims_scope.md by hand (original values are not overwritten).
The excitability test of the grid points uses mcluster_reduced.excitable() (no integration).
  -> analysis/paper4_figs/aggregates.json
  python3 scripts/paper4_aggregates.py
"""
from collections import Counter, defaultdict
from pathlib import Path
import glob
import json
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcluster_reduced as mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
U = ROOT / "analysis/universality"
OUT = ROOT / "analysis/paper4_figs/aggregates.json"
PARTS = ("34-33-33", "40-30-30", "50-25-25")
E = (0.2, 0.25, 0.3, 0.35, 0.4)
B = np.round(np.arange(.6, 1.4001, .04), 2)
G = np.round(np.arange(.1, .9501, .05), 2)


def load(pattern):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(U / pattern)))]


def key(r):
    return (round(r["eps"], 2), round(r["beta"], 2), round(r["gamma"], 2))


def med(x):
    return float(np.median(x)) if len(x) else None


# ---------------------------------------------------------------- claim A
def claim_a():
    grid = []
    for e in E:
        for b in B:
            for g in G:
                mr.E, mr.B, mr.G = e, b, g
                if mr.excitable():
                    grid.append((e, float(b), float(g)))
    data = {s: {key(r): r for r in load("map/*_%s.json" % s)} for s in PARTS}
    m = data["34-33-33"]
    a = dict(A1=dict(claimed="1655 excitable points, 34:33:33 start orbit found at about 1000 points",
                     excitable_grid_points=len(grid),
                     files={s: len(data[s]) for s in PARTS},
                     grid_points_without_file_34=[list(k) for k in grid if k not in m],
                     files_not_finished={s: sorted("%g_%g_%g" % k for k, r in data[s].items() if "seconds" not in r)
                                         for s in PARTS},
                     status_34=dict(Counter(r["status"] for r in m.values()))))
    ok = {k: r for k, r in m.items() if r["status"] == "ok"}
    ek = Counter(r["end"]["kind"] for r in ok.values())
    pd = [list(k) + [r.get("paper1_scenario")] for k, r in ok.items() if r["end"]["kind"] == "PD"]
    a["A2"] = dict(claimed="1005 points; internal first about 79%, fold 207, collective PD 1 at (0.3, 0.68, 0.55)",
                   ok=len(ok), end_kind=dict(ek), internal_fraction=ek["internal"] / len(ok), pd_points=pd)
    # A3: fraction internal first per eps and partition; common points and medians of f at the first end
    frac = {}
    for s in PARTS:
        frac[s] = {}
        for e in E:
            rs = [r for k, r in data[s].items() if k[0] == e and r["status"] == "ok" and "end" in r]
            frac[s][str(e)] = (sum(r["end"]["kind"] == "internal" for r in rs) / len(rs)) if rs else None
    common = [k for k in ok if all(data[s].get(k, {}).get("status") == "ok" and "end" in data[s][k] for s in PARTS)]
    a["A3"] = dict(claimed="internal first 61-90%, 91-97%, 96-99% by eps; 832 common points; median f 0.46, 0.54, 0.67",
                   internal_fraction_by_eps=frac,
                   ok_files={s: sum(r["status"] == "ok" for r in data[s].values()) for s in PARTS},
                   common_points=len(common),
                   median_f_first_end={s: med([data[s][k]["end"]["f_last_good"] for k in common]) for s in PARTS})
    # A4: which cluster reaches 1 first at the internal ends of 34:33:33; full-system check at 26 points
    cl = Counter()
    for r in ok.values():
        if r["end"]["kind"] == "internal":
            im = r["end"]["last_bad_audit"]["internal_moduli"]
            cl[int(np.argmax(im))] += 1
    full = load("internal_end_full/*.json")
    fk = Counter()
    for r in full:
        mu = [z[0] for z in r["internal_multipliers"]]
        sign = "+1" if max(mu) > .5 else "-1" if min(mu) < -.5 else "?"
        fk[(sign, "largest" if r["critical_size"] == max(r["sizes"]) else "not largest")] += 1
    a["A4"] = dict(claimed="largest cluster at 667 of 797 internal ends (34:33:33); full system 26 points: 21 at +1, "
                           "4 at -1 (33-cluster, eps 0.35-0.4)",
                   internal_ends=sum(cl.values()), crossing_cluster_index=dict(cl), crossing_34=cl[0],
                   full_points=len(full), full_kinds={"%s %s" % k: v for k, v in fk.items()})
    # A5: collective end past the internal instability
    a5 = {}
    for s in PARTS:
        rs = [r for r in data[s].values() if r["status"] == "ok" and "end" in r and r["end"]["kind"] == "internal"
              and "collective_end" in r]
        c = Counter((r["collective_end"]["kind"], r["collective_end"].get("pd_type", {}).get("type")) for r in rs)
        gap = [abs(r["end"]["K_last_good"] - r["collective_end"]["K_last_good"]) / abs(r["K1"]) for r in rs
               if r["collective_end"]["kind"] == "PD"]
        a5[s] = dict(n=len(rs), kinds={"%s %s" % k: v for k, v in c.items()}, median_gap_pd_over_K1=med(gap))
    a["A5"] = dict(claimed="sub:super = 56:3, 16:18, 20:20; undetermined 97, 185, 189; median gap 0.020, 0.039, 0.098 "
                           "|K1|", by_partition=a5)
    # A6: fold test at first ends with shooting failure
    me = load("map_ends/*.json")
    lab = Counter((r["name"].split("_")[-1][:-5], r["which"], r["label"]) for r in me)
    a["A6"] = dict(claimed="189 of 195 first ends without internal instability are fold-like",
                   shoot_failed_first_ends_34=ek["shoot_failed"],
                   labels={"%s %s %s" % k: v for k, v in sorted(lab.items())})
    # values quoted in Sec. III of the draft (2026-11-17)
    s3 = {}
    for s in PARTS:
        rs = [r for r in data[s].values() if r["status"] == "ok"]
        s3[s] = dict(
            status=dict(Counter(r["status"] for r in data[s].values())),
            start_factor=dict(Counter(str(r.get("start_fac")) for r in rs)),
            end_kind=dict(Counter(r["end"]["kind"] for r in rs)),
            f_below_1={k: "%d/%d" % (sum(r["end"]["f_last_good"] < 1 for r in rs if r["end"]["kind"] == k),
                                     sum(r["end"]["kind"] == k for r in rs)) for k in ("internal", "shoot_failed")},
            crossing_size=dict(Counter(str(r["sizes"][int(np.argmax(r["end"]["last_bad_audit"]["internal_moduli"]))])
                                       for r in rs if r["end"]["kind"] == "internal")))
    no = [r for r in m.values() if r["status"] == "no start orbit (no oscillation)"]
    s3["no_oscillation_medians_beta_gamma"] = [med([r["beta"] for r in no]), med([r["gamma"] for r in no])]
    s3["start_orbit_medians_beta_gamma"] = [med([r["beta"] for r in ok.values()]), med([r["gamma"] for r in ok.values()])]
    s3["pd_point_neighbours"] = {"%g_%g" % (b_, g_): (m[(0.3, b_, g_)]["end"]["kind"] if m.get((0.3, b_, g_), {})
                                                    .get("status") == "ok" else m.get((0.3, b_, g_), {}).get("status"))
                                 for b_, g_ in ((0.64, 0.55), (0.72, 0.55), (0.68, 0.5), (0.68, 0.6))}
    out = Counter()
    for r in full:
        init = sorted(r["sizes"], reverse=True)
        for q in r["runs"]:
            last = q.get("t20000")
            o = ("no reduced orbit" if last is None else "rest" if last.get("rest") else
                 "same" if last.get("sizes") == init else "changed")
            out["d=%g %s" % (q["d"], o)] += 1
    s3["internal_end_full_t20000"] = dict(out)
    a["A_sec3"] = s3
    return a


# ---------------------------------------------------------------- claim B
def sweep_summary(r):
    S = [s for s in r["steps"] if "sizes" in s or s.get("rest")]
    first_change = next((s["f"] for s in S if s.get("sizes") != r["sizes"] or s.get("rest")), None)
    return dict(stop=r.get("stop"), f_end=S[-1]["f"] if S else None, f_first_change=first_change,
                min_largest_3=min((max(s["sizes"]) for s in S if s.get("sizes") and len(s["sizes"]) == 3),
                                  default=None))


def claim_b():
    b = {}
    reb = {r["name"][:-5]: sweep_summary(r) for r in load("rebalance/*.json")}
    b["B1_rebalance"] = dict(claimed="11 of 12 end in synchronized rest; (0.35, 0.84, 0.55) stays at 36:33:31 for 300 "
                                     "steps", points=len(reb), stop=dict(Counter(v["stop"] for v in reb.values())),
                             not_rest=[k for k, v in reb.items() if v["stop"] != "rest"], detail=reb)
    var = defaultdict(list)
    pat = re.compile(r"^(.*_\d+-\d+-\d+)_s(\d+)_N(\d+)(?:_T(\d+))?(?:_D([\d.]+))?\.json$")
    for f in sorted(glob.glob(str(U / "rebalance_var/*.json"))):
        mt = pat.match(Path(f).name)
        if not mt:
            continue
        r = json.loads(Path(f).read_text())
        tag = "T%s" % mt[4] if mt[4] else "D%s" % mt[5] if mt[5] else "N%s" % mt[3]
        var[tag].append((mt[1], int(mt[2]), sweep_summary(r)))
    b["B1_variants"] = dict(
        claimed="12 points x 10 seeds, N = 200, 400; T = 5000, 20000 (72 runs); D = 0.0025, 0.01 (72 runs); 11 points "
                "rest in all, (0.35, 0.84, 0.55) never",
        runs={t: len(v) for t, v in sorted(var.items())},
        rest={t: "%d/%d" % (sum(x[2]["stop"] == "rest" for x in v), len(v)) for t, v in sorted(var.items())},
        not_rest_points={t: sorted({x[0] for x in v if x[2]["stop"] != "rest"}) for t, v in sorted(var.items())})
    # two peaks of (0.25, 1.32, 0.2): f at rest over all N = 100 realizations
    fr = [x[2]["f_end"] for x in var["N100"] if x[0] == "e0.25_b1.32_g0.2_34-33-33" and x[2]["stop"] == "rest"]
    b["B1_two_peaks"] = dict(claimed="f about 0.42 in 34/41, 0.803 in 6/41 (41 realizations, N = 100 and more)",
                             n_N100=len(fr), f_rest_N100=sorted(round(x, 3) for x in fr),
                             high=sum(x > .7 for x in fr), low=sum(x < .5 for x in fr))
    esc = defaultdict(lambda: [0, 0])
    for r in load("escape/*_r*.json"):
        for q in r["runs"]:
            esc[(r["seed"], r["sd"])][0] += bool(q["rest"])
            esc[(r["seed"], r["sd"])][1] += 1
    b["B1_escape"] = dict(claimed="rest 14-18% of 500 repeats of the step f = 0.808 -> 0.803, any start state and sd",
                          by_seed_sd={"s%s sd%s" % k: "%d/%d" % tuple(v) for k, v in sorted(esc.items())},
                          total="%d/%d" % (sum(v[0] for v in esc.values()), sum(v[1] for v in esc.values())))
    # Table II of the draft (Sec. IV, 2026-11-17): per point, N = 100 realizations (original sweep = s0)
    def tab(r):
        S = r["steps"]
        init = sorted(r["sizes"], reverse=True)
        f1 = next((s["f"] for s in S if s.get("rest") or (s.get("sizes") and sorted(s["sizes"], reverse=True) != init)),
                  None)
        three = [s for s in S if s.get("sizes") and len(s["sizes"]) == 3]
        return dict(f0=S[0]["f"], f1=f1, stop=r.get("stop"), f_end=S[-1]["f"],
                    largest_balanced=min(max(s["sizes"]) for s in three) if three else None)
    t2 = {}
    for r in load("rebalance/*.json"):
        if not r.get("steps"):
            continue
        p = r["name"][:-5]
        rs = [tab(r)]
        for f in sorted(glob.glob(str(U / "rebalance_var" / (p + "_s*_N100.json")))):
            if Path(f).name != p + "_s0_N100.json":
                q = json.loads(Path(f).read_text())
                if q.get("steps"):
                    rs.append(tab(q))
        fr = [x["f_end"] for x in rs if x["stop"] == "rest"]
        lb = [x["largest_balanced"] for x in rs if x["largest_balanced"]]
        t2[p] = dict(f0=rs[0]["f0"], f1_s0=rs[0]["f1"], n=len(rs), rest=len(fr),
                     f_rest=[min(fr), max(fr)] if fr else None, largest_balanced=[min(lb), max(lb)])
    b["B_sec4_table"] = dict(claimed="Table II of the draft", detail=t2)
    # B2: no return to the initial partition
    def back(r):
        bw = [s for s in r["backward"] if s.get("sizes")]
        init = sorted(r["sizes"], reverse=True)
        return dict(returned=any(s["sizes"] == init for s in bw), last_sizes=bw[-1]["sizes"] if bw else None,
                    f_last=bw[-1]["f"] if bw else None, f0=r.get("f0", r["forward"][0]["f"]))
    hy = {r["name"][:-5]: back(r) for r in load("hysteresis/*.json")}
    hv = [back(r) for r in load("hysteresis_var/*.json")]
    b["B2"] = dict(claimed="10 points and 45 realizations: none returns to the initial partition",
                   points=len(hy), returned=sum(v["returned"] for v in hy.values()),
                   variants=len(hv), variants_returned=sum(v["returned"] for v in hv), detail=hy)
    # B3: strong-coupling end of balanced partitions (reduced system)
    se = {}
    for f in sorted(glob.glob(str(U / "hysteresis_ext/*_strong_end*.json"))):
        r = json.loads(Path(f).read_text())
        fb = r.get("first_bad") or {}
        im = fb.get("internal_moduli")
        se[Path(f).stem] = dict(end=r.get("end"), sizes=r.get("sizes"), f_last_good=r.get("f_last_good"),
                                crossing=int(np.argmax(im)) if im else None,
                                crossing_is_largest=(r["sizes"][int(np.argmax(im))] == max(r["sizes"])) if im else None,
                                collective_max_last_good=(r.get("last_good") or {}).get("collective_max"))
    b["B3"] = dict(claimed="8 points; ends at f = 1.92-4.16, internal +1 of the largest cluster; collective < 1",
                   files=len(se), end=dict(Counter(v["end"] for v in se.values())),
                   base=(lambda bs: dict(points=len(bs), f_range=[min(v["f_last_good"] for v in bs),
                                                                  max(v["f_last_good"] for v in bs)],
                                         crossing_is_largest=dict(Counter(str(v["crossing_is_largest"]) for v in bs)),
                                         collective_max_range=[min(v["collective_max_last_good"] for v in bs),
                                                               max(v["collective_max_last_good"] for v in bs)]))(
                       [v for k, v in se.items() if k.endswith("_strong_end")]),
                   variants={k: v for k, v in se.items() if not k.endswith("_strong_end")}, detail=se)
    # Table III of the draft (Sec. V, 2026-11-17): steps of 0.005 |K1| past the strong end after which the full system
    # had changed the partition (_strong_end full_beyond = step 1, then _strong_sweep steps), and the extension sweeps
    t3 = {}
    for f in sorted(glob.glob(str(U / "hysteresis_ext/*_strong_end.json"))):
        r = json.loads(Path(f).read_text())
        p = Path(f).name.replace("_strong_end.json", "")
        init = sorted(r["full_inside"]["r"]["sizes"], reverse=True)
        seq = [r["full_beyond"]["r"].get("sizes")]
        sw = U / "hysteresis_ext" / (p + "_strong_sweep.json")
        if sw.exists():
            seq += [s.get("sizes") for s in json.loads(sw.read_text())["steps"][1:]]
        n = next((i + 1 for i, z in enumerate(seq) if z != init), None)
        t3[p] = dict(f_end=r["f_last_good"], inside_kept=r["full_inside"]["r"]["sizes"] == r["sizes"],
                     steps_to_change=n)
    ext = {}
    for f in sorted(glob.glob(str(U / "hysteresis_ext/e*_[0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"))):
        r = json.loads(Path(f).read_text())
        S = r.get("steps")
        if isinstance(S, list) and S and isinstance(S[0], dict):
            ch = next((s for s in S if s.get("rest") or s.get("sizes") != S[0]["sizes"]), None)
            ext[Path(f).stem] = dict(start=S[0]["sizes"], f_range=[S[0]["f"], S[-1]["f"]],
                                     first_change=[ch.get("sizes"), ch["f"]] if ch else None)
    b["B_sec5"] = dict(claimed="Table III of the draft; extension sweeps", table=t3, extension=ext)
    # B6 (supplement S3, 2026-11-17): states met on the extension sweeps after the balanced partition broke, and the
    # four-cluster orbits of the 60000-time-unit runs beyond the strong end of (0.25, 1, 0.35)
    b6 = {}
    for f in sorted(glob.glob(str(U / "hysteresis_ext/*_states.json"))):
        r = json.loads(Path(f).read_text())
        b6[Path(f).stem] = [dict(sizes=st["sizes"], result=st["result"], kind=st.get("kind"),
                                 f=[st["f_first"], st["f_last"]], steps=st["step_last"] - st["step_first"] + 1,
                                 dist_full_to_orbit=st.get("dist_full_to_orbit"),
                                 collective_max=st.get("collective_max", st.get("collective_max_re")),
                                 internal=st.get("internal_moduli", st.get("internal_max_re")),
                                 x=st.get("x")) for st in r["states"]]
    bey = json.loads((U / "hysteresis_ext/e0.25_b1_g0.35_50-25-25_beyond.json").read_text())
    b6["e0.25_b1_g0.35_50-25-25_beyond_60000"] = [
        dict(seed=q["seed"], sizes=q["final"]["sizes"], motion=q["final"].get("motion"), period=q["final"].get("period"),
             collective_max=q["final"].get("collective_max"),
             internal_max=max(q["final"]["internal_moduli"]) if q["final"].get("internal_moduli") else None)
        for q in bey["runs"]]
    b["B6"] = dict(claimed="58:42 stable two-cluster orbit; 75:25 nonoscillating equilibrium (x = -1.91, 2.14); stable "
                           "four-cluster orbits such as 34:32:18:16 (the last from the 60000-unit runs of B5)",
                   detail=b6)
    # B5: first cluster whose spread exceeds 1e-4, and the clusters that release elements (rows of the membership
    # table: original cluster -> cluster at the end; a cluster releases elements if its row has more than one nonzero)
    b5 = {}
    for f in sorted(glob.glob(str(U / "hysteresis_ext/*_members*.json"))):
        r = json.loads(Path(f).read_text())
        sz = r["sizes"]
        largest = [i for i, s in enumerate(sz) if s == max(sz)]
        first, rel, ends = Counter(), Counter(), Counter()
        for q in r["runs"]:
            fb = q.get("first_break")
            first["none" if not fb else "largest" if all(c in largest for c in fb["clusters"]) else "other"] += 1
            t = q.get("table")
            if t:
                lost = [i for i, row in enumerate(t) if sum(x > 0 for x in row) > 1]
                rel["none" if not lost else "largest" if all(i in largest for i in lost) else
                    "other: " + ",".join(str(sz[i]) for i in lost)] += 1
            ends["rest" if q.get("rest") else ":".join(map(str, q.get("sizes") or []))] += 1
        b5[Path(f).stem] = dict(sizes=sz, first_break=dict(first), released=dict(rel), end_partitions=len(ends))
    b["B5"] = dict(claimed="9 places x 10 noises; break in 8; first spread > 1e-4 always in the largest cluster; "
                           "elements released by the largest cluster at 7 of 8 (exception (0.2, 1.28, 0.45): the 36 "
                           "breaks); 9 end partitions in 10 at (0.35, 0.92, 0.65)",
                   places=len(b5), detail=b5)
    # internal mechanism: the cluster reaching modulus 1 against the cluster of largest I_a = int (1 - x_a^2) dt
    it = json.loads((U / "internal_trace.json").read_text())
    rows = []
    for r in it:
        for side in ("weak_end", "strong_end"):
            e = r.get(side)
            if not e:
                continue
            cs = e["clusters"]
            mm = [q["max_modulus"] for q in cs]
            ii = [q["I_over_T"] for q in cs]
            p = [q["p"] for q in cs]
            k = int(np.argmax(mm))
            ti = [q["time_inside"] for q in cs]
            rows.append(dict(name=r["name"][:-5], side=side, sizes=e["sizes"], crossing=k,
                             crossing_is_max_I=k == int(np.argmax(ii)), crossing_is_largest=p[k] == max(p),
                             crossing_is_max_time_inside=k == int(np.argmax(ti))))
    b["internal_trace"] = dict(claimed="22 end orbits; crossing cluster = largest I_a in 20/22, = largest cluster in "
                                       "20/22; exceptions both sides of (0.35, 0.84, 0.55); crossing cluster = longest "
                                       "time with |x_a| < 1 per period in 22/22 (added 2026-11-19)",
                               orbits=len(rows), max_I=sum(q["crossing_is_max_I"] for q in rows),
                               max_time_inside=sum(q["crossing_is_max_time_inside"] for q in rows),
                               largest=sum(q["crossing_is_largest"] for q in rows),
                               exceptions=[q for q in rows if not (q["crossing_is_max_I"] and q["crossing_is_largest"])])
    return b


# ---------------------------------------------------------------- claim C
def claim_c():
    fs = load("fold_saddle/*.json")
    cls, det = Counter(), {}
    for r in fs:
        t = next((q["two_sided"] for q in r["coexistence"] if "two_sided" in q), [])
        sides = {s: sorted({q["outcome"] for q in t if q["sign"] == s}) for s in (1, -1)}
        o = {s: v[0] if len(v) == 1 else "mixed" for s, v in sides.items()}
        so = sorted(o.values())
        c = ("rest | oscillation" if so == ["rest", "stable orbit"] else
             "other oscillation | oscillation" if so == ["other", "stable orbit"] else
             "oscillation | oscillation" if so == ["stable orbit", "stable orbit"] else " | ".join(so))
        cls[c] += 1
        q4 = next((q for q in r["coexistence"] if "two_sided" in q), {})
        det[r["name"][:-5]] = dict(sides=o, slope_distance=r.get("slope_distance"), slope_log_mu=r.get("slope_log_mu"),
                                   f_fold=r["f_fold"], turning_points=len(r["turning_points"]),
                                   middle_width_over_K1=r["middle_segment"]["width_over_K1"], mu_u=q4.get("mu_u"),
                                   saddle_internal_max=max(q4.get("saddle_internal", [0])),
                                   returns_near_saddle_vs_predicted=[(t["returns_near_saddle"], t["predicted"])
                                                                     for t in q4.get("two_sided", [])
                                                                     if t["outcome"] != "stable orbit"])
    sl = [v["slope_distance"] for v in det.values() if v["slope_distance"]]
    fu = {}
    for r in load("fold_saddle_full/*.json"):
        fu[r["name"][:-5]] = sorted({(q["sign"], "rest" if q["t20000"]["rest"] else ":".join(map(str, q["t20000"]["sizes"]
                                     or []))) for q in r["runs"]})
    return dict(C1_C2=dict(claimed="12 points; exponent 0.49-0.53; sides rest|oscillation 6, other oscillation 4, "
                                   "both original oscillation 2", points=len(fs), classes=dict(cls),
                           slope_distance_range=[min(sl), max(sl)], detail=det),
                C3=dict(claimed="full system at 3 points, same destinations", outcomes=fu))


def main():
    res = dict(A=claim_a(), B=claim_b(), C=claim_c())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))

    def short(d):
        return {k: v for k, v in d.items() if k not in ("detail",)}
    for c, blocks in res.items():
        for k, v in blocks.items():
            print("==", k)
            print(json.dumps(short(v), default=str)[:1500])


if __name__ == "__main__":
    main()
