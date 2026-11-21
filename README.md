# Code for "Equalization of Cluster Partitions and the Ends of Three-Cluster States in Globally Coupled Excitable Elements with Repulsive Interactions"

Author: Hirotaka Tominaga. Code version: v1.0.0.
Data DOI: https://doi.org/10.5281/zenodo.24414126.

This repository contains the scripts from which the figures, tables and the numbers quoted in the paper and its
supplementary material can be regenerated from stored numerical outputs (Stage A), and the scripts that produced those
outputs (Stage B). The stored outputs are archived separately on Zenodo (https://doi.org/10.5281/zenodo.24414126) under the same version
number. Code: MIT License (`LICENSE`); data: CC BY 4.0.

## Getting the data

Download the Zenodo archive `cluster-partition-equalization-data-v1.0.0` and copy its `analysis/` directory into the
root of this repository, so that the scripts find their inputs under `analysis/`. The archive also contains
`reference_outputs/`, the figure files and JSON outputs used in the paper, for comparison with the regenerated files.

```
pip install -r requirements.txt
cp -r <path to the Zenodo archive>/analysis .
```

Paths below are relative to the repository root.

## Two levels of reproduction

**Stage A: regeneration from stored outputs (no integration).** The figures, Tables I and II and the numbers quoted
in the text are recomputed from the files under `analysis/` (`aggregates.json` gathers them with the value quoted in
the paper next to each). The values in the tables of the supplementary material are read from the stored outputs named
in the table below. `paper4_aggregates.py` also evaluates the excitability test of each grid point (no integration).
See the section "Stage A: check" for the result of the check.

**Stage B: recomputation.** The scripts that produced the stored outputs are included, with the parameter points, start
orbits, random seeds, integration method and step settings (paper, Sec. II, and Supplementary Material; docstrings of
the scripts). Stage B has not been verified from this repository as a whole. The noisy sweeps of the full system depend
on rounding: a sweep repeated on the same computer reproduces the stored sweep bit for bit, but on a computer whose
OpenBLAS selects a different kernel (AVX-512 versus AVX2) it can depart from it after tens of steps; in the checks of
the paper (Sec. II E) such sweeps gave the same qualitative course and values of f at rest within 0.04. Setting
`OPENBLAS_CORETYPE=Haswell` (see `scripts/pf_avx2.sh`) selects the AVX2 kernel on any x86-64 computer.

## Stage A: commands

```
python scripts/paper4_aggregates.py             # values of the text and the supplement -> analysis/paper4_figs/aggregates.json
python scripts/partition_flow_summary.py        # f_c of the ordered partitions at the three points of Fig. 3
                                                #   -> analysis/universality/partition_flow_summary.json (and draft figures)
python scripts/partition_flow_order_summary.py  # exits compared with f_c in the recorded firing order (Sec. IV B, Table S)
                                                #   -> analysis/universality/partition_flow_order_summary.json
python scripts/plot_paper4_figs.py              # Figs. 1-4 -> analysis/paper4_figs/fig_{map,sweep,flow,rest}.{pdf,png}
```

The figure files: `fig_map` (Fig. 1), `fig_sweep` (Fig. 2), `fig_flow` (Fig. 3), `fig_rest` (Fig. 4).
`collect_order_canonical.py` is not part of Stage A: it selected, among repeated sweeps fetched from several computers,
the ones identical to the original sweeps, and wrote `analysis/universality/rebalance_order_canonical/` (its
`manifest.json` lists the source of each file).

| Item | Inputs (under `analysis/universality/`) | Produced by (Stage B) |
|---|---|---|
| Sec. III, Fig. 1, Table I, S1 | `map/*.json`, `map_ends/*.json` | `universality_map.py` (launched by `universality_map_launch.py`), `universality_map_ends.py` |
| Sec. II C, IV B, Fig. 3, S2 (exits) | `partition_flow/*.json`, `rebalance_order_canonical/*.json` | `partition_flow.py` (through `pf_avx2.sh` on the computer with AVX-512), `rebalance_order.py`, `collect_order_canonical.py` |
| Sec. IV, Table II, Figs. 2 and 4, S2 | `rebalance/`, `rebalance_var/`, `hysteresis/`, `hysteresis_var/`, `internal_end_full/`, `escape/` | `rebalance_sweep.py`, `rebalance_variants.py`, `hysteresis_sweep.py`, `hysteresis_variants.py`, `internal_end_full.py`, `escape_prepare.py`, `escape_runs.py` |
| Sec. V, S3, S4 | `hysteresis_ext/`, `internal_trace.json` | `hysteresis_extend.py`, `hysteresis_new_states.py`, `balanced_strong_end.py`, `strong_end_sweep.py`, `end_arclength.py`, `period2_check.py`, `beyond_end_detail.py`, `break_statistics.py`, `break_membership.py`, `kick_test.py`, `nonperiodic_state.py`, `transient_detail.py`, `internal_trace.py` |
| Sec. VI | `fold_saddle/`, `fold_saddle_full/` | `fold_saddle_test.py`, `fold_saddle_full.py` |

## File list of the paper

`core_files_paper4.txt` lists the scripts and outputs needed for each figure, table and number of the paper, grouped by
the part of the paper that uses them; `core_files_paper4_manifest.tsv` gives the size and SHA-256 checksum of each of
the 6476 files. Both are included in the code repository and in the data archive. Their paths are those of the
original project: `scripts/` is in the code repository, `analysis/` in the data archive, and the outputs regenerated in
Stage A (`analysis/paper4_figs/aggregates.json`, `analysis/paper4_figs/fig_*`,
`analysis/universality/partition_flow_summary.json`, `analysis/universality/partition_flow_order_summary.json`) are in
`reference_outputs/` of the data archive. Every published file agrees with its checksum except three that were changed
for publication: `scripts/universality_map_launch.py` and `scripts/pf_avx2.sh` (host list and local paths removed) and
`analysis/universality/rebalance_order_canonical/manifest.json` (local paths in the "source" fields replaced by
computer and file name). `MANIFEST.sha256` gives the checksums of the published files themselves.

## Settings of the full-system sweeps

- Start: the last stable orbit of the map at 0.02|K1| before its end, every cluster filled by its elements (N = 100
  unless stated; the partition is scaled for N = 200 and 400).
- Steps of 0.005|K1| in K toward weak coupling (variants 0.0025|K1| and 0.01|K1|), T = 10 000 time units per step
  (variants 5000 and 20 000); at each step Gaussian noise of standard deviation 1e-8 is added to every coordinate.
- The noise of step j in realization s is drawn from NumPy `default_rng(100000 s + j)`.
- Integration: SciPy `solve_ivp`, DOP853, rtol 1e-10, atol 1e-12, maximum step 0.15.

## Software and computers

- Regeneration (Stage A) on 2026-11-19: Python 3.12.15, NumPy 2.5.4, SciPy 1.17.1, matplotlib 3.11.2, Pillow 13.0.0
  (`requirements.txt`).
- Original computations: Python 3.12 with NumPy 2.5.4, SciPy 1.17.1 and OpenBLAS 0.3.34 on all computers; Numba 0.68.0
  for the map.
- The "host" field of the outputs gives the computer on which each was computed:

| host | CPU | AVX-512 |
|---|---|---|
| `origin-horse` | Intel Core i9-14900KF | no |
| `origin2022` | Intel Core i9-12900K | no |
| `origin201805` | Intel Core i7-6700K | no (map points only) |
| `origin-boar` | Intel Core i9-9960X | yes |
| `gpgpu2022` | Intel Core i7-11700 | yes |

- `universality_map_launch.py` is a public version of the launcher: the author's list of computers and local paths
  were removed, and it runs the tasks on the local computer.
- Docstrings and comments refer to the author's working notes (for example `docs/RESULTS_NOTES.md`) and to file names
  of the original project; these notes are not part of this repository.

## Stage A: check

Checked on 2026-11-19 by placing this code and the `analysis/` directory of the data archive in a new empty directory
and running the four commands above in order (about 10 s in all): `aggregates.json`, `partition_flow_summary.json`,
`partition_flow_order_summary.json` and the figure PNGs were identical byte for byte to `reference_outputs/` of the data
archive, and the figure PDFs were identical apart from their creation dates.
