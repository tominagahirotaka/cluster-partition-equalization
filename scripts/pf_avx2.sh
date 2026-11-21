#!/bin/sh
# Run a script with the AVX2 code paths of NumPy and OpenBLAS, so that results on computers with AVX-512 match those
# on computers without it bit for bit (checked 2026-11-18 for the sweeps; paper, Sec. II E). Used for
# partition_flow.py on the computer with AVX-512.
#   sh scripts/pf_avx2.sh scripts/partition_flow.py <map file> <args>
export OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
cd "$(dirname "$0")/.." || exit 1
exec timeout 30h python "$@"
