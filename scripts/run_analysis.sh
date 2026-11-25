#!/usr/bin/env bash
# End-to-end analysis pipeline for isolated Hsc70 NBD trajectories.
# Usage: bash scripts/run_analysis.sh [--skip 20] [--until NS] [--reps 1,2] [--csp path/to/csp.tsv]
set -euo pipefail

PYTHON=${PYTHON:-python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

for s in atp adp; do
  for r in 1 2 3; do
    [[ -f "work_${s}/rep${r}/prod.xtc" ]] || continue
    bash scripts/an_prep.sh "$s" "$r"
    "$PYTHON" scripts/an_rep.py "$s" "$r"
  done
done

if [[ -d analysis/snaps ]]; then
  "$PYTHON" scripts/an_snap.py
fi
"$PYTHON" scripts/an_compare.py "$@"
"$PYTHON" scripts/an_figures.py
