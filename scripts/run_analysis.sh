#!/usr/bin/env bash
# End-to-end analysis pipeline for Hsc70 NBD trajectories.
# Usage:
#   bash scripts/run_analysis.sh [--states atp,adp] [--reps 1,2] [--skip 20] [--until NS] [--csp path/to/csp.tsv]
set -euo pipefail

PYTHON=${PYTHON:-python}
STATES_CSV="atp,adp"
REPS_CSV="1,2"
COMPARE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --states)
      STATES_CSV=$2
      shift 2
      ;;
    --reps)
      REPS_CSV=$2
      COMPARE_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      COMPARE_ARGS+=("$1")
      shift
      ;;
  esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

IFS=',' read -r -a STATES <<< "$STATES_CSV"
IFS=',' read -r -a REPS <<< "$REPS_CSV"

for s in "${STATES[@]}"; do
  for r in "${REPS[@]}"; do
    [[ -f "work_${s}/rep${r}/prod.xtc" ]] || continue
    bash scripts/an_prep.sh "$s" "$r"
    "$PYTHON" scripts/an_rep.py "$s" "$r"
  done
done

if [[ -d analysis/snaps ]]; then
  "$PYTHON" scripts/an_snap.py
fi
"$PYTHON" scripts/an_compare.py "${COMPARE_ARGS[@]}"
"$PYTHON" scripts/an_figures.py
