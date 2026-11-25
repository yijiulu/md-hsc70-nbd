#!/usr/bin/env bash
# Periodically export full-system checkpoint snapshots (including bulk K+ and water) while mdrun is active.
# Usage: bash scripts/snap_sampler.sh [EXPECTED_REPLICAS_PER_STATE]
set -euo pipefail

GMX=${GMX:-gmx}
NREPS=${1:-2}
TOTAL_RUNS=$((NREPS * 2))
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SN="$ROOT/analysis/snaps"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$SN"

while :; do
  done_runs=0
  for s in atp adp; do
    [[ -f "$SN/${s}_system.pdb" ]] || {
      [[ -f "$ROOT/work_${s}/rep1/prod.tpr" ]] && "$GMX" editconf -f "$ROOT/work_${s}/rep1/prod.tpr" -o "$SN/${s}_system.pdb" > /dev/null 2>&1
    }
    for r in $(seq 1 "$NREPS"); do
      R="$ROOT/work_${s}/rep${r}"
      grep -qs "Finished mdrun" "$R/prod.log" 2>/dev/null && done_runs=$((done_runs + 1))
      [[ -f "$R/prod.cpt" ]] || continue
      m=$(stat -c %Y "$R/prod.cpt" 2>/dev/null || stat -f %m "$R/prod.cpt")
      [[ "$m" == "$(cat "$SN/.${s}_rep${r}.mtime" 2>/dev/null || true)" ]] && continue
      cp "$R/prod.cpt" "$TMP/c.cpt" || continue
      mkdir -p "$SN/${s}_rep${r}"
      t=$(printf 'Protein\nSystem\n' | "$GMX" trjconv -s "$R/prod.tpr" -f "$TMP/c.cpt" -o "$TMP/s.xtc" -pbc mol -center -ur compact 2>&1 \
          | awk '/Last written: frame/ {printf "%d", $NF}')
      if [[ -n "$t" ]]; then
        mv "$TMP/s.xtc" "$SN/${s}_rep${r}/t$(printf %07d "$t").xtc"
        echo "$m" > "$SN/.${s}_rep${r}.mtime"
        echo "$(date '+%F %T') $s rep$r t=$t ps"
      fi
    done
  done
  (( done_runs >= TOTAL_RUNS )) && { echo "All $TOTAL_RUNS production runs finished."; break; }
  sleep 300
done
