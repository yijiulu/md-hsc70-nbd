#!/usr/bin/env bash
# Whole-molecule, protein-centred copy of one replica's production trajectory (Protein_LIG group: protein, LIG, MGB, KB).
# Usage: an_prep.sh <state> REP   (e.g. atp|adp|lk_atp|lk_adp|nl_atp|nl_adp; safe to rerun while running)
# Output: analysis/<state>_rep<REP>/{sub.tpr, top.pdb, whole.xtc}
# MDAnalysis cannot read GROMACS 2026 tpr files (tpx 138), so the topology goes to Python as top.pdb. LIG and the ions
# can still end up one box vector away from the protein after -pbc mol; an_rep.py moves them to the nearest image.
set -euo pipefail
s=$1 r=$2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/work_$s
O=$ROOT/analysis/${s}_rep$r
mkdir -p "$O"
[[ -f $O/sub.tpr ]] || echo Protein_LIG | gmx convert-tpr -s "$W/rep$r/prod.tpr" -n "$W/build/index.ndx" -o "$O/sub.tpr" > "$O/prep.log" 2>&1
[[ -f $O/top.pdb ]] || gmx editconf -f "$O/sub.tpr" -o "$O/top.pdb" >> "$O/prep.log" 2>&1
printf 'Protein\nSystem\n' | gmx trjconv -s "$O/sub.tpr" -f "$W/rep$r/prod.xtc" -o "$O/whole.xtc" -pbc mol -center -ur compact >> "$O/prep.log" 2>&1
echo "$s rep$r: $(gmx check -f "$O/whole.xtc" 2>&1 | awk '/^Coords/ {print $2 " frames"}')"
