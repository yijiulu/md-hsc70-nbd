#!/usr/bin/env bash
# Self-contained GROMACS equilibration (NVT -> NPT) and checkpoint-resumable production MD runner.
#
# Usage:
#   bash scripts/run_md.sh <NAME> [--replicas N] [--ns LENGTH_NS] [--nt THREADS]
# Example:
#   bash scripts/run_md.sh atp --replicas 2 --ns 150 --nt 8
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <NAME> [--replicas N] [--ns LENGTH_NS] [--nt THREADS]" >&2
  exit 1
fi

NAME=$1
shift
REPLICAS=2
PROD_NS=150
NT=${NT:-8}
GMX=${GMX:-gmx}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --replicas) REPLICAS=$2; shift 2 ;;
    --ns)       PROD_NS=$2; shift 2 ;;
    --nt)       NT=$2; shift 2 ;;
    *)          echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MDP="$ROOT/mdp"
WORK="$ROOT/work_$NAME"
BUILD="$WORK/build"

if [[ ! -f "$BUILD/em.gro" ]]; then
  echo "Missing $BUILD/em.gro. Run scripts/build_system.sh first." >&2
  exit 1
fi

NSTEPS=$(awk -v ns="$PROD_NS" 'BEGIN { printf "%d", ns * 1000.0 / 0.004 }')

for r in $(seq 1 "$REPLICAS"); do
  RDIR="$WORK/rep$r"
  mkdir -p "$RDIR"
  cd "$RDIR"
  echo "=== [$NAME rep$r] NVT -> NPT -> Production (${PROD_NS} ns, ${NSTEPS} steps at 4 fs HMR) ==="

  if [[ ! -f nvt.gro ]]; then
    cp "$MDP/nvt.mdp" ./nvt.mdp
    "$GMX" grompp -f nvt.mdp -c "$BUILD/em.gro" -r "$BUILD/em.gro" -p "$BUILD/topol.top" -n "$BUILD/index.ndx" \
      -o nvt.tpr -po nvt_mdout.mdp > grompp_nvt.log 2>&1
    "$GMX" mdrun -deffnm nvt -ntmpi 1 -ntomp "$NT" > nvt.out 2>&1
  fi

  if [[ ! -f npt.gro ]]; then
    cp "$MDP/npt.mdp" ./npt.mdp
    "$GMX" grompp -f npt.mdp -c nvt.gro -r "$BUILD/em.gro" -t nvt.cpt -p "$BUILD/topol.top" -n "$BUILD/index.ndx" \
      -o npt.tpr -po npt_mdout.mdp > grompp_npt.log 2>&1
    "$GMX" mdrun -deffnm npt -ntmpi 1 -ntomp "$NT" > npt.out 2>&1
  fi

  if [[ ! -f prod.tpr ]]; then
    sed "s/^nsteps[[:space:]]*=.*/nsteps                   = ${NSTEPS}/" "$MDP/prod.mdp" > ./prod.mdp
    "$GMX" grompp -f prod.mdp -c npt.gro -t npt.cpt -p "$BUILD/topol.top" -n "$BUILD/index.ndx" \
      -o prod.tpr -po prod_mdout.mdp > grompp_prod.log 2>&1
  fi

  if [[ -f prod.cpt ]]; then
    echo "    Resuming $NAME rep$r from prod.cpt ..."
    "$GMX" mdrun -deffnm prod -cpi prod.cpt -append -ntmpi 1 -ntomp "$NT" >> prod.out 2>&1
  else
    "$GMX" mdrun -deffnm prod -ntmpi 1 -ntomp "$NT" > prod.out 2>&1
  fi
done
