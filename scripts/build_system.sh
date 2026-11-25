#!/usr/bin/env bash
# Build Hsc70 NBD (1KAX with M71K reverted, or 7RAX-templated NBD+linker model) + ATP/ADP + Mg2+ + 2 K+
# for GROMACS up to energy minimisation.
#
# Usage:
#   bash scripts/build_system.sh atp|adp
#
# Optional environment overrides:
#   PDB=structures/1KAX.pdb   NAME=<state>   PH=7.0   PAD=1.0   SALT=0.15   NT=8   GMX=gmx   PYTHON=python
set -euo pipefail

STATE=${1:?usage: build_system.sh atp|adp}
PH=${PH:-7.0}
PAD=${PAD:-1.0}
SALT=${SALT:-0.15}
NT=${NT:-8}
GMX=${GMX:-gmx}
PYTHON=${PYTHON:-python}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
P="$ROOT/params"
MDP="$ROOT/mdp"
PDB=$(realpath "${PDB:-$ROOT/structures/1KAX.pdb}")
NAME=${NAME:-$STATE}
B="$ROOT/work_$NAME/build"
FFDIR=$(dirname "$(command -v "$GMX")")/../share/gromacs/top/amber19sb.ff

mkdir -p "$B" && cd "$B"
step(){ echo ">> $(date '+%H:%M:%S') $*"; }

step "0/5 split $(basename "$PDB") ($STATE)"
"$PYTHON" "$ROOT/scripts/prep_1kax.py" "$PDB" "$STATE" .

# ---- 1. Protein ------------------------------------------------------------------------------
step "1/5 protein: M71K, missing heavy atoms (PDBFixer), PROPKA pH $PH, pdb2gmx (amber19sb, opc)"
"$PYTHON" - <<'EOF'
from pdbfixer import PDBFixer
from openmm.app import PDBFile
f = PDBFixer(filename="protein_raw.pdb")
f.applyMutations(["MET-71-LYS"], "A")
f.findMissingResidues()
f.missingResidues = {}  # residues 1-3 are disordered in 1KAX; not modelled
f.findNonstandardResidues()
f.replaceNonstandardResidues()
f.findMissingAtoms()
f.addMissingAtoms()
with open("fixed.pdb", "w") as fh:
    PDBFile.writeFile(f.topology, f.positions, fh, keepIds=True)
r71 = [r.name for r in f.topology.residues() if r.id == "71"]
assert r71 == ["LYS"], r71
EOF

pdb2pqr --ff=AMBER --ffout=AMBER --titration-state-method=propka --with-ph="$PH" --keep-chain \
        --pdb-output=protonated.pdb fixed.pdb protonated.pqr > pdb2pqr.log 2>&1

"$PYTHON" - protonated.pdb protein_in.pdb <<'EOF'
import sys, collections, math
# Metal-free PROPKA protonates carboxylates coordinating Mg2+/K+ (Asp10, Asp199, Asp206);
# restore any Asp/Glu within 4.0 A of a retained cation to its deprotonated state.
cations = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open("ions.pdb") if l.startswith("HETATM")]
near = {(l[21], l[22:27]) for l in open(sys.argv[1]) if l.startswith("ATOM")
        and min(math.dist([float(l[30:38]), float(l[38:46]), float(l[46:54])], c) for c in cations) < 4.0}
names, forced = collections.Counter(), set()
with open(sys.argv[2], "w") as out:
    for l in open(sys.argv[1]):
        if l.startswith(("ATOM", "HETATM")):
            res = l[17:21].strip()
            if len(res) == 4 and res[0] in "NC":
                res = res[1:]
            if res in ("ASH", "GLH") and (l[21], l[22:27]) in near:
                res = {"ASH": "ASP", "GLH": "GLU"}[res]
                forced.add(f"{res}{l[22:26].strip()}")
                if l[12:16].strip() in ("HD2", "HE2"):
                    continue
            names[res] += 1 if l[12:16].strip() == "CA" else 0
            l = "ATOM  " + l[6:17] + f"{res:<3s}" + l[20:]
        if l.startswith(("ATOM", "TER", "END")):
            out.write(l)
special = {k: v for k, v in names.items() if k in ("HID", "HIE", "HIP", "ASH", "GLH", "LYN", "CYX", "CYM")}
print("   protonation/tautomer states:", special or "all standard")
print("   cation-coordinating carboxylates kept charged:", sorted(forced) or "none needed")
EOF

"$GMX" pdb2gmx -f protein_in.pdb -o protein.gro -p topol.top -i posre.itp -ff amber19sb -water opc -ignh > pdb2gmx.log 2>&1

# ---- 2. Nucleotide ---------------------------------------------------------------------------
NUC=${STATE^^}
QNUC=$([[ $STATE == atp ]] && echo -4 || echo -3)
step "2/5 $NUC: Meagher 2003 parameters (tleap) -> GROMACS (ParmEd), charge $QNUC"
cat > tleap.in <<EOF
source leaprc.protein.ff19SB
loadamberparams $P/frcmod.phos
loadamberprep $P/$NUC.prep
$NUC = $STATE
m = loadpdb nuc_heavy.pdb
check m
charge m
saveamberparm m nuc.prmtop nuc.inpcrd
quit
EOF
tleap -f tleap.in > tleap.log 2>&1
grep -qE "^Total unperturbed charge: *${QNUC}\.0+" tleap.log || { echo "unexpected $NUC charge (see $B/tleap.log)"; exit 1; }
! grep -qiE "could not find (bond|angle|torsion)|FATAL" tleap.log || { echo "missing parameters (see $B/tleap.log)"; exit 1; }

"$PYTHON" - "$FFDIR/ffnonbonded.itp" <<'EOF'
import sys, re, parmed as pmd
s = pmd.load_file("nuc.prmtop", "nuc.inpcrd")
for a in s.atoms:
    a.residue.name = "LIG"
s.save("nuc_full.top", format="gromacs", overwrite=True)
s.save("nuc.gro", overwrite=True)
t = open("nuc_full.top").read()
ff = {}
for l in open(sys.argv[1]):
    f = l.split()
    if len(f) >= 7 and not l.lstrip().startswith((";", "[")):
        ff[f[0]] = (float(f[-2]), float(f[-1]))
block = re.search(r"\[ atomtypes \]\n(.*?)\n\n", t, re.S).group(1)
new = []
for l in block.splitlines():
    f = l.split()
    if not f or f[0].startswith(";"):
        continue
    sig, eps = float(f[-2]), float(f[-1])
    if f[0] in ff:
        assert abs(ff[f[0]][0] - sig) < 1e-4 and abs(ff[f[0]][1] - eps) < 1e-4, f"LJ mismatch for {f[0]}: {ff[f[0]]} vs {sig, eps}"
    else:
        new.append(l)
open("LIG_atomtypes.itp", "w").write("[ atomtypes ]\n; Meagher et al. 2003 polyphosphate type(s) not in amber19sb.ff\n" + "\n".join(new) + "\n")
mol = t[t.index("[ moleculetype ]"):t.index("[ system ]")]
mol = re.sub(r"(\[ moleculetype \]\n;[^\n]*\n)\S+", r"\1LIG", mol, count=1)
heavy = [a.idx + 1 for a in s.atoms if a.element > 1]
mol += "#ifdef POSRES\n#include \"posre_LIG.itp\"\n#endif\n\n"
open("LIG.itp", "w").write(mol)
open("posre_LIG.itp", "w").write("[ position_restraints ]\n; atom  type  fx    fy    fz\n"
                                 + "".join(f"{i:6d}     1  1000  1000  1000\n" for i in heavy))
print(f"   LIG: {len(s.atoms)} atoms, net charge {sum(a.charge for a in s.atoms):+.3f}; new atom types: "
      f"{[l.split()[0] for l in new]}; {len(heavy)} heavy atoms restrained under POSRES")
EOF

# ---- 3. Bound ions, crystal waters, solvation & KCl ------------------------------------------
step "3/5 complex: Mg2+ + 2 K+ + crystal waters, dodecahedron (d = $PAD nm), OPC, $SALT M KCl"
cat > bound.itp <<'EOF'
; Crystal Mg2+, active-site K+ (Li/Merz OPC), and first-shell crystal waters (OPC) as separate molecule types.
; GROMACS permits SETTLE on only one molecule type (SOL); CWAT uses constrained O-H bonds and a flexible H-O-H angle.
[ moleculetype ]
MGB  1
[ atoms ]
1  Mg2+_opc  1  MGB  MG  1  2.0  24.305
#ifdef POSRES
[ position_restraints ]
1  1  1000  1000  1000
#endif

[ moleculetype ]
KB  1
[ atoms ]
1  K+_opc  1  KB  K  1  1.0  39.1
#ifdef POSRES
[ position_restraints ]
1  1  1000  1000  1000
#endif

[ moleculetype ]
CWAT  2
[ atoms ]
1  OW_opc  1  HOH  OW   1   0          16.00000
2  HW_opc  1  HOH  HW1  1   0.679142    1.00800
3  HW_opc  1  HOH  HW2  1   0.679142    1.00800
4  MW      1  HOH  MW   1  -1.358284    0.00000
[ bonds ]
1  2  1  0.08724331  462750.4
1  3  1  0.08724331  462750.4
[ angles ]
2  1  3  1  103.6  836.8
[ virtual_sites3 ]
4  1  2  3  1  0.14772234  0.14772234
[ exclusions ]
1  2  3  4
2  1  3  4
3  1  2  4
4  1  2  3
#ifdef POSRES
[ position_restraints ]
1  1  1000  1000  1000
#endif
EOF

"$PYTHON" - <<'EOF'
def gro_atoms(fn):
    L = open(fn).read().splitlines()
    return [(l[5:10].strip(), l[10:15].strip(), [float(l[20:28]), float(l[28:36]), float(l[36:44])])
            for l in L[2:2 + int(L[1])]], L[-1]
def pdb_atoms(fn):
    return [(l[17:20].strip(), l[12:16].strip(), [float(l[30:38]) / 10, float(l[38:46]) / 10, float(l[46:54]) / 10])
            for l in open(fn) if l.startswith("HETATM")]
prot, box = gro_atoms("protein.gro")
lig, _ = gro_atoms("nuc.gro")
ions, cw = pdb_atoms("ions.pdb"), pdb_atoms("cwat.pdb")
ions = [({"MG": "MGB", "K": "KB"}[r], a, x) for r, a, x in ions]
lines, res = [], 0
def add(atoms, per_res):
    global res
    prev = None
    for i, (rn, an, x) in enumerate(atoms):
        key = i // per_res
        if key != prev:
            res += 1
            prev = key
        lines.append(f"{res % 100000:5d}{rn:<5s}{an:>5s}{(len(lines) + 1) % 100000:5d}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}")
p_lines = open("protein.gro").read().splitlines()
prev = None
for l in p_lines[2:2 + int(p_lines[1])]:
    if l[:5] != prev:
        res += 1
        prev = l[:5]
    lines.append(f"{res % 100000:5d}{l[5:10]}{l[10:15]}{(len(lines) + 1) % 100000:5d}{l[20:44]}")
add(lig, len(lig))
add(ions, 1)
add(cw, 4)
open("complex.gro", "w").write(f"Hsc70 NBD + nucleotide + Mg/K + crystal waters\n{len(lines)}\n" + "\n".join(lines) + f"\n{box}\n")
t = open("topol.top").read()
ffinc = '#include "amber19sb.ff/forcefield.itp"\n'
assert ffinc in t and "; Include water topology" in t, "unexpected topol.top layout"
t = t.replace(ffinc, ffinc + '\n; Meagher polyphosphate atom type(s)\n#include "LIG_atomtypes.itp"\n', 1)
t = t.replace("; Include water topology",
              '; nucleotide, bound ions, crystal waters\n#include "LIG.itp"\n#include "bound.itp"\n\n; Include water topology', 1)
nk = sum(1 for r, *_ in ions if r == "KB")
t = t.rstrip("\n") + f"\nLIG                 1\nMGB                 1\nKB                  {nk}\nCWAT                {len(cw) // 4}\n"
open("topol.top", "w").write(t)
print(f"   complex: {len(prot)} protein + {len(lig)} LIG + {len(ions)} ions ({nk} K+) + {len(cw) // 4} crystal waters")
EOF

grep -q 'ions_opc.itp\|ions.itp' topol.top || { echo "topol.top does not include an ion topology"; exit 1; }
"$GMX" editconf -f complex.gro -o box.gro -bt dodecahedron -d "$PAD" -c > editconf.log 2>&1
"$GMX" solvate -cp box.gro -cs tip4p.gro -o solv.gro -p topol.top > solvate.log 2>&1
"$GMX" grompp -f "$MDP/ions.mdp" -c solv.gro -p topol.top -o ions.tpr -po ions_mdout.mdp > grompp_ions.log 2>&1
echo SOL | "$GMX" genion -s ions.tpr -o system.gro -p topol.top -pname K -nname CL -neutral -conc "$SALT" > genion.log 2>&1

"$GMX" select -s system.gro -on index.ndx -select '"Protein_LIG" group "Protein" or resname LIG MGB KB;
  "Water_and_ions" not (group "Protein" or resname LIG MGB KB); "LIG_heavy" resname LIG and not name "H.*"' > make_ndx.log 2>&1
for g in Protein_LIG Water_and_ions; do
  grep -qE "^\[ $g \]" index.ndx || { echo "index group $g missing (see $B/make_ndx.log)"; exit 1; }
done

# ---- 4. Energy minimisation ------------------------------------------------------------------
step "4/5 energy minimisation"
"$GMX" grompp -f "$MDP/em.mdp" -c system.gro -r system.gro -p topol.top -n index.ndx -o em.tpr -po em_mdout.mdp > grompp_em.log 2>&1
"$GMX" mdrun -deffnm em -ntmpi 1 -ntomp "$NT" > em.out 2>&1
grep -q 'converged to Fmax <' em.log || { echo "EM did not converge (see $B/em.log)"; exit 1; }
echo "   $(grep -m1 'converged to Fmax' em.log | sed 's/^ *//'); $(sed -n 2p system.gro | tr -d ' ') atoms"

# ---- 5. Check active-site geometry after EM --------------------------------------------------
step "5/5 active-site geometry after EM"
"$PYTHON" - <<'EOF'
import math
L = open("em.gro").read().splitlines()
at = [(l[5:10].strip(), l[10:15].strip(), [float(l[20:28]), float(l[28:36]), float(l[36:44])], i)
      for i, l in enumerate(L[2:2 + int(L[1])])]
cat = [a for a in at if a[0] in ("MGB", "KB")]
for rn, an, x, i in cat:
    shell = sorted((math.dist(x, b[2]) * 10, f"{b[0]}:{b[1]}") for b in at
                   if b[1][0] == "O" and b[3] != i and math.dist(x, b[2]) < 0.32)
    print(f"   {rn:2s} atom {i + 1}: " + ", ".join(f"{n} {d:.2f}" for d, n in shell))
EOF
