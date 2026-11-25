"""Split 1KAX (bovine Hsc70 NBD K71M + ATP, Mg2+, K+; residues 4-381, identical to human HSPA8) into build inputs.

usage: python prep_1kax.py 1KAX.pdb {atp|adp} OUTDIR

Writes to OUTDIR:
  protein_raw.pdb   chain A protein atoms (M71 is reverted to K later by PDBFixer)
  nuc_heavy.pdb     nucleotide heavy atoms with AMBER prep names (ATP.prep / ADP.prep, Meagher et al. 2003);
                    for adp the gamma-phosphate (PG, O1G, O2G, O3G) is removed
  ions.gro-lines    Mg2+ and the two active-site K+ (K490: Asp10/Tyr15/ATP beta; K491: Asp199/Thr204/Asp206/ATP gamma)
  cwat.pdb          first-shell crystal waters of Mg2+ and the two K+ (O-cation < 3.2 A), with OPC hydrogens and
                    virtual site placed pointing away from the cation; for adp one extra water sits where O3G
                    coordinated Mg2+, completing its octahedral shell
"""
import sys, os
import numpy as np

pdb, state, out = sys.argv[1], sys.argv[2], sys.argv[3]
assert state in ("atp", "adp")
os.makedirs(out, exist_ok=True)

KEEP_K = (490, 491)          # canonical active-site K+ sites; K492 (Tyr149/Thr204/Asp206, no nucleotide contact) dropped
GAMMA = {"PG", "O1G", "O2G", "O3G"}
DOH, HOH_DEG, VS_A = 0.8724331, 103.6, 0.14772234      # OPC geometry (Å) and virtual-site coefficient (amber19sb.ff/opc.itp)

prot, nuc, ions, wat = [], [], [], {}
for l in open(pdb):
    if not l.startswith(("ATOM", "HETATM")) or l[21] != "A":
        continue
    res, num, name = l[17:20].strip(), int(l[22:26]), l[12:16].strip()
    xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    if l.startswith("ATOM"):
        prot.append(l)
    elif res == "ATP":
        nuc.append((name, xyz))
    elif res == "MG" or (res == "K" and num in KEEP_K):
        ions.append((res, num, xyz))
    elif res == "HOH":
        wat[num] = xyz

with open(f"{out}/protein_raw.pdb", "w") as f:
    f.writelines(prot)
    f.write("TER\nEND\n")

resn = state.upper()
kept = [(n, x) for n, x in nuc if not (state == "adp" and n in GAMMA)]
with open(f"{out}/nuc_heavy.pdb", "w") as f:
    for i, (n, x) in enumerate(kept, 1):
        n = n.replace("'", "*")
        f.write(f"HETATM{i:5d} {n:<4s} {resn} A 500    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00\n")
    f.write("END\n")

cations = [(r, x) for r, _, x in ions]
shell = sorted(k for k, x in wat.items() if min(np.linalg.norm(x - c) for _, c in cations) < 3.2)
waters = [(f"HOH{k}", wat[k]) for k in shell]
if state == "adp":
    waters.append(("O3G->HOH", dict(nuc)["O3G"]))

def place_h(o, away):
    """OPC water with its bisector pointing from `away` through O; returns O, H1, H2, MW (Å)."""
    u = (o - away) / np.linalg.norm(o - away)
    t = np.cross(u, [1.0, 0.0, 0.0])
    if np.linalg.norm(t) < 0.1:
        t = np.cross(u, [0.0, 1.0, 0.0])
    t /= np.linalg.norm(t)
    half = np.radians(HOH_DEG / 2)
    h1 = o + DOH * (np.cos(half) * u + np.sin(half) * t)
    h2 = o + DOH * (np.cos(half) * u - np.sin(half) * t)
    mw = o + VS_A * (h1 - o) + VS_A * (h2 - o)
    return [o, h1, h2, mw]

with open(f"{out}/cwat.pdb", "w") as f:
    i = 0
    for j, (label, o) in enumerate(waters, 1):
        near = min(cations, key=lambda c: np.linalg.norm(o - c[1]))[1]
        for name, x in zip(("OW", "HW1", "HW2", "MW"), place_h(o, near)):
            i += 1
            f.write(f"HETATM{i:5d} {name:<4s} HOH W{j:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00\n")
    f.write("END\n")

with open(f"{out}/ions.pdb", "w") as f:
    for i, (r, num, x) in enumerate(ions, 1):
        f.write(f"HETATM{i:5d} {r:<4s} {r:>3s} I{i:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00\n")
    f.write("END\n")

print(f"{state}: {len(prot)} protein atoms, {resn} {len(kept)} heavy atoms, ions {[f'{r}{n}' for r, n, _ in ions]}, "
      f"{len(waters)} crystal waters ({', '.join(w for w, _ in waters)})")
