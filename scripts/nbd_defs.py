"""Shared definitions for the Hsc70 NBD ATP-vs-ADP analysis (see docs/ANALYSIS_PLAN.md).

Residue numbers are Hsc70 numbering; the simulated construct is residues 4-381 (1KAX numbering, kept by the build).
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATES = ("atp", "adp")
RES = np.arange(4, 382)                      # simulated residues

# Subdomains (IA 1-39, 116-188, 361-381; IB 40-115; IIA 189-228, 307-360; IIB 229-306)
SUBDOMAINS = {
    "IA": list(range(4, 40)) + list(range(116, 189)) + list(range(361, 382)),
    "IB": list(range(40, 116)),
    "IIA": list(range(189, 229)) + list(range(307, 361)),
    "IIB": list(range(229, 307)),
}

# NBD residues within 4 Å of the linker / SBDbeta / SBDalpha in ATP-state DnaK (4B9Q), mapped to Hsc70 numbering
# (388-390 lie beyond the construct and are left out)
INTERFACES = {
    "linker": [148, 149, 151, 174, 177, 181, 216, 217, 218, 219, 222, 376, 379],
    "SBDb": [77, 100, 102, 104, 113, 117, 151, 152, 155, 156, 159, 171, 172, 221, 222, 319, 323],
    "SBDa": [46, 47, 48, 49, 104, 122, 125, 160, 163, 164],
}
# Two walls of the linker-binding crevice (IA side vs IIA side), for the crevice-width observable
CREVICE_IA = [148, 149, 151, 174, 177, 181, 376, 379]
CREVICE_IIA = [216, 217, 218, 219, 222]
IIA_STRAND = [216, 217, 218, 219]            # Gołaś 2015: proposed to be immobilised by ATP (K214-V218 region)
PLOOP = list(range(10, 16))                  # phosphate-binding loop compared with ADP-Pi 3HSC

# Protein ligands of the active-site K+ sites in 1KAX (waters are not in the trajectory)
K_SITES = {
    "K1": [(10, "OD1"), (10, "OD2"), (15, "O")],                                   # K490; also beta-phosphate O
    "K2": [(199, "OD1"), (199, "OD2"), (204, "O"), (204, "OG1"), (206, "OD1"), (206, "OD2")],  # K491; also gamma-P O
    "K3": [(149, "OH"), (204, "OG1"), (206, "OD1"), (206, "OD2")],                 # K492, omitted from the model
}
K_CUT = 3.3      # Å, K+ ligand contact
MG_CUT = 2.5     # Å, Mg2+ inner-shell contact

MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "P": 30.974, "H": 1.008}


def rep_dir(state, rep):
    return os.path.join(ROOT, "analysis", f"{state}_rep{rep}")


def read_pdb(path, chain=None, altloc=" A"):
    """Minimal PDB reader -> list of (resname, resid, name, xyz); first altloc only."""
    out = []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")) and line[16] in altloc and (chain is None or line[21] == chain):
            out.append((line[17:20].strip(), int(line[22:26]), line[12:16].strip(),
                        np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    return out


def pick(atoms, resid, name):
    for a in atoms:
        if a[1] == resid and a[2] == name:
            return a[3]
    raise KeyError(f"{resid}:{name}")


def kabsch(mobile, ref):
    """Rotation R and translation so that (mobile - cm) @ R + cr best fits ref; returns (R, cm, cr)."""
    cm, cr = mobile.mean(0), ref.mean(0)
    h = (mobile - cm).T @ (ref - cr)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt, cm, cr


def fit_apply(x, rot):
    R, cm, cr = rot
    return (x - cm) @ R + cr


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def angle(a, b, c):
    v1, v2 = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))
