"""Rotation of lobe II (IIA+IIB) relative to lobe I (IA+IB), measured from 1KAX and projected onto the
linker/SBD-docked ATP-state axis of DnaK (mean of 4B9Q, 4JNE, 7KRT, 7KO2, 5NRO). DnaK chains are mapped into
Hsc70 numbering via global BLOSUM62 alignment with 1KAX (51% identity).

Reference values (deg along the docked-state axis): 3HSC -0.9, 7RAX (NBD + docked linker, ATP) +11.6,
7KRU +14.6, SBD-docked states +22.6 to +24.1.

Usage as a standalone check:
  python scripts/lobe_rotation.py model.pdb [chain]
"""
import functools
import os
import sys

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import index_to_one, three_to_index
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
from nbd_defs import ROOT, SUBDOMAINS  # noqa: E402

DNAK = f"{ROOT}/structures/dnak"
DOCKED = ("4B9Q", "4JNE", "7KRT", "7KO2", "5NRO")
LOBE1 = SUBDOMAINS["IA"] + SUBDOMAINS["IB"]
LOBE2 = SUBDOMAINS["IIA"] + SUBDOMAINS["IIB"]
_P = PDBParser(QUIET=True)


def chain_res(path, ch=None):
    s = _P.get_structure("x", path)[0]
    ch = ch or [c.id for c in s if len([r for r in c if r.id[0] == " "]) > 300][0]
    return [r for r in s[ch] if r.id[0] == " " and "CA" in r]


def _seq(res):
    return "".join("M" if r.get_resname() == "MSE" else index_to_one(three_to_index(r.get_resname())) for r in res)


@functools.lru_cache(maxsize=1)
def _kax_ref():
    kax = chain_res(f"{ROOT}/structures/1KAX.pdb", "A")
    ref = {r.id[1]: r["CA"].coord.astype(float) for r in kax}
    al = Align.PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -10, -0.5
    return kax, ref, al


def hsc70_map(res):
    """Map DnaK/Hsp70 NBD residues -> {Hsc70 resid: residue} via global BLOSUM62 alignment with 1KAX 4-381."""
    kax, _, al = _kax_ref()
    a = al.align(_seq(kax), _seq(res))[0]
    m = {}
    for (i0, i1), (j0, _) in zip(*a.aligned):
        for k in range(i1 - i0):
            m[kax[i0 + k].id[1]] = res[j0 + k]
    return m


def fit(mob, tgt):
    """Row-vector rotation R with (mob - <mob>) @ R + <tgt> ~ tgt."""
    U, _, Vt = np.linalg.svd((mob - mob.mean(0)).T @ (tgt - tgt.mean(0)))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def rotvec(x):
    """Return axis-angle rotation vector (deg, in 1KAX frame) taking 1KAX lobe II onto x's lobe II after lobe I fit."""
    _, ref, _ = _kax_ref()
    c1, c2 = [i for i in LOBE1 if i in x], [i for i in LOBE2 if i in x]
    A1, B1 = np.array([x[i] for i in c1]), np.array([ref[i] for i in c1])
    y2 = (np.array([x[i] for i in c2]) - A1.mean(0)) @ fit(A1, B1) + B1.mean(0)
    return np.degrees(Rotation.from_matrix(fit(np.array([ref[i] for i in c2]), y2).T).as_rotvec())


def dnak_ca(code):
    m = hsc70_map([r for r in chain_res(f"{DNAK}/{code}.pdb") if r.id[1] <= 383])
    return {i: r["CA"].coord.astype(float) for i, r in m.items()}


@functools.lru_cache(maxsize=1)
def docked_axis():
    v = np.mean([rotvec(dnak_ca(c)) for c in DOCKED], 0)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    u = docked_axis()
    x = {r.id[1]: r["CA"].coord.astype(float) for r in chain_res(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)}
    v = rotvec(x)
    print(f"{sys.argv[1]}: lobe II vs lobe I rotation from 1KAX |{np.linalg.norm(v):.1f}| deg, along the docked axis "
          f"{v @ u:+.1f} deg (7RAX +11.6, docked +22.6..+24.1)")
