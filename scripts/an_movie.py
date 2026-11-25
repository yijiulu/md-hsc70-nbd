#!/usr/bin/env python3
"""Interactive trajectory movie: ATP and ADP side by side (3Dmol.js page built from scripts/movie_template.html).

Usage: an_movie.py [--rep 1] [--block 1.0]
Per state: protein backbone (N CA C O), Trp90 and Arg261 side chains, nucleotide, Mg2+ and the two K+. Every frame is fitted
on the IA+IIA core CA onto 1KAX (so IB and IIB visibly swing over a fixed base), then averaged in non-overlapping blocks
(default 1 ns): thermal jitter otherwise hides the few-degree domain motion. Coordinates are stored as int16 hundredths
of an Å around the 1KAX core centre. Needs an_prep.sh and an_rep.py output (whole.xtc, top.pdb, timeseries.csv).
Writes analysis/movie/nbd_movie.html.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors

sys.path.insert(0, os.path.dirname(__file__))
from nbd_defs import ROOT, STATES, SUBDOMAINS, CREVICE_IA, CREVICE_IIA, read_pdb, pick, kabsch, fit_apply  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rep", type=int, default=1)
ap.add_argument("--block", type=float, default=1.0, help="averaging block, ns")
a = ap.parse_args()
OUT = f"{ROOT}/analysis/movie"
os.makedirs(OUT, exist_ok=True)

# GROMACS/Amber residue names -> PDB names (termini, protonation and disulfide variants)
RESNAME = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}
SIDE = (90, 261)                                    # Trp90 and Arg261 (the Woo 2009 contact) shown as sticks
CORE = [r for k in ("IA", "IIA") for r in SUBDOMAINS[k]]

kax = read_pdb(f"{ROOT}/structures/1KAX.pdb", chain="A")
core_ref = np.array([pick(kax, r, "CA") for r in CORE])
origin = core_ref.mean(0)
# canonical front view in the default camera: lobe I -> lobe II along +x, base (IA+IIA) -> top (IB+IIB) along +y
com = {k: np.array([pick(kax, r, "CA") for r in v]).mean(0) for k, v in SUBDOMAINS.items()}
e1 = (com["IIA"] + com["IIB"]) - (com["IA"] + com["IB"])
e1 /= np.linalg.norm(e1)
e2 = (com["IB"] + com["IIB"]) - (com["IA"] + com["IIA"])
e2 -= (e2 @ e1) * e1
e2 /= np.linalg.norm(e2)
E = np.stack([e1, e2, np.cross(e1, e2)], 1)
ss = "".join(l for l in open(f"{ROOT}/structures/1KAX.pdb") if l.startswith(("HELIX", "SHEET")) and
             (l[19] == "A" if l.startswith("HELIX") else l[21] == "A"))


def pdb_line(i, name, resname, resid, xyz, elem, het=False):
    nm = name if len(name) == 4 else " " + name
    return (f"{'HETATM' if het else 'ATOM  '}{i:5d} {nm:<4s} {resname:>3s} A{resid:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2s}\n")


def state_data(s):
    d = f"{ROOT}/analysis/{s}_rep{a.rep}"
    u = mda.Universe(f"{d}/top.pdb", f"{d}/whole.xtc")
    bb = u.select_atoms("resid 4:381 and name N CA C O OC1")
    side = u.select_atoms(f"resid {' '.join(map(str, SIDE))} and not backbone and not name H* OC2")
    lig = u.select_atoms("resname LIG and not name H*")
    ions = u.select_atoms("resname MGB KB")
    main = bb | side | lig                          # union is sorted by index, i.e. residue order
    allsel = main | ions
    pheavy = u.select_atoms("resid 4:381 and not name H*")
    ligall = u.select_atoms("resname LIG")
    pb = u.select_atoms("resname LIG and name PB")[0].index
    core = u.select_atoms("name CA and resid " + " ".join(map(str, CORE)))
    assert len(core) == len(core_ref)

    nblk = int(round(u.trajectory[-1].time / 1000 / a.block))
    acc = np.zeros((nblk, len(allsel), 3))
    cnt = np.zeros(nblk)
    for ts in u.trajectory:
        b = min(int(ts.time / 1000 / a.block), nblk - 1)
        x = u.atoms.positions.astype(np.float64)
        v = x[ligall.indices].mean(0) - x[pheavy.indices].mean(0)
        x[ligall.indices] += minimize_vectors(v[None].astype(np.float32), ts.dimensions)[0] - v
        for i in ions.indices:
            v = x[i] - x[pb]
            x[i] += minimize_vectors(v[None].astype(np.float32), ts.dimensions)[0] - v
        rot = kabsch(x[core.indices], core_ref)
        acc[b] += fit_apply(x[allsel.indices], rot)
        cnt[b] += 1
    assert (allsel.indices[:len(main)] == main.indices).all()
    xyz = (acc / cnt[:, None, None] - origin) @ E

    lines, ion_lines = [], []
    for k, at in enumerate(main):
        rn = at.resname[-3:] if at.resname.startswith(("N", "C")) and len(at.resname) == 4 else at.resname
        rn = "ATP" if at.resname == "LIG" and s == "atp" else "ADP" if at.resname == "LIG" else RESNAME.get(rn, rn)
        name = "O" if at.name == "OC1" else at.name.replace("*", "'")
        lines.append(pdb_line(k + 1, name, rn, at.resid, xyz[0, k], name[0], at.resname == "LIG"))
    for k, at in enumerate(ions):
        el = "MG" if at.resname == "MGB" else "K"
        ion_lines.append(pdb_line(k + 1, el, el, at.resid, xyz[0, len(main) + k], el, True))

    t = pd.read_csv(f"{d}/timeseries.csv")
    blk = np.minimum((t.time_ns / a.block).astype(int), nblk - 1)
    m = t.groupby(blk)[["delta", "w90_r261", "crevice"]].mean()
    q = np.round(xyz * 100).astype("<i2")
    assert np.abs(xyz).max() < 327
    return {"pdb": ss + "".join(lines) + "END\n", "ions_pdb": "".join(ion_lines) + "END\n",
            "n_main": len(main), "n_ions": len(ions), "xyz": base64.b64encode(q.tobytes()).decode(),
            "delta": m.delta.round(2).tolist(), "w90": m.w90_r261.round(2).tolist(), "crev": m.crevice.round(2).tolist()}, nblk


data = {"rep": a.rep, "block_ns": a.block, "skip_ns": 20, "states": {},
        "sub": {k: [[int(g[0]), int(g[-1])] for g in np.split(np.array(v), np.where(np.diff(v) != 1)[0] + 1)]
                for k, v in SUBDOMAINS.items()},
        "crevice": CREVICE_IA + CREVICE_IIA, "side": list(SIDE)}
for s in STATES:
    data["states"][s], nblk = state_data(s)
    print(s, data["states"][s]["n_main"], "atoms +", data["states"][s]["n_ions"], "ions,", nblk, "frames")
data["t"] = [round((i + 0.5) * a.block, 3) for i in range(nblk)]
data["ghost_pdb"] = ss + "".join(pdb_line(k + 1, "CA", rn, r, (xyz - origin) @ E, "C")
                                 for k, (rn, r, n, xyz) in enumerate(kax) if n == "CA" and 4 <= r <= 381) + "END\n"

html = open(f"{ROOT}/scripts/movie_template.html").read()
assert html.count("/*MOVIE_DATA*/null") == 1
html = html.replace("/*MOVIE_DATA*/null", json.dumps(data, separators=(",", ":")))
open(f"{OUT}/nbd_movie.html", "w").write(html)
print(f"{OUT}/nbd_movie.html  {os.path.getsize(f'{OUT}/nbd_movie.html') / 1e6:.1f} MB")
