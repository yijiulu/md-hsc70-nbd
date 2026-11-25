#!/usr/bin/env python3
"""Linker docking in the NBD + linker runs (work_lk_atp, work_lk_adp; model from build_linker_model.py).

Usage: an_linker.py STATE REP [--stride 10]       STATE = lk_atp | lk_adp; needs an_prep.sh output
       an_linker.py --summary                     table over every analysed replica -> analysis/summary/linker_stats.csv
Per frame (default every 10th saved frame = 100 ps), written to analysis/<STATE>_rep<REP>/linker.csv:
  q_linker     fraction of the reference linker-NBD residue contacts still formed (any heavy-atom pair within 4.5 A);
               reference = the residue pairs between linker 388-395 and NBD 4-381 in the minimised start
               (work_<STATE>/build/em.gro, 21 pairs; 16 of the 19 7RAX pairs are among them)
  n_contacts   linker-NBD residue pairs within 4.5 A, reference or not
  rmsd_lk      linker backbone (388-395, N CA C O) RMSD from the start after fitting the slot-wall CA, A
  d391..d394   minimum distance from each leucine side chain (CB CG CD1 CD2) to the slot residues, A
  rot_docked   lobe II vs lobe I rotation from 1KAX along the DnaK docked-state direction, deg (start +12)
Defined on 2026-10-29 before any production frame existed:
  released     q_linker < 0.3 throughout a 5 ns window; release time = start of the first such window
  summary      per replica: release time (or none), mean q_linker / rmsd_lk / rot_docked over the last 20 ns, and the
               change of rot_docked from the first to the last 10 ns. With 2 replicas per state the comparison is
               descriptive: a state difference is reported only when both ATP replicas lie on the same side of both ADP.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.lib.distances import capped_distance

sys.path.insert(0, os.path.dirname(__file__))
import lobe_rotation as lr  # noqa: E402
from nbd_defs import ROOT, INTERFACES, CREVICE_IA, CREVICE_IIA  # noqa: E402

LINKER = "resid 388:395"
NBD = "resid 4:381"
SLOT = INTERFACES["linker"]
CUT, Q_OFF, WIN_NS, LAST_NS = 4.5, 0.3, 5.0, 20.0

ap = argparse.ArgumentParser()
ap.add_argument("state", nargs="?")
ap.add_argument("rep", nargs="?", type=int)
ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--summary", action="store_true")
ap.add_argument("--top", help="override topology (testing)")
ap.add_argument("--traj", help="override trajectory (testing)")
a = ap.parse_args()


def heavy(u, sel):
    return u.select_atoms(f"({sel}) and not name H*")


def residue_pairs(lk, nbd, box=None):
    p = capped_distance(lk.positions, nbd.positions, CUT, box=box, return_distances=False)
    return {(int(lk.resids[i]), int(nbd.resids[j])) for i, j in p}


def release_time(t, q):
    """start of the first WIN_NS window in which q stays below Q_OFF (frames are evenly spaced)"""
    dt = np.median(np.diff(t))
    w = max(1, int(round(WIN_NS / dt)))
    low = (q < Q_OFF).astype(int)
    run = np.convolve(low, np.ones(w, int), "valid")
    hit = np.where(run == w)[0]
    return float(t[hit[0]]) if len(hit) else np.nan


if a.summary:
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/analysis/lk_*_rep*/linker.csv")):
        state, rep = os.path.basename(os.path.dirname(f)).rsplit("_rep", 1)
        d = pd.read_csv(f)
        end = d.time_ns.max()
        last, first = d[d.time_ns > end - LAST_NS], d[d.time_ns <= 10]
        rows.append({"state": state, "rep": int(rep), "length_ns": end, "release_ns": release_time(d.time_ns.values, d.q_linker.values),
                     "q_last20": last.q_linker.mean(), "rmsd_lk_last20": last.rmsd_lk.mean(), "n_contacts_last20": last.n_contacts.mean(),
                     "rot_first10": first.rot_docked.mean(), "rot_last20": last.rot_docked.mean(),
                     "rot_change": d[d.time_ns > end - 10].rot_docked.mean() - first.rot_docked.mean()})
    s = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/analysis/summary", exist_ok=True)
    s.to_csv(f"{ROOT}/analysis/summary/linker_stats.csv", index=False, float_format="%.3g")
    print(s.round(2).to_string(index=False))
    for c in ("q_last20", "rmsd_lk_last20", "rot_last20", "rot_change"):
        x, y = s[s.state == "lk_atp"][c].values, s[s.state == "lk_adp"][c].values
        if len(x) and len(y):
            sep = "ATP > ADP" if x.min() > y.max() else "ATP < ADP" if x.max() < y.min() else "overlapping"
            print(f"{c:16s} ATP {np.round(x, 2)}  ADP {np.round(y, 2)}  -> {sep}")
    sys.exit()

D = f"{ROOT}/analysis/{a.state}_rep{a.rep}"
u = mda.Universe(a.top or f"{D}/top.pdb", a.traj or f"{D}/whole.xtc")
ref = mda.Universe(f"{ROOT}/work_{a.state}/build/em.gro")
lk, nbd, rlk, rnbd = heavy(u, LINKER), heavy(u, NBD), heavy(ref, LINKER), heavy(ref, NBD)
REFPAIRS = residue_pairs(rlk, rnbd, ref.dimensions)
slot_ca = u.select_atoms("name CA and resid " + " ".join(map(str, CREVICE_IA + CREVICE_IIA)))
rslot_ca = ref.select_atoms("name CA and resid " + " ".join(map(str, CREVICE_IA + CREVICE_IIA)))
lk_bb, rlk_bb = u.select_atoms(f"{LINKER} and name N CA C O"), ref.select_atoms(f"{LINKER} and name N CA C O")
slot = heavy(u, "resid " + " ".join(map(str, SLOT)))
leu = {r: u.select_atoms(f"resid {r} and name CB CG CD1 CD2") for r in (391, 392, 393, 394)}
ca = u.select_atoms("name CA and resid 4:381")
assert len(slot_ca) == len(rslot_ca) and len(lk_bb) == len(rlk_bb) == 32 and len(ca) == 378
assert all(len(g) == 4 for g in leu.values())
axis = lr.docked_axis()
r0 = rslot_ca.positions - rslot_ca.positions.mean(0)
lk0 = rlk_bb.positions - rslot_ca.positions.mean(0)
print(f"{a.state} rep{a.rep}: {len(REFPAIRS)} reference linker-NBD residue pairs")

rows = []
for ts in u.trajectory[::a.stride]:
    pairs = residue_pairs(lk, nbd, ts.dimensions)
    c = slot_ca.positions.mean(0)
    R, _ = rotation_matrix(slot_ca.positions - c, r0)
    lkf = (lk_bb.positions - c) @ R.T
    row = {"time_ns": ts.time / 1000, "q_linker": len(pairs & REFPAIRS) / len(REFPAIRS), "n_contacts": len(pairs),
           "rmsd_lk": float(np.sqrt(((lkf - lk0) ** 2).sum(1).mean()))}
    for r, g in leu.items():
        row[f"d{r}"] = float(capped_distance(g.positions, slot.positions, 30.0, box=ts.dimensions)[1].min())
    row["rot_docked"] = float(lr.rotvec(dict(zip(ca.resids, ca.positions.astype(float)))) @ axis)
    rows.append(row)
df = pd.DataFrame(rows)
if not (a.top or a.traj):
    df.to_csv(f"{D}/linker.csv", index=False, float_format="%.4g")
print(df.describe().loc[["mean", "min", "max"]].round(2).to_string())
