#!/usr/bin/env python3
"""Lobe II vs lobe I rotation (lobe_rotation.py, projected on the DnaK docked-state direction) for every analysed
replica of the isolated NBD (atp, adp), the NBD + linker runs (lk_atp, lk_adp) and the linker-free control started from
the same rotated model (nl_atp, nl_adp; docs/ANALYSIS_PLAN.md section 11), from the CA coordinates that an_rep.py writes
(ca.npy, residues 4-381).

Usage: an_rotation.py
Writes analysis/<state>_rep<r>/rotation.csv (time_ns, rot_docked, delta) and analysis/summary/rotation_stats.csv:
per replica the mean and SD over the window fixed in docs/ANALYSIS_PLAN.md before the data were seen
(isolated NBD 20-150 ns, section 10 / an_compare.py --skip 20; NBD + linker and the control the last 20 ns,
an_linker.py and section 11).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import lobe_rotation as lr  # noqa: E402
from nbd_defs import ROOT  # noqa: E402

RES = np.arange(4, 382)
axis = lr.docked_axis()
rows = []
for d in sorted(glob.glob(f"{ROOT}/analysis/*_rep[0-9]")):
    state, rep = os.path.basename(d).rsplit("_rep", 1)
    if state not in ("atp", "adp", "lk_atp", "lk_adp", "nl_atp", "nl_adp") or not os.path.exists(f"{d}/ca.npy"):
        continue
    ca, ts = np.load(f"{d}/ca.npy"), pd.read_csv(f"{d}/timeseries.csv")
    assert ca.shape[1] == len(RES) and len(ca) == len(ts)
    rot = np.array([lr.rotvec(dict(zip(RES, f.astype(float)))) @ axis for f in ca])
    out = pd.DataFrame({"time_ns": ts.time_ns, "rot_docked": rot, "delta": ts.delta})
    out.to_csv(f"{d}/rotation.csv", index=False, float_format="%.4g")
    end = out.time_ns.max()
    w = out[out.time_ns >= 20] if state in ("atp", "adp") else out[out.time_ns > end - 20]
    rows.append({"state": state, "rep": int(rep), "window": f"{w.time_ns.min():.0f}-{end:.0f} ns",
                 "rot_mean": w.rot_docked.mean(), "rot_sd": w.rot_docked.std(), "rot_min": w.rot_docked.min(),
                 "rot_max": w.rot_docked.max(), "delta_mean": w.delta.mean(), "delta_sd": w.delta.std()})
s = pd.DataFrame(rows)
os.makedirs(f"{ROOT}/analysis/summary", exist_ok=True)
s.to_csv(f"{ROOT}/analysis/summary/rotation_stats.csv", index=False, float_format="%.3g")
print(s.round(2).to_string(index=False))
