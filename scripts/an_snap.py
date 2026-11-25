#!/usr/bin/env python3
"""Ion checks on the full-system checkpoint snapshots written by snap_sampler.sh.

Usage: an_snap.py            (all states and replicas found in analysis/snaps)
Writes analysis/snaps/snap_ions.csv, one row per snapshot:
  mg_cn, mg_wat        Mg2+ inner shell (O within 2.5 Å) and how many of those are water
  k1_cn, k2_cn         K+ shell (O within 3.3 Å), any molecule
  k1_wat, k2_wat       ... of which water
  site3_k              distance from the site-3 centre (Tyr149 OH, Thr204 OG1, Asp206 OD1/OD2) to the nearest K+, Å
  site3_k_id           that K+: KB (a modelled active-site K+) or K (bulk)
  site3_wat            water O within 3.3 Å of the site-3 centre
  pi_bulk_k            bulk K+ (not KB) within 3.3 Å of a phosphate O
"""
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import MDAnalysis as mda  # noqa: E402
from MDAnalysis.lib.distances import distance_array  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from nbd_defs import ROOT, K_SITES, K_CUT, MG_CUT, read_pdb, pick  # noqa: E402

SN = f"{ROOT}/analysis/snaps"
kax = read_pdb(f"{ROOT}/structures/1KAX.pdb", chain="A")
s3 = np.array([pick(kax, r, n) for r, n in K_SITES["K3"]])
ref_s3 = float(np.linalg.norm([a[3] for a in kax if a[0] == "K" and a[1] == 492][0] - s3.mean(0)))

rows = []
for state in ("atp", "adp"):
    top = f"{SN}/{state}_system.pdb"
    if not os.path.exists(top):
        continue
    u0 = mda.Universe(top)
    sel = lambda u, s: u.select_atoms(s)  # noqa: E731
    for rdir in sorted(glob.glob(f"{SN}/{state}_rep*")):
        rep = int(rdir.rsplit("rep", 1)[1])
        files = sorted(glob.glob(f"{rdir}/t*.xtc"))
        if not files:
            continue
        u = mda.Universe(top, files)
        mg, kb = sel(u, "resname MGB"), sel(u, "resname KB")
        kall, kbulk = sel(u, "resname KB K"), sel(u, "resname K")
        oxy = sel(u, "(not resname MGB KB K CL) and name O*")
        wat = sel(u, "resname SOL HOH CWAT and name OW")
        lig = sel(u, "resname LIG")
        po = lig[[n.startswith("O") and n[1:2].isdigit() and n[-1] in "ABG" for n in lig.names]]
        site = {k: sel(u, " or ".join(f"(resid {r} and name {n})" for r, n in v)) for k, v in K_SITES.items()}
        for ts in u.trajectory:
            box = ts.dimensions
            r = {"state": state, "rep": rep, "time_ns": ts.time / 1000}
            dmg = distance_array(mg.positions, oxy.positions, box)[0]
            r["mg_cn"] = int((dmg < MG_CUT).sum())
            r["mg_wat"] = int((distance_array(mg.positions, wat.positions, box)[0] < MG_CUT).sum())
            # which KB atom sits in which site: nearest to the site's ligand centroid (same rule as an_rep.py)
            for k in ("K1", "K2"):
                c = site[k].positions.mean(0)[None]   # site atoms are within one residue span, so no PBC split
                i = int(np.argmin(distance_array(c, kb.positions, box)[0]))
                ki = kb[i:i + 1]
                r[f"{k.lower()}_cn"] = int((distance_array(ki.positions, oxy.positions, box)[0] < K_CUT).sum())
                r[f"{k.lower()}_wat"] = int((distance_array(ki.positions, wat.positions, box)[0] < K_CUT).sum())
            c3 = site["K3"].positions.mean(0)[None]
            d3 = distance_array(c3, kall.positions, box)[0]
            r["site3_k"] = float(d3.min())
            r["site3_k_id"] = kall.resnames[int(np.argmin(d3))]
            r["site3_wat"] = int((distance_array(c3, wat.positions, box)[0] < K_CUT).sum())
            r["pi_bulk_k"] = int((distance_array(kbulk.positions, po.positions, box) < K_CUT).any(1).sum())
            rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(f"{SN}/snap_ions.csv", index=False, float_format="%.3g")
print(f"site-3 reference: K492 is {ref_s3:.2f} Å from the site-3 centre in 1KAX")
print(df.groupby(["state", "rep"]).agg(n=("time_ns", "size"), t_max=("time_ns", "max"),
                                       mg_cn=("mg_cn", "mean"), mg_wat=("mg_wat", "mean"),
                                       k1_cn=("k1_cn", "mean"), k2_cn=("k2_cn", "mean"),
                                       site3_k_min=("site3_k", "min"), site3_wat=("site3_wat", "mean"),
                                       pi_bulk_k=("pi_bulk_k", "max")).round(2).to_string())
