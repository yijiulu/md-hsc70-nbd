#!/usr/bin/env python3
"""Per-frame observables for one replica (run scripts/an_prep.sh first).

Usage: an_rep.py atp|adp REP
Reads analysis/<state>_rep<REP>/{top.pdb, whole.xtc}; writes in the same folder:
  timeseries.csv   one row per frame (columns described in COLUMNS below)
  ca.npy           CA coordinates (Å, float32, frames x residues 4-381 x 3) after a CA fit onto 1KAX, so that both
                   states share one reference frame
  sites.txt        which KB atom is K1/K2, the phosphate oxygens used for Mg2+/K+ contacts, reference values from 1KAX
"""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import MDAnalysis as mda  # noqa: E402
from MDAnalysis.lib.distances import minimize_vectors  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, __import__("os").path.dirname(__file__))
from nbd_defs import (RES, SUBDOMAINS, CREVICE_IA, CREVICE_IIA, PLOOP, K_SITES, K_CUT, MG_CUT, MASS, ROOT,  # noqa: E402
                      rep_dir, read_pdb, pick, kabsch, fit_apply, rmsd, angle, dihedral)

COLUMNS = {
    "time_ns": "simulation time",
    "rmsd_ca": "CA RMSD from 1KAX after a CA fit, Å",
    "delta": "domain angle IB-(IA+IIA)-IIB between heavy-atom centres of mass (Woo 2009), degrees",
    "tau": "dihedral IB-IA-IIA-IIB between centres of mass (twist), degrees",
    "d_ib_iib": "IB-IIB centre-of-mass distance (cleft width), Å",
    "th_ib/th_iib": "lobe angles IB-IA-IIA and IA-IIA-IIB: each lobe's own rotation about the IA-IIA base, degrees "
                    "(delta and tau share the IB and IIB centres, so they are partly coupled by construction)",
    "w90_r261": "min distance Trp90 HE1 - Arg261 NH1/NH2 (Woo 2009 definition), Å",
    "w90ring_r261cz": "Trp90 indole centroid - Arg261 CZ, Å",
    "crevice": "distance between CA centroids of the IA and IIA walls of the linker crevice, Å",
    "mg_oa/mg_ob/mg_og": "non-bridging alpha/beta/gamma phosphate O within 2.5 Å of Mg2+",
    "mg_prot": "protein O/N within 2.5 Å of Mg2+",
    "k1_prot/k2_prot": "1KAX protein ligand atoms of the site within 3.3 Å of the K+",
    "k1_nuc/k2_nuc": "non-bridging phosphate O within 3.3 Å of the K+",
    "k1_d/k2_d": "K+ to the centroid of its site's protein ligand atoms, Å",
    "nuc_rmsd": "nucleotide heavy-atom RMSD from frame 0 after fitting pocket CA (within 10 Å of the nucleotide), Å",
    "ploop_3hsc/ploop_1kax": "P-loop (10-15) backbone RMSD after an IA-core CA fit onto 3HSC / 1KAX, Å",
    "pbc_min": "minimum protein heavy-atom distance to its periodic images (every 1 ns), Å",
}

state, rep = sys.argv[1], int(sys.argv[2])
D = rep_dir(state, rep)
u = mda.Universe(f"{D}/top.pdb", f"{D}/whole.xtc")
nf = len(u.trajectory)
idx = {(a.resid, a.name): a.index for a in u.atoms}
prot = u.select_atoms("resid 4:381")
lig, mg, kb = u.select_atoms("resname LIG"), u.select_atoms("resname MGB"), u.select_atoms("resname KB")
assert len(mg) == 1 and len(kb) == 2 and len(lig) in (39, 43), (len(mg), len(kb), len(lig))
ca = np.array([idx[(r, "CA")] for r in RES])
heavy = lambda ag: ag[[not n.startswith("H") for n in ag.names]]  # noqa: E731
pheavy, lheavy = heavy(prot), heavy(lig)
mass = np.array([MASS.get(n[0], 0.0) for n in u.atoms.names])   # only heavy protein atoms are used
sub = {k: heavy(u.select_atoms("resid " + " ".join(map(str, v)))).indices for k, v in SUBDOMAINS.items()}
sub["IA+IIA"] = np.concatenate([sub["IA"], sub["IIA"]])
w90_he1 = idx[(90, "HE1")]
r261_nh = [idx[(261, "NH1")], idx[(261, "NH2")]]
w90_ring = [idx[(90, n)] for n in ("CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ3", "CH2", "CZ2")]
r261_cz = idx[(261, "CZ")]
crev_a = np.array([idx[(r, "CA")] for r in CREVICE_IA])
crev_b = np.array([idx[(r, "CA")] for r in CREVICE_IIA])

# References: 1KAX CA (common fit frame for both states) and the P-loop vs 1KAX / 3HSC (ADP-Pi)
kax = read_pdb(f"{ROOT}/structures/1KAX.pdb", chain="A")
hsc = read_pdb(f"{ROOT}/structures/3HSC.pdb", chain="A")
ref_ca = np.array([pick(kax, r, "CA") for r in RES])
ia_core = [r for r in SUBDOMAINS["IA"] if not 8 <= r <= 18]
ia_core_i = np.array([idx[(r, "CA")] for r in ia_core])
pl_names = [(r, n) for r in PLOOP for n in ("N", "CA", "C", "O")]
pl_i = np.array([idx[k] for k in pl_names])
pl_ref = {}
for tag, s in (("3hsc", hsc), ("1kax", kax)):
    pl_ref[tag] = (np.array([pick(s, r, "CA") for r in ia_core]), np.array([pick(s, r, n) for r, n in pl_names]))
rot = kabsch(pl_ref["1kax"][0], pl_ref["3hsc"][0])
ploop_kax_vs_hsc = rmsd(fit_apply(pl_ref["1kax"][1], rot), pl_ref["3hsc"][1])

# Frame 0: phosphate oxygens, K site assignment, nucleotide pocket
u.trajectory[0]
x0 = u.atoms.positions.copy()


def put_near(x, box):
    """Move LIG (as a whole) next to the protein and each ion next to LIG PB, by whole box vectors."""
    pc = x[pheavy.indices].mean(0)
    v = x[lig.indices].mean(0) - pc
    x[lig.indices] += minimize_vectors(v[None].astype(np.float32), box)[0] - v
    pb = x[idx[(lig.resids[0], "PB")]]
    for i in (mg.indices[0], kb.indices[0], kb.indices[1]):
        v = x[i] - pb
        x[i] += minimize_vectors(v[None].astype(np.float32), box)[0] - v
    return x


x0 = put_near(x0, u.trajectory.ts.dimensions)
lp = [i for i in lig.indices if u.atoms.names[i].startswith("P")]
term_o = {}                                   # non-bridging phosphate O, by phosphate (A/B/G from the P name)
for i in lig.indices:
    n = u.atoms.names[i]
    if not n.startswith("O"):
        continue
    near_p = [j for j in lp if np.linalg.norm(x0[i] - x0[j]) < 1.8]
    near_c = [j for j in lig.indices if u.atoms.names[j].startswith("C") and np.linalg.norm(x0[i] - x0[j]) < 1.7]
    if len(near_p) == 1 and not near_c:
        term_o.setdefault(u.atoms.names[near_p[0]][1], []).append(i)
all_term_o = np.array(sorted(i for v in term_o.values() for i in v))
site_atoms = {k: np.array([idx[a] for a in v]) for k, v in K_SITES.items() if k != "K3"}
kmap = {}
for k, ai in site_atoms.items():
    d = [np.linalg.norm(x0[i] - x0[ai].mean(0)) for i in kb.indices]
    kmap[k] = kb.indices[int(np.argmin(d))]
assert kmap["K1"] != kmap["K2"], "both KB atoms closest to one site"
mg_i = mg.indices[0]
po_i = heavy(prot).indices[[n[0] in "ON" for n in heavy(prot).names]]
pocket = np.array([i for i in ca if np.min(np.linalg.norm(x0[lheavy.indices] - x0[i], axis=1)) < 10])
pocket0, lig0 = x0[pocket], x0[lheavy.indices]

with open(f"{D}/sites.txt", "w") as f:
    f.write(f"{state} rep{rep}: {nf} frames\n")
    for k, i in kmap.items():
        f.write(f"{k} = atom {i + 1} (KB resid {u.atoms.resids[i]}); protein ligands {K_SITES[k]}\n")
    f.write("non-bridging phosphate O: " + "; ".join(f"{p}: {[u.atoms.names[i] for i in v]}" for p, v in term_o.items()) + "\n")
    f.write(f"nucleotide pocket: {len(pocket)} CA within 10 Å\n")
    f.write(f"reference: P-loop backbone RMSD 1KAX vs 3HSC after IA-core fit = {ploop_kax_vs_hsc:.2f} Å\n")
    for c, d in COLUMNS.items():
        f.write(f"  {c}: {d}\n")


def com(x, ii):
    return (x[ii] * mass[ii, None]).sum(0) / mass[ii].sum()


def pbc_min(x, tric):
    h = x[pheavy.indices]
    tree = cKDTree(h)
    best = np.inf
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                if a == b == c == 0:
                    continue
                d, _ = tree.query(h + a * tric[0] + b * tric[1] + c * tric[2], distance_upper_bound=min(best, 40.0))
                best = min(best, d.min())
    return best


rows, cas = [], np.empty((nf, len(RES), 3), np.float32)
for f, ts in enumerate(u.trajectory):
    x = put_near(u.atoms.positions.copy(), ts.dimensions)
    r = {"time_ns": ts.time / 1000}
    R = kabsch(x[ca], ref_ca)
    xf = fit_apply(x, R)
    cas[f] = xf[ca]
    r["rmsd_ca"] = rmsd(xf[ca], ref_ca)
    c = {k: com(x, v) for k, v in sub.items()}
    r["delta"] = angle(c["IB"], c["IA+IIA"], c["IIB"])
    r["tau"] = dihedral(c["IB"], c["IA"], c["IIA"], c["IIB"])
    r["d_ib_iib"] = float(np.linalg.norm(c["IB"] - c["IIB"]))
    r["th_ib"] = angle(c["IB"], c["IA"], c["IIA"])
    r["th_iib"] = angle(c["IA"], c["IIA"], c["IIB"])
    r["w90_r261"] = float(np.linalg.norm(x[r261_nh] - x[w90_he1], axis=1).min())
    r["w90ring_r261cz"] = float(np.linalg.norm(x[w90_ring].mean(0) - x[r261_cz]))
    r["crevice"] = float(np.linalg.norm(x[crev_a].mean(0) - x[crev_b].mean(0)))
    dmg = np.linalg.norm(x - x[mg_i], axis=1)
    for p in ("A", "B", "G"):
        r[f"mg_o{p.lower()}"] = int((dmg[term_o.get(p, [])] < MG_CUT).sum())
    r["mg_prot"] = int((dmg[po_i] < MG_CUT).sum())
    for k, i in kmap.items():
        dk = np.linalg.norm(x - x[i], axis=1)
        r[f"{k.lower()}_prot"] = int((dk[site_atoms[k]] < K_CUT).sum())
        r[f"{k.lower()}_nuc"] = int((dk[all_term_o] < K_CUT).sum())
        r[f"{k.lower()}_d"] = float(np.linalg.norm(x[i] - x[site_atoms[k]].mean(0)))
    Rp = kabsch(x[pocket], pocket0)
    r["nuc_rmsd"] = rmsd(fit_apply(x[lheavy.indices], Rp), lig0)
    for tag, (ref_core, ref_pl) in pl_ref.items():
        r[f"ploop_{tag}"] = rmsd(fit_apply(x[pl_i], kabsch(x[ia_core_i], ref_core)), ref_pl)
    r["pbc_min"] = pbc_min(x, ts.triclinic_dimensions) if f % 100 == 0 else np.nan
    rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(f"{D}/timeseries.csv", index=False, float_format="%.4g")
np.save(f"{D}/ca.npy", cas)
print(f"{state} rep{rep}: {nf} frames to {df.time_ns.iloc[-1]:.1f} ns -> {D}/timeseries.csv, ca.npy")
