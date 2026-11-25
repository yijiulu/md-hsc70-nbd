#!/usr/bin/env python3
"""Build Hsc70 NBD + interdomain linker (residues 4-395) in the linker-docked geometry of ATP-bound DnaK
crystal structure 7RAX (Wang et al. 2021), written in 1KAX conventions for prep_1kax.py.

Usage:
  python scripts/build_linker_model.py structures/hsc70_nbd_linker.pdb [--relax-ps 200] [--platform auto]
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))
import lobe_rotation as lr  # noqa: E402
from nbd_defs import ROOT, SUBDOMAINS  # noqa: E402


def select_openmm_platform(mm, choice: str):
    avail = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    order = ["CUDA", "OpenCL", "CPU"] if choice == "auto" else [choice]
    for name in order:
        if name in avail:
            props = {"Precision": "mixed"} if name in ("CUDA", "OpenCL") else {}
            return mm.Platform.getPlatformByName(name), props
    raise RuntimeError(f"Requested OpenMM platform '{choice}' unavailable (available: {avail}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="Output heavy-atom PDB path.")
    ap.add_argument("--relax-ps", type=float, default=200.0)
    ap.add_argument("--platform", default="auto", choices=["auto", "CUDA", "OpenCL", "CPU"])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log_fh = open(os.path.splitext(a.out)[0] + ".log", "w")

    def say(*x):
        print(*x)
        print(*x, file=log_fh)

    kax_pdb, rax_pdb = f"{ROOT}/structures/1KAX.pdb", f"{ROOT}/structures/dnak/7RAX.pdb"
    tail = {382: 384, 388: 386, 389: 387, 390: 388, 391: 389, 392: 390, 393: 391, 394: 392, 395: 393}
    hsc_tail = dict(zip(range(382, 396), "GLY ASP LYS SER GLU ASN VAL GLN ASP LEU LEU LEU LEU ASP".split()))
    insert = list(range(383, 388))
    hinges = list(range(186, 192)) + list(range(357, 365)) + list(range(378, 382))
    linker = list(range(388, 396))
    backbone = {"N", "CA", "C", "O", "OXT"}
    atp_fit = ["PG", "PB", "PA", "O3B", "O3A", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
               "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]

    def xyz_of(line):
        return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    def apply(x, T):
        R, c0, c1 = T
        return (x - c0) @ R + c1

    def pdb_line(rec, i, name, resn, resid, x, elem):
        nm = name if len(name) == 4 else " " + name
        return (f"{rec:<6s}{i:5d} {nm:<4s} {resn:>3s} A{resid:4d}    {x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00  0.00"
                f"          {elem:>2s}\n")

    prot, het = {}, []
    for l in open(kax_pdb):
        if not l.startswith(("ATOM", "HETATM")) or l[21] != "A":
            continue
        if l.startswith("ATOM"):
            if l[12:16].strip() != "OXT":
                prot.setdefault(int(l[22:26]), [l[17:20], {}])[1][l[12:16].strip()] = xyz_of(l)
        elif l[17:20].strip() != "CL":
            het.append(l)

    rax = lr.chain_res(rax_pdb, "A")
    amap = lr.hsc70_map([r for r in rax if r.id[1] <= 383])

    def subdomain_fit(k):
        rs = [i for i in SUBDOMAINS[k] if i in amap and i in prot]
        A = np.array([prot[i][1]["CA"] for i in rs])
        B = np.array([amap[i]["CA"].coord for i in rs])
        T = (lr.fit(A, B), A.mean(0), B.mean(0))
        say(f"{k} fit onto 7RAX: {len(rs)} CA, RMSD {np.sqrt(((apply(A, T) - B) ** 2).sum(1).mean()):.2f} A")
        return T

    T1, T2 = subdomain_fit("IA"), subdomain_fit("IIA")
    say(f"angle between lobe I and lobe II placements: "
        f"{np.degrees(np.arccos((np.trace(T1[0].T @ T2[0]) - 1) / 2)):.1f} deg")
    lobe1 = set(lr.LOBE1)
    model = {i: [rn, {n: apply(x, T1 if i in lobe1 else T2) for n, x in at.items()}] for i, (rn, at) in prot.items()}

    rres = {r.id[1]: r for r in rax}
    for h, d in tail.items():
        r, rn = rres[d], hsc_tail[h]
        if r.get_resname() == rn:
            names = [x.get_id() for x in r if x.element != "H"]
        else:
            names = [n for n in ("N", "CA", "C", "O", "CB") if n in r]
        if rn == "GLY":
            names = [n for n in names if n != "CB"]
        model[h] = [rn, {n: r[n].coord.astype(float) for n in names}]
        say(f"linker {rn}{h} <- 7RAX {r.get_resname()}{d}: {'full side chain' if r.get_resname() == rn else 'backbone + CB'}")
    say(f"S381 C - G382 N after placement: {np.linalg.norm(model[381][1]['C'] - model[382][1]['N']):.2f} A")

    kax_atp = {l[12:16].strip(): xyz_of(l) for l in het if l[17:20] == "ATP"}
    rax_atp = [r for r in lr._P.get_structure("r", rax_pdb)[0]["A"] if r.get_resname() == "ATP"][0]
    A = np.array([kax_atp[n] for n in atp_fit])
    B = np.array([rax_atp[n].coord for n in atp_fit])
    TN = (lr.fit(A, B), A.mean(0), B.mean(0))
    say(f"1KAX ATP onto 7RAX ATP ({len(atp_fit)} non-symmetric heavy atoms): RMSD {np.sqrt(((apply(A, TN) - B) ** 2).sum(1).mean()):.2f} A")
    het_new = [l[:30] + "{:8.3f}{:8.3f}{:8.3f}".format(*apply(xyz_of(l), TN)) + l[54:] for l in het]

    import openmm as mm
    import openmm.unit as unit
    from openmm.app import ForceField, HBonds, NoCutoff, Simulation
    from pdbfixer import PDBFixer

    tmp = os.path.splitext(a.out)[0] + "_tmp.pdb"
    with open(tmp, "w") as f:
        k = 0
        for i in sorted(model):
            rn, at = model[i]
            for n, x in at.items():
                k += 1
                f.write(pdb_line("ATOM", k, n, rn, i, x, n[0]))
        f.write("TER\nEND\n")

    fx = PDBFixer(filename=tmp)
    order = [int(r.id) for r in fx.topology.residues()]
    fx.missingResidues = {(0, order.index(388)): [hsc_tail[i] for i in insert]}
    fx.findMissingAtoms()
    fx.addMissingAtoms()
    fx.addMissingHydrogens(7.0)
    res_list = list(fx.topology.residues())
    resid = {r.index: 4 + j for j, r in enumerate(res_list)}
    assert resid[res_list[-1].index] == 395, resid[res_list[-1].index]

    atoms = list(fx.topology.atoms())
    pos = np.array(fx.positions.value_in_unit(unit.nanometer))
    heavy = np.array([at.element.symbol != "H" for at in atoms])
    rid = np.array([resid[at.residue.index] for at in atoms])
    bb = np.array([at.name in backbone for at in atoms])
    lk = np.isin(rid, linker) & heavy
    near = cKDTree(pos[lk]).query_ball_point(pos[heavy & (rid <= 381)], 0.6)
    shell = sorted({int(r) for r, hit in zip(rid[heavy & (rid <= 381)], near) if hit})
    say(f"side chains within 6 A of linker (free during relaxation): {shell}")
    free = np.isin(rid, hinges + [382] + insert) | (np.isin(rid, linker) & ~bb) | (np.isin(rid, shell) & ~bb)
    restr = heavy & ~free

    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(fx.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
    cf = mm.CustomExternalForce("0.5*kr*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    cf.addGlobalParameter("kr", 1000.0)
    for p in ("x0", "y0", "z0"):
        cf.addPerParticleParameter(p)
    for i in np.where(restr)[0]:
        cf.addParticle(int(i), pos[i].tolist())
    system.addForce(cf)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    plat, props = select_openmm_platform(mm, a.platform)
    sim = Simulation(fx.topology, system, integ, plat, props)
    sim.context.setPositions(fx.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    sim.minimizeEnergy(tolerance=10)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    sim.step(int(a.relax_ps / 0.002))
    sim.minimizeEnergy(tolerance=10)
    st = sim.context.getState(getEnergy=True, getPositions=True)
    say(f"relaxation ({plat.getName()}): {np.count_nonzero(restr)} restrained / {np.count_nonzero(heavy & free)} free heavy atoms, "
        f"{a.relax_ps:g} ps at 300 K; E {e0:.3g} -> {st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.4g} kJ/mol")
    xyz = np.array(st.getPositions().value_in_unit(unit.angstrom))

    H = np.where(heavy)[0]
    idx = {(rid[i], atoms[i].name): i for i in H}
    cn = [(r, np.linalg.norm(xyz[idx[(r, "C")]] - xyz[idx[(r + 1, "N")]])) for r in range(4, 395)]
    say(f"peptide C-N: max {max(d for _, d in cn):.2f} A")
    v = lr.rotvec({r: xyz[idx[(r, "CA")]] for r in range(4, 382)})
    say(f"lobe II vs lobe I rotation from 1KAX along docked axis: {v @ lr.docked_axis():+.1f} deg (7RAX +11.6)")

    with open(a.out, "w") as f:
        f.write("REMARK   Hsc70 NBD + linker (4-395) on 7RAX lobe/linker geometry; see build_linker_model.py\n")
        for k, i in enumerate(H, 1):
            at = atoms[i]
            f.write(pdb_line("ATOM", k, at.name, at.residue.name, rid[i], xyz[i], at.element.symbol))
        f.write("TER\n")
        f.writelines(het_new)
        f.write("END\n")
    os.remove(tmp)
    say(f"wrote {a.out}: {len(H)} protein heavy atoms (residues 4-395), {len(het_new)} hetero atoms")
    log_fh.close()


if __name__ == "__main__":
    main()
