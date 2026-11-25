# Analysis Plan: Nucleotide-Dependent Allostery in Hsc70 NBD

## 1. Research Question & Objective Interface Definitions

The central mechanistic question is whether ATP binding (relative to ADP) induces conformational or dynamical changes that propagate beyond the nucleotide pocket to the NBD surfaces responsible for docking the interdomain linker, $\mathrm{SBD\beta}$, and $\mathrm{SBD\alpha}$.

Docking interfaces are defined objectively from the full-length ATP-bound *E. coli* DnaK structure (`4B9Q`) by selecting all NBD residues with any heavy atom within `4.0 Å` of the linker or SBD, then mapping them onto bovine/human Hsc70 numbering via global sequence alignment (`54%` NBD sequence identity; residues `388–390` lie outside the `4–381` isolated-NBD construct):

| Interface | Hsc70 Residues |
| :--- | :--- |
| Linker-binding crevice (`IA`/`IIA`) | A148, Y149, N151, N174, T177, I181, I216, F217, E218, V219, T222, Q376, I379 |
| $\mathrm{SBD\beta}$ docking face | R77, R100, K102, Q104, S113, E117, N151, D152, R155, Q156, K159, R171, I172, S221, T222, K319, D323 |
| $\mathrm{SBD\alpha}$ (lid) docking face | D46, T47, E48, R49, Q104, M122, T125, D160, T163, I164 |

Subdomain boundaries: `IA` (`4–39`, `116–188`, `361–381`), `IB` (`40–115`), `IIA` (`189–228`, `307–360`), `IIB` (`229–306`).

## 2. Pre-Registered Hypotheses & Observables

- **H1 — Cleft closure & IB–IIB coupling** (benchmarking Woo *et al.*, 2009):
  - Domain angle $\delta$ (`IB–(IA+IIA)–IIB` heavy-atom COM angle) and its SD, twist dihedral $\tau$ (`IB–IA–IIA–IIB`), `Trp90 HE1 – Arg261 NH1/NH2` minimum distance (Woo contact definition; not a donor–acceptor H-bond), and `Trp90` indole centroid to `Arg261 CZ` distance.
- **H2 — Propagation to allosteric docking surfaces** (benchmarking Gołaś *et al.*, 2015):
  - Subdomain-fitted per-residue $\mathrm{C\alpha}$ RMSF and $\Delta\mathrm{RMSF}(\mathrm{ATP} - \mathrm{ADP})$, excess mean-structure shift (`between-state` minus `within-state` replica noise), IIA strand `216–219` flexibility, linker-crevice width, and **lobe II vs lobe I rotation projected onto the 5-structure DnaK docked axis** (`scripts/lobe_rotation.py`; reference values: `3HSC -0.9°`, `1KAX 0.0°`, `7RAX +11.6°`, `7KRU +14.6°`, SBD-docked ensemble `+22.6°` to `+24.1°`).
  - Distance-matched null test: because perturbation decays radially from the $\gamma$-phosphate, each interface set is tested against `5,000` random non-interface residue sets stratified into `5 Å` radial shells from `ATP:PG`.
- **H3 — Lobe-angle correlation**:
  - Linearly detrended Pearson correlation between independent lobe angles $\theta_{\mathrm{IB}}$ (`IB–IA–IIA`) and $\theta_{\mathrm{IIB}}$ (`IA–IIA–IIB`). Raw $(\delta, \tau)$ correlation is reported for completeness but not used for H3 because $\delta$ and $\tau$ share `IB` and `IIB` centres by construction.
- **H4 — Optional comparison with NMR chemical shift perturbations (CSP)**:
  - When an external two-column TSV (`resid<TAB>csp_ppm`) is supplied via `--csp`, computes both raw Spearman $\rho$ and partial Spearman correlation controlling for distance from `ATP:PG` to separate genuine structural coupling from shared radial distance decay.

## 3. Quality Control & Statistical Criteria

- **Mandatory QC** (`qc.csv`, `an_snap.py`): $\mathrm{Mg^{2+}}$ coordination (inner-shell waters + phosphate oxygens via full-system checkpoint snapshots), `K1` and `K2` active-site retention, bulk $\mathrm{K^+}$ entry at site 3, nucleotide RMSD, P-loop (`10–15`) RMSD against `1KAX` and `3HSC`, and minimum periodic-image distance (`pbc_min`).
- **Replica-level inference**: With small replica counts ($n = 2$–$3$ per state), classical $p < 0.05$ thresholds are underpowered; differences are called only when ATP and ADP replica distributions separate completely (all cross-state pairs have identical sign), with Welch $p$-values reported purely as reference metrics.

## 4. Extension to NBD + Interdomain Linker & Linker-Free Control

1. **NBD + linker (`work_lk_atp`, `work_lk_adp`, `2 × 100 ns`)**: Built on `7RAX` lobe/linker geometry (`+11.1°` initial rotation after restrained OpenMM relaxation, preserving `16/19` `7RAX` linker–NBD contacts). Pre-registered observables (`scripts/an_linker.py`): native contact fraction $Q_{\mathrm{linker}}$ (`4.5 Å` heavy-atom cutoff), slot-aligned linker backbone RMSD, `L391–L394` side-chain distances to slot residues, and lobe rotation. Pre-registered release criterion: $Q_{\mathrm{linker}} < 0.3$ sustained over a continuous `5 ns` window.
2. **Linker-stripped rotated control (`work_nl_atp`, `work_nl_adp`)**: Removes residues `382–395` from `hsc70_nbd_linker.pdb` while keeping identical `+12°` NBD, nucleotide, ion, and crystal-water coordinates to test whether persistent rotation in the last `20 ns` requires the docked linker or reflects slow relaxation on the `100 ns` timescale.
