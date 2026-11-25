#!/usr/bin/env python3
"""ATP vs ADP across replicas: quality control and hypotheses H1-H4 (see docs/ANALYSIS_PLAN.md).

Usage:
  python scripts/an_compare.py [--skip 20] [--until NS] [--reps 1,2] [--csp path/to/csp.tsv]
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from nbd_defs import ROOT, STATES, RES, SUBDOMAINS, INTERFACES, IIA_STRAND, read_pdb, pick  # noqa: E402


def fit_batch(X, ref):
    """Least-squares fit of every frame of X (F, N, 3) onto ref (N, 3)."""
    Xc = X - X.mean(1, keepdims=True)
    rc = ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(np.einsum("fni,nj->fij", Xc, rc))
    D = np.ones((len(X), 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    return np.einsum("fni,fij->fnj", Xc, (U * D[:, None, :]) @ Vt) + ref.mean(0)


def detrend(t, y):
    return y - np.polyval(np.polyfit(t, y, 1), t)


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-replica ATP vs ADP QC, H1-H3 statistics, interface nulls, and optional NMR CSP correlation.")
    ap.add_argument("--skip", type=float, default=20.0, help="Equilibration burn-in cut in ns (default: 20.0).")
    ap.add_argument("--until", type=float, default=None, help="Upper window bound in ns (default: shortest replica).")
    ap.add_argument("--reps", default=None, help="Comma-separated replica indices (default: all analysed replicas).")
    ap.add_argument("--csp", default=None, help="Optional two-column TSV (resid<TAB>csp_ppm) for H4 NMR CSP comparison.")
    a = ap.parse_args()

    out_dir = f"{ROOT}/analysis/summary"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(1)

    use = {int(x) for x in a.reps.split(",")} if a.reps else None
    reps = {}
    for s in STATES:
        for d in sorted(glob.glob(f"{ROOT}/analysis/{s}_rep*")):
            if os.path.exists(f"{d}/timeseries.csv") and (use is None or int(d.rsplit("rep", 1)[1]) in use):
                reps[(s, int(d.rsplit("rep", 1)[1]))] = (pd.read_csv(f"{d}/timeseries.csv"), np.load(f"{d}/ca.npy", mmap_mode="r"))
    if not reps:
        sys.exit("No analysed replicas found (run scripts/an_prep.sh and scripts/an_rep.py first).")
    until = a.until or min(ts.time_ns.max() for ts, _ in reps.values())
    if until <= a.skip:
        sys.exit(f"Window empty: shortest replica ends at {until:.1f} ns, cut is {a.skip} ns.")

    window = f"{a.skip:g}-{until:.1f} ns"
    print(f"replicas: {sorted(reps)}; analysis window {window}")
    kax = read_pdb(f"{ROOT}/structures/1KAX.pdb", chain="A")
    ref_ca = np.array([pick(kax, r, "CA") for r in RES])
    ri = {r: i for i, r in enumerate(RES)}

    def win(ts):
        return (ts.time_ns >= a.skip) & (ts.time_ns <= until)

    # QC
    qc_rows = []
    for (s, r), (ts, _) in sorted(reps.items()):
        w = win(ts)
        mg_ok = (ts.mg_ob >= 1) & (ts.mg_prot == 0) & ((ts.mg_og >= 1) if s == "atp" else True)
        q = {"state": s, "rep": r, "t_end_ns": ts.time_ns.max(),
             "mg_expected_coord": mg_ok.mean(),
             "mg_alpha_contact": (ts.mg_oa > 0).mean()}
        for k in ("k1", "k2"):
            q[f"{k}_in_site"] = ((ts[f"{k}_d"] < 4.0) & (ts[f"{k}_prot"] >= 1)).mean()
            q[f"{k}_d_max"] = ts[f"{k}_d"].max()
        q.update({"rmsd_ca_mean": ts.rmsd_ca[w].mean(), "rmsd_ca_max": ts.rmsd_ca[w].max(),
                  "nuc_rmsd_mean": ts.nuc_rmsd[w].mean(), "nuc_rmsd_max": ts.nuc_rmsd[w].max(),
                  "ploop_vs_1kax": ts.ploop_1kax[w].mean(), "ploop_vs_3hsc": ts.ploop_3hsc[w].mean(),
                  "pbc_min_A": ts.pbc_min.min()})
        qc_rows.append(q)
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(f"{out_dir}/qc.csv", index=False, float_format="%.3g")

    # Per-replica RMSF and mean structures
    sub_idx = {k: np.array([ri[r] for r in v]) for k, v in SUBDOMAINS.items()}
    rmsf_g, rmsf_l, mean_g, mean_l = {}, {}, {}, {}
    for key, (ts, ca) in reps.items():
        X = np.asarray(ca[win(ts).values], dtype=np.float64)
        mean_g[key] = X.mean(0)
        rmsf_g[key] = np.sqrt(((X - mean_g[key]) ** 2).sum(-1).mean(0))
        rl, ml = np.empty(len(RES)), np.empty((len(RES), 3))
        for k, ii in sub_idx.items():
            Y = fit_batch(X[:, ii], ref_ca[ii])
            ml[ii] = Y.mean(0)
            rl[ii] = np.sqrt(((Y - ml[ii]) ** 2).sum(-1).mean(0))
        rmsf_l[key], mean_l[key] = rl, ml

    def disp(A, B, local):
        if not local:
            return np.linalg.norm(fit_batch(B[None], A)[0] - A, axis=1)
        d = np.empty(len(RES))
        for ii in sub_idx.values():
            d[ii] = np.linalg.norm(fit_batch(B[None, ii], A[ii])[0] - A[ii], axis=1)
        return d

    keys = {s: [k for k in sorted(reps) if k[0] == s] for s in STATES}
    pr = pd.DataFrame({"resid": RES})
    pr["subdomain"] = [next(k for k, v in SUBDOMAINS.items() if r in v) for r in RES]
    for tag, rm in (("global", rmsf_g), ("local", rmsf_l)):
        for s in STATES:
            pr[f"rmsf_{tag}_{s}"] = np.mean([rm[k] for k in keys[s]], axis=0)
        pr[f"drmsf_{tag}"] = pr[f"rmsf_{tag}_atp"] - pr[f"rmsf_{tag}_adp"]
        cross = np.array([rm[i] - rm[j] for i in keys["atp"] for j in keys["adp"]])
        pr[f"drmsf_{tag}_same_sign"] = (np.all(cross > 0, 0) | np.all(cross < 0, 0)) if len(cross) > 1 else False
    for tag, ms in (("global", mean_g), ("local", mean_l)):
        loc = tag == "local"
        between = np.array([disp(ms[i], ms[j], loc) for i in keys["atp"] for j in keys["adp"]])
        within = np.array([disp(ms[i], ms[j], loc) for s in STATES for n, i in enumerate(keys[s]) for j in keys[s][n + 1:]])
        pr[f"disp_{tag}_between"] = between.mean(0)
        pr[f"disp_{tag}_within"] = within.mean(0) if len(within) else np.nan
        pr[f"disp_{tag}_excess"] = pr[f"disp_{tag}_between"] - pr[f"disp_{tag}_within"]
        pr[f"disp_{tag}_robust"] = (between.min(0) > within.max(0)) if len(within) else False

    pg = [x[3] for x in kax if x[0] == "ATP" and x[2] == "PG"][0]
    pr["dist_pg"] = [min(np.linalg.norm(x[3] - pg) for x in kax if x[1] == r and x[0] != "HOH" and not x[2].startswith("H"))
                     for r in RES]
    for k, v in INTERFACES.items():
        pr[f"if_{k}"] = pr.resid.isin(v)
    pr["if_any"] = pr[[f"if_{k}" for k in INTERFACES]].any(axis=1)

    has_csp = bool(a.csp and os.path.isfile(a.csp))
    if has_csp:
        csp = pd.read_csv(a.csp, sep="\t", header=None, names=["resid", "csp"])
        csp["csp"] = pd.to_numeric(csp.csp, errors="coerce")
        pr = pr.merge(csp, on="resid", how="left")
        pr["csp_untracked"] = pr.csp >= 0.99
    pr.to_csv(f"{out_dir}/per_residue.csv", index=False, float_format="%.4g")

    # Scalars & stats
    strand = np.array([ri[r] for r in IIA_STRAND])
    rows = []
    for (s, r), (ts, _) in sorted(reps.items()):
        w = ts[win(ts)]
        rows.append({"state": s, "rep": r,
                     "delta_mean": w.delta.mean(), "delta_sd": w.delta.std(),
                     "tau_mean": w.tau.mean(), "tau_sd": w.tau.std(),
                     "d_ib_iib": w.d_ib_iib.mean(),
                     "th_ib_mean": w.th_ib.mean(), "th_iib_mean": w.th_iib.mean(),
                     "r_lobes_detrended": stats.pearsonr(detrend(w.time_ns, w.th_ib), detrend(w.time_ns, w.th_iib))[0],
                     "w90_r261": w.w90_r261.mean(), "w90_r261_contact_frac": (w.w90_r261 < 3.0).mean(),
                     "w90ring_r261cz": w.w90ring_r261cz.mean(),
                     "crevice_mean": w.crevice.mean(), "crevice_sd": w.crevice.std(),
                     "r_delta_tau": stats.pearsonr(w.delta, w.tau)[0],
                     "r_delta_tau_detrended": stats.pearsonr(detrend(w.time_ns, w.delta), detrend(w.time_ns, w.tau))[0],
                     "rmsf_iia_strand_local": rmsf_l[(s, r)][strand].mean(),
                     **{f"rmsf_{k}_local": rmsf_l[(s, r)][[ri[x] for x in v]].mean() for k, v in INTERFACES.items()},
                     "rmsf_all_local": rmsf_l[(s, r)].mean()})
    sc = pd.DataFrame(rows)
    sc.to_csv(f"{out_dir}/scalars.csv", index=False, float_format="%.4g")

    st_rows = []
    for c in sc.columns[2:]:
        x, y = sc[sc.state == "atp"][c].values, sc[sc.state == "adp"][c].values
        sep = len(x) > 1 and len(y) > 1 and (x.min() > y.max() or x.max() < y.min())
        st_rows.append({"observable": c, "atp_mean": x.mean(), "atp_se": stats.sem(x) if len(x) > 1 else np.nan,
                        "adp_mean": y.mean(), "adp_se": stats.sem(y) if len(y) > 1 else np.nan,
                        "diff_atp_minus_adp": x.mean() - y.mean(),
                        "welch_p": stats.ttest_ind(x, y, equal_var=False).pvalue if len(x) > 1 and len(y) > 1 else np.nan,
                        "separated": sep, "atp_values": " ".join(f"{v:.3g}" for v in x), "adp_values": " ".join(f"{v:.3g}" for v in y)})
    st = pd.DataFrame(st_rows)
    st.to_csv(f"{out_dir}/stats.csv", index=False, float_format="%.4g")

    # H2: distance-matched null
    metrics = ["disp_local_excess", "disp_global_excess", "drmsf_local", "drmsf_global"]
    bins = np.digitize(pr.dist_pg, np.arange(0, 60, 5))
    ifr = []
    for k in list(INTERFACES) + ["any"]:
        m = pr[f"if_{k}"].values
        pool = {b: np.where((bins == b) & ~pr.if_any.values)[0] for b in np.unique(bins)}
        draws = [np.concatenate([rng.choice(pool[b], size=(bins[m] == b).sum(), replace=len(pool[b]) < (bins[m] == b).sum())
                                 for b in np.unique(bins[m])]) for _ in range(5000)]
        for mt in metrics:
            v = pr[mt].values
            f = np.abs if mt.startswith("drmsf") else (lambda z: z)
            obs = f(v[m]).mean()
            null = np.array([f(v[d]).mean() for d in draws])
            ifr.append({"set": k, "n": int(m.sum()), "mean_dist_pg": pr.dist_pg[m].mean(),
                        "metric": mt + (" (abs)" if mt.startswith("drmsf") else ""),
                        "observed": obs, "null_mean": null.mean(),
                        "p_upper": (null >= obs).mean() if np.isfinite(obs) else np.nan,
                        "signed_drmsf": v[m].mean() if mt.startswith("drmsf") else np.nan})
    pd.DataFrame(ifr).to_csv(f"{out_dir}/interfaces.csv", index=False, float_format="%.4g")

    # H4: optional NMR CSP correlation
    if has_csp:
        cr = []
        for subset, mask in (("tracked", pr.csp.notna() & ~pr.csp_untracked), ("with_untracked", pr.csp.notna())):
            d = pr[mask]
            for mt in ["dist_pg"] + metrics:
                v = d[mt].abs() if mt.startswith("drmsf") else d[mt]
                rho, p = stats.spearmanr(d.csp, v)
                row = {"csp_set": subset, "n": len(d), "md_metric": mt, "spearman": rho, "p": p}
                if mt != "dist_pg":
                    pr_, pp = partial_spearman(d.csp.values, v.values, d.dist_pg.values)
                    row.update({"partial_given_dist": pr_, "partial_p": pp})
                cr.append(row)
        pd.DataFrame(cr).to_csv(f"{out_dir}/csp.csv", index=False, float_format="%.4g")

    with open(f"{out_dir}/window.txt", "w") as f:
        f.write(f"{window}; replicas {sorted(reps)}\n")
    print(f"[OK] Comparison summaries written to {out_dir}")


if __name__ == "__main__":
    main()
