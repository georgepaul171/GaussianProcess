"""Aggregate raw JSONL -> results.csv, winner tables, one figure per d.

Winner rule, per (d, kernel, beta, n):
  candidates = configurations that ran with status ok in every rep of the cell (band radii outside {c*, c*+1}
               only ran on rep 0 and are therefore not candidates, but they still define the validity threshold c*);
  best       = min over candidates of the mean *validation* RMSE to f0 (selection never uses the test set);
  eligible   = candidates with mean validation RMSE <= 1.05 * best;
  winner     = eligible configuration with the smallest mean wall-clock (fit + mean + 200 variances),
               and separately with the smallest approximate flop count.
Test RMSE, coverage and the RMSE gap to exact (n = 1e4) are reported for the winners.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY_STYLE = {   # fixed categorical order (validated palette), plus a distinct marker per family
    "band": ("#2a78d6", "o", "hard banding"),
    "sgpr": ("#eb6834", "s", "SGPR / Nystrom"),
    "vecchia_latent": ("#1baf7a", "^", "Vecchia (latent, maxmin)"),
    "vecchia_pred": ("#eda100", "v", "NN-Vecchia prediction"),
    "pcg": ("#e87ba4", "D", "pivoted-Cholesky PCG"),
    "exact": ("#008300", "*", "exact GP"),
}


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def cell_tables(df):
    keys = ["d", "kernel", "beta", "n"]
    nrep = df.groupby(keys)["rep"].nunique().rename("nrep")
    ok = df[df.status == "ok"]
    agg = ok.groupby(keys + ["family", "method"]).agg(
        reps_ok=("rep", "nunique"), rmse_val=("rmse_val", "mean"), rmse_f0=("rmse_f0", "mean"),
        rmse_f0_sd=("rmse_f0", "std"), cover=("cover", "mean"), t_total=("t_total", "mean"),
        flops=("flops", "mean"), nnz_used=("nnz_used", "mean"), var_ratio=("var_ratio", "mean"),
        rmse_to_exact=("rmse_to_exact", "mean"), peak_mem_mb=("peak_mem_mb", "mean"),
        nnz_row=("info_nnz_row", "mean") if "info_nnz_row" in ok else ("rep", "size")).reset_index()
    agg = agg.join(nrep, on=keys)
    cand = agg[agg.reps_ok == agg.nrep].copy()
    best = cand.groupby(keys)["rmse_val"].min().rename("best_val")
    cand = cand.join(best, on=keys)
    cand["eligible"] = cand.rmse_val <= 1.05 * cand.best_val
    return agg, cand


def band_threshold(df):
    """c* = smallest radius with a valid (PD) banded system on rep 0; nnz/row there; status of the c sweep."""
    b = df[(df.family == "band") & (df.rep == 0)]
    out = []
    for key, g in b.groupby(["d", "kernel", "beta", "n"]):
        okc = sorted(g[g.status == "ok"].c)
        sweep = ",".join(f"{c:g}:{s[:3]}" for c, s in zip(g.c, g.status))
        row = dict(zip(["d", "kernel", "beta", "n"], key), c_star=okc[0] if okc else np.nan, sweep=sweep)
        if okc:
            row["nnz_row_cstar"] = float(g[g.c == okc[0]].info_nnz_row.iloc[0])
        out.append(row)
    return pd.DataFrame(out)


def winners(cand, df):
    keys = ["d", "kernel", "beta", "n"]
    out = []
    for key, g in cand.groupby(keys):
        e = g[g.eligible]
        wt = e.loc[e.t_total.idxmin()]; wf = e.loc[e.flops.idxmin()]
        fam_best = {}
        for fam, gf in e.groupby("family"):
            fam_best[fam] = gf.t_total.min()
        bandrow = g[g.family == "band"]
        cell = df[(df.d == key[0]) & (df.kernel == key[1]) & (df.beta == key[2]) & (df.n == key[3])]
        out.append(dict(zip(keys, key),
                        rho_ell_d=float(cell.pts_per_ell.iloc[0]), L=float(cell.L.iloc[0]),
                        best_val_rmse=float(g.best_val.iloc[0]),
                        winner_time=wt.method, time_s=wt.t_total, rmse_f0=wt.rmse_f0, cover=wt.cover,
                        winner_flops=wf.method, flops=wf.flops, rmse_f0_flopwin=wf.rmse_f0, cover_flopwin=wf.cover,
                        band_eligible=bool((e.family == "band").any()),
                        band_best_time=fam_best.get("band", np.nan),
                        band_rel_rmse=float(bandrow.rmse_val.min() / g.best_val.iloc[0]) if len(bandrow) else np.nan,
                        speedup_vs_band=fam_best.get("band", np.nan) / wt.t_total if "band" in fam_best else np.nan))
    return pd.DataFrame(out)


def statuses(df):
    s = df[df.family != "exact"].groupby(["d", "kernel", "beta", "n", "method"]).status.agg(
        lambda x: "/".join(sorted(set(x)))).reset_index()
    return s


def fmt_table(w, th):
    w = w.merge(th[["d", "kernel", "beta", "n", "c_star", "nnz_row_cstar"]], on=["d", "kernel", "beta", "n"], how="left")
    lines = []
    for d, gd in w.groupby("d"):
        lines.append(f"\n**d = {d}**\n")
        lines.append("| kernel | beta | n | pts/ell^d | c* (nnz/row) | winner (wall-clock) | s | RMSE | cover | winner (flops) | flops | band within 5%? | band time / winner |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in gd.sort_values(["kernel", "beta", "n"]).iterrows():
            cs = "none valid" if np.isnan(r.c_star) else f"{r.c_star:g} ({r.nnz_row_cstar:.0f})"
            sp = "-" if np.isnan(r.speedup_vs_band) else f"{r.speedup_vs_band:.1f}x"
            lines.append(f"| {r.kernel} | {r.beta:g} | {r.n:.0e} | {r.rho_ell_d:.3g} | {cs} | {r.winner_time} | {r.time_s:.2f} | "
                         f"{r.rmse_f0:.4f} | {r.cover:.2f} | {r.winner_flops} | {r.flops:.1e} | "
                         f"{'yes' if r.band_eligible else 'no'} | {sp} |")
    return "\n".join(lines)


def figures(cand, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    paths = []
    for d in sorted(cand.d.unique()):
        gd = cand[cand.d == d]
        fig, axes = plt.subplots(2, len(C.BETAS), figsize=(3.1 * len(C.BETAS), 5.4), sharey=True, sharex=True)
        for i, kern in enumerate(C.KERNELS):
            for jb, beta in enumerate(C.BETAS):
                ax = axes[i, jb]
                g = gd[(gd.kernel == kern) & (gd.beta == beta)]
                for fam, (col, mk, lab) in FAMILY_STYLE.items():
                    gf = g[(g.family == fam)]
                    if gf.empty:
                        continue
                    el = gf[gf.eligible].groupby("n").t_total.min()
                    ne = gf[~gf.eligible].groupby("n").t_total.min()
                    if len(el):
                        ax.plot(el.index, el.values, color=col, marker=mk, ms=6, lw=2, label=lab,
                                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
                    ne = ne[~ne.index.isin(el.index)]
                    if len(ne):
                        ax.plot(ne.index, ne.values, ls="none", marker=mk, ms=6, mfc="none", mec=col, mew=1.2, zorder=2)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(f"{'SE' if kern == 'se' else 'Matérn-3/2'}, β = {beta:g}"
                             + (" (infill, oracle ℓ)" if beta == 0 else " (ℓ = 1)"), fontsize=9)
                if jb == 0:
                    ax.set_ylabel("wall-clock s (fit+predict)")
                if i == 1:
                    ax.set_xlabel("n")
        h, l = [], []
        for ax in axes.ravel():
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in l:
                    h.append(hh); l.append(ll)
        from matplotlib.lines import Line2D
        h.append(Line2D([], [], ls="none", marker="o", mfc="none", mec="#666")); l.append("hollow: not within 5% of best RMSE")
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.suptitle(f"d = {d}: cheapest configuration per method family (filled = RMSE within 5% of best)", fontsize=10)
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
        p = os.path.join(ROOT, "figures", f"winners_d{d}.png")
        fig.savefig(p, dpi=160); plt.close(fig); paths.append(p)
    return paths


def main():
    df = load()
    df.to_csv(os.path.join(ROOT, "results.csv"), index=False)
    agg, cand = cell_tables(df)
    th = band_threshold(df)
    w = winners(cand, df)
    agg.to_csv(os.path.join(ROOT, "results", "summary_by_config.csv"), index=False)
    w.to_csv(os.path.join(ROOT, "results", "winners.csv"), index=False)
    th.to_csv(os.path.join(ROOT, "results", "band_threshold.csv"), index=False)
    statuses(df).to_csv(os.path.join(ROOT, "results", "status_by_config.csv"), index=False)
    with open(os.path.join(ROOT, "results", "winners.md"), "w") as fh:
        fh.write(fmt_table(w, th) + "\n")
    figs = figures(cand, df)
    print(fmt_table(w, th))
    print("\nband thresholds:\n", th.to_string())
    print("figures:", figs)
