"""Aggregate raw results + saved predictions -> results.csv, winner tables, figures.

PRIMARY metric = fidelity to the reference posterior:
  reference = exact GP (n = 1e4) or, where exact is unavailable, hard banding at c = 5 (ref_band_c5; checked
  against exact on every n = 1e4 cell, see results/ref_check.csv).
  err_ref   = RMSE between the method's and the reference posterior mean (test / validation points),
  var_ratio = mean over the 200 variance points of v_method / v_ref.
Eligibility (selection uses validation means; variances exist only on the 200-point test subset):
  err_ref(val) <= TOL_MEAN * RMSE(reference mean vs f0, val)  and  |mean var_ratio - 1| <= TOL_VAR.
Winner = eligible configuration (status ok in every rep of the cell) with the smallest mean wall-clock, and
separately the smallest approximate flop count. SECONDARY: the original "test-RMSE within 5% of best" rule and
coverage of f0.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import config as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL_MEAN, TOL_VAR = 0.05, 0.10
KEYS = ["d", "kernel", "beta", "n"]
JOINT = ["exact", "band", "vecchia_latent", "sgpr", "pcg"]       # return a joint (multivariate) posterior
POINTWISE = JOINT + ["vecchia_pred"]                               # any method usable for pointwise prediction
FAMILY_STYLE = {   # fixed categorical order (validated palette) + distinct marker per family
    "band": ("#2a78d6", "o", "hard banding"),
    "sgpr": ("#eb6834", "s", "SGPR / Nyström"),
    "vecchia_latent": ("#1baf7a", "^", "Vecchia (latent, maxmin)"),
    "vecchia_pred": ("#eda100", "v", "NN-Vecchia prediction"),
    "pcg": ("#e87ba4", "D", "pivoted-Cholesky PCG"),
    "exact": ("#008300", "*", "exact GP"),
}


def jid(r):
    return "d{d}_{kernel}_b{beta}_n{n}_r{rep}".format(**r)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    df = pd.DataFrame(rows)
    df = df[df.method != "diag"]
    extra = []
    for _, g in df.groupby(KEYS + ["rep"]):
        r0 = g.iloc[0]
        p = os.path.join(ROOT, "results", "preds", jid(r0) + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        if "mv" not in z.files:          # produced before validation means were saved: job must be rerun
            continue
        names = list(z["names"])
        ref = "exact" if "exact" in names else ("ref_band_c5" if "ref_band_c5" in names else None)
        if ref is None:
            for nm in names:
                extra.append(dict(r0[KEYS + ["rep"]], method=nm, ref_name="none"))
            continue
        i0 = names.index(ref)
        mt0, mv0, v0 = z["mt"][i0], z["mv"][i0], z["vsub"][i0]
        ref_rmse_f0_v = float(np.sqrt(np.mean((mv0 - z["fv"]) ** 2)))
        ref_rmse_f0_t = float(np.sqrt(np.mean((mt0 - z["ft"]) ** 2)))
        for i, nm in enumerate(names):
            extra.append(dict(r0[KEYS + ["rep"]], method=nm, ref_name=ref,
                              err_ref_t=float(np.sqrt(np.mean((z["mt"][i] - mt0) ** 2))),
                              err_ref_v=float(np.sqrt(np.mean((z["mv"][i] - mv0) ** 2))),
                              var_ratio_ref=float(np.mean(z["vsub"][i] / v0)),
                              var_ratio_ref_min=float(np.min(z["vsub"][i] / v0)),
                              var_ratio_ref_max=float(np.max(z["vsub"][i] / v0)),
                              ref_rmse_f0_v=ref_rmse_f0_v, ref_rmse_f0_t=ref_rmse_f0_t))
    if extra:
        df = df.merge(pd.DataFrame(extra), on=KEYS + ["rep", "method"], how="left")
    return df


def ref_check(df):
    """At n = 1e4: does hard banding at c = 5 reproduce the exact posterior?"""
    e = df[(df.method == "ref_band_c5") & (df.ref_name == "exact")]
    return e.groupby(KEYS).agg(reps=("rep", "nunique"), status=("status", lambda s: "/".join(sorted(set(s)))),
                               err_mean_vs_exact=("err_ref_t", "max"), rel_to_stat_err=("ref_rmse_f0_t", "mean"),
                               var_ratio_min=("var_ratio_ref_min", "min"), var_ratio_max=("var_ratio_ref_max", "max")
                               ).assign(rel_to_stat_err=lambda x: x.err_mean_vs_exact / x.rel_to_stat_err).reset_index()


def config_table(df, tol_mean=None):
    tol_mean = TOL_MEAN if tol_mean is None else tol_mean
    nrep = df.groupby(KEYS)["rep"].nunique().rename("nrep")
    ok = df[(df.status == "ok") & (df.family != "ref")]
    agg = ok.groupby(KEYS + ["family", "method"]).agg(
        reps_ok=("rep", "nunique"), err_ref_v=("err_ref_v", "mean"), err_ref_t=("err_ref_t", "mean"),
        var_ratio=("var_ratio_ref", "mean"), ref_rmse_f0_v=("ref_rmse_f0_v", "mean"),
        rmse_val=("rmse_val", "mean"), rmse_f0=("rmse_f0", "mean"), cover=("cover", "mean"),
        t_total=("t_total", "mean"), t_xcov=("t_xcov", "mean"), flops=("flops", "mean"),
        nnz_used=("nnz_used", "mean"), nnz_row=("info_nnz_row", "mean"), peak_mem_mb=("peak_mem_mb", "mean"),
        ref_name=("ref_name", "first")).reset_index().join(nrep, on=KEYS)
    agg["all_reps"] = agg.reps_ok == agg.nrep
    agg["rel_err"] = agg.err_ref_v / agg.ref_rmse_f0_v
    agg["eligible"] = agg.all_reps & (agg.rel_err <= tol_mean) & ((agg.var_ratio - 1).abs() <= TOL_VAR)
    best = agg[agg.all_reps].groupby(KEYS)["rmse_val"].min().rename("best_rmse_val")
    agg = agg.join(best, on=KEYS)
    agg["eligible_f0rule"] = agg.all_reps & (agg.rmse_val <= 1.05 * agg.best_rmse_val)
    return agg


def band_threshold(df):
    b = df[(df.family == "band") & (df.rep == 0) & ~df.method.str.contains("_metis")]
    out = []
    for key, g in b.groupby(KEYS):
        okc = sorted(g[g.status == "ok"].c)
        row = dict(zip(KEYS, key), c_star=okc[0] if okc else np.nan,
                   sweep=" ".join(f"{c:g}:{s[:4]}" for c, s in sorted(zip(g.c, g.status))))
        if okc:
            row["nnz_row_cstar"] = float(g[g.c == okc[0]].info_nnz_row.iloc[0])
        out.append(row)
    return pd.DataFrame(out)


def winners(agg, df, col="eligible", families=POINTWISE):
    out = []
    agg = agg[agg.family.isin(families)]
    for key, g in agg.groupby(KEYS):
        cell = df[(df.d == key[0]) & (df.kernel == key[1]) & (df.beta == key[2]) & (df.n == key[3])]
        e = g[g[col]]
        row = dict(zip(KEYS, key), pts_per_ell=float(cell.pts_per_ell.iloc[0]), L=float(cell.L.iloc[0]),
                   ref=g.ref_name.dropna().iloc[0] if g.ref_name.notna().any() else "none")
        if e.empty:
            row.update(winner_time="(none eligible)", winner_flops="(none eligible)")
        else:
            wt = e.loc[e.t_total.idxmin()]; wf = e.loc[e.flops.idxmin()]
            row.update(winner_time=wt.method, time_s=wt.t_total, err_rel=wt.rel_err, var_ratio=wt.var_ratio,
                       rmse_f0=wt.rmse_f0, cover=wt.cover, winner_flops=wf.method, flops=wf.flops)
        for fam in ["band", "vecchia_latent", "sgpr", "vecchia_pred", "pcg"]:
            ef = e[e.family == fam]
            row[f"t_{fam}"] = ef.t_total.min() if len(ef) else np.nan
            row[f"f_{fam}"] = ef.flops.min() if len(ef) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def md_winners(w, th):
    w = w.merge(th[KEYS + ["c_star", "nnz_row_cstar"]], on=KEYS, how="left")
    L = []
    for d, gd in w.groupby("d"):
        L.append(f"\n**d = {d}**\n")
        L.append("| kernel | β | n | pts/ℓ^d | ref | c* (nnz/row) | winner: wall-clock | s | err/stat-err | var ratio | coverage | winner: flops | band eligible? (s) | latent Vecchia eligible? (s) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in gd.sort_values(["kernel", "beta", "n"]).iterrows():
            cs = "none valid" if np.isnan(r.c_star) else f"{r.c_star:g} ({r.nnz_row_cstar:.0f})"
            f = lambda x: "-" if pd.isna(x) else f"{x:.3g}"
            L.append(f"| {r.kernel} | {r.beta:g} | {r.n:.0e} | {r.pts_per_ell:.3g} | {r.ref.replace('ref_band_c5', 'band c5')} | {cs} | "
                     f"{r.winner_time} | {f(r.get('time_s'))} | {f(r.get('err_rel'))} | {f(r.get('var_ratio'))} | "
                     f"{f(r.get('cover'))} | {r.winner_flops} | {'yes (' + f(r.t_band) + ')' if not pd.isna(r.t_band) else 'no'} | "
                     f"{'yes (' + f(r.t_vecchia_latent) + ')' if not pd.isna(r.t_vecchia_latent) else 'no'} |")
    return "\n".join(L)


def matched_cost(agg):
    """Banding vs latent Vecchia in the informative cells (SE any d; any kernel in d = 2): all configurations,
    all reps available (band radii other than c*, c*+1 are rep 0 only)."""
    sel = agg[agg.family.isin(["band", "vecchia_latent"]) & ((agg.kernel == "se") | (agg.d == 2))]
    out = []
    for key, g in sel.groupby(KEYS):
        b = g[g.family == "band"]; v = g[g.family == "vecchia_latent"]
        verdict = []
        for _, rv in v.iterrows():
            # band configuration that is at most as expensive (flops) as this Vecchia configuration
            cheaper = b[(b.flops <= rv.flops) & b.rel_err.notna()]
            if pd.isna(rv.rel_err):
                verdict.append(f"{rv.method}: no reference")
            elif len(cheaper):
                bb = cheaper.loc[cheaper.rel_err.idxmin()]
                verdict.append(f"{rv.method}: band@≤flops {bb.method} err {bb.rel_err:.3f} vs {rv.rel_err:.3f}")
            else:
                verdict.append(f"{rv.method}: no band config this cheap")
        out.append(dict(zip(KEYS, key), verdict="; ".join(verdict)))
    return sel[KEYS + ["method", "reps_ok", "flops", "nnz_used", "peak_mem_mb", "t_total", "rel_err", "var_ratio",
                       "nnz_row"]], pd.DataFrame(out)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    return plt


def fig_winners(agg):
    plt = _style()
    from matplotlib.lines import Line2D
    paths = []
    for d in sorted(agg.d.unique()):
        gd = agg[agg.d == d]
        fig, axes = plt.subplots(2, len(C.BETAS), figsize=(3.2 * len(C.BETAS), 5.6), sharey=True, sharex=True)
        for i, kern in enumerate(C.KERNELS):
            for jb, beta in enumerate(C.BETAS):
                ax = axes[i, jb]
                g = gd[(gd.kernel == kern) & (gd.beta == beta) & gd.all_reps]
                for fam, (col, mk, lab) in FAMILY_STYLE.items():
                    gf = g[g.family == fam]
                    if gf.empty:
                        continue
                    el = gf[gf.eligible].groupby("n").t_total.min()
                    ne = gf[~gf.eligible].groupby("n").t_total.min()
                    ne = ne[~ne.index.isin(el.index)]
                    if len(el):
                        ax.plot(el.index, el.values, color=col, marker=mk, ms=6, lw=2, label=lab,
                                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
                    if len(ne):
                        ax.plot(ne.index, ne.values, ls="none", marker=mk, ms=6, mfc="none", mec=col, mew=1.2, zorder=2)
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(f"{'SE' if kern == 'se' else 'Matérn-3/2'}, β={beta:g}" + (" (infill, oracle ℓ)" if beta == 0 else " (ℓ=1)"), fontsize=9)
                if jb == 0:
                    ax.set_ylabel("wall-clock s, single thread")
                if i == 1:
                    ax.set_xlabel("n")
        h = [Line2D([], [], color=c, marker=m, lw=2, ms=6, markeredgecolor="white") for c, m, _ in FAMILY_STYLE.values()]
        l = [lab for _, _, lab in FAMILY_STYLE.values()]
        h.append(Line2D([], [], ls="none", marker="o", mfc="none", mec="#666")); l.append("hollow: cheapest config, not eligible")
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.suptitle(f"d = {d}: cheapest configuration per family reproducing the reference posterior "
                     f"(mean err ≤ {TOL_MEAN:.0%} of stat. error, |var ratio−1| ≤ {TOL_VAR:.0%})", fontsize=9.5)
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))
        p = os.path.join(ROOT, "figures", f"winners_d{d}.png"); fig.savefig(p, dpi=160); plt.close(fig); paths.append(p)
    return paths


def fig_matched(sel):
    """Error-vs-flops frontiers, banding vs latent Vecchia, informative cells."""
    plt = _style()
    cells = sel.groupby(KEYS).size().index.tolist()
    if not cells:
        return None
    nc = 4; nr = int(np.ceil(len(cells) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.5 * nr), squeeze=False)
    for ax, key in zip(axes.ravel(), cells):
        g = sel[(sel.d == key[0]) & (sel.kernel == key[1]) & (sel.beta == key[2]) & (sel.n == key[3])]
        for fam in ["band", "vecchia_latent"]:
            col, mk, lab = FAMILY_STYLE[fam]
            gf = g[g.method.str.startswith("band" if fam == "band" else "vecchia_latent")].sort_values("flops")
            gf = gf[gf.rel_err.notna()]
            if gf.empty:
                continue
            ax.plot(gf.flops, np.maximum(gf.rel_err, 1e-6), color=col, marker=mk, lw=2, ms=6, label=lab,
                    markeredgecolor="white")
            for _, r in gf.iterrows():
                tag = f"c{r.method.split('_c')[-1]}" if fam == "band" else f"m{r.method.split('_m')[-1]}"
                ax.annotate(tag, (r.flops, max(r.rel_err, 1e-6)), fontsize=6.5, color="#444",
                            xytext=(3, 3), textcoords="offset points")
        if not ax.lines:
            ax.text(0.5, 0.5, "no reference posterior\n(c=5 band infeasible)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="#666")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axhline(TOL_MEAN, color="#999", lw=0.8, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"d={key[0]} {'SE' if key[1] == 'se' else 'M3/2'} β={key[2]:g} n={key[3]:.0e}", fontsize=8)
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel("approx. flops"); fig.supylabel("mean error / statistical error (dashed: 5% tolerance)")
    fig.tight_layout()
    p = os.path.join(ROOT, "figures", "band_vs_vecchia_matched_cost.png"); fig.savefig(p, dpi=160); plt.close(fig)
    return p


def fig_frontiers(agg):
    """Primary evidence: error to the reference posterior vs approximate flops, every configuration, per cell."""
    plt = _style()
    from matplotlib.lines import Line2D
    paths = []
    for d in sorted(agg.d.unique()):
        gd = agg[agg.d == d]
        ns = sorted(gd.n.unique()); rows = [(k, b) for k in C.KERNELS for b in C.BETAS]
        fig, axes = plt.subplots(len(rows), len(ns), figsize=(3.0 * len(ns), 2.15 * len(rows)), squeeze=False)
        for i, (kern, beta) in enumerate(rows):
            for jn, n in enumerate(ns):
                ax = axes[i, jn]
                g = gd[(gd.kernel == kern) & (gd.beta == beta) & (gd.n == n) & gd.rel_err.notna()]
                for fam, (col, mk, lab) in FAMILY_STYLE.items():
                    if fam == "exact":
                        continue
                    gf = g[g.family == fam].sort_values("flops")
                    if gf.empty:
                        continue
                    ax.plot(gf.flops, np.maximum(gf.rel_err, 1e-7), color=col, lw=1.5, marker=mk, ms=5,
                            markeredgecolor="white", markeredgewidth=0.6, zorder=3)
                    bad = gf[(gf.var_ratio - 1).abs() > TOL_VAR]
                    ax.plot(bad.flops, np.maximum(bad.rel_err, 1e-7), ls="none", marker="x", ms=4, color="#333", zorder=4)
                if g.empty:
                    ax.text(0.5, 0.5, "no reference posterior", ha="center", va="center", transform=ax.transAxes,
                            fontsize=7, color="#666"); ax.set_xticks([]); ax.set_yticks([])
                else:
                    for t, ls in ((0.03, ":"), (0.05, "--"), (0.10, ":")):
                        ax.axhline(t, color="#999", lw=0.7, ls=ls)
                    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-7, 30)
                ax.set_title(f"{'SE' if kern == 'se' else 'M3/2'} β={beta:g} n={n:.0e}", fontsize=8)
        h = [Line2D([], [], color=c, marker=m, lw=1.5, ms=5, markeredgecolor="white")
             for f, (c, m, _) in FAMILY_STYLE.items() if f != "exact"]
        l = [lab for f, (_, _, lab) in FAMILY_STYLE.items() if f != "exact"]
        h.append(Line2D([], [], ls="none", marker="x", color="#333")); l.append("variance ratio off by >10%")
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.supxlabel("approximate flops (fit + mean + 200 variances)", y=0.035, fontsize=9)
        fig.supylabel("‖mean − reference mean‖ / statistical error (lines: 3%, 5%, 10%)", fontsize=9)
        fig.suptitle(f"d = {d}: error–cost frontiers per method family (each marker = one configuration)", fontsize=10)
        fig.tight_layout(rect=(0.01, 0.06, 1, 0.975))
        p = os.path.join(ROOT, "figures", f"frontiers_d{d}.png"); fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def threshold_sensitivity(df):
    out = []
    out_all = []
    for use, fams in (("joint", JOINT), ("pointwise", POINTWISE)):
        out_all += _sens(df, use, fams)
    return pd.DataFrame(out_all)


def _sens(df, use, fams):
    out = []
    ws = {t: winners(config_table(df, t), df, families=fams) for t in (0.03, 0.05, 0.10)}
    base = ws[0.05].set_index(KEYS)
    for t in (0.03, 0.10):
        w = ws[t].set_index(KEYS)
        for k in base.index:
            a, b = base.loc[k, "winner_time"], w.loc[k, "winner_time"]
            fam = lambda s: s.rsplit("_", 1)[0] if s.startswith(("band", "sgpr", "vecchia", "pcg")) else s
            if a != b:
                out.append(dict(zip(KEYS, k), use=use, tol=t, winner_at_5pct=a, winner=b, family_flip=fam(a) != fam(b)))
    return out


def main():
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    df = load()
    df.to_csv(os.path.join(ROOT, "results.csv"), index=False)
    rc = ref_check(df); rc.to_csv(os.path.join(ROOT, "results", "ref_check.csv"), index=False)
    agg = config_table(df); agg.to_csv(os.path.join(ROOT, "results", "summary_by_config.csv"), index=False)
    th = band_threshold(df); th.to_csv(os.path.join(ROOT, "results", "band_threshold.csv"), index=False)
    w = winners(agg, df); w.to_csv(os.path.join(ROOT, "results", "winners_pointwise.csv"), index=False)
    wj = winners(agg, df, families=JOINT); wj.to_csv(os.path.join(ROOT, "results", "winners_joint.csv"), index=False)
    w2 = winners(agg, df, "eligible_f0rule"); w2.to_csv(os.path.join(ROOT, "results", "winners_f0rule.csv"), index=False)
    sel, verdict = matched_cost(agg)
    sel.to_csv(os.path.join(ROOT, "results", "band_vs_vecchia.csv"), index=False)
    verdict.to_csv(os.path.join(ROOT, "results", "band_vs_vecchia_verdict.csv"), index=False)
    with open(os.path.join(ROOT, "results", "winners.md"), "w") as fh:
        fh.write("## Full posterior (joint-posterior methods only)\n" + md_winners(wj, th) +
                 "\n\n## Pointwise prediction (all methods incl. NN-Vecchia prediction)\n" + md_winners(w, th) + "\n")
    figs = fig_winners(agg) + [fig_matched(sel)] + fig_frontiers(agg)
    ts = threshold_sensitivity(df); ts.to_csv(os.path.join(ROOT, "results", "threshold_sensitivity.csv"), index=False)
    print("\nthreshold sensitivity (winner by wall-clock differs from the 5% rule):\n", ts.to_string())
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200); pd.set_option("display.max_rows", 500)
    print("REFERENCE CHECK (band c=5 vs exact, n=1e4):\n", rc.to_string())
    print("FULL POSTERIOR"); print(md_winners(wj, th)); print("POINTWISE"); print(md_winners(w, th))
    print("\nf0-rule winners:\n", w2[KEYS + ["winner_time", "winner_flops"]].to_string())
    print("\nband thresholds:\n", th.to_string())
    print("\nmatched cost:\n", verdict.to_string())
    print("figures:", figs)
