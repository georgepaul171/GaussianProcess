"""results.csv, figures (RMSE vs number of selected features; coverage by method) and summary tables."""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# fixed categorical order (validated palette from the banding study) + marker per method
STYLE = {
    ("exact_gp", "-"): ("#008300", "*", "exact GP (SE, ML-II)"),
    ("rff_blr", "iid"): ("#8a8a8a", "o", "RFF-BLR, all p (iid)"),
    ("rff_blr", "lattice"): ("#8a8a8a", "D", "RFF-BLR, all p (lattice)"),
    ("svb_individual", "iid"): ("#2a78d6", "o", "SnS-VB individual (iid)"),
    ("svb_individual", "lattice"): ("#2a78d6", "D", "SnS-VB individual (lattice)"),
    ("svb_group", "iid"): ("#eb6834", "o", "SnS-VB group (iid)"),
    ("svb_group", "lattice"): ("#eb6834", "D", "SnS-VB group (lattice)"),
    ("ssgp", "optimised"): ("#1baf7a", "^", "SSGP (optimised freq.)"),
}


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def summary(df):
    g = df[df.status == "ok"].groupby(["d", "truth", "p", "method", "design"], dropna=False)
    s = g.agg(reps=("rep", "nunique"), rmse=("rmse", "mean"), rmse_se=("rmse", lambda x: x.std() / np.sqrt(len(x))),
              coverage=("coverage", "mean"), n_selected=("n_selected", "mean"), n_sel_sd=("n_selected", "std"),
              runtime=("runtime", "median"), converged=("converged", "mean"),
              max_abs_corr=("max_abs_corr", "mean"), median_col_max_corr=("median_col_max_corr", "mean"), min_freq_gap=("min_freq_gap", "mean")).reset_index()
    ex = s[s.method == "exact_gp"][["d", "truth", "rmse"]].rename(columns={"rmse": "rmse_exact"})
    return s.merge(ex, on=["d", "truth"]).assign(rmse_rel_exact=lambda x: x.rmse / x.rmse_exact)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    return plt


def fig_rmse_vs_selected(s):
    plt = _plt()
    from matplotlib.lines import Line2D
    cells = [(d, t) for d in (1, 2) for t in ("sparse", "matern52")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (d, t) in zip(axes, cells):
        g = s[(s.d == d) & (s.truth == t)]
        for (m, des), (col, mk, lab) in STYLE.items():
            gm = g[(g.method == m) & (g.design == des)].sort_values("p")
            if gm.empty:
                continue
            if m == "exact_gp":
                ax.axhline(gm.rmse.iloc[0], color=col, lw=1.2, ls="--")
                continue
            ax.plot(gm.n_selected, gm.rmse, color=col, marker=mk, ms=6, lw=1.5 if m != "rff_blr" else 0.8,
                    ls="-" if des != "lattice" else ":", markeredgecolor="white", markeredgewidth=0.6)
            for _, r in gm.iterrows():
                ax.annotate(f"p={r.p:.0f}", (r.n_selected, r.rmse), fontsize=6, color="#555",
                            xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"d={d}, {'spectrally sparse' if t == 'sparse' else 'Matérn-5/2 draw'}", fontsize=9)
        ax.set_xlabel("number of features used (selected: inclusion > 0.5)")
    axes[0].set_ylabel("test RMSE to f0 (mean over reps)")
    h = [Line2D([], [], color=c, marker=m, ls=":" if k[1] == "lattice" else ("--" if k[0] == "exact_gp" else "-"),
                markeredgecolor="white") for k, (c, m, _) in STYLE.items()]
    fig.legend(h, [l for *_, l in STYLE.values()], loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    p = os.path.join(ROOT, "figures", "rmse_vs_selected.png"); fig.savefig(p, dpi=160); plt.close(fig)
    return p


def fig_coverage(s):
    plt = _plt()
    cells = [(d, t) for d in (1, 2) for t in ("sparse", "matern52")]
    keys = [k for k in STYLE if k[0] != "exact_gp"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    ps = sorted(s.p[s.p > 0].unique())
    w = 0.8 / len(keys)
    for ax, (d, t) in zip(axes, cells):
        g = s[(s.d == d) & (s.truth == t)]
        ex = g[g.method == "exact_gp"].coverage
        for i, k in enumerate(keys):
            col, mk, lab = STYLE[k]
            gm = g[(g.method == k[0]) & (g.design == k[1])].set_index("p").reindex(ps)
            x = np.arange(len(ps)) + (i - len(keys) / 2 + 0.5) * w
            ax.bar(x, gm.coverage.values, width=w * 0.9, color=col, alpha=1.0 if k[1] != "lattice" else 0.55,
                   hatch="//" if k[1] == "lattice" else None, edgecolor="white", linewidth=0.5, label=lab)
        if len(ex):
            ax.axhline(ex.iloc[0], color="#008300", ls="--", lw=1.2, label="exact GP")
        ax.axhline(0.95, color="#333", lw=0.8)
        ax.set_xticks(range(len(ps))); ax.set_xticklabels([f"p={p:.0f}" for p in ps])
        ax.set_title(f"d={d}, {'spectrally sparse' if t == 'sparse' else 'Matérn-5/2 draw'}", fontsize=9)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("mean pointwise 95% coverage of f0")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    p = os.path.join(ROOT, "figures", "coverage_by_method.png"); fig.savefig(p, dpi=160); plt.close(fig)
    return p


def claim3(df):
    """Predicted (fixed-support VIF) vs observed coverage of exact-GP function values, per VB fit, aggregated.
    Undercoverage U = 0.95 - observed is split as
      VIF part (coefficient)  0.95 - pred_cov_coef   (Claim 3 as stated: per-coefficient deflation)
      VIF part (function)     0.95 - pred_cov_fun    (same deflation for f(x) = phi_S' theta_S)
      width part              0.95 - width_only      (VB band width vs GP sd, incl. inclusion variance)
      remainder               width_only - observed  (VB mean offset from the GP mean)."""
    v = df[df.method.isin(["svb_individual", "svb_group"]) & (df.status == "ok")]
    g = v.groupby(["truth", "d", "design", "method", "p"]).agg(
        reps=("rep", "nunique"), support=("support_size", "mean"),
        cov_f0=("coverage", "mean"), cov_gp_observed=("cov_gp_observed", "mean"),
        cov_gp_width_only=("cov_gp_width_only", "mean"), pred_cov_coef=("pred_cov_coef", "mean"),
        pred_cov_fun=("pred_cov_fun", "mean"), vif_median=("vif_median", "median"),
        fun_vif_median=("fun_vif_median", "median"), frac_R_below_1=("frac_fun_vif_below_1", "mean"),
        singular_support=("support_singular", "mean"), mean_shift_sd=("mean_shift_over_gp_sd", "median")).reset_index()
    U = 0.95 - g.cov_gp_observed
    g["undercov"] = U
    g["share_vif_coef"] = (0.95 - g.pred_cov_coef) / U
    g["share_vif_fun"] = (0.95 - g.pred_cov_fun) / U
    g["share_width"] = (0.95 - g.cov_gp_width_only) / U
    g["share_mean_offset"] = (g.cov_gp_width_only - g.cov_gp_observed) / U
    return g


def fig_claim3(c3):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (col, lab) in zip(axes, [("pred_cov_coef", "coefficient-level VIF prediction"),
                                     ("pred_cov_fun", "function-level fixed-support prediction")]):
        for (m, des), gg in c3.groupby(["method", "design"]):
            c, mk, _ = STYLE[(m, des)]
            for t, fill in (("sparse", c), ("matern52", "none")):
                h = gg[gg.truth == t]
                ax.scatter(h[col], h.cov_gp_observed, marker=mk, s=40, facecolors=fill, edgecolors=c, linewidths=1.2,
                           label=f"{STYLE[(m, des)][2]}, {'sparse' if t == 'sparse' else 'Matérn'}")
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--")
        ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
        ax.set_xlabel(f"predicted coverage ({lab})")
        ax.set_title(lab, fontsize=9)
    axes[0].set_ylabel("observed coverage of exact-GP function values")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    p = os.path.join(ROOT, "figures", "claim3_predicted_vs_observed.png"); fig.savefig(p, dpi=160); plt.close(fig)
    return p


def main():
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    df = load()
    df.to_csv(os.path.join(ROOT, "results.csv"), index=False)
    s = summary(df)
    s.to_csv(os.path.join(ROOT, "results", "summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    cols = ["d", "truth", "p", "method", "design", "reps", "rmse", "rmse_rel_exact", "coverage", "n_selected",
            "runtime", "converged", "max_abs_corr", "median_col_max_corr", "min_freq_gap"]
    print(s[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(fig_rmse_vs_selected(s), fig_coverage(s))
    c3 = claim3(df)
    c3.to_csv(os.path.join(ROOT, "results", "claim3.csv"), index=False)
    print(c3.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(fig_claim3(c3))
