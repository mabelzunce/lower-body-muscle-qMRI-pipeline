"""
Group comparison (control, trasplante, diálisis) of muscle volume and fat fraction.

- Reads data/all_subjects_volumes_and_ffs.csv and merges data/study_info.csv
  (matching Subject <-> ID).
- ANCOVA per variable (FF and volume normalised by body weight), corrected by age and sex:
      y ~ C(group) + age + C(sex)
  Omnibus group effect = Type II F test. Pairwise group contrasts come from the same
  model (adjusted means) with Holm correction over the three pairs.
- Boxplots of the muscles with a significant group effect, annotated with the
  significant pairwise comparisons.

Usage:
    python src/analysis/group_comparison_ancova.py [--alpha 0.05] [--fdr] [--include-shortfov]
"""
import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEFAULT_OUT_DIR = ROOT / "results" / "group_comparison_ancova"

GROUPS = ["control", "trasplante", "diálisis"]
GROUP_COLORS = {"control": "#4c9f70", "trasplante": "#4c78a8", "diálisis": "#e45756"}
VOLUME_UNIT_LABEL = "Volume / weight (cm³/kg)"  # volumes in mm³ -> cm³ (÷1000)

STUDY_WEIGHT_COL = "¿Cuál es su peso aproximadamente?\nPoner el peso en kilgramos"


def load_data():
    vols = pd.read_csv(DATA_DIR / "all_subjects_volumes_and_ffs.csv")
    study = pd.read_csv(DATA_DIR / "study_info.csv")

    study = study.rename(columns={"ID": "Subject", "Grupo": "group_study", "Sexo": "sex_study",
                                  "Edad": "age", STUDY_WEIGHT_COL: "weight_study"})
    study = study[["Subject", "group_study", "sex_study", "age", "weight_study"]]
    df = vols.merge(study, on="Subject", how="inner", validate="one_to_one")

    df["group"] = (df["group_study"].str.strip().str.lower()
                   .replace({"dialisis": "diálisis"}))
    df["sex"] = df["sex_study"]
    # Weight: use the volumes table, fall back to study_info when missing
    df["weight"] = df["Weight (kg)"].fillna(df["weight_study"])
    df = df[df["group"].isin(GROUPS)].copy()
    df["group"] = pd.Categorical(df["group"], categories=GROUPS, ordered=True)
    return df


def select_variables(df, include_shortfov):
    """Muscle volume and FF columns (skin/fat, T2 and empty columns are excluded)."""
    def keep(col):
        if not col.startswith(("Vol ", "FF ")) or "skinFat" in col:
            return False
        if "ShortFOV" in col:
            return include_shortfov
        if "Lumbar_" in col or "Pelvis_" in col:
            return False
        return df[col].notna().any()

    vol_cols = [c for c in df.columns if c.startswith("Vol ") and keep(c)]
    ff_cols = [c for c in df.columns if c.startswith("FF ") and keep(c)]
    return vol_cols, ff_cols


def holm_adjust(pvals):
    return multipletests(pvals, method="holm")[1]


def ancova(data, column):
    """Fit y ~ group + age + sex. Returns dict with omnibus and pairwise results."""
    d = data[["group", "age", "sex", column]].dropna().rename(columns={column: "y"})
    d["group"] = d["group"].astype(str)
    model = smf.ols("y ~ C(group, Treatment('control')) + age + C(sex)", data=d).fit()
    table = anova_lm(model, typ=2)
    group_row = [i for i in table.index if i.startswith("C(group")][0]

    names = list(model.params.index)
    pairs = {}
    raw_p = []
    pair_list = list(itertools.combinations(GROUPS, 2))
    for a, b in pair_list:
        c = np.zeros(len(names))
        for g, sign in ((b, 1.0), (a, -1.0)):
            key = f"C(group, Treatment('control'))[T.{g}]"
            if key in names:
                c[names.index(key)] += sign
        t = model.t_test(c)
        pairs[(a, b)] = (float(np.squeeze(t.effect)), float(np.squeeze(t.pvalue)))
        raw_p.append(pairs[(a, b)][1])
    adj_p = holm_adjust(raw_p)

    res = {"variable": column, "n": len(d),
           "F_group": table.loc[group_row, "F"], "p_group": table.loc[group_row, "PR(>F)"],
           "p_age": table.loc["age", "PR(>F)"], "p_sex": table.loc[[i for i in table.index if i.startswith("C(sex")][0], "PR(>F)"]}
    for (a, b), p_adj in zip(pair_list, adj_p):
        diff, p_raw = pairs[(a, b)]
        res[f"diff_{b}_vs_{a}"] = diff
        res[f"p_{b}_vs_{a}"] = p_raw
        res[f"p_holm_{b}_vs_{a}"] = p_adj
    return res


def run_family(df, columns, family):
    rows = [ancova(df, c) for c in columns]
    out = pd.DataFrame(rows)
    out.insert(0, "family", family)
    out["p_group_fdr"] = multipletests(out["p_group"], method="fdr_bh")[1]
    return out


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def boxplot_significant(df, results, family, ylabel, out_path, alpha, use_fdr, transform=None):
    p_col = "p_group_fdr" if use_fdr else "p_group"
    sig = results[results[p_col] < alpha].sort_values(p_col)
    if sig.empty:
        print(f"[{family}] no significant muscles (alpha={alpha}, {p_col}).")
        return

    n = len(sig)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.8 * nrows), squeeze=False)
    rng = np.random.default_rng(0)

    for ax, (_, row) in zip(axes.ravel(), sig.iterrows()):
        col = row["variable"]
        d = df[["group", "age", "sex", col]].dropna()
        values = d[col] / 1000.0 if transform == "ml" else d[col]
        data = [values[d["group"] == g].to_numpy() for g in GROUPS]
        bp = ax.boxplot(data, positions=range(len(GROUPS)), widths=0.55, patch_artist=True,
                        showfliers=False, medianprops={"color": "black"})
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(GROUP_COLORS[g])
            patch.set_alpha(0.5)
        for i, (g, v) in enumerate(zip(GROUPS, data)):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14,
                       color=GROUP_COLORS[g], edgecolor="k", linewidth=0.3, zorder=3)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f"{g}\n(n={len(v)})" for g, v in zip(GROUPS, data)])
        ax.set_ylabel(ylabel)

        # Significance brackets (Holm-adjusted pairwise ANCOVA contrasts)
        ymin, ymax = np.nanmin(np.concatenate(data)), np.nanmax(np.concatenate(data))
        span = ymax - ymin
        step = 0.09 * span
        level = 0
        for (a, b) in sorted(itertools.combinations(range(3), 2), key=lambda ab: ab[1] - ab[0]):
            ga, gb = GROUPS[a], GROUPS[b]
            p = row[f"p_holm_{gb}_vs_{ga}"]
            if p >= alpha:
                continue
            y = ymax + step * (level + 1)
            ax.plot([a, a, b, b], [y - 0.3 * step, y, y, y - 0.3 * step], color="k", lw=1)
            ax.text((a + b) / 2, y, stars(p), ha="center", va="bottom", fontsize=11)
            level += 1
        ax.set_ylim(top=ymax + step * (level + 2))
        ax.set_title(f"{col}\nANCOVA p={row['p_group']:.3g}", fontsize=10)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{family}: significant muscles ({'FDR ' if use_fdr else ''}p<{alpha}); "
                 "brackets: Holm-adjusted pairwise comparisons, adjusted by age and sex\n"
                 "* p<0.05  ** p<0.01  *** p<0.001", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[{family}] saved {n} muscle boxplots to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--fdr", action="store_true",
                        help="Select significant muscles using Benjamini-Hochberg FDR-adjusted omnibus p-values")
    parser.add_argument("--include-shortfov", action="store_true", help="Also analyse ShortFOV_* columns")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Subjects after merge: {len(df)}")
    print(df["group"].value_counts().to_string())
    print(f"Missing weight: {df['weight'].isna().sum()}, age: {df['age'].isna().sum()}, "
          f"sex: {df['sex'].isna().sum()}")

    vol_cols, ff_cols = select_variables(df, args.include_shortfov)
    # Volume normalised by weight, mm³/kg
    norm_vol_cols = []
    for c in vol_cols:
        name = f"{c} / kg"
        df[name] = df[c] / df["weight"]
        norm_vol_cols.append(name)

    res_vol = run_family(df, norm_vol_cols, "Volume/weight")
    res_ff = run_family(df, ff_cols, "FF")
    results = pd.concat([res_vol, res_ff], ignore_index=True)
    results.to_csv(args.out_dir / "ancova_results.csv", index=False)
    print(f"Results saved to {args.out_dir / 'ancova_results.csv'}")

    p_col = "p_group_fdr" if args.fdr else "p_group"
    for fam, res in (("Volume/weight", res_vol), ("FF", res_ff)):
        print(f"\n{fam}: significant ({p_col}<{args.alpha}):")
        print(res[res[p_col] < args.alpha][["variable", "n", "F_group", "p_group", "p_group_fdr"]]
              .to_string(index=False))

    boxplot_significant(df, res_vol, "Volume/weight", VOLUME_UNIT_LABEL,
                        args.out_dir / "boxplots_volume_per_weight.png", args.alpha, args.fdr, transform="ml")
    boxplot_significant(df, res_ff, "Fat fraction", "Fat fraction",
                        args.out_dir / "boxplots_ff.png", args.alpha, args.fdr)


if __name__ == "__main__":
    main()
