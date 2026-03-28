#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless environment
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

# ---------- Configuration ----------
CSV_PATH = Path("./data/split_files/lib2_ALICE_new.csv")
OUTDIR   = Path("./data")  # can be changed to Path("./outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Column name constants (must match the provided CSV)
COL_AA        = "AA_sequence"
COL_FIT       = "fitness"  # Note: will be recalculated and overwritten
COL_FIT_PRED  = "fitness prediction"
COL_HT_PRED   = "hTFR1 enrichment prediction"
COL_HU_EXP    = "huTfR1_fc_log2_enrichment"
COL_FMB       = "final_merge_brain_log2_enrichment"
COL_H68       = "H68_brain_log2_enrichment"
COL_VIRUS3    = "virus3_mean"
COL_PLASMID   = "plasmid_mean"

# ---------- Utility Functions ----------
def to_float(s: pd.Series) -> pd.Series:
    """Generic and robust string-to-numeric conversion."""
    ss = s.astype(str).str.strip()
    ss = ss.str.replace("\u2212", "-", regex=False).str.replace(",", "", regex=False)
    num = ss.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(num, errors="coerce")

def corr_stats(y_true, y_pred):
    """Return (Pearson, Spearman, R², n)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(m.sum())
    if n < 2:
        return np.nan, np.nan, np.nan, n
    yt, yp = y_true[m], y_pred[m]
    p = stats.pearsonr(yp, yt)[0]
    s = stats.spearmanr(yp, yt)[0]
    r2 = r2_score(yt, yp)
    return float(p), float(s), float(r2), n

def plot_one(data: pd.DataFrame, x: str, y: str, title: str, outfile_stem: str, dpi: int = 320):
    """Plot one scatter plot with statistics annotated; save as PNG+SVG."""
    v = data[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(v[x], v[y], s=30, alpha=0.85, color="#75C2DF")
    p, s, r2, n = corr_stats(v[y].values, v[x].values)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title)
    ax.text(0.02, 0.98, f"Pearson={p:.4f}\nSpearman={s:.4f}\nR²={r2:.4f}\nn={n}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout()
    png = OUTDIR / f"{outfile_stem}.png"
    svg = OUTDIR / f"{outfile_stem}.svg"
    fig.savefig(png, dpi=dpi)
    fig.savefig(svg)
    plt.close(fig)
    print(f"[SAVE] {png}")
    print(f"[SAVE] {svg}")

# ---------- Main Pipeline ----------
def main():
    # Read
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Standardize AA_sequence
    if COL_AA not in df.columns:
        raise KeyError(f"Missing required column: {COL_AA}")
    df[COL_AA] = df[COL_AA].astype(str).str.upper().str.strip()

    # Check duplicates
    dup_mask = df.duplicated(subset=[COL_AA], keep=False)
    dup_df = df.loc[dup_mask, [COL_AA]].copy()
    if not dup_df.empty:
        dup_counts = dup_df.value_counts(subset=[COL_AA]).reset_index(name="count")
        dup_csv = OUTDIR / "duplicates_AA_sequence.csv"
        dup_counts.to_csv(dup_csv, index=False)
        print(f"[DUPLICATES] Found {dup_counts.shape[0]} duplicate AA_sequence(s):")
        print(dup_counts.head(20))
        print(f"[DUPLICATES] Full list exported to: {dup_csv}")
    else:
        print("[DUPLICATES] No duplicates in AA_sequence.")

    # Numeric conversion / column check
    for c in [COL_FIT_PRED, COL_HT_PRED, COL_HU_EXP, COL_FMB, COL_H68, COL_VIRUS3, COL_PLASMID]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")
        df[c] = to_float(df[c])

    # Replace original fitness with log2(virus3_mean / plasmid_mean)
    df[COL_FIT] = np.log2(df[COL_VIRUS3] / df[COL_PLASMID])

    # Derived columns
    df["__y_mean__"] = (df[COL_FIT] + df[COL_HU_EXP]) / 2.0
    df["__x_mean__"] = (df[COL_FIT_PRED] + df[COL_HT_PRED]) / 2.0
    df["delta_enr"]  = df[COL_FMB] - df[COL_H68]

    # 1) Fitness vs Fitness Prediction
    plot_one(df, COL_FIT_PRED, COL_FIT,
             "Fitness vs Fitness Prediction (v6 set)",
             "scatter_fit_vs_fitpred_v6", dpi=320)

    # 2) huTfR1fc (log2) vs hTFR1 enrichment prediction
    plot_one(df, COL_HT_PRED, COL_HU_EXP,
             "huTfR1fc (log2) vs hTFR1 Prediction (v6 set)",
             "scatter_hu_vs_htpred_v6", dpi=320)

    # 3) Mean((fit, huTfR1fc)) vs Mean((fit_pred, ht_pred))
    plot_one(df, "__x_mean__", "__y_mean__",
             "Mean((fit, huTfR1fc)) vs Mean((fit_pred, ht_pred)) (v6 set)",
             "scatter_mean_pair_v6", dpi=320)


if __name__ == "__main__":
    main()
