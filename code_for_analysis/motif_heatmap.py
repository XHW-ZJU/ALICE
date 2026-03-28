#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------- utils -----------------------
AA_STD = set("ACDEFGHIKLMNPQRSTVWY")


def get_seq_col(df: pd.DataFrame, prefer: list[str]) -> str:
    for c in prefer:
        if c in df.columns:
            return c
    raise KeyError(f"Sequence column not found, tried: {prefer}")


def clean_seqs(series: pd.Series) -> list[str]:
    seqs = []
    for s in series.astype(str).str.strip().str.upper():
        if s and all(ch in AA_STD for ch in s):
            seqs.append(s)
    return seqs


def kmer_counts(seqs: list[str], k: int) -> tuple[Counter, int]:
    """Return (k-mer counts, total number of k-mer windows in this source)"""
    cnt = Counter()
    total_windows = 0
    for s in seqs:
        if len(s) >= k:
            total_windows += (len(s) - k + 1)
            for i in range(len(s) - k + 1):
                cnt[s[i:i + k]] += 1
    return cnt, total_windows


def motif_frequency_matrix(train_cnt: Counter, train_den: int,
                           ai_cnt: Counter, ai_den: int,
                           motifs: list[str]) -> np.ndarray:
    """Rows: Train, AI; Columns: motifs; Values: frequency (count / denom)"""
    mat = np.zeros((2, len(motifs)), dtype=float)
    if train_den > 0:
        mat[0, :] = [train_cnt.get(m, 0) / train_den for m in motifs]
    if ai_den > 0:
        mat[1, :] = [ai_cnt.get(m, 0) / ai_den for m in motifs]
    return mat


def save_freq_table(motifs: list[str], mat: np.ndarray, out_csv: Path):
    df = pd.DataFrame(mat, index=["Train", "AI"], columns=motifs)
    df.to_csv(out_csv, index=True, encoding="utf-8-sig")


def plot_heatmap(motifs: list[str], mat: np.ndarray, k: int, out_png: Path):
    plt.figure(figsize=(max(10, 0.22 * len(motifs)), 3.4))
    sns.heatmap(
        mat,
        cmap="YlGnBu",  # color = frequency
        cbar=True,
        yticklabels=["Train", "AI"],
        xticklabels=motifs,
        vmin=0.0,
        vmax=0.1,
        linewidths=.3
    )
    plt.title(f"Motif frequency heatmap (k={k})  [color = frequency]")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="./data/train_ref_pos.csv", help="Filtered Train CSV")
    ap.add_argument("--ai", default="./data/stage2_sum/seq-epoch-19_pred.csv", help="Filtered AI CSV")
    ap.add_argument("--outdir", default="motif_heatmaps", help="Output directory")
    ap.add_argument("--k", nargs="+", type=int, default=[3, 4, 5], help="List of k-mer lengths")
    ap.add_argument("--topn", type=int, default=20, help="For each k, select topN motifs (by Train+AI total counts)")
    ap.add_argument("--motifs", type=str, default="", help="Comma-separated motif list (overrides topN)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_tr = pd.read_csv(args.train)
    df_ai = pd.read_csv(args.ai)[:500]

    # Find sequence column (Train: iAAs -> seq -> sequence; AI: seq -> sequence)
    tr_col = get_seq_col(df_tr, ["iAAs", "seq", "sequence"])
    ai_col = get_seq_col(df_ai, ["seq", "sequence"])

    train_seqs = clean_seqs(df_tr[tr_col])
    ai_seqs = clean_seqs(df_ai[ai_col])

    # Deduplicate sequences (does not affect denominator definition = sliding windows within each sequence)
    train_seqs = list(dict.fromkeys(train_seqs))
    ai_seqs = list(dict.fromkeys(ai_seqs))

    print(f"[LOAD] Train sequences: {len(train_seqs)} | AI sequences: {len(ai_seqs)}")

    # If user provided fixed motifs, parse them
    fixed_motifs = [m.strip().upper() for m in args.motifs.split(",") if m.strip()] if args.motifs else []

    for k in args.k:
        if fixed_motifs:
            motifs_k = [m for m in fixed_motifs if len(m) == k]
            if not motifs_k:
                print(f"[SKIP] No fixed motifs of length k={k} found; skip this k")
                continue

        # Count k-mers
        tr_cnt, tr_den = kmer_counts(train_seqs, k)
        ai_cnt, ai_den = kmer_counts(ai_seqs, k)
        if tr_den == 0 and ai_den == 0:
            print(f"[INFO] k={k} has no valid windows in either source, skip.")
            continue

        # Select motifs: fixed list or topN
        if fixed_motifs:
            motifs = motifs_k
        else:
            all_cnt = tr_cnt + ai_cnt  # Counter supports addition
            motifs = [m for m, _ in all_cnt.most_common(args.topn)]

        # Frequency matrix
        mat = motif_frequency_matrix(tr_cnt, tr_den, ai_cnt, ai_den, motifs)

        # Sort motifs by AI frequency (descending)
        ai_freqs = mat[1, :]  # AI row
        sort_indices = np.argsort(-ai_freqs)
        motifs_sorted = [motifs[i] for i in sort_indices]
        mat_sorted = mat[:, sort_indices]

        # Export frequency table & plot heatmap
        csv_path = outdir / f"motif_frequency_k{k}.csv"
        png_path = outdir / f"motif_heatmap_k{k}.svg"
        save_freq_table(motifs_sorted, mat_sorted, csv_path)
        plot_heatmap(motifs_sorted, mat_sorted, k, png_path)

        print(f"[DONE] k={k} frequency table → {csv_path.name}; heatmap → {png_path.name}")

    print(f"[DONE] All outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
