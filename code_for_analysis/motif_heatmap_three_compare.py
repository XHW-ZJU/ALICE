#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse, sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

AA_STD = set("ACDEFGHIKLMNPQRSTVWY")

# ---------------- utils ----------------
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
    cnt = Counter()
    total_windows = 0
    for s in seqs:
        if len(s) >= k:
            total_windows += (len(s) - k + 1)
            for i in range(len(s) - k + 1):
                cnt[s[i:i+k]] += 1
    return cnt, total_windows

def motif_frequency_matrix(cnts: list[Counter], dens: list[int], motifs: list[str]) -> np.ndarray:
    mat = np.zeros((len(cnts), len(motifs)), dtype=float)
    for i, (cnt, den) in enumerate(zip(cnts, dens)):
        if den > 0:
            mat[i, :] = [cnt.get(m, 0) / den for m in motifs]
    return mat

def save_freq_table(sources: list[str], motifs: list[str], mat: np.ndarray, out_csv: Path):
    df = pd.DataFrame(mat, index=sources, columns=motifs)
    df.to_csv(out_csv, index=True, encoding="utf-8-sig")

def plot_heatmap(sources: list[str], motifs: list[str], mat: np.ndarray, k: int, out_png: Path):
    plt.figure(figsize=(max(10, 0.22*len(motifs)), 4.0))
    sns.heatmap(
        mat,
        cmap="YlGnBu",
        cbar=True,
        yticklabels=sources,
        xticklabels=motifs,
        vmin=0.0,
        vmax=0.1,
        linewidths=.3
    )
    plt.title(f"Motif frequency heatmap (k={k})  [AI-sorted]")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic", default="./data/all_train_and_tst_data.csv")
    ap.add_argument("--train",    default="./data/train_ref_pos.csv")
    ap.add_argument("--ai",       default="./data/lib2_ALICE_new.csv")
    ap.add_argument("--outdir",   default="motif_heatmaps")
    ap.add_argument("--k", nargs="+", type=int, default=[3,4,5])
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--motifs", type=str, default="", help="Fixed motif list, comma-separated (overrides topN)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_sem = pd.read_csv(args.semantic)
    df_tr  = pd.read_csv(args.train)
    df_ai  = pd.read_csv(args.ai)

    sem_col = get_seq_col(df_sem, ["iAAs", "AA_sequence", "sequence"])
    tr_col  = get_seq_col(df_tr,  ["iAAs", "AA_sequence", "sequence"])
    ai_col  = get_seq_col(df_ai,  ["AA_sequence", "sequence"])

    sem_seqs = list(dict.fromkeys(clean_seqs(df_sem[sem_col])))
    tr_seqs  = list(dict.fromkeys(clean_seqs(df_tr[tr_col])))
    ai_seqs  = list(dict.fromkeys(clean_seqs(df_ai[ai_col])))

    print(f"[LOAD] Semantic={len(sem_seqs)} Train={len(tr_seqs)} AI={len(ai_seqs)}")

    fixed_motifs = [m.strip().upper() for m in args.motifs.split(",") if m.strip()] if args.motifs else []
    sources = ["Semantic", "Train", "AI"]

    for k in args.k:
        # Count k-mers
        sem_cnt, sem_den = kmer_counts(sem_seqs, k)
        tr_cnt,  tr_den  = kmer_counts(tr_seqs,  k)
        ai_cnt,  ai_den  = kmer_counts(ai_seqs,  k)
        if all(den == 0 for den in [sem_den, tr_den, ai_den]):
            print(f"[INFO] k={k} has no valid windows, skip.")
            continue

        # Select motifs
        if fixed_motifs:
            motifs = [m for m in fixed_motifs if len(m) == k]
            if not motifs:
                print(f"[SKIP] No fixed motifs of length {k}")
                continue
        else:
            motifs = [m for m, _ in ai_cnt.most_common(args.topn)]

        # Frequency matrix
        mat = motif_frequency_matrix([sem_cnt, tr_cnt, ai_cnt],
                                     [sem_den, tr_den, ai_den],
                                     motifs)

        # Export
        csv_path = outdir / f"motif_frequency_k{k}.csv"
        png_path = outdir / f"motif_heatmap_k{k}.svg"
        save_freq_table(sources, motifs, mat, csv_path)
        plot_heatmap(sources, motifs, mat, k, png_path)

        print(f"[DONE] k={k} → {csv_path.name}, {png_path.name}")

    print(f"[DONE] Output directory: {outdir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
