#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, warnings, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from collections import Counter
from difflib import SequenceMatcher
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ============ Styling Parameters ============
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# ============ Basic Functions ============
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_epoch_files(exp_dir: Path, target_epochs: list):
    pat = re.compile(r"seq-epoch-(\d+)_pred\.csv$", re.I)
    files = []
    for p in exp_dir.glob("seq-epoch-*_pred.csv"):
        m = pat.match(p.name)
        if m:
            epoch_num = int(m.group(1))
            if epoch_num in target_epochs:
                files.append((epoch_num, p))
    files.sort(key=lambda x: x[0])
    return files

def load_data_top_n(files, top_n=500):
    dfs = []
    for e, p in files:
        df = pd.read_csv(p, nrows=top_n)
        df["epoch"] = e
        dfs.append(df)
        print(f"Epoch {e}: Loaded first {len(df)} samples.")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ============ Distance / Similarity ============
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def calculate_positional_frequencies(sequences: list, pseudocount=1e-9):
    if not sequences or not all(isinstance(s, str) for s in sequences):
        return None
    seq_len = len(sequences[0])
    sequences = [s for s in sequences if len(s) == seq_len]
    if not sequences:
        return None
    freq_matrix = np.zeros((seq_len, len(AMINO_ACIDS)))
    for seq in sequences:
        for pos, aa in enumerate(seq):
            if aa in AA_TO_INT:
                freq_matrix[pos, AA_TO_INT[aa]] += 1
    freq_matrix += pseudocount
    freq_matrix /= freq_matrix.sum(axis=1, keepdims=True)
    return freq_matrix

def calculate_kl_divergence(p_freq, q_freq):
    if p_freq is None or q_freq is None:
        return np.nan
    return np.sum(p_freq * np.log(p_freq / q_freq))

# BLOSUM62 (simplified demo, a full matrix should be used in practice)
BLOSUM62 = {('A','A'):4, ('R','R'):5, ('N','N'):6, ('D','D'):6, ('C','C'):9,
            ('Q','Q'):5, ('E','E'):5, ('G','G'):6, ('H','H'):8, ('I','I'):4,
            ('L','L'):4, ('K','K'):5, ('M','M'):5, ('F','F'):6, ('P','P'):7,
            ('S','S'):4, ('T','T'):5, ('W','W'):11, ('Y','Y'):7, ('V','V'):4}

def blosum62_similarity(seq1, seq2):
    score = 0
    for a, b in zip(seq1, seq2):
        if (a, b) in BLOSUM62: score += BLOSUM62[(a, b)]
        elif (b, a) in BLOSUM62: score += BLOSUM62[(b, a)]
        else: score += BLOSUM62.get((a, a), 0)
    return score

def ngram_similarity(seq1, seq2, n=3):
    ngrams1 = Counter([seq1[i:i+n] for i in range(len(seq1)-n+1)])
    ngrams2 = Counter([seq2[i:i+n] for i in range(len(seq2)-n+1)])
    return sum((ngrams1 & ngrams2).values()) / max(sum(ngrams1.values()), sum(ngrams2.values()), 1)

def lcs_ratio(seq1, seq2):
    matcher = SequenceMatcher(None, seq1, seq2)
    match = matcher.find_longest_match(0, len(seq1), 0, len(seq2))
    return match.size / max(len(seq1), len(seq2))

def cross_entropy(seq1, seq2):
    p = Counter(seq1)
    q = Counter(seq2)
    total_p, total_q = len(seq1), len(seq2)
    return -sum((p[ch]/total_p) * math.log2(q.get(ch,1)/total_q) for ch in p)

def mutual_information(seq1, seq2):
    joint_prob = Counter(zip(seq1, seq2))
    prob1, prob2 = Counter(seq1), Counter(seq2)
    total = len(seq1)
    mi = 0
    for (x,y), count in joint_prob.items():
        p_xy = count / total
        p_x = prob1[x] / total
        p_y = prob2[y] / total
        mi += p_xy * math.log2(p_xy/(p_x*p_y))
    return mi

# ============ Beautified Trajectory Plotting ============
def plot_ref_distance_trajectory(df, outdir, ref_pos, ref_neg, metric_fn, metric_name):
    print(f"Plotting trajectory using {metric_name}...")

    epochs = sorted(df['epoch'].unique())
    results = []
    for epoch in epochs:
        seqs = df[df['epoch']==epoch]['seq'].dropna().tolist()
        if not seqs: continue
        sim_pos = np.mean([np.mean([metric_fn(s,r) for r in ref_pos]) for s in seqs])
        sim_neg = np.mean([np.mean([metric_fn(s,r) for r in ref_neg]) for s in seqs])
        results.append({"epoch":epoch, "sim_pos":sim_pos, "sim_neg":sim_neg})

    if not results: return
    res_df = pd.DataFrame(results)

    # Gradient colors
    cmap = plt.cm.plasma
    norm = plt.Normalize(min(res_df["epoch"]), max(res_df["epoch"]))
    colors = [cmap(norm(e)) for e in res_df["epoch"]]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(res_df["sim_neg"], res_df["sim_pos"], color="lightgray", linewidth=2.5, zorder=1)
    sc = ax.scatter(res_df["sim_neg"], res_df["sim_pos"],
                    c=res_df["epoch"], cmap=cmap,
                    s=80, edgecolors="k", linewidths=0.7, zorder=2)

    # Start and End
    ax.scatter(res_df.iloc[0]["sim_neg"], res_df.iloc[0]["sim_pos"],
               s=180, facecolors="none", edgecolors="green", linewidths=2, label="Start")
    ax.scatter(res_df.iloc[-1]["sim_neg"], res_df.iloc[-1]["sim_pos"],
               s=180, color="red", marker="*", label="End")

    # Annotations
    for _, row in res_df.iterrows():
        if row["epoch"] % 5 == 0 or row["epoch"] in [min(res_df["epoch"]), max(res_df["epoch"])]:
            ax.annotate(str(int(row["epoch"])), (row["sim_neg"], row["sim_pos"]),
                        xytext=(5,5), textcoords="offset points", fontsize=10)

    ax.set_xlabel(f"{metric_name} to Ref-")
    ax.set_ylabel(f"{metric_name} to Ref+")
    ax.set_title(f"Evolutionary Trajectory ({metric_name})", weight="bold")
    # ax.grid(alpha=0.3, linestyle="--")

    fig.colorbar(sc, ax=ax, label="Epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"D6_ref_distance_{metric_name}.svg")
    plt.close(fig)
    print(f"{metric_name} plot saved.")

# ============ Main Pipeline ============
def main():
    exp_dir = Path("./data/stage2_sum/")
    outdir = Path("./ultimate_analysis_multi_metric_beauty")
    ensure_dir(outdir)
    target_epochs = list(range(20))

    # Reference sets
    ref_file = Path("iAAs_dataset.json")
    if not ref_file.exists():
        print(f"Error: Reference file {ref_file} not found."); return
    ref_data = json.load(open(ref_file))
    ref_pos, ref_neg = ref_data.get("Positive", []), ref_data.get("Negative", [])
    if not ref_pos or not ref_neg:
        print("Error: Positive/Negative empty."); return

    # Data
    files = find_epoch_files(exp_dir, target_epochs)
    if not files:
        print("No files found."); return
    ALL_DATA = load_data_top_n(files, top_n=500)
    if ALL_DATA.empty:
        print("No data loaded."); return
    ALL_DATA['seq'] = ALL_DATA['seq'].astype(str).replace('nan','')

    # Multi-metric trajectories
    funcs = [
        partial(plot_ref_distance_trajectory, ref_pos=ref_pos, ref_neg=ref_neg,
                metric_fn=lambda s,r: calculate_kl_divergence(
                    calculate_positional_frequencies([s]),
                    calculate_positional_frequencies([r])
                ), metric_name="KL"),
        partial(plot_ref_distance_trajectory, ref_pos=ref_pos, ref_neg=ref_neg,
                metric_fn=ngram_similarity, metric_name="Ngram"),
        partial(plot_ref_distance_trajectory, ref_pos=ref_pos, ref_neg=ref_neg,
                metric_fn=lcs_ratio, metric_name="LCS"),
    ]

    for f in funcs:
        try:
            f(ALL_DATA, outdir)
        except Exception as e:
            print(f"[ERROR] {f} failed: {e}")

    print(f"\nAnalysis complete. Beautified plots saved in {outdir}")


if __name__ == "__main__":
    main()
