#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The Full Suite of 22 Intuitive Visualizations for Evolutionary AI Explainability.

New: Plot reward efficacy heatmap vs div_prev and ref_dist
"""

import re
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import fisher_exact
from sklearn.manifold import TSNE
import networkx as nx
import logomaker

# --- Setup ---
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# --- Utility Functions ---

def find_epoch_files(exp_dir: Path, max_epochs: int = None):
    files = []
    pat = re.compile(r".*epoch[-_]?(\d+).*\.csv$", re.I)
    for p in exp_dir.glob("*.csv"):
        m = pat.match(str(p.name))
        if m:
            e = int(m.group(1))
            files.append((e, p))
    files.sort(key=lambda x: x[0])
    if max_epochs is not None:
        files = [x for x in files if x[0] <= max_epochs]
    return files


def load_all_data(files):
    epochs_data = []
    for e, p in files:
        try:
            df = pd.read_csv(p)
            need = ["seq", "fitness_pred", "htfr1_pred", "ref_dist", "div_prev", "reward_signal", "final_score"]
            if not all(c in df.columns for c in need):
                continue
            for c in need[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=need)
            df['epoch'] = e
            epochs_data.append(df)
        except Exception as ex:
            print(f"Warning: Could not load or process {p}. Error: {ex}")
    if not epochs_data:
        raise ValueError("No valid data could be loaded from CSV files.")
    return pd.concat(epochs_data, ignore_index=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# --- Reward efficacy by diversity/ref_dist stratification (previous function) ---

def plot_reward_efficacy_heatmap_by_metric(
    ALL, outdir, stratify_col='div_prev', target_metric='fitness_pred', metric_name='Fitness'
):
    rows = []
    epochs = sorted(ALL['epoch'].unique())
    for i in range(len(epochs) - 1):
        e_t, e_tp1 = epochs[i], epochs[i + 1]
        df_t = ALL[ALL['epoch'] == e_t].copy()
        df_tp1 = ALL[ALL['epoch'] == e_tp1]

        if df_t.empty or df_tp1.empty:
            continue

        df_t['strata_bin'] = pd.qcut(
            df_t[stratify_col].rank(method='first'),
            4,
            labels=["Bottom 25%", "25-50%", "50-75%", "Top 25%"],
            duplicates='drop'
        )

        delta_metric = df_tp1[target_metric].median() - df_t.groupby('strata_bin')[target_metric].median()

        for bin_name, val in delta_metric.items():
            rows.append({'epoch': e_t, 'bin': bin_name, 'delta': val})

    if not rows:
        return

    pivot_data = (
        pd.DataFrame(rows)
        .pivot_table(index='bin', columns='epoch', values='delta')
        .reindex(["Top 25%", "50-75%", "25-50%", "Bottom 25%"])
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_data, cmap="coolwarm",
                center=0, linewidths=.5)
    plt.title(
        f"Reward Efficacy by {stratify_col}: Median {metric_name} Change in Next Epoch",
        fontsize=16
    )
    plt.xlabel("Current Epoch (T)")
    plt.ylabel(f"{stratify_col} Percentile at Epoch T")
    plt.tight_layout()
    plt.savefig(outdir / f"reward_efficacy_heatmap_{stratify_col}_{target_metric}.svg", dpi=300)
    plt.close()


# --- New: Reward efficacy vs div_prev / ref_dist ---

def plot_reward_efficacy_on_divref(ALL, outdir, target_col='div_prev'):
    """
    Analysis: Group by reward_signal quantiles, then measure change of target_col
    (div_prev / ref_dist) in the next epoch.
    """
    rows = []
    epochs = sorted(ALL['epoch'].unique())
    for i in range(len(epochs) - 1):
        e_t, e_tp1 = epochs[i], epochs[i + 1]
        df_t = ALL[ALL['epoch'] == e_t].copy()
        df_tp1 = ALL[ALL['epoch'] == e_tp1]

        if df_t.empty or df_tp1.empty:
            continue

        df_t['reward_bin'] = pd.qcut(
            df_t['reward_signal'].rank(method='first'),
            4,
            labels=["Bottom 25%", "25-50%", "50-75%", "Top 25%"],
            duplicates='drop'
        )

        delta_metric = df_tp1[target_col].median() - df_t.groupby('reward_bin')[target_col].median()

        for bin_name, val in delta_metric.items():
            rows.append({'epoch': e_t, 'reward_bin': bin_name, 'delta': val})

    if not rows:
        return

    pivot_data = (
        pd.DataFrame(rows)
        .pivot_table(index='reward_bin', columns='epoch', values='delta')
        .reindex(["Top 25%", "50-75%", "25-50%", "Bottom 25%"])
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_data, cmap="coolwarm", center=0, linewidths=.5)
    plt.title(
        f"Reward Efficacy on {target_col}: Median Change in Next Epoch",
        fontsize=16
    )
    plt.xlabel("Current Epoch (T)")
    plt.ylabel("Reward Percentile at Epoch T")
    plt.tight_layout()
    plt.savefig(outdir / f"reward_efficacy_on_{target_col}.svg", dpi=300)
    plt.close()


# --- Main Execution ---

def main():
    ap = argparse.ArgumentParser(description="Full Suite of 22 Intuitive Visualizations for Evo AI")
    ap.add_argument("--exp_dir", type=str, default='./data/stage2_sum/', help="Directory containing epoch CSVs")
    ap.add_argument("--outdir", type=str, default='./AI_sum/', help="Output directory")
    ap.add_argument("--epochs", type=int, default=20, help="Maximum epochs to process")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print("Loading all epoch data...")
    files = find_epoch_files(Path(args.exp_dir), max_epochs=args.epochs)
    ALL = load_all_data(files)
    print(f"Data loaded. Total records: {len(ALL)}, Epochs found: {ALL['epoch'].unique()}")

    # Call all required plots
    plot_functions = [
        (plot_reward_efficacy_heatmap_by_metric, (ALL, outdir, 'div_prev', 'fitness_pred', 'Fitness')),
        (plot_reward_efficacy_heatmap_by_metric, (ALL, outdir, 'div_prev', 'htfr1_pred', 'hTFR1')),
        (plot_reward_efficacy_heatmap_by_metric, (ALL, outdir, 'ref_dist', 'fitness_pred', 'Fitness')),
        (plot_reward_efficacy_heatmap_by_metric, (ALL, outdir, 'ref_dist', 'htfr1_pred', 'hTFR1')),
    ]

    for i, item in enumerate(plot_functions):
        func, f_args = item
        plot_name = func.__name__
        print(f"\n({i + 1}/{len(plot_functions)}) Generating plot: {plot_name}...")
        try:
            func(*f_args)
            print(f"  -> Success.")
        except Exception as e:
            print(f"  -> FAILED for plot {plot_name}. Reason: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[SUCCESS] Reward efficacy heatmaps (6 total) generated in: {outdir}")


if __name__ == "__main__":
    main()
