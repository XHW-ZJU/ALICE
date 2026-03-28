#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from typing import Dict, Tuple, List

# -------------------- Configuration --------------------
FILES = {
    "ALICE_new":   "./data/processed/lib2_ALICE_new.csv",
    "ALICE_old":   "./data/processed/lib2_ALICE_old.csv",
    "Mutate":      "./data/processed/lib2_Mutate.csv",
    "ST":          "./data/processed/lib2_ST.csv",
    "ST_Ranking":  "./data/processed/lib2_ST_Ranking.csv",
}
SEQ_COL_CAND = ("AA_sequence", "iAAs", "sequence")
FITNESS_COL = "fitness"
ENRICH_COL  = "huTfR1fc_log2_enrichment"

K = 3                 # k-mer length
TOPN = 15             # number of top motifs per file
SIZE_MIN_MAX = (200, 1000)   # node size range in pixels
HD_THR = 1            # Hamming distance threshold (≤ HD_THR → connect)

OUT_ROOT = Path("motif_networks_effect_axes")
OUT_ROOT.mkdir(exist_ok=True)

AA_STD = set("ACDEFGHIKLMNPQRSTVWY")

# -------------------- Utility functions --------------------
def get_seq_col(df: pd.DataFrame, prefer=SEQ_COL_CAND) -> str:
    for c in prefer:
        if c in df.columns:
            return c
    raise KeyError(f"Sequence column not found, tried {prefer}")

def clean_seq(s: str) -> str:
    s = str(s).strip().upper()
    return s if s and all(ch in AA_STD for ch in s) else ""

def extract_kmers(seq: str, k: int) -> List[str]:
    return [seq[i:i+k] for i in range(len(seq)-k+1)] if len(seq) >= k else []

def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))  # should be equal length, fallback with max length
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

def normalize_01_from_minmax(x: np.ndarray, mn: float, mx: float) -> np.ndarray:
    denom = (mx - mn)
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.zeros_like(x) + 0.5
    return (x - mn) / denom

def map_size_from_norm(norm_vals: np.ndarray, vmin=SIZE_MIN_MAX[0], vmax=SIZE_MIN_MAX[1]) -> np.ndarray:
    return norm_vals * (vmax - vmin) + vmin

# -------------------- Load & global normalization --------------------
def load_all_files_with_global_norm(files: Dict[str, str]) -> Tuple[Dict[str, pd.DataFrame],
                                                                    float, float,
                                                                    Tuple[float,float], Tuple[float,float]]:
    """
    Read all files, clean sequences, collect raw fitness/enrich values across all rows to compute
    global min/max, then normalize each file's rows to [0,1].
    Returns:
      - per_file_df_norm: {tag: df_norm} containing [seq_col, FITNESS_COL, ENRICH_COL, fitness_norm, enrich_norm]
      - g_fit_mean_norm, g_en_mean_norm: global mean after normalization (for reference 0-line)
      - fit_minmax_raw, en_minmax_raw: raw global (min, max) (for consistency check if needed)
    """
    cleaned = {}
    fit_all, en_all = [], []

    for tag, path in files.items():
        df = pd.read_csv(path)
        seq_col = get_seq_col(df)
        df = df.dropna(subset=[seq_col, FITNESS_COL, ENRICH_COL]).copy()
        df[seq_col] = df[seq_col].map(clean_seq)
        df = df[df[seq_col] != ""]
        if df.empty:
            cleaned[tag] = pd.DataFrame(columns=[seq_col, FITNESS_COL, ENRICH_COL])
            continue
        df[FITNESS_COL] = pd.to_numeric(df[FITNESS_COL], errors="coerce")
        df[ENRICH_COL]  = pd.to_numeric(df[ENRICH_COL],  errors="coerce")
        df = df.dropna(subset=[FITNESS_COL, ENRICH_COL])
        cleaned[tag] = df
        if not df.empty:
            fit_all.append(df[FITNESS_COL].values)
            en_all.append(df[ENRICH_COL].values)

    if not fit_all:
        raise RuntimeError("No valid raw data for global normalization.")

    fit_all = np.concatenate(fit_all)
    en_all  = np.concatenate(en_all)

    fit_min, fit_max = float(np.min(fit_all)), float(np.max(fit_all))
    en_min,  en_max  = float(np.min(en_all)),  float(np.max(en_all))

    per_file_df_norm = {}
    for tag, df in cleaned.items():
        if df.empty:
            per_file_df_norm[tag] = df.assign(fitness_norm=[], enrich_norm=[])
            continue
        f_norm = normalize_01_from_minmax(df[FITNESS_COL].values.astype(float), fit_min, fit_max)
        e_norm = normalize_01_from_minmax(df[ENRICH_COL].values.astype(float),  en_min,  en_max)
        df2 = df.copy()
        df2["fitness_norm"] = f_norm
        df2["enrich_norm"]  = e_norm
        per_file_df_norm[tag] = df2

    g_fit_mean_norm = float(np.mean(normalize_01_from_minmax(fit_all, fit_min, fit_max)))
    g_en_mean_norm  = float(np.mean(normalize_01_from_minmax(en_all,  en_min,  en_max)))

    return per_file_df_norm, g_fit_mean_norm, g_en_mean_norm, (fit_min, fit_max), (en_min, en_max)

# -------------------- Motif aggregation (based on normalized rows) --------------------
def motif_stats_on_file(df_norm: pd.DataFrame,
                        k: int,
                        topn: int,
                        seq_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Input: normalized row-level data for one file
    Output:
      - stats_df: motif-level summary table (motif, fitness_norm_mean, enrich_norm_mean, count, freq)
      - motifs_all: all motifs that appeared in this file (deduplicated)
    """
    if df_norm.empty:
        return (pd.DataFrame(columns=["motif","fitness_norm_mean","enrich_norm_mean","count","freq"]), [])

    motifs_records = []
    motifs_all = set()
    total_seqs = len(df_norm)

    for _, r in df_norm.iterrows():
        ms = extract_kmers(r[seq_col], k)
        ms = sorted(set(ms))
        for m in ms:
            motifs_records.append((m, r["fitness_norm"], r["enrich_norm"]))
            motifs_all.add(m)

    if not motifs_records:
        return (pd.DataFrame(columns=["motif","fitness_norm_mean","enrich_norm_mean","count","freq"]), [])

    mdf = pd.DataFrame(motifs_records, columns=["motif","f_norm","e_norm"])
    stats = (
        mdf.groupby("motif")
           .agg(fitness_norm_mean=("f_norm","mean"),
                enrich_norm_mean =("e_norm","mean"),
                count            =("motif","size"))
           .reset_index()
    )

    stats["freq"] = stats["count"] / total_seqs

    stats = (
        stats.sort_values("freq", ascending=False)
             .head(topn)
             .reset_index(drop=True)
    )

    return stats, sorted(motifs_all)

# -------------------- Effect-aware coordinates (relative to global means) --------------------
def coords_effect_axes(stats_df: pd.DataFrame,
                       g_fit_mean_norm: float,
                       g_en_mean_norm: float) -> Dict[str, Tuple[float,float]]:
    """
    Subtract global normalized means, then divide by SD to obtain z-scores:
      X = z( fitness_norm_mean - g_fit_mean_norm )
      Y = z( enrich_norm_mean  - g_en_mean_norm  )
    """
    x = stats_df["fitness_norm_mean"].values - g_fit_mean_norm
    y = stats_df["enrich_norm_mean"].values  - g_en_mean_norm
    sx = np.std(x) + 1e-9
    sy = np.std(y) + 1e-9
    xz = x / sx
    yz = y / sy
    return {stats_df.loc[i, "motif"]: (xz[i], yz[i]) for i in range(len(stats_df))}

# -------------------- Hamming edge construction --------------------
def build_edges_by_hamming(motifs: List[str], hd_thr: int = HD_THR) -> List[Tuple[str,str,int]]:
    edges = []
    L = len(motifs)
    for i in range(L):
        for j in range(i+1, L):
            d = hamming(motifs[i], motifs[j])
            if d <= hd_thr:
                edges.append((motifs[i], motifs[j], d))
    return edges

# -------------------- Plotting --------------------
def draw_graph(tag: str,
               stats_df: pd.DataFrame,
               edges_w: List[Tuple[str,str,int]],
               pos: Dict[str, Tuple[float,float]],
               out_dir: Path,
               g_fit_mean_norm: float,
               g_en_mean_norm: float,
               show_axes: bool = False):
    """
    Node attributes:
      - node_size: mapped from fitness_norm_mean
      - node_color: enrich_norm_mean ∈ [0,1] → colormap
    Axes:
      - effect-aware z-score coordinates, 0 lines = global normalized means (dashed)
    """
    G = nx.Graph()
    for _, r in stats_df.iterrows():
        G.add_node(r["motif"],
                   fmean_norm=float(r["fitness_norm_mean"]),
                   emean_norm=float(r["enrich_norm_mean"]))
    for u, v, w in edges_w:
        if u in G and v in G:
            G.add_edge(u, v, weight=w)

    fvals = np.array([G.nodes[n]["fmean_norm"] for n in G.nodes], dtype=float)
    evals = np.array([G.nodes[n]["emean_norm"] for n in G.nodes], dtype=float)

    sizes = map_size_from_norm(fvals, *SIZE_MIN_MAX) * 1.0
    colors = evals

    fig, ax = plt.subplots(figsize=(10, 10))

    if G.number_of_edges() > 0:
        w = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
        max_d = max(1.0, float(HD_THR))
        sim = 1.0 - (w / max_d)
        sim = np.clip(sim, 0.0, 1.0)
        nx.draw_networkx_edges(G, pos, alpha=0.35, width=(0.5 + 2.5 * sim), ax=ax)

    nx.draw_networkx_nodes(G, pos,
                           node_size=sizes,
                           node_color=colors,
                           cmap="viridis",
                           vmin=0.0, vmax=1.0,
                           alpha=0.88,
                           ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Mean huTfR1fc_log2_enrichment (globally normalized to [0,1])")

    if show_axes:
        ax.set_axis_on()
        for spine in ax.spines.values():
            spine.set_visible(True)

        xs = np.array([pos[n][0] for n in G.nodes()]) if G.nodes else np.array([0.0])
        ys = np.array([pos[n][1] for n in G.nodes()]) if G.nodes else np.array([0.0])

        def _limits(arr, pad_ratio=0.10, min_span=1.0):
            a_min, a_max = float(np.min(arr)), float(np.max(arr))
            span = max(a_max - a_min, min_span)
            pad = span * pad_ratio
            return (a_min - pad, a_max + pad)

        xlim = _limits(xs)
        ylim = _limits(ys)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        xticks = np.linspace(xlim[0], xlim[1], 7)
        yticks = np.linspace(ylim[0], ylim[1], 7)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{t:.2f}" for t in xticks])
        ax.set_yticklabels([f"{t:.2f}" for t in yticks])

        ax.set_xlabel("ΔFitness (z-score vs global mean of normalized values)")
        ax.set_ylabel("ΔhuTfR1fc_log2_enrichment (z-score vs global mean of normalized values)")

        ax.axhline(0, color="grey", lw=0.9, ls="--", zorder=0)
        ax.axvline(0, color="grey", lw=0.9, ls="--", zorder=0)
        ax.tick_params(axis="both", which="both", labelsize=10, direction="out", length=4)
    else:
        ax.axis("off")

    ax.set_title(f"{tag} | effect-aware motif layout", fontsize=13)
    plt.tight_layout(pad=1.2)
    out_fp = out_dir / f"motif_network_{tag}_{'axes' if show_axes else 'noaxes'}.svg"
    plt.savefig(out_fp, dpi=300)
    plt.close()
    print(f"[OK] {tag:>12s} → {out_fp}")

# -------------------- Main pipeline --------------------
def main():
    # 1) Load and globally normalize all rows ([0,1] separately for fitness/enrich)
    per_file_df_norm, g_fit_mean_norm, g_en_mean_norm, fit_minmax_raw, en_minmax_raw = \
        load_all_files_with_global_norm(FILES)

    # 2) For each file, compute TopN motif stats and counts
    per_file_stats = {}
    per_file_motif_all = {}
    for tag, df_norm in per_file_df_norm.items():
        if df_norm is None or df_norm.empty:
            continue
        seq_col = get_seq_col(df_norm)
        stats, motifs_all = motif_stats_on_file(df_norm, k=K, topn=TOPN, seq_col=seq_col)
        if not stats.empty:
            per_file_stats[tag] = stats
            per_file_motif_all[tag] = motifs_all

    if not per_file_stats:
        print("[ERROR] No valid motif aggregation data")
        return

    # 3) Output directories
    out_noaxes = OUT_ROOT / "layout_effect_noaxes"
    out_axes   = OUT_ROOT / "layout_effect_axes"
    out_noaxes.mkdir(parents=True, exist_ok=True)
    out_axes.mkdir(parents=True,   exist_ok=True)

    # 4) Global motif edge pool (compute Hamming distance for all motifs seen)
    all_motifs_union = sorted(set(m for ml in per_file_motif_all.values() for m in ml))
    global_edges_by_hd = build_edges_by_hamming(all_motifs_union, hd_thr=HD_THR)

    # 5) Draw per file
    for tag, stats_df in per_file_stats.items():
        node_set = set(stats_df["motif"])
        edges_w = [(u, v, d) for (u, v, d) in global_edges_by_hd if (u in node_set and v in node_set)]

        pos = coords_effect_axes(stats_df, g_fit_mean_norm, g_en_mean_norm)

        draw_graph(tag, stats_df, edges_w, pos, out_noaxes, g_fit_mean_norm, g_en_mean_norm, show_axes=False)
        draw_graph(tag, stats_df, edges_w, pos, out_axes,   g_fit_mean_norm, g_en_mean_norm, show_axes=True)


if __name__ == "__main__":
    main()
