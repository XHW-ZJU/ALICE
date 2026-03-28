#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Input / Output
IN_JSON = "./data/ref_dataset.json"   # Path to JSON file
OUT_POS_SVG = "positive_weblogo.svg"
OUT_NEG_SVG = "negative_weblogo.svg"

# 20 standard amino acids
AA_STD = set("ACDEFGHIKLMNPQRSTVWY")

def build_color_scheme():
    return {
        # Acidic
        "D": "#E41A1C", "E": "#E41A1C",
        # Basic
        "K": "#377EB8", "R": "#377EB8", "H": "#377EB8",
        # Polar uncharged
        "S": "#984EA3", "T": "#984EA3", "N": "#984EA3", "Q": "#984EA3",
        "Y": "#984EA3", "C": "#984EA3",
        # Hydrophobic
        "A": "#4DAF4A", "V": "#4DAF4A", "L": "#4DAF4A", "I": "#4DAF4A",
        "M": "#4DAF4A", "F": "#4DAF4A", "W": "#4DAF4A",
        "P": "#4DAF4A", "G": "#4DAF4A"
    }

def filter_standard_amino_acids(seqs: pd.Series) -> pd.Series:
    mask = seqs.apply(lambda s: all(ch in AA_STD for ch in s))
    return seqs[mask].reset_index(drop=True)

def select_modal_length(seqs: pd.Series) -> pd.Series:
    length_counts = seqs.map(len).value_counts()
    modal_len = int(length_counts.idxmax())
    return seqs[seqs.map(len) == modal_len].reset_index(drop=True)

def plot_weblogo(seqs: pd.Series, title: str, out_svg: Path):
    import logomaker

    alignment = list(seqs.values)
    # Count matrix
    counts_mat = logomaker.alignment_to_matrix(
        alignment,
        to_type="counts",
        characters_to_ignore=""
    )
    # Convert to probability matrix
    prob_mat = counts_mat.div(counts_mat.sum(axis=1), axis=0)

    colors = build_color_scheme()

    fig, ax = plt.subplots(figsize=(max(6, prob_mat.shape[1] * 0.35), 4.2))
    logo = logomaker.Logo(
        prob_mat,
        ax=ax,
        color_scheme=colors,
        fade_probabilities=False,
        show_spines=False,
        vpad=0.02,
        width=0.8,
    )

    logo.style_spines(visible=False)
    logo.style_xticks(anchor=0, spacing=1, rotation=0)
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"[SAVED] {out_svg}")

def main():
    # Load JSON
    with open(IN_JSON, "r") as f:
        data = json.load(f)

    for group, out_svg in [
        ("Positive", OUT_POS_SVG),
        ("Negative", OUT_NEG_SVG),
    ]:
        seqs = pd.Series(data.get(group, []), dtype=str)
        if seqs.empty:
            print(f"[WARN] No {group} sequences in JSON")
            continue

        # Cleaning
        seqs = seqs.str.strip().str.upper()
        seqs = filter_standard_amino_acids(seqs)
        seqs = select_modal_length(seqs)

        print(f"[INFO] {group} sequences: {len(seqs)}, length: {len(seqs.iloc[0]) if len(seqs)>0 else 'NA'}")

        if len(seqs) < 2:
            print(f"[ERROR] Not enough {group} sequences to draw WebLogo")
            continue

        plot_weblogo(seqs, f"{group} WebLogo", out_svg)

if __name__ == "__main__":
    main()
