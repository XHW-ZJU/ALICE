import os, re, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================
# Utility Functions
# =========================
def ensure_dir(p: Path):
    """Ensure the directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def find_epoch_files(exp_dir: Path, target_epochs: list):
    """
    Find target epoch CSV files in the format 'seq-epoch-X_pred.csv'.
    """
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
    """
    Load only the top N rows from each file.
    """
    dfs = []
    for e, p in files:
        df = pd.read_csv(p, nrows=top_n)
        df["epoch"] = e
        dfs.append(df)
        print(f"Epoch {e}: Loaded first {len(df)} samples.")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# =========================
# KL Divergence Functions
# =========================
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def calculate_positional_frequencies(sequences: list, pseudocount=1e-9):
    """
    Compute positional amino acid frequency matrix.
    Shape: (seq_length, num_amino_acids).
    """
    if not sequences or not all(isinstance(s, str) for s in sequences):
        return None

    seq_len = len(sequences[0])
    if not all(len(s) == seq_len for s in sequences):
        # Keep only consistent-length sequences
        sequences = [s for s in sequences if len(s) == seq_len]
        if not sequences:
            return None

    freq_matrix = np.zeros((seq_len, len(AMINO_ACIDS)))
    for seq in sequences:
        for pos, aa in enumerate(seq):
            if aa in AA_TO_INT:
                freq_matrix[pos, AA_TO_INT[aa]] += 1

    # Add pseudocount and normalize
    freq_matrix += pseudocount
    freq_matrix /= freq_matrix.sum(axis=1, keepdims=True)
    return freq_matrix


def calculate_kl_divergence(p_freq, q_freq):
    """
    Compute KL divergence D_KL(P || Q).
    """
    if p_freq is None or q_freq is None:
        return np.nan
    return np.sum(p_freq * np.log(p_freq / q_freq))


# =========================
# Main Plot Function
# =========================
def plot_ref_distance_trajectory(df, outdir, ref_pos, ref_neg):
    """
    Plot trajectory of KL distances to Positive and Negative reference sets.
    """
    print("Plotting: KL Distance Trajectory to Reference Sets...")

    # Compute frequency matrices for references
    freq_pos = calculate_positional_frequencies(ref_pos)
    freq_neg = calculate_positional_frequencies(ref_neg)
    if freq_pos is None or freq_neg is None:
        print("[ERROR] Reference frequency calculation failed.")
        return

    # Iterate over epochs
    distances = []
    for epoch in sorted(df['epoch'].unique()):
        epoch_seqs = df[df['epoch'] == epoch]['seq'].dropna().tolist()
        if not epoch_seqs:
            continue

        freq_epoch = calculate_positional_frequencies(epoch_seqs)
        if freq_epoch is None:
            continue

        distances.append({
            "epoch": epoch,
            "dist_to_pos": calculate_kl_divergence(freq_epoch, freq_pos),
            "dist_to_neg": calculate_kl_divergence(freq_epoch, freq_neg)
        })

    if not distances:
        print("[WARN] No distances computed.")
        return

    # Convert to DataFrame for plotting
    dist_df = pd.DataFrame(distances)

    plt.figure(figsize=(10, 8))
    plt.plot(dist_df["dist_to_neg"], dist_df["dist_to_pos"], marker="o", color="purple", label="Trajectory")

    # Annotate epochs
    for _, row in dist_df.iterrows():
        plt.annotate(str(int(row["epoch"])), (row["dist_to_neg"], row["dist_to_pos"]), xytext=(5, 5), textcoords="offset points")

    # Mark start and end
    plt.scatter(dist_df.iloc[0]["dist_to_neg"], dist_df.iloc[0]["dist_to_pos"], s=150, facecolors='none', edgecolors='g', linewidth=2, label="Start")
    plt.scatter(dist_df.iloc[-1]["dist_to_neg"], dist_df.iloc[-1]["dist_to_pos"], s=100, c='red', marker='*', label="End")

    plt.xlabel("KL Distance to Negative Set")
    plt.ylabel("KL Distance to Positive Set")
    plt.title("KL Distance Trajectory (First 500 per epoch)")
    plt.legend()
    plt.savefig(outdir / "ref_distance_trajectory_first500.svg")
    plt.close()
    print("plot saved.")


# =========================
# Main Workflow
# =========================
def main():
    exp_dir = Path("./data/stage2_sum/")
    outdir = Path("./trajectory_analysis/")
    ensure_dir(outdir)

    target_epochs = list(range(20))

    # Load reference sequences
    ref_file = Path("ref_dataset.json")
    if not ref_file.exists():
        print("Error: Reference JSON not found.")
        return
    with open(ref_file, 'r') as f:
        ref_data = json.load(f)
    ref_pos, ref_neg = ref_data.get("Positive", []), ref_data.get("Negative", [])
    if not ref_pos or not ref_neg:
        print("Error: Empty reference sets.")
        return

    # Load data
    files = find_epoch_files(exp_dir, target_epochs)
    if not files:
        print("Error: No epoch files found.")
        return
    df = load_data_top_n(files, top_n=500)
    if df.empty:
        print("Error: No data loaded.")
        return

    # Preprocess sequences
    if 'seq' in df.columns:
        df['seq'] = df['seq'].astype(str).replace('nan', '')

    print(f"Total samples: {len(df)} | Epochs: {sorted(df['epoch'].unique())}")

    # Run only D6 plot
    plot_ref_distance_trajectory(df, outdir, ref_pos, ref_neg)

    print(f"\nAnalysis complete. Plot saved in: {outdir}")


if __name__ == "__main__":
    main()
