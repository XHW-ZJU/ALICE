import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import List, Tuple, Dict
import matplotlib

matplotlib.use('TkAgg')

word_dict = {}

# Define amino acid alphabet including padding and unknown token
alphabet = ['<PAD>', 'X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'V', 'W', 'Y']

for i, aa in enumerate(alphabet):
    word_dict[aa] = i


# Convert amino acid sequence into numeric representation
def sequence_to_numeric(sequence, ngram=1):
    # Convert sequence to CNN-style numeric input
    sequence = sequence.upper()
    word_list = [sequence[i:i + ngram] for i in range(len(sequence) - ngram + 1)]
    output = []
    for word in word_list:
        if word not in alphabet:
            output.append(word_dict['X'])
        else:
            output.append(word_dict[word])

    return np.array(output, np.int32)


def processed_seqs(seqs_dict: Dict,
                   index_seqs: List = ['gen', 'nnk', 'normal'],
                   annotate: str = None):
    """
    Convert sequence dictionary into numeric arrays and optionally return index of annotated sequence.
    """
    all_seqs_lst, all_numeric_seqs_lst, all_array_records = [], [], []
    for seq_name in index_seqs:
        tmp_seqs = seqs_dict[seq_name]
        all_seqs_lst.append(tmp_seqs)
        numeric_tmp_seqs = [sequence_to_numeric(seq) for seq in tmp_seqs]
        all_numeric_seqs_lst.append(numeric_tmp_seqs)
        all_array_records.append(np.array(numeric_tmp_seqs))

    if annotate:
        # Find index of the annotated sequence in flattened list
        flat_list = [item for sublist in all_seqs_lst for item in sublist]
        sequence_index = flat_list.index(annotate)
        return seqs_dict, all_seqs_lst, all_numeric_seqs_lst, all_array_records, sequence_index

    return seqs_dict, all_seqs_lst, all_numeric_seqs_lst, all_array_records, None


def load_data(data_path: str or pd.DataFrame,
              index_seqs: List = ['gen', 'nnk', 'normal'],
              annotate: str = None):
    """
    Load sequence data from Excel or DataFrame and convert into numeric arrays.
    """
    if isinstance(data_path, str):
        df = pd.read_excel(data_path)
    else:
        df = data_path

    all_seqs_lst, all_numeric_seqs_lst, all_array_records = [], [], []
    for seq_name in index_seqs:
        tmp_seqs = df[seq_name].astype(str).tolist()
        all_seqs_lst.append(tmp_seqs)
        numeric_tmp_seqs = [sequence_to_numeric(seq) for seq in tmp_seqs]
        all_numeric_seqs_lst.append(numeric_tmp_seqs)
        all_array_records.append(np.array(numeric_tmp_seqs))

    if annotate:
        # Find index of annotated sequence in DataFrame
        sequence_index = df.index[df['nnk'] == annotate].tolist()[0]
        return df, all_seqs_lst, all_numeric_seqs_lst, all_array_records, sequence_index

    return df, all_seqs_lst, all_numeric_seqs_lst, all_array_records, None


def umap_processing(input_array: np.array,
                    n_neighbors: int,
                    min_dist: float,
                    rand_stat: int,
                    epoch_n: int,
                    learning_rate: float,
                    sequence_index=None):
    """
    Run UMAP dimensionality reduction on sequence data.
    """
    # Standardize feature vectors
    scaler = StandardScaler()
    scaled_feature_vectors = scaler.fit_transform(input_array)

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='jaccard',
        n_epochs=epoch_n,
        learning_rate=learning_rate,
        random_state=rand_stat,
        init='pca'
    )

    embedding = umap_model.fit_transform(scaled_feature_vectors)

    # Density estimation (for visualization)
    kde = gaussian_kde(embedding.T)

    if sequence_index:
        # Coordinates of highlighted sequence
        highlight_x = embedding[sequence_index, 0]
        highlight_y = embedding[sequence_index, 1]
        return embedding, kde, highlight_x, highlight_y
    else:
        return embedding, kde, None, None


def umap_plot(numeric_seqs_lst: List,
              color_map: List,
              label_name: List,
              embedding,
              kde,
              outpath: str,
              annotate: str = None,
              x_and_y: Tuple = None):
    """
    Plot UMAP embedding with optional density contours and sequence annotation.
    """
    assert len(numeric_seqs_lst) == len(color_map) == len(label_name)
    c_ = [color for color, numeric_seq in zip(color_map, numeric_seqs_lst) for _ in numeric_seq]

    plt.scatter(embedding[:, 0], embedding[:, 1], c=c_, s=5)

    # Set extended plot boundaries for density
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x, y = np.mgrid[x_min - 0.1 * x_range: x_max + 0.1 * x_range: 100j,
                    y_min - 0.1 * y_range: y_max + 0.1 * y_range: 100j]

    # Legend
    handle_lst = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i],
                             markersize=5, label=label_name[i]) for i in range(len(color_map))]

    if annotate:
        highlight_x, highlight_y = x_and_y
        # Mark specific reference sequence
        plt.scatter(highlight_x, highlight_y, c='red', s=50, label='RefSeq')
        plt.annotate(annotate, (highlight_x, highlight_y), textcoords="offset points", xytext=(0, 10),
                     ha='center')
        # Add custom legend entry
        custom_handle = plt.scatter([], [], c='red', s=50, label='RefSeq')
        handle_lst.append(custom_handle)

    plt.legend(handles=handle_lst, loc='best')
    plt.title('UMAP Embedding with Density')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.colorbar(label='Density')
    plt.savefig(outpath)
    plt.close()


if __name__ == '__main__':
    # ---------------------------- Load Sequences ----------------------------
    seqs_name = ['Pos Ref+', 'Neg Ref-', 'epoch 0', 'epoch 5', 'epoch 10', 'epoch 15', 'epoch 20']
    total_seqs_lst = []

    # Load reference positive/negative sets
    json_read = json.load(open('./data/ref_dataset.json', 'r'))
    pos_seqs = json_read['Positive']
    neg_seqs = json_read['Negative']

    # Load evolution sequences (first 500 per epoch)
    ep0_seqs = pd.read_csv('./data/stage2_sum/seq-epoch-0_pred.csv')['seq'].tolist()[:500]
    ep5_seqs = pd.read_csv('./data/stage2_sum/seq-epoch-4_pred.csv')['seq'].tolist()[:500]
    ep10_seqs = pd.read_csv('./data/stage2_sum/seq-epoch-9_pred.csv')['seq'].tolist()[:500]
    ep15_seqs = pd.read_csv('./data/stage2_sum/seq-epoch-14_pred.csv')['seq'].tolist()[:500]
    ep20_seqs = pd.read_csv('./data/stage2_sum/seq-epoch-19_pred.csv')['seq'].tolist()[:500]

    total_seqs_lst.extend([pos_seqs, neg_seqs, ep0_seqs, ep5_seqs, ep10_seqs, ep15_seqs, ep20_seqs])

    color_map = [
        '#FF00FF',
        '#516770',
        '#fad0d4',  # red
        '#faf7ca',  # yellow
        '#bcf1d5',  # green
        '#c4ddf1',  # blue
        '#e4d8e6',  # purple
    ]

    assert len(seqs_name) == len(total_seqs_lst)
    seqs_dict = {k: v for k, v in zip(seqs_name, total_seqs_lst)}

    seq_df, all_seqs, all_numeric_seqs, all_seqs_array, _ = processed_seqs(seqs_dict, seqs_name)

    # Merge all sequences into one array
    sequences_array = np.concatenate(all_seqs_array)

    neighbors = 18
    min_dists = 0.9
    rand_states = 22
    lr = 0.3
    epochs = 100

    embedding, kde, highlight_x, highlight_y = umap_processing(
        sequences_array, neighbors, min_dists, rand_states, epochs, lr
    )

    t_outpath = './umap_plot/epoch_among_evolution.svg'
    umap_plot(all_numeric_seqs, color_map, seqs_name, embedding, kde, t_outpath)
