# ALICE Evolution Pipeline

This repository contains the two-step evolutionary optimization workflow for ALICE-designed protein/capsid sequences.

The process consists of:

1. **Stage 1 — Function-guided Evolution (`function_guided_evolution.py`)**
   A classic evolutionary search that initializes a sequence population, evaluates candidates against reference positive/negative sets, and optimizes them with evolutionary operators (GA, EDA, SA, EDG).

2. **Stage 2 — Reward-based Evolution (`stage2_evolve.py`)**
   A refinement stage that incorporates predictive models (production fitness and hTfR1 binding) with diversity and distance penalties, exploring reward signal combinations to select promising variants.

---

## 1. Stage 1 — Function-guided Evolution

### Description

* Initializes the population from a CSV containing sequences.
* Evaluates them against **Positive/Negative reference samples** (`sequences.json`).
* Runs for multiple epochs (generations), saving intermediate populations.
* Supports different group/evolutionary strategies (GA, EDA, EDG, SA).

### Usage

```bash
python function_guided_evolution.py \
    --input_file ../data/infer_gen_seqs.csv \
    --sampath ../data/sequences.json \
    --matpath BLOSSUM62.txt \
    --group_type EDG \
    --loss_type basic \
    --epochs 25 \
    --seed 11
```

### Key Arguments

* `--input_file`: Input CSV file with a `seq` column. Only 7-mer sequences are kept.
* `--sampath`: Path to JSON file containing `Positive` and `Negative` reference sequences.
* `--matpath`: Substitution matrix file (default: `BLOSSUM62.txt`).
* `--group_type`: Evolutionary group type (`GA`, `EDA`, `EDG`, `SA`).
* `--loss_type`: Scoring type (`basic`, `triplet`, `npair`, `nce`, `info_nce`).
* `--epochs`: Number of evolutionary generations.
* `--seed`: Random seed for reproducibility.

### Outputs

* **CSV per generation**: `../output/<group_type>/<loss_type>/<...>/seq-epoch-<n>.csv`
* **Figures**: Training curves in `../Figures/<group_type>/<loss_type>/.../`
* **Summary**: `run_summary.json` recording parameters and final outputs.

---

## 2. Stage 2 — Exploration Stage

### Description

* Takes the **final Stage 1 CSV** as input.
* Loads pretrained AutoGluon predictors for **production fitness** and **hTfR1 binding enrichment**.
* Uses external JSON (`sequences.json`) for Ref+/Ref- references.
* Runs multiple modes of reward combination:

  * `sum` (fitness + hTfR1)
  * `min` (minimum of the two)
  * `rank_sum`
  * `sum+div`, `sum+ref`, `min+div`, `min+ref`
  * `all` (combined signals)

### Usage

```bash
python stage2_evolve.py \
    --stage1_final ../data/top1000_sequences.csv \
    --model_root ./gen_model_research/aggregate/hTfR1_model \
    --refjson ../data/sequences.json \
    --epochs 20 \
    --baseline_q 0.5 \
    --out_root ./evolve_sum/
```

### Key Arguments

* `--stage1_final`: The final output CSV from Stage 1 (must include `seq` column).
* `--model_root`: Root directory containing trained AutoGluon predictors.

  * Expected subfolders: `fitness/` and `hT_mean__V_mean_log2_enr/`
* `--refjson`: Reference positive/negative JSON file.
* `--epochs`: Number of generations to evolve.
* `--baseline_q`: Quantile used as baseline threshold for penalizing low predictions.
* `--out_root`: Output directory for Stage 2 results.

### Outputs

* **Per-epoch CSVs**: Contain predicted fitness, hTfR1 binding, diversity scores, and final reward values.
* **Stage-specific folders**: One subfolder per reward mode, e.g. `./evolve_sum/stage2_sum/`.
* **Final evolved sequences** ready for downstream evaluation.

---

## Example Workflow

1. **Run Stage 1 Evolution**

```bash
python function_guided_evolution.py \
    --input_file ../data/infer_gen_seqs.csv \
    --sampath ../data/sequences.json \
    --epochs 25
```

2. **Take the last CSV from Stage 1 and run Stage 2**

```bash
python stage2_evolve.py \
    --stage1_final ../output/EDG/basic/roberta2SeqGAN/seq-epoch-19.csv \
    --model_root ./gen_model_research/aggregate/hTfR1_model \
    --epochs 20
```
