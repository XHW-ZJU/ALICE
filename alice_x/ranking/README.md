# AutoGluon GBM Training for Fitness and hTfR1 Evaluation Models

This script trains **GBM-only regression models** using **AutoGluon Tabular** for two tasks:

* **Production fitness** (`V_mean__P_mean_log2_enr`)
* **hTfR1 binding enrichment** (`hT_mean__V_mean_log2_enr`)

Both models are trained on preprocessed CSV datasets located in `./datapack`.

---

## How It Works

1. **Data loading**

   * From `./datapack/train_fitness_data.csv` and `test_fitness_data.csv` for **fitness**.
   * From `./datapack/train_hT_mean__V_mean_log2_enr_data.csv` and `test_hT_mean__V_mean_log2_enr_data.csv` for **hTfR1**.
   * Non-numeric columns (e.g., `AA_sequence`) are dropped.

2. **Model training**

   * Uses **GBM** models under AutoGluon’s Tabular framework.
   * Runs hyperparameter optimization (HPO) with either `random` search or `bayesopt`.
   * Optimizes **Pearson correlation (`pearsonr`)** by default.

3. **Outputs**

   * Best models saved under `./gen_model_research/aggregate/hTfR1_model/<label>/`.
   * Leaderboards (`*_leaderboard.csv`) and training summaries (`*_fit_summary.json`).
   * Best parameters (`*_best_params.json`).

---

## Usage

Example run:

```bash
python train_eval_models.py \
  --datapack_dir ./datapack \
  --model_dir ./gen_model_research/aggregate/hTfR1_model \
  --time_limit 600 \
  --num_trials 120 \
  --searcher random \
  --optimize_metric pearsonr \
  --presets medium_quality \
  --random_state 10
```

## Notes

* By default, the script will **train both models sequentially**:

  * `V_mean__P_mean_log2_enr` (fitness)
  * `hT_mean__V_mean_log2_enr` (hTfR1 binding)
* Each run produces a leaderboard with metrics: RMSE, MAE, R², Pearson, Spearman, etc.
* Training logs are saved to `logging.txt`.
