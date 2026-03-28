# coding=utf-8
import os
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Dict, Any
import json
import random

import numpy as np
import pandas as pd


try:
    from autogluon.core import space as ag_space
    AG_HAS_SPACE = True
except Exception:
    ag_space = None
    AG_HAS_SPACE = False

from autogluon.tabular import TabularDataset, TabularPredictor


logging.basicConfig(
    filename='./logging.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    label: str
    model_path: Path
    model_name: str
    time_limit: int = 600
    presets: str = 'medium_quality'
    num_trials: int = 120
    searcher: str = 'random'        
    optimize_metric: str = 'pearsonr'
    random_state: int = 10


def load_processed_datapack(datapack_dir: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label == "V_mean__P_mean_log2_enr":
        train_path = datapack_dir / "train_fitness_data.csv"
        test_path  = datapack_dir / "test_fitness_data.csv"
    elif label == "hT_mean__V_mean_log2_enr":
        train_path = datapack_dir / "train_hT_mean__V_mean_log2_enr_data.csv"
        test_path  = datapack_dir / "test_hT_mean__V_mean_log2_enr_data.csv"
    else:
        raise ValueError(f"Unsupported label: {label}")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed CSVs not found: {train_path} / {test_path}")

    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    for df in (df_tr, df_te):
        df.drop(columns=[c for c in ["AA_sequence"] if c in df.columns],
                inplace=True, errors="ignore")
    return df_tr, df_te


class AutoGBMTrainer:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.cfg.model_path.mkdir(parents=True, exist_ok=True)
        random.seed(self.cfg.random_state)
        np.random.seed(self.cfg.random_state)

    def _get_eval_metric_name(self) -> str:
        met = self.cfg.optimize_metric.lower()
        if met in ['pearson', 'pearsonr', 'pcc', 'corr']:
            return 'pearsonr'  
        return 'rmse'

    def _gbm_search(self) -> Dict[str, Any]:
        if AG_HAS_SPACE:
            return {
                'GBM': {
                    'num_boost_round': ag_space.Int(400, 3000),
                    'num_leaves': ag_space.Int(16, 512),
                    'max_depth': ag_space.Int(-1, 12),
                    'learning_rate': ag_space.Real(1e-3, 0.15, log=True),
                    'bagging_fraction': ag_space.Real(0.5, 1.0),
                    'bagging_freq': ag_space.Int(1, 7),
                    'feature_fraction': ag_space.Real(0.5, 1.0),
                    'min_data_in_leaf': ag_space.Int(5, 200),
                    'min_sum_hessian_in_leaf': ag_space.Real(1e-3, 10.0, log=True),
                    'lambda_l1': ag_space.Real(1e-6, 10.0, log=True),
                    'lambda_l2': ag_space.Real(1e-6, 10.0, log=True),
                    'objective': 'regression',
                    'metric': 'rmse',
                }
            }
        else:
            
            def sample_one() -> Dict[str, Any]:
                return {
                    'num_boost_round': int(np.random.randint(400, 3001)),
                    'num_leaves': int(np.random.randint(16, 513)),
                    'max_depth': int(np.random.choice([-1] + list(range(3, 13)))),
                    'learning_rate': float(np.exp(np.random.uniform(np.log(1e-3), np.log(0.15)))),
                    'bagging_fraction': float(np.random.uniform(0.5, 1.0)),
                    'bagging_freq': int(np.random.randint(1, 8)),
                    'feature_fraction': float(np.random.uniform(0.5, 1.0)),
                    'min_data_in_leaf': int(np.random.randint(5, 201)),
                    'min_sum_hessian_in_leaf': float(np.exp(np.random.uniform(np.log(1e-3), np.log(10.0)))),
                    'lambda_l1': float(np.exp(np.random.uniform(np.log(1e-6), np.log(10.0)))),
                    'lambda_l2': float(np.exp(np.random.uniform(np.log(1e-6), np.log(10.0)))),
                    'objective': 'regression',
                    'metric': 'rmse',
                }
            candidates = [sample_one() for _ in range(max(8, self.cfg.num_trials))]
            return {'GBM': candidates}

    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        label = self.cfg.label

        for df in (train_df, test_df):
            if label not in df.columns:
                raise ValueError(f"Label '{label}' not in dataframe columns")
            df[label] = pd.to_numeric(df[label], errors='coerce')
            keep_cols = [c for c in df.columns if (c == label or np.issubdtype(df[c].dtype, np.number))]
            df.drop(columns=[c for c in df.columns if c not in keep_cols], inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(axis=0, inplace=True)

        train_data = TabularDataset(train_df)
        test_data = TabularDataset(test_df)

        eval_metric_name = self._get_eval_metric_name()
        hyperparameters = self._gbm_search()

        logger.info(f"Start GBM HPO for '{label}' (AG_HAS_SPACE={AG_HAS_SPACE}) "
                    f"Train={train_data.shape}, Test={test_data.shape}, eval_metric={eval_metric_name}")
        predictor = TabularPredictor(
            label=label,
            path=str(self.cfg.model_path),
            problem_type='regression',
            eval_metric=eval_metric_name,   
        )

        tune_kwargs = None
        if AG_HAS_SPACE:
            tune_kwargs = {
                'num_trials': self.cfg.num_trials,
                'scheduler': 'local',
                'searcher': self.cfg.searcher,
            }

        predictor.fit(
            train_data=train_data,
            tuning_data=test_data,               
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=tune_kwargs,
            presets=self.cfg.presets,
            time_limit=self.cfg.time_limit,
            num_bag_folds=None,
            num_stack_levels=0,
            keep_only_best=True,
            refit_full=False
        )

        leaderboard = predictor.leaderboard(
            test_data,
            extra_metrics=['rmse', 'mae', 'r2', 'pearsonr', 'spearmanr', 'median_absolute_error'],
            silent=False
        )
        lb_csv = self.cfg.model_path / f"{label}_leaderboard.csv"
        leaderboard.to_csv(lb_csv, index=False)
        logger.info(f"Leaderboard exported to '{lb_csv}'")
        print(leaderboard)

        summary = predictor.fit_summary(verbosity=1)
        with open(self.cfg.model_path / f"{label}_fit_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        try:
            best_model = predictor._get_model_best()
            best_cfg = {}
            try:
                best_cfg = predictor._trainer.model_graph.models_best[0].params
            except Exception:
                pass
            with open(self.cfg.model_path / f"{label}_best_params.json", "w", encoding="utf-8") as f:
                json.dump({'best_model': best_model, 'best_params': best_cfg}, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Extract best params failed: {e}")

        predictor.save(str(self.cfg.model_path / self.cfg.model_name))
        logger.info(f"Model saved to {self.cfg.model_path / self.cfg.model_name}")


def run_task(datapack_dir: Path,
             label: str,
             model_root: Path,
             time_limit: int,
             num_trials: int,
             searcher: str,
             optimize_metric: str,
             presets: str,
             random_state: int):
    df_tr, df_te = load_processed_datapack(datapack_dir, label)

    task_dir = model_root / label
    cfg = ModelConfig(
        label=label,
        model_path=task_dir,
        model_name=f'{label}_GBM_only',
        time_limit=time_limit,
        presets=presets,
        num_trials=num_trials,
        searcher=searcher,
        optimize_metric=optimize_metric,
        random_state=random_state
    )
    trainer = AutoGBMTrainer(cfg)
    trainer.fit(df_tr, df_te)


def main():
    parser = argparse.ArgumentParser(description="AutoGluon GBM-only HPO (uses preprocessed datapack CSVs).")
    parser.add_argument('--datapack_dir', type=str, default='./datapack',
                        help="Directory containing processed train/test CSVs.")
    parser.add_argument('--model_dir', type=str, default='./gen_model_research/aggregate/hTfR1_model',
                        help="Output root; subfolders per label will be created.")
    parser.add_argument('--time_limit', type=int, default=600,
                        help="Time limit per task (seconds).")
    parser.add_argument('--num_trials', type=int, default=120,
                        help="HPO trials (or candidate configs when no 'space').")
    parser.add_argument('--searcher', type=str, default='random', choices=['random', 'bayesopt'])
    parser.add_argument('--optimize_metric', type=str, default='pearsonr', choices=['pearsonr', 'rmse'])
    parser.add_argument('--presets', type=str, default='medium_quality')
    parser.add_argument('--random_state', type=int, default=10)
    args = parser.parse_args()

    datapack_dir = Path(args.datapack_dir)
    model_root = Path(args.model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    for label in ['V_mean__P_mean_log2_enr', 'hT_mean__V_mean_log2_enr']:
        run_task(
            datapack_dir=datapack_dir,
            label=label,
            model_root=model_root,
            time_limit=args.time_limit,
            num_trials=args.num_trials,
            searcher=args.searcher,
            optimize_metric=args.optimize_metric,
            presets=args.presets,
            random_state=args.random_state
        )


if __name__ == '__main__':
    main()
