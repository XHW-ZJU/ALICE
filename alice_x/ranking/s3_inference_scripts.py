# coding: utf-8
import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from autogluon.tabular import TabularPredictor

import parameters  

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    filename='./inference_logging.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------
# Helpers
# -----------------------
def _infer_feature_names(n: int) -> List[str]:
    if hasattr(parameters, "FEATURE_NAMES"):
        names = list(parameters.FEATURE_NAMES)
        if len(names) != n:
            raise ValueError(f"FEATURE_NAMES length {len(names)} 与特征维度 {n} 不一致")
        return names
    if hasattr(parameters, "get_feature_names") and callable(parameters.get_feature_names):
        names = list(parameters.get_feature_names())
        if len(names) != n:
            raise ValueError(f"get_feature_names() 长度 {len(names)} 与特征维度 {n} 不一致")
        return names
    raise ValueError("Not found FEATURE_NAMES or get_feature_names()")


def compute_features_df(seqs: pd.Series) -> Tuple[pd.DataFrame, pd.Index]:
    feats = {}
    feature_names = None

    for idx, s in tqdm(seqs.items(), desc="Feature selection", unit="seq"):
        try:
            f = parameters.cal_pep(str(s))
            if isinstance(f, dict):
                row = f
            elif isinstance(f, pd.Series):
                row = f.to_dict()
            else:
                arr = np.asarray(f)
                if feature_names is None:
                    feature_names = _infer_feature_names(arr.shape[0])
                row = {feature_names[i]: arr[i] for i in range(arr.shape[0])}
            feats[idx] = row
        except Exception as e:
            logger.warning(f"[Feature Failed] row {idx}, seq {s}, reason:{e}")

    if not feats:
        return pd.DataFrame(), pd.Index([])

    df = pd.DataFrame.from_dict(feats, orient='index')  
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df)
    df = df.dropna(axis=0, how='any')
    dropped = before - len(df)
    if dropped > 0:
        bad_idx = set(feats.keys()) - set(df.index)
        logger.warning(f"[Feature NaN/Inf drop] {dropped} 行：{sorted(list(bad_idx))[:10]}{'...' if len(bad_idx) > 10 else ''}")
    return df, df.index


def load_predictor(dir_path: Path) -> TabularPredictor:
    dir_path = Path(dir_path)
    if not (dir_path / "predictor.pkl").exists():
        raise FileNotFoundError(f"Model not found：{dir_path / 'predictor.pkl'}")
    pred = TabularPredictor.load(str(dir_path))
    logger.info(f"Loaded model: {dir_path}")
    return pred


def align_features(features: pd.DataFrame, predictor: TabularPredictor) -> pd.DataFrame:
    # Interface
    try:
        used_cols = list(predictor.feature_metadata.get_features())
    except Exception:
        used_cols = getattr(predictor, 'feature_metadata_in', None)
        if used_cols is None:
            used_cols = list(features.columns)

    df = features.copy()
    missing = [c for c in used_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    df = df[used_cols]
    return df


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Inference for fitness & hT enrichment on aggxx.csv-like files.")
    parser.add_argument('--data', type=str, default='./input_file.csv')
    parser.add_argument('--model_root', type=str, default='./gen_model_research/aggregate/hTfR1_model')
    parser.add_argument('--out', type=str, default='./data/final_predict_lib2.csv')
    parser.add_argument('--filter_underscore', action='store_true', default=True)
    args = parser.parse_args()

    data_path = Path(args.data)
    model_root = Path(args.model_root)
    out_path = Path(args.out)

    df_raw = pd.read_csv(data_path)

    df_work = df_raw.copy()
    df_work['_row_id'] = np.arange(len(df_work))  

    if args.filter_underscore:
        valid_mask = ~df_work['AA_sequence'].astype(str).str.contains('_', na=False)
        removed = int((~valid_mask).sum())

    else:
        valid_mask = pd.Series(True, index=df_work.index)

    valid_idx = df_work.index[valid_mask]

    feat_df, ok_idx = compute_features_df(df_work.loc[valid_idx, 'AA_sequence'].astype(str))
    
    fitness_dir = model_root / 'V_mean__P_mean_log2_enr'
    if not fitness_dir.exists():
        alt = model_root / 'fitness'
        if alt.exists():
            fitness_dir = alt
        else:
            raise FileNotFoundError(f"cannot find fitness directory: {model_root}/V_mean__P_mean_log2_enr or {model_root}/fitness")
    predictor_fit = load_predictor(fitness_dir)

    ht_dir = model_root / 'hT_mean__V_mean_log2_enr'
    if not ht_dir.exists():
        raise FileNotFoundError(f"Cannot find hT model directory：{ht_dir}")
    predictor_ht = load_predictor(ht_dir)

    X_fit = align_features(feat_df, predictor_fit)
    X_ht = align_features(feat_df, predictor_ht)

    preds_fit = pd.to_numeric(predictor_fit.predict(X_fit), errors='coerce')
    preds_ht = pd.to_numeric(predictor_ht.predict(X_ht), errors='coerce')

    df_work['fitness prediction'] = np.nan
    df_work['hTFR1 enrichment prediction'] = np.nan
    df_work.loc[ok_idx, 'fitness prediction'] = preds_fit.values
    df_work.loc[ok_idx, 'hTFR1 enrichment prediction'] = preds_ht.values


    out_df = df_work.sort_values('_row_id').drop(columns=['_row_id'])

    out_df.to_csv(out_path, index=False)



if __name__ == "__main__":
    main()
