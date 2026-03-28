# coding: utf-8
"""
process_sequences.py  (FIXED: 保留具名特征列 + 输出目标结构 + 修复 global BUG)
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    filename='./logging.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import parameters  # 必须提供 cal_pep(seq)，若返回数组需提供 FEATURE_NAMES 或 get_feature_names()

# -------------------------
# 常量配置
# -------------------------
TARGET_LABELS = ["hT_mean__V_mean_log2_enr", "V_mean__P_mean_log2_enr"]
TRAIN_RATIO = 0.96
RANDOM_STATE = 10


@dataclass
class DataConfig:
    data_path: Path
    outdir: Path
    train_test_ratio: float = TRAIN_RATIO
    random_state: int = RANDOM_STATE
    drop_underscore: bool = True


class DataProcessor:
    def __init__(self, config: DataConfig):
        self.data_path = config.data_path
        self.outdir = config.outdir
        self.train_test_ratio = config.train_test_ratio
        self.random_state = config.random_state
        self.drop_underscore = config.drop_underscore

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: List[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"输入数据缺少必要列：{missing}")

    @staticmethod
    def _infer_feature_names(n: int) -> List[str]:
        if hasattr(parameters, "FEATURE_NAMES"):
            names = list(parameters.FEATURE_NAMES)
            if len(names) != n:
                raise ValueError(f"FEATURE_NAMES 长度 {len(names)} 与实际特征数 {n} 不一致")
            return names
        if hasattr(parameters, "get_feature_names") and callable(parameters.get_feature_names):
            names = list(parameters.get_feature_names())
            if len(names) != n:
                raise ValueError(f"get_feature_names() 长度 {len(names)} 与实际特征数 {n} 不一致")
            return names
        raise ValueError("cal_pep 返回数组/列表，但 parameters 中未提供 FEATURE_NAMES 或 get_feature_names()")

    def _compute_features(self, seqs: pd.Series) -> pd.DataFrame:
        feats = []
        feature_names = None
        for s in seqs:
            f = parameters.cal_pep(s)
            if isinstance(f, dict):
                feats.append(f)
            elif isinstance(f, pd.Series):
                feats.append(f.to_dict())
            else:
                arr = np.asarray(f)
                if feature_names is None:
                    feature_names = self._infer_feature_names(arr.shape[0])
                feats.append({feature_names[i]: arr[i] for i in range(arr.shape[0])})
        return pd.DataFrame(feats)

    def _prepare_one_label(self, base_df: pd.DataFrame, label: str) -> Tuple[Path, Path]:
        sub = base_df.dropna(subset=[label]).copy()
        meta_cols = ["AA_sequence"]
        feature_cols = [c for c in sub.columns if c not in (meta_cols + TARGET_LABELS)]
        export_cols = meta_cols + feature_cols + [label]
        sub = sub[export_cols]

        train_df, test_df = train_test_split(
            sub,
            test_size=1 - self.train_test_ratio,
            random_state=self.random_state,
            shuffle=True,
        )

        if label == "hT_mean__V_mean_log2_enr":
            train_name = "train_hT_mean__V_mean_log2_enr_data.csv"
            test_name = "test_hT_mean__V_mean_log2_enr_data.csv"
        elif label == "V_mean__P_mean_log2_enr":
            train_name = "train_fitness_data.csv"
            test_name = "test_fitness_data.csv"
        else:
            train_name = f"train_{label}_data.csv"
            test_name = f"test_{label}_data.csv"

        train_path = self.outdir / train_name
        test_path = self.outdir / test_name
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"[OK] 已导出 {train_path} (shape={train_df.shape})")
        print(f"[OK] 已导出 {test_path} (shape={test_df.shape})")
        return train_path, test_path

    def process(self):
        df = pd.read_csv(self.data_path)
        self._validate_columns(df, ["iAAs"] + TARGET_LABELS)

        if self.drop_underscore:
            before = len(df)
            df = df[~df["iAAs"].astype(str).str.contains("_", na=False)].copy()
            print(f"[INFO] 过滤含 '_'：{before} -> {len(df)}")

        df = df.drop_duplicates(subset="iAAs", keep="first").copy()

        print("[INFO] 计算特征 ...")
        features_df = self._compute_features(df["iAAs"].astype(str))
        base = pd.concat([features_df.reset_index(drop=True),
                          df[TARGET_LABELS].reset_index(drop=True)], axis=1)

        base.insert(0, "AA_sequence", df["iAAs"].astype(str).reset_index(drop=True))
        # base.insert(0, "Unnamed: 0", np.arange(len(base)))

        self.outdir.mkdir(parents=True, exist_ok=True)
        for label in TARGET_LABELS:
            self._prepare_one_label(base, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data/oligo2_final_change_name.csv")
    parser.add_argument("--outdir", type=str, default="./datapack")
    parser.add_argument("--keep_underscore", action="store_true")
    args = parser.parse_args()

    cfg = DataConfig(
        data_path=Path(args.data),
        outdir=Path(args.outdir),
        drop_underscore=not args.keep_underscore
    )
    DataProcessor(cfg).process()


if __name__ == "__main__":
    main()
