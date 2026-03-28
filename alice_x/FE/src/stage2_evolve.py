#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
import json

import mixture
import group
import evaluation as eva
import parameters


def load_stage1_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seq" not in df.columns:
        for c in ["AA_sequence", "Sequence", "aa_seq", "AA_SEQ"]:
            if c in df.columns:
                df = df.rename(columns={c: "seq"})
                break
    if "seq" not in df.columns:
        raise ValueError("Stage1 file must contain 'seq' column")
    df["seq"] = df["seq"].astype(str).str.strip().str.upper()
    df = df[df["seq"].map(len) == 7].reset_index(drop=True)
    return df[["seq"]].drop_duplicates(ignore_index=True)


def extract_features(seqs: pd.Series) -> pd.DtaFrame:
    feats = {}
    for s in seqs:
        try:
            f = parameters.cal_pep(s)
            if isinstance(f, dict):
                feats[s] = f
            else:
                arr = np.asarray(f)
                names = getattr(parameters, "FEATURE_NAMES", None)
                if names is None and hasattr(parameters, "get_feature_names"):
                    names = parameters.get_feature_names()
                if names is None:
                    names = [f"f{i}" for i in range(len(arr))]
                feats[s] = {names[i]: arr[i] for i in range(len(arr))}
        except Exception:
            continue
    return pd.DataFrame.from_dict(feats, orient="index")


def align_features(features: pd.DataFrame, predictor: TabularPredictor) -> pd.DataFrame:
    try:
        used_cols = list(predictor.feature_metadata.get_features())
    except Exception:
        used_cols = getattr(predictor, 'feature_metadata_in', None)
        if used_cols is None:
            used_cols = list(features.columns)
    df = features.copy()
    for c in used_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[used_cols]


def minmax01(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors='coerce')
    m, M = x.min(), x.max()
    if not np.isfinite(m) or not np.isfinite(M) or M==m:
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    return (x - m) / (M - m)


def load_refs_json(path: str):
    J = json.load(open(path, 'r'))
    refplus = list(map(str, J.get("Positive", [])))
    refminus = list(map(str, J.get("Negative", [])))
    return refplus, refminus


def stage2_run_one_mode(mode: str,
                        df_start: pd.DataFrame,
                        predictor_fit: TabularPredictor,
                        predictor_ht: TabularPredictor,
                        refplus: list,
                        refminus: list,
                        epochs: int,
                        lambdas: dict,
                        out_root: Path,
                        baseline_quantile: float = 0.5):
    outdir = out_root / f"stage2_{mode}"
    outdir.mkdir(parents=True, exist_ok=True)

    g = mixture.EDGGroup(len(df_start), 7)
    g.set_auto_eval(False)
    g.setBasicReg(w_pos=0.0, w_neg=0.0, w_novel=0.0)
    g.initFromFile(df_start, col="seq")

    evaluator = g.evamet  

    feat0 = extract_features(df_start["seq"])
    X0_fit = align_features(feat0, predictor_fit)
    X0_ht  = align_features(feat0, predictor_ht)
    base_fit_vals = pd.Series(predictor_fit.predict(X0_fit), index=feat0.index)
    base_ht_vals  = pd.Series(predictor_ht.predict(X0_ht),  index=feat0.index)
    fit_base = float(base_fit_vals.quantile(baseline_quantile))
    ht_base  = float(base_ht_vals.quantile(baseline_quantile))

    prev_gen_list = df_start["seq"].tolist()

    for epoch in range(epochs):
        g.evolution()
        seqs = pd.Series(g.inds)

        feat_df = extract_features(seqs)
        if feat_df.empty:
            continue
        X_fit = align_features(feat_df, predictor_fit)
        X_ht  = align_features(feat_df, predictor_ht)
        preds_fit_raw = pd.Series(predictor_fit.predict(X_fit), index=feat_df.index)
        preds_ht_raw  = pd.Series(predictor_ht.predict(X_ht),  index=feat_df.index)

        preds_fit = minmax01(preds_fit_raw)
        preds_ht  = minmax01(preds_ht_raw)

        final_scores, ref_dists, div_prevs, rewards = [], [], [], []

        for s in seqs:
            ref_dist = evaluator.ref_distance_norm(s, refplus, refminus)
            div_prev = evaluator.diversity_prev_norm(s, prev_gen_list)
            score = evaluator.evaluate_stage2_scalar(
                seq=s,
                fit_pred=float(preds_fit.get(s,0.5)),
                ht_pred=float(preds_ht.get(s,0.5)),
                refplus=refplus,
                refminus=refminus,
                prev_gen=prev_gen_list,
                mode=mode,
                fit_base=fit_base,
                ht_base=ht_base,
                lambdas=lambdas
            )
            if mode.startswith("sum"):
                rew = float(preds_fit.get(s,0.5) + preds_ht.get(s,0.5))
            elif mode.startswith("min"):
                rew = float(min(preds_fit.get(s,0.5), preds_ht.get(s,0.5)))
            else:
                rew = float(preds_fit.get(s,0.5) + preds_ht.get(s,0.5))
            final_scores.append(score)
            ref_dists.append(ref_dist)
            div_prevs.append(div_prev)
            rewards.append(rew)

        g.score = np.array(final_scores, dtype=float)

        df_out = pd.DataFrame({
            "seq": seqs.values,
            "fitness_pred": preds_fit.reindex(seqs.values).values,
            "htfr1_pred": preds_ht.reindex(seqs.values).values,
            "ref_dist": ref_dists,
            "div_prev": div_prevs,
            "reward_signal": rewards,
            "final_score": final_scores
        })
        df_out.to_csv(outdir / f"seq-epoch-{epoch}.csv", index=False)

        prev_gen_list = seqs.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_final", type=str, default='../data/top1000_sequences.csv', help="Stage1 last generation CSV")
    ap.add_argument("--model_root", type=str, default="./gen_model_research/aggregate/hTfR1_model")
    ap.add_argument("--refjson", type=str, default="../data/sequences.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--baseline_q", type=float, default=0.5)
    ap.add_argument("--out_root", type=str, default="./evolve_sum/")
    args = ap.parse_args()

    df_start = load_stage1_df(args.stage1_final)

    fitness_dir = Path(args.model_root) / "V_mean__P_mean_log2_enr"
    if not fitness_dir.exists():
        alt = Path(args.model_root) / "fitness"
        if not alt.exists():
            raise FileNotFoundError(f"Can't find fitness model dialogue：{fitness_dir} or {alt}")
        fitness_dir = alt
    predictor_fit = TabularPredictor.load(str(fitness_dir))
    predictor_ht  = TabularPredictor.load(str(Path(args.model_root) / "hT_mean__V_mean_log2_enr"))

    refplus, refminus = load_refs_json(args.refjson)

    modes = ["sum"]
    lambdas = dict(w_ref=0.25, w_div=0.25, w_ref_pen=0.35, w_fit_pen=0.3, w_ht_pen=0.3)
    out_root = Path(args.out_root)

    for mode in modes:
        stage2_run_one_mode(
            mode=mode,
            df_start=df_start,
            predictor_fit=predictor_fit,
            predictor_ht=predictor_ht,
            refplus=refplus,
            refminus=refminus,
            epochs=args.epochs,
            lambdas=lambdas,
            out_root=out_root,
            baseline_quantile=args.baseline_q
        )


if __name__ == "__main__":
    main()