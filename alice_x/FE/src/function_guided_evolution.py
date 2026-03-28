#!/usr/bin/env python
# coding: utf-8
import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import group
import GA
import EDA
import mixture
import SA


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seq" not in df.columns:
        for cand in ["AA_sequence", "Sequence", "aa_seq", "AA_SEQ"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "seq"})
                break
    if "seq" not in df.columns:
        raise ValueError(f"Input CSV {path} must have 'seq' column")
    df["seq"] = df["seq"].astype(str).str.strip().str.upper()
    return df[["seq"]].dropna().reset_index(drop=True)


def build_parser():
    p = argparse.ArgumentParser(description="Function-guided Evolution (repro-ready)")

    p.add_argument("--group_type", type=str, default='EDG', choices=['SA', 'GA', 'EDA', 'EDG', 'NONE'])
    p.add_argument("--gan_date", type=str, default='20250520')
    p.add_argument("--pregan_date", type=str, default='20250520')
    p.add_argument("--model_name", type=str, default='SeqGAN', choices=['SeqGAN','Random'])
    p.add_argument("--pretrain_model", type=str, default='roberta', choices=['bert','roberta','NONE'])
    p.add_argument("--input_file", type=str, default='./infer_gen_seqs/infer_gen_seqs.csv',
                   help="initialize group")
    p.add_argument("--sampath", type=str, default=None,
                   help="JSON(Positive/Negative);")
    p.add_argument("--matpath", type=str, default='BLOSSUM62.txt',
                   help="matrix of blossum62")
    p.add_argument("--seed", type=int, default=1234, help="random seed")
    p.add_argument("--epochs", type=int, default=11, help="evolution epoch")
    p.add_argument("--loss_type", type=str, default='basic',
                   choices=['basic','triplet','npair','nce','info_nce'])

    p.add_argument("--edg_r_eda", type=float, default=None)
    p.add_argument("--edg_r_ga_copy", type=float, default=None)
    p.add_argument("--ga_rmutate", type=float, default=None)
    p.add_argument("--ga_rcopy", type=float, default=None)
    p.add_argument("--ga_doublecross", type=int, default=None)

    p.add_argument("--repro_config", type=str, default=None,
                   help="path to search_success/repro_config.json")
    return p


def main():
    args = build_parser().parse_args()

    if args.repro_config and os.path.exists(args.repro_config):
        cfg = json.load(open(args.repro_config, "r", encoding="utf-8"))
        rcfg = cfg.get("config", {})

        args.input_file = rcfg.get("input_file", args.input_file)
        args.sampath = rcfg.get("sampath", args.sampath)
        args.matpath = rcfg.get("matpath", args.matpath)
        args.seed = int(rcfg.get("seed", args.seed))
        args.epochs = int(rcfg.get("epochs", args.epochs))
        args.loss_type = rcfg.get("loss_type", args.loss_type)
        args.edg_r_eda = rcfg.get("edg_r_eda", args.edg_r_eda)
        args.edg_r_ga_copy = rcfg.get("edg_r_ga_copy", args.edg_r_ga_copy)
        args.ga_rmutate = rcfg.get("ga_rmutate", args.ga_rmutate)
        args.ga_rcopy = rcfg.get("ga_rcopy", args.ga_rcopy)
        args.ga_doublecross = rcfg.get("ga_doublecross", args.ga_doublecross)

    set_global_seed(args.seed)

    if args.group_type == 'GA':
        group_ty = GA.GAGroup
    elif args.group_type == 'EDA':
        group_ty = EDA.EDAGroup
    elif args.group_type == 'EDG':
        group_ty = mixture.EDGGroup
    elif args.group_type == 'SA':
        group_ty = SA.SAGroup
    elif args.group_type == 'NONE':
        group_ty = group.Group
    else:
        raise ValueError("Unknown group_type")

    df = load_df(args.input_file)

    df = df[df["seq"].map(len) == 7].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No 7-mer sequences found in {args.input_file}")

    g = group_ty(len(df), 7)

    if args.sampath:
        g.setMet(args.sampath, args.matpath)

    g.initFromFile(df, col="seq", update_length=True, cut=False, loss_type=args.loss_type)
    g.showMsg('File Group')

    if hasattr(g, "r_eda") and args.edg_r_eda is not None:
        g.r_eda = float(args.edg_r_eda)
    if hasattr(g, "r_ga_copy") and args.edg_r_ga_copy is not None:
        g.r_ga_copy = float(args.edg_r_ga_copy)
    if hasattr(g, "setRmutate") and args.ga_rmutate is not None:
        g.setRmutate(float(args.ga_rmutate))
    if hasattr(g, "setRcopy") and args.ga_rcopy is not None:
        g.setRcopy(float(args.ga_rcopy))
    if hasattr(g, "setDoublecross") and args.ga_doublecross is not None:
        g.setDoublecross(bool(args.ga_doublecross))

    base = f'../{args.group_type}/{args.loss_type}/{args.pretrain_model}2{args.model_name}/'
    path = os.path.join('../output', base)
    os.makedirs(path, exist_ok=True)
    fig_path = os.path.join('../Figures', base)
    os.makedirs(fig_path, exist_ok=True)

    with open(os.path.join(path, 'RA_input_filepath.txt'), 'a+', encoding='utf-8') as f:
        f.write(str(args.input_file) + '\n')

    nbest = max(1, math.floor(0.25 * len(df)))  # top 25%
    score_hist = []

    total_epoch = int(args.epochs)
    for epoch in range(total_epoch):
        g.evolution()
        g.evaluate(args.loss_type)
        score_hist.append(g.sortedScore()[:nbest].mean())

        g.outputMsg(os.path.join(path, f'seq-epoch-{epoch}.csv'))

    try:
        x = np.arange(total_epoch)
        plt.figure()
        plt.plot(x, score_hist, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Top 25% Score Average')
        plt.tight_layout()
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(os.path.join(fig_path, f'{args.loss_type}_score.png'), dpi=150)
        plt.savefig(os.path.join(fig_path, f'{args.loss_type}_score.svg'))
        plt.close()
    except Exception:
        pass

    run_sum = {
        "seed": args.seed,
        "epochs": args.epochs,
        "group_type": args.group_type,
        "loss_type": args.loss_type,
        "edg_r_eda": getattr(g, "r_eda", None),
        "edg_r_ga_copy": getattr(g, "r_ga_copy", None),
        "ga_rmutate": getattr(g, "rmutate", None) if hasattr(g, "rmutate") else None,
        "ga_rcopy": getattr(g, "rcopy", None) if hasattr(g, "rcopy") else None,
        "ga_doublecross": getattr(g, "doublecross", None) if hasattr(g, "doublecross") else None,
        "history_top25": score_hist,
        "final_csv": os.path.join(path, f'seq-epoch-{total_epoch-1}.csv')
    }
    with open(os.path.join(path, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_sum, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
