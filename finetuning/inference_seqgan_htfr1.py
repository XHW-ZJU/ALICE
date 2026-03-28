# coding: utf-8
import os
import argparse
import numpy as np
import pandas as pd
import torch

# —— 与训练脚本保持一致的导入 —— #
import parameters as parameters  # 你原代码里就是这样引用的
from SeqGAN import Generator
from minibert import BERT  # 仅为类型标注/兼容

# -------------------------
# 基础设置/常量
# -------------------------
AA_LIST = ['<PAD>', 'X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _safe_load_generator(ckpt_path, bert, GAN_param, Pretrain_param):
    """
    根据训练脚本保存方式，优先按 state_dict 加载；如果用户之前保存的是整模型，也做兼容。
    """
    # 构建同构的 Generator
    generator = Generator(bert, GAN_param, Pretrain_param).to(DEVICE)
    generator.eval()

    # 读取 checkpoint
    obj = torch.load(ckpt_path, map_location=DEVICE)

    # 1) 训练脚本最新版保存的是 {'model': state_dict}
    if isinstance(obj, dict) and 'model' in obj and isinstance(obj['model'], dict):
        generator.load_state_dict(obj['model'], strict=True)
        return generator

    # 2) 兼容：如果直接保存了整个模型（不推荐，但有可能）
    try:
        # 如果是整模型，这里会是 nn.Module
        if hasattr(obj, 'state_dict'):
            generator = obj.to(DEVICE)
            generator.eval()
            return generator
    except Exception:
        pass

    # 3) 兼容：直接就是 state_dict
    if isinstance(obj, dict):
        try:
            generator.load_state_dict(obj, strict=True)
            return generator
        except Exception as e:
            raise RuntimeError(f"无法将该 dict 作为 state_dict 加载到 Generator：{e}")

    raise RuntimeError("未能识别的 Generator checkpoint 格式，请检查 ckpt 文件。")


@torch.no_grad()
def generate_sequences(generator, train_len, n_samples=1000, batch_gen=64, forbid_tokens=(0, 1)):
    """
    调用 generator.gen_sample(train_len) 多次生成；按 batch 聚合到 n_samples。
    过滤任何包含 forbid_tokens(默认: PAD=0, X=1) 的序列。
    返回：list[str]
    """
    seqs_out = []
    while len(seqs_out) < n_samples:
        # 多次调用 gen_sample；大多数实现会返回 (B, L) 的 LongTensor
        cur = []
        n_calls = int(np.ceil(batch_gen / 1.0))
        for _ in range(n_calls):
            sample = generator.gen_sample(train_len)  # 训练脚本就是这么用的
            if isinstance(sample, torch.Tensor):
                if sample.ndim == 1:
                    sample = sample[None, :]  # (1, L)
                sample = sample.detach().to('cpu').numpy()
            elif isinstance(sample, np.ndarray):
                if sample.ndim == 1:
                    sample = sample[None, :]
            else:
                raise RuntimeError("generator.gen_sample 返回了未知类型，请检查实现。")

            cur.append(sample)

        cur = np.concatenate(cur, axis=0)  # (B, L)
        # 过滤包含 PAD(0) 或 X(1) 的序列
        mask_ok = ~np.isin(cur, np.array(forbid_tokens)).any(axis=1)
        cur = cur[mask_ok]
        # 映射到氨基酸字符
        for row in cur:
            chars = [AA_LIST[int(idx)] for idx in row]
            seqs_out.append(''.join(chars))
            if len(seqs_out) >= n_samples:
                break

    return seqs_out[:n_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='../checkpoint/pretrain/train/20230506/roberta2SeqGAN/G/G_repeat_11.pth',
                        help="训练阶段保存的 Generator checkpoint 路径，如 ../checkpoint/.../G_repeat_8.pth")
    parser.add_argument('--pretrain_pth', type=str, default='./model/words_model_ep3.pth',
                        help="Roberta/BERT 预训练权重路径（与训练时一致）")
    parser.add_argument('--outdir', type=str, default='./outputs/infer_gen_seqs.csv',
                        help="输出目录（会写 infer_gen_seqs.csv）")
    parser.add_argument('--n_samples', type=int, default=1000,
                        help="希望生成的序列条数")
    parser.add_argument('--batch_gen', type=int, default=64,
                        help="每轮尝试生成的粗批大小（越大越快）")
    args = parser.parse_args()

    # 读取与训练一致的参数
    GAN_param = parameters.GAN           # 训练脚本就是用 parameters.GAN
    Pretrain_param = parameters.roberta  # 与训练时一致
    train_len = int(GAN_param.get('train_len', 7))

    # 构建并冻结 roberta（与训练保持一致）
    bert = torch.load(args.pretrain_pth, map_location=DEVICE, weights_only=False)
    for p in bert.parameters():
        p.requires_grad = False

    # 构建 Generator 并加载训练好的权重
    generator = _safe_load_generator(args.ckpt, bert, GAN_param, Pretrain_param)
    generator.eval()

    # 生成
    seqs = generate_sequences(
        generator=generator,
        train_len=train_len,
        n_samples=args.n_samples,
        batch_gen=args.batch_gen,
        forbid_tokens=(0, 1)  # 过滤任何包含 <PAD>(0) 或 X(1) 的序列
    )

    # 保存
    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, 'infer_gen_seqs_ep11_3.csv')
    pd.DataFrame({'seq': seqs}).to_csv(out_csv, index=False, encoding='utf-8')
    print(f"[OK] 生成完毕，共 {len(seqs)} 条；已保存到：{out_csv}")


if __name__ == '__main__':
    main()
