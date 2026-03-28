# coding: utf-8
import os
import math
import time
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import parameters
from SeqGAN import *              
from minibert import BERT         
from utils import *               


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

aa_list = ['<PAD>', 'X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
AA2IDX = {aa: i for i, aa in enumerate(aa_list)}
VOCAB_SIZE = len(aa_list)  # = 22

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.normal_(m.weight.data, mean=0, std=min(1.0 / math.sqrt(m.weight.data.shape[-1]), 0.1))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def one_hot_train(y, num_columns, train_len=None):
    if isinstance(y, torch.Tensor):
        y = y.detach().to('cpu').numpy()
    y = np.asarray(y)

    if y.ndim == 1:
        y = y[None, :]  # -> (1, L)

    B, L = y.shape
    if train_len is None:
        train_len = L

    if L < train_len:
        pad = np.zeros((B, train_len - L), dtype=np.int64)
        y = np.concatenate([y, pad], axis=1)
        L = train_len
    elif L > train_len:
        y = y[:, :train_len]
        L = train_len

    y_cat = np.zeros((B, L, num_columns), dtype=np.float32)
    y_clip = np.clip(y, 0, num_columns - 1).astype(np.int64)
    rows = np.arange(L)
    for i in range(B):
        y_cat[i, rows, y_clip[i]] = 1.0
    out = torch.tensor(y_cat, dtype=torch.float32, device=device)
    return out


def encode_seq(seq: str, train_len: int) -> np.ndarray:
    seq = (seq or "").strip().upper()
    idxs = [AA2IDX.get(ch, AA2IDX['X']) for ch in seq]
    if len(idxs) >= train_len:
        idxs = idxs[:train_len]
    else:
        idxs = idxs + [AA2IDX['<PAD>']] * (train_len - len(idxs))
    return np.array(idxs, dtype=np.int64)


def load_csv_and_split(csv_path: str, train_len: int, test_ratio: float = 0.1, seed: int = 42):
    df = pd.read_csv(csv_path)
    if 'iAAs' not in df.columns:
        raise ValueError("Not Found 'iAAs' column in CSV File!")

    seqs = df['iAAs'].astype(str).tolist()
    X = np.stack([encode_seq(s, train_len) for s in seqs], axis=0)  # (N, train_len)

    rng = np.random.RandomState(seed)
    N = X.shape[0]
    indices = np.arange(N)
    rng.shuffle(indices)
    split = int(N * (1.0 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    VS_D_XTS = X[train_idx]
    VS_D_XTS_TEST = X[test_idx]

    print(f"[DATA] Total Data: {N}, Train: {VS_D_XTS.shape[0]}, Test: {VS_D_XTS_TEST.shape[0]}, Sequence length(train_len): {train_len}")
    return VS_D_XTS, VS_D_XTS_TEST


class AdaptiveBCE:
    def __init__(self):
        self.bce = nn.BCELoss().to(device)
        self.bce_logits = nn.BCEWithLogitsLoss().to(device)

    def __call__(self, d_out, targets):
        with torch.no_grad():
            minv = float(d_out.min().item())
            maxv = float(d_out.max().item())
        if 0.0 <= minv <= 1.0 and 0.0 <= maxv <= 1.0:
            return self.bce(d_out, targets)
        else:
            return self.bce_logits(d_out, targets)


# -------------------------
# Gumbel-Softmax Sampling
# -------------------------
@torch.enable_grad()
def differentiable_autoregressive_onehots(
    generator: nn.Module,
    B: int,
    L: int,
    start_id: int = 1,      # Still use 'X' as BOS
    tau: float = 1.0,       # Gumbel Temperature
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Initialize auto-regression
    ar_input = torch.full((B, 1), int(start_id), dtype=torch.long, device=device)  # (B,1)
    onehots = []
    ent_list = []

    for t in range(L):
        if not isinstance(ar_input, torch.LongTensor) and not (isinstance(ar_input, torch.Tensor) and ar_input.dtype == torch.long):
            ar_input = ar_input.long()
        if ar_input.device != device:
            ar_input = ar_input.to(device)

        logits_full = generator.forward(ar_input)  # shape (B,V) / (B,T,V) / (T,V)

        # ----  logits -> (B,V) ----
        if logits_full is None:
            raise RuntimeError("generator.forward(ar_input) 返回 None，请检查 Generator 实现。")

        if logits_full.dim() == 3:
            # (B, T, V) 
            logits_t = logits_full[:, -1, :]
        elif logits_full.dim() == 2:
            if logits_full.size(0) == B:
                logits_t = logits_full
            elif logits_full.size(0) == L or logits_full.size(0) == t + 1:
                logits_t = logits_full[-1, :].unsqueeze(0).expand(B, -1).contiguous()
            else:
                logits_t = logits_full
                if logits_t.size(0) != B:
                    logits_t = logits_t[:1, :].expand(B, -1).contiguous()
        else:
            raise RuntimeError(f"Unsurported logits dimention: {tuple(logits_full.shape)}")

        # ---- Gumbel-Softmax (straight-through) generation one-hot ----
        y_t = F.gumbel_softmax(logits_t, tau=tau, hard=True, dim=-1)  # (B,V)
        onehots.append(y_t.unsqueeze(1))  # (B,1,V)

        # ---- 熵正则（用 softmax，不加 gumbel）----
        probs = F.softmax(logits_t, dim=-1)
        ent = -(probs * (probs.clamp_min(1e-8).log())).sum(-1).mean()
        ent_list.append(ent)

        # ---- 下一个 token 的条件输入（取 argmax 的 id）----
        next_ids = y_t.argmax(-1)  # (B,)
        ar_input = torch.cat([ar_input, next_ids.unsqueeze(1)], dim=1)  # (B, t+2)

    Y_onehot = torch.cat(onehots, dim=1)          # (B,L,V)
    mean_entropy = torch.stack(ent_list).mean()   # 标量
    return Y_onehot, mean_entropy


# -------------------------
# 训练主循环（修复版）
# -------------------------
def train_and_eval(bert: BERT,
                   vs_G_datapack,
                   vs_G_datapack_TEST,
                   GAN_param,
                   Pretrain_param,
                   columns,
                   aa_list):

    # Loss
    adv_loss_adaptive = AdaptiveBCE()
    mle_loss = nn.CrossEntropyLoss().to(device)

    # 模型
    generator = Generator(bert, GAN_param, Pretrain_param)
    discriminator = Discriminator(GAN_param, Pretrain_param)
    generator.to(device)
    discriminator.to(device)
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # 参数量
    total_G_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_D_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print('total G params {} {}: {}'.format(GAN_param.get('date', 'na'), GAN_param.get('model_name', 'na'), total_G_params))
    print('total D params {} {}: {}'.format(GAN_param.get('date', 'na'), GAN_param.get('model_name', 'na'), total_D_params))

    # 学习率
    learning_rate = float(GAN_param.get('lr', 1e-4))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler_GV = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=0.98)
    scheduler_DV = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=0.98)

    # 数据
    batch_size = int(GAN_param['batch_size'])
    train_len = int(GAN_param['train_len'])
    X_train = vs_G_datapack[0]
    X_test = vs_G_datapack_TEST[0]

    num_train_batches = max(1, int(len(X_train) / batch_size))
    num_test_batches = int(len(X_test) / batch_size)

    shuffle_index = np.arange(len(X_train))
    shuffle_index_TEST = np.arange(len(X_test))

    # 日志
    G_losses, D_losses = [], []
    G_losses_TEST, D_losses_TEST = [], []
    D_acc_hist, D_acc_TEST_hist = [], []
    d_real_losses, d_fake_losses = [], []
    d_real_losses_TEST, d_fake_losses_TEST = [], []
    lr_GV_log, lr_DV_log = [], []

    # 训练参数
    n_epochs = int(GAN_param['n_epochs'])
    n_critic = int(GAN_param.get('n_critic', 2))
    n_pretrain = int(GAN_param.get('n_pretrain', 5))   # 新增：MLE 预训练轮数
    lambda_mle = float(GAN_param.get('lambda_mle', 1.0))
    lambda_adv = float(GAN_param.get('lambda_adv', 0.25))  # 对抗权重建议较小
    lambda_ent = float(GAN_param.get('lambda_ent', 0.01))  # 熵正则促进多样性
    gumbel_tau_init = float(GAN_param.get('gumbel_tau_init', 1.5))
    gumbel_tau_min = float(GAN_param.get('gumbel_tau_min', 0.7))
    gumbel_tau_gamma = float(GAN_param.get('gumbel_tau_gamma', 0.97))  # 每个 epoch 退火

    for epoch in range(n_epochs):
        np.random.shuffle(shuffle_index)
        np.random.shuffle(shuffle_index_TEST)

        generator.train()
        discriminator.train()

        epoch_g_losses, epoch_d_losses = [], []
        epoch_d_real_losses, epoch_d_fake_losses = [], []
        epoch_d_preds, epoch_d_reals = [], []

        # 切 batch
        seqs_trn = [X_train[shuffle_index[i * batch_size:(i + 1) * batch_size]] for i in range(num_train_batches)]
        seqs_tst = [X_test[shuffle_index_TEST[i * batch_size:(i + 1) * batch_size]] for i in range(num_test_batches)]

        # Gumbel 温度（退火）
        tau = max(gumbel_tau_min, gumbel_tau_init * (gumbel_tau_gamma ** epoch))

        start_time = time.perf_counter()
        for num in range(num_train_batches):
            real_seqs = torch.as_tensor(seqs_trn[num], dtype=torch.long, device=device)
            B = real_seqs.shape[0]

            # label smoothing
            valid = torch.full((B, 1), 0.9, device=device)
            fake = torch.full((B, 1), 0.1, device=device)

            # ---------------------
            #  判别器训练 n_critic 次
            # ---------------------
            for _ in range(n_critic):
                optimizer_D.zero_grad(set_to_none=True)

                # REAL
                real_oh = one_hot_train(real_seqs, columns, train_len=train_len)  # (B,L,V)
                real_pred = discriminator(real_oh)
                d_real_loss = adv_loss_adaptive(real_pred, valid)

                # FAKE（使用硬采样，不回传到 G）
                with torch.no_grad():
                    # 使用你原有的离散采样接口（或可替换成温度/核采样）
                    fake_seqs = generator.gen_sample(train_len)  # (B,L) or (L,)
                    if isinstance(fake_seqs, torch.Tensor) and fake_seqs.ndim == 1:
                        fake_seqs = fake_seqs[None, :]
                fake_oh = one_hot_train(fake_seqs, columns, train_len=train_len)
                fake_pred = discriminator(fake_oh)
                d_fake_loss = adv_loss_adaptive(fake_pred, fake)

                d_loss = 0.5 * (d_real_loss + d_fake_loss)
                d_loss.backward()
                clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
                optimizer_D.step()

            # -----------------
            #  生成器训练
            # -----------------
            optimizer_G.zero_grad(set_to_none=True)

            # 1) Teacher-Forcing MLE（始终启用；预训练阶段仅此项）
            start_tokens = torch.full((B, 1), 1, dtype=torch.long, device=device)  # 'X' 的 id=1 作为 BOS
            ar_input = torch.cat([start_tokens, real_seqs], dim=1)[:, :-1].contiguous()
            gen_logits = generator.forward(ar_input)  # (B, L, V)
            g_mle_loss = mle_loss(gen_logits.reshape(-1, columns), real_seqs.reshape(-1))

            # 2) 可微对抗（预训练阶段关闭）
            if epoch >= n_pretrain:
                # 用 Gumbel-Softmax (straight-through) 采样得到 (B,L,V) one-hot，梯度可回传到 logits
                y_soft_onehot, mean_entropy = differentiable_autoregressive_onehots(
                    generator, B=B, L=train_len, start_id=1, tau=tau
                )  # (B,L,V)
                validity = discriminator(y_soft_onehot)  # (B,1) or (B,?)，取决于你的 D 实现
                g_adv_loss = adv_loss_adaptive(validity, torch.ones_like(validity, device=device) * 0.9)
                g_loss = lambda_mle * g_mle_loss + lambda_adv * g_adv_loss - lambda_ent * mean_entropy
            else:
                # 仅做 MLE 预训练
                mean_entropy = torch.zeros((), device=device)
                g_adv_loss = torch.zeros((), device=device)
                g_loss = g_mle_loss

            g_loss.backward()
            clip_grad_norm_(generator.parameters(), max_norm=5.0)
            optimizer_G.step()

            # --- Logging for the epoch ---
            epoch_g_losses.append(float(g_loss.item()))
            epoch_d_losses.append(float(d_loss.item()))
            epoch_d_real_losses.append(float(d_real_loss.item()))
            epoch_d_fake_losses.append(float(d_fake_loss.item()))

            # 判别器精度估计（阈值0.5）
            with torch.no_grad():
                def to_prob(x):
                    if x.min() < 0 or x.max() > 1:
                        return torch.sigmoid(x)
                    return x
                rp = to_prob(real_pred).detach().flatten().cpu().numpy()
                fp = to_prob(fake_pred).detach().flatten().cpu().numpy()
            epoch_d_preds.extend(rp)
            epoch_d_reals.extend(np.ones_like(rp))
            epoch_d_preds.extend(fp)
            epoch_d_reals.extend(np.zeros_like(fp))

        end_time = time.perf_counter()
        print(f"train epoch {epoch} elapsed: {end_time - start_time:0.3f}s")

        # --- Epoch summary (train) ---
        D_losses.append(float(np.mean(epoch_d_losses)))
        G_losses.append(float(np.mean(epoch_g_losses)))
        d_real_losses.append(float(np.mean(epoch_d_real_losses)))
        d_fake_losses.append(float(np.mean(epoch_d_fake_losses)))
        preds_binary = (np.array(epoch_d_preds) >= 0.5).astype(int)
        d_acc = float(np.mean(preds_binary == np.array(epoch_d_reals)))
        D_acc_hist.append(d_acc)

        # ---------- Testing ----------
        generator.eval()
        discriminator.eval()
        epoch_g_losses_test, epoch_d_losses_test = [], []
        epoch_d_real_losses_test, epoch_d_fake_losses_test = [], []
        epoch_d_preds_test, epoch_d_reals_test = [], []

        start_time = time.perf_counter()
        if num_test_batches > 0:
            with torch.no_grad():
                for num_test in range(num_test_batches):
                    real_seqs_test = torch.as_tensor(seqs_tst[num_test], dtype=torch.long, device=device)
                    B_test = real_seqs_test.shape[0]
                    valid_test = torch.full((B_test, 1), 1.0, device=device)
                    fake_test = torch.full((B_test, 1), 0.0, device=device)

                    # D
                    real_oh_t = one_hot_train(real_seqs_test, columns, train_len=train_len)
                    real_pred_test = discriminator(real_oh_t)
                    d_real_loss_test = adv_loss_adaptive(real_pred_test, valid_test)

                    fake_seqs_test = generator.gen_sample(train_len)
                    if isinstance(fake_seqs_test, torch.Tensor) and fake_seqs_test.ndim == 1:
                        fake_seqs_test = fake_seqs_test[None, :]
                    fake_oh_t = one_hot_train(fake_seqs_test, columns, train_len=train_len)
                    fake_pred_test = discriminator(fake_oh_t)
                    d_fake_loss_test = adv_loss_adaptive(fake_pred_test, fake_test)
                    d_loss_test = 0.5 * (d_real_loss_test + d_fake_loss_test)

                    # G（测试阶段仅做 MLE + 评估对抗）
                    st = torch.full((B_test, 1), 1, dtype=torch.long, device=device)
                    ar_input_test = torch.cat([st, real_seqs_test], dim=1)[:, :-1].contiguous()
                    gen_logits_test = generator.forward(ar_input_test)
                    g_mle_loss_test = mle_loss(gen_logits_test.reshape(-1, columns), real_seqs_test.reshape(-1))
                    validity_test = discriminator(fake_oh_t)
                    g_adv_loss_test = adv_loss_adaptive(validity_test, torch.ones_like(validity_test, device=device))
                    g_loss_test = lambda_mle * g_mle_loss_test + lambda_adv * g_adv_loss_test

                    # Logging
                    epoch_g_losses_test.append(float(g_loss_test.item()))
                    epoch_d_losses_test.append(float(d_loss_test.item()))
                    epoch_d_real_losses_test.append(float(d_real_loss_test.item()))
                    epoch_d_fake_losses_test.append(float(d_fake_loss_test.item()))

                    def to_prob(x):
                        if x.min() < 0 or x.max() > 1:
                            return torch.sigmoid(x)
                        return x
                    epoch_d_preds_test.extend(to_prob(real_pred_test).flatten().cpu().numpy())
                    epoch_d_reals_test.extend(np.ones(B_test))
                    epoch_d_preds_test.extend(to_prob(fake_pred_test).flatten().cpu().numpy())
                    epoch_d_reals_test.extend(np.zeros(B_test))

            D_losses_TEST.append(float(np.mean(epoch_d_losses_test)))
            G_losses_TEST.append(float(np.mean(epoch_g_losses_test)))
            d_real_losses_TEST.append(float(np.mean(epoch_d_real_losses_test)))
            d_fake_losses_TEST.append(float(np.mean(epoch_d_fake_losses_test)))
            preds_binary_test = (np.array(epoch_d_preds_test) >= 0.5).astype(int)
            d_acc_test = float(np.mean(preds_binary_test == np.array(epoch_d_reals_test)))
            D_acc_TEST_hist.append(d_acc_test)
        else:
            D_losses_TEST.append(np.nan); G_losses_TEST.append(np.nan)
            d_real_losses_TEST.append(np.nan); d_fake_losses_TEST.append(np.nan)
            D_acc_TEST_hist.append(np.nan)

        end_time = time.perf_counter()
        if num_test_batches > 0:
            print(f"test epoch {epoch} elapsed: {end_time - start_time:0.3f}s")

        # --- Print epoch results ---
        print(
            "[data: {}][train_name:{}] [Epoch {}/{}] [D loss: {:.4f}] [G loss: {:.4f}] [D_acc: {:.4f}] [tau: {:.3f}] [lr_DV: {:.6f}] [lr_GV: {:.6f}]".format(
                GAN_param.get('date', 'na'), GAN_param.get('model_name', 'na'),
                epoch + 1, n_epochs,
                D_losses[-1], G_losses[-1], D_acc_hist[-1], tau,
                optimizer_D.param_groups[0]['lr'], optimizer_G.param_groups[0]['lr']
            ))
        print(
            "[data_TEST: {}][train_name_TEST:{}] [Epoch_TEST {}/{}] [D loss_TEST: {:.4f}] [G loss_TEST: {:.4f}] [D_acc_TEST: {}]".format(
                GAN_param.get('date', 'na'), GAN_param.get('model_name', 'na'),
                epoch + 1, n_epochs,
                D_losses_TEST[-1], G_losses_TEST[-1],
                'nan' if np.isnan(D_acc_TEST_hist[-1]) else f"{D_acc_TEST_hist[-1]:.4f}"
            ))

        # --- Write to log files ---
        with open(Model_Eval_Fig + "LOSS_ACC.txt", 'a+', encoding='utf-8') as f:
            f.writelines("date: {} train_name: {} Epoch :{}  D loss:  {:.6f}   G loss:  {:.6f}  d_acc: {:.6f}\n".format(
                GAN_param.get('date', 'na'), GAN_param.get('model_name', 'na'), epoch + 1, D_losses[-1], G_losses[-1], D_acc_hist[-1]))
        with open(Model_Eval_Fig + "LOSS_ACC_TEST.txt", 'a+', encoding='utf-8') as f:
            f.writelines(
                "date_TEST: {} train_name: {} Epoch_TEST :{}  D loss_TEST:  {:.6f}   G loss_TEST:  {:.6f}  d_acc_TEST: {}\n".format(
                    GAN_param.get('date', 'na'), GAN_param.get('model_name', 'na'), epoch + 1, D_losses_TEST[-1], G_losses_TEST[-1],
                    'nan' if np.isnan(D_acc_TEST_hist[-1]) else f"{D_acc_TEST_hist[-1]:.6f}"))

        # --- Plotting ---
        try:
            plot_losses(
                [D_losses, d_real_losses, d_fake_losses, D_losses_TEST, d_real_losses_TEST, d_fake_losses_TEST],
                ["D_loss", "D_real", 'D_fake', 'D_loss_TEST', 'D_real_TEST', 'D_fake_TEST'],
                Model_Eval_Fig + f'd_loss_components_{GAN_param.get("model_name", "")}{GAN_param.get("date", "")}.png'
            )
            plot_losses(
                [D_acc_hist, D_acc_TEST_hist],
                ["D_acc", 'D_acc_TEST'],
                Model_Eval_Fig + f'D_acc_{GAN_param.get("model_name", "")}{GAN_param.get("date", "")}.png'
            )
            plot_losses(
                [D_losses, D_losses_TEST],
                ["D_losses", 'D_losses_TEST'],
                Model_Eval_Fig + f'D_losses_{GAN_param.get("model_name", "")}{GAN_param.get("date", "")}.png'
            )
            plot_losses(
                [G_losses, G_losses_TEST],
                ["G_losses", 'G_losses_TEST'],
                Model_Eval_Fig + f'G_losses_{GAN_param.get("model_name", "")}{GAN_param.get("date", "")}.png'
            )
        except Exception as e:
            print("[WARN] plot_losses 出错（跳过本轮绘图）：", e)

        # --- 学习率 step ---
        lr_GV_log.append(optimizer_G.param_groups[0]['lr'])
        lr_DV_log.append(optimizer_D.param_groups[0]['lr'])
        scheduler_GV.step()
        scheduler_DV.step()

        # 保存 checkpoint
        torch.save({'model': discriminator.state_dict()}, checkpoint_dir_D + f'D_repeat_{epoch}.pth')
        torch.save({'model': generator.state_dict()}, checkpoint_dir_G + f'G_repeat_{epoch}.pth')

    # 训练结束后再画一次 LR
    try:
        plot_losses([lr_GV_log, lr_DV_log], ["lr_G", "lr_D"], Model_Eval_Fig + f'lr_{GAN_param.get("model_name", "")}{GAN_param.get("date", "")}.png')
    except Exception as e:
        print("[WARN] lr plot 出错：", e)

    print('Finished Training')


# -------------------------
# 入口
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='./data/semantic_tuning_data.csv', help="包含列 iAAs 的CSV路径")
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 读取超参
    GAN_param = parameters.GAN
    Pretrain_param = parameters.roberta
    # 如果 parameters.GAN 中未配置以下键，可在此给出默认值
    GAN_param.setdefault('n_pretrain', 5)
    GAN_param.setdefault('lambda_mle', 1.0)
    GAN_param.setdefault('lambda_adv', 0.25)
    GAN_param.setdefault('lambda_ent', 0.01)
    GAN_param.setdefault('gumbel_tau_init', 1.5)
    GAN_param.setdefault('gumbel_tau_min', 0.7)
    GAN_param.setdefault('gumbel_tau_gamma', 0.97)
    GAN_param.setdefault('n_critic', 2)
    GAN_param.setdefault('lr', 1e-4)
    print(GAN_param)

    # 预训练模型（roberta）
    pretrain_model = 'roberta'
    pretrain_pth_file = './model/words_model_ep3.pth'

    # 输出与ckpt目录
    date_tag = str(GAN_param.get('date', 'unknown'))
    model_name = str(GAN_param.get('model_name', 'model'))
    Model_Eval_Fig = f'../Figures/PreGan/{date_tag}/{pretrain_model}2{model_name}/'
    os.makedirs(Model_Eval_Fig, exist_ok=True)
    checkpoint_dir_D = f'../checkpoint/pretrain/train/{date_tag}/{pretrain_model}2{model_name}/D/'
    checkpoint_dir_G = f'../checkpoint/pretrain/train/{date_tag}/{pretrain_model}2{model_name}/G/'
    os.makedirs(checkpoint_dir_D, exist_ok=True)
    os.makedirs(checkpoint_dir_G, exist_ok=True)

    # 载入预训练BERT/Roberta（冻结）
    pretrain_model_obj = torch.load(pretrain_pth_file, map_location=device, weights_only=False)
    for para in pretrain_model_obj.parameters():
        para.requires_grad = False

    # 加载 CSV -> 编码 -> 切分
    train_len = int(GAN_param['train_len'])
    VS_D_XTS, VS_D_XTS_TEST = load_csv_and_split(args.csv_path, train_len, args.test_ratio, args.seed)

    # 组装成原有数据包接口（输入=目标）
    vs_G_datapack = (VS_D_XTS, VS_D_XTS)
    vs_G_datapack_TEST = (VS_D_XTS_TEST, VS_D_XTS_TEST)

    # 开训
    columns = VOCAB_SIZE
    train_and_eval(pretrain_model_obj, vs_G_datapack, vs_G_datapack_TEST, GAN_param, Pretrain_param, columns, aa_list)
