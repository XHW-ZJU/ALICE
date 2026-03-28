# evaluation.py
"""
basic-only 版本 + Stage2 扩展：
- 保留 basic（Ref+/Ref- 差异）用于历史/Stage1。
- 新增 Stage2 专用接口：不再以 Ref+/Ref- 相似度为正向信号，仅作惩罚/距离计算；
  目标转为：奖励“预测高 + 远离 Ref+/Ref- + 远离上一代（多样性）”。
- 所有分数 clamp 至 [0,1]。
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple, List, Sequence, Optional, Dict
import numpy as np
import json, math

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def _w(fn): return fn
        return _w

import basic  # 复用 p-范数/mean 实现（calculate_mean 等）


# ==================== Config ====================
@dataclass
class BasicRegConfig:
    # Stage1/basic 用
    w_pos: float = 0.0
    w_neg: float = 0.0
    w_novel: float = 0.2
    kl_tau: float = 1.0
    kl_eps: float = 1e-4
    kl_scheme: int = 1
    sim_margin_pos: float = 0.0


# ==================== SW 内核 ====================
@njit(cache=True)
def _sw_score_numba(seqa_idx, seqb_idx, matrix, gap, ext):
    la=seqa_idx.size; lb=seqb_idx.size
    H=np.zeros((la+1,lb+1),dtype=np.float64)
    E=np.full((la+1,lb+1),-np.inf); F=np.full((la+1,lb+1),-np.inf)
    best=0.0
    for i in range(1,la+1):
        ai=seqa_idx[i-1]
        for j in range(1,lb+1):
            bj=seqb_idx[j-1]; s=matrix[ai,bj]
            E[i,j]=max(H[i,j-1]+gap,E[i,j-1]+ext)
            F[i,j]=max(H[i-1,j]+gap,F[i-1,j]+ext)
            h=max(0.0,H[i-1,j-1]+s,E[i,j],F[i,j]); H[i,j]=h
            if h>best: best=h
    return best


# ==================== 评估器 ====================
class MSAEva:
    """Smith-Waterman + basic + Stage2 扩展"""
    def __init__(self):
        self.gap=-4.0; self.extend=-2.0
        self.sampath=''; self.samples=([],[])
        self.matpath=''; self.matseq=''
        self.matrix=np.zeros((20,20),dtype=np.float64)
        self.scorebound=(-1.0,1.0)
        self.alphabet="ACDEFGHIKLMNPQRSTVWY"
        self._c2i={c:i for i,c in enumerate(self.alphabet)}
        self.regcfg=BasicRegConfig()
        self._pair_cache={}
        # 默认载入
        self.loadSample('./datapack/sequences.json')
        self.loadMatrix('BLOSSUM62.txt')
        self._refresh_encoded_samples()

    # ---------- 公共接口 ----------
    def setBasicReg(self, **kw):
        for k,v in kw.items():
            if hasattr(self.regcfg,k):
                setattr(self.regcfg,k,v)

    def loadSample(self, path:str):
        self.sampath=path
        J=json.load(open(path,'r'))
        pos=list(map(str,J.get('Positive',J.get('positive',[]))))
        neg=list(map(str,J.get('Negative',J.get('negative',[]))))
        self.samples=(pos,neg)
        self._refresh_encoded_samples()
        self._build_posneg_freq()

    def loadMatrix(self, path:str):
        self.matpath=path
        with open(path,'r') as f:
            seq=f.readline().strip().upper(); self.matseq=seq
            if seq=='NOPE':
                mat=list(map(int,f.readline().split()))
                self.scorebound=(float(mat[0]),float(mat[1]))
                self.matrix=np.array(mat,float)
            else:
                data=[list(map(float,f.readline().split())) for _ in range(len(seq))]
                self.matrix=np.array(data,float)
                self.scorebound=(self.extend,float(np.diag(self.matrix).mean()))
        self._pair_cache={}

    # ---------- 内部工具 ----------
    def _encode_seq(self,s:str)->np.ndarray:
        return np.fromiter((self._c2i.get(ch.upper(),0) for ch in s),dtype=np.int32)

    def _refresh_encoded_samples(self):
        pos,neg=self.samples
        self._pos_idx=[self._encode_seq(s) for s in pos]
        self._neg_idx=[self._encode_seq(s) for s in neg]

    def _build_posneg_freq(self):
        def freq(seqs,L=None):
            if not seqs: return np.full((L or 7,20),1/20,dtype=float)
            if L is None: L=max(map(len,seqs))
            M=np.full((L,20),1e-9)
            for s in seqs:
                for i,ch in enumerate(s[:L]):
                    j=self._c2i.get(ch.upper(),None)
                    if j is not None: M[i,j]+=1.0
            M=M/M.sum(1,keepdims=True); return M
        L=7
        if self.samples[0]: L=max(L,max(map(len,self.samples[0])))
        if self.samples[1]: L=max(L,max(map(len,self.samples[1])))
        self._pos_freq=freq(self.samples[0],L)
        self._neg_freq=freq(self.samples[1],L)

    def _score_pair(self,a:str,b:str)->float:
        ms=int(self.matrix.size+int(self.matrix.sum()))
        key=(a,b,ms,self.gap,self.extend)
        if key in self._pair_cache: return self._pair_cache[key]
        if self.matseq==self.alphabet:
            ai=self._encode_seq(a); bi=self._encode_seq(b)
        else:
            m={c:i for i,c in enumerate(self.matseq)}
            ai=np.fromiter((m.get(ch.upper(),0) for ch in a),dtype=np.int32)
            bi=np.fromiter((m.get(ch.upper(),0) for ch in b),dtype=np.int32)
        s=_sw_score_numba(ai,bi,self.matrix,self.gap,self.extend)
        self._pair_cache[key]=float(s); return float(s)

    # ========== Stage1/basic ==========
    def evaluate(self, seq: str, loss_type='basic', dim=1) -> float:
        posSeqs, negSeqs = self.samples
        pos = [self._score_pair(seq, s) for s in posSeqs] if posSeqs else [0.0]
        neg = [self._score_pair(seq, s) for s in negSeqs] if negSeqs else [0.0]
        nf = max(1.0, len(seq) * (self.scorebound[1] - self.scorebound[0]))
        pos = [s / nf for s in pos]
        neg = [s / nf for s in neg]

        pm = basic.calculate_mean(dim, pos)
        nm = basic.calculate_mean(dim, neg)
        base = (pm - nm) / 2.0 + 0.5  # ~0.5±

        # 简化：Stage1 推荐 w_pos=w_neg=0，仅保留轻度新颖性惩罚（可设 0）
        cfg = self.regcfg
        w_pos = getattr(cfg, 'w_pos', 0.0)
        w_neg = getattr(cfg, 'w_neg', 0.0)
        w_novel = getattr(cfg, 'w_novel', 0.0)

        # 轻量新颖性（相对 Ref+）
        novelty = 0.0
        if w_novel != 0.0:
            L = max(len(seq), self._pos_freq.shape[0])
            Q = np.full((L,20), cfg.kl_eps)
            for i,ch in enumerate(seq[:L]):
                j=self._c2i.get(ch.upper(),None)
                if j is not None: Q[i,j]+=1.0
            Q = Q / Q.sum(1,keepdims=True)
            Pp = np.clip(self._pos_freq[:L],1e-12,1)
            Pp = Pp / Pp.sum(1,keepdims=True)
            # JS 范围 [0, ln2]，做个归一
            def _kl(A,B):
                eps=1e-12; A=np.clip(A,eps,1); B=np.clip(B,eps,1)
                return float(np.sum(A*(np.log(A)-np.log(B))))
            def _js(A,B):
                M=0.5*(A+B); return 0.5*_kl(A,M)+0.5*_kl(B,M)
            novelty = min(1.0, _js(Q,Pp)/(math.log(2.0)+1e-12))

        score = base + w_pos*pm + w_neg*(-nm) - w_novel*novelty
        return float(max(0.0, min(1.0, score)))

    # ========== Stage2/外扩 ==========
    @staticmethod
    def hamming(a: str, b: str) -> int:
        L = min(len(a), len(b))
        return sum(1 for i in range(L) if a[i] != b[i]) + abs(len(a)-len(b))

    def ref_distance_norm(self, seq: str, refplus: Sequence[str], refminus: Sequence[str]) -> float:
        if not refplus and not refminus:
            return 1.0
        L = max(1, len(seq))
        dists = []
        if refplus:
            dists += [self.hamming(seq, r) for r in refplus]
        if refminus:
            dists += [self.hamming(seq, r) for r in refminus]
        dmin = float(min(dists)) if dists else L
        return max(0.0, min(1.0, dmin / L))

    def diversity_prev_norm(self, seq: str, prev_gen: Sequence[str]) -> float:
        if not prev_gen:
            return 1.0
        L = max(1, len(seq))
        dmin = min(self.hamming(seq, s) for s in prev_gen)
        return max(0.0, min(1.0, dmin / L))

    def evaluate_stage2_scalar(self,
                               seq: str,
                               fit_pred: float,
                               ht_pred: float,
                               refplus: Sequence[str],
                               refminus: Sequence[str],
                               prev_gen: Sequence[str],
                               mode: str,
                               fit_base: float,
                               ht_base: float,
                               lambdas: Dict[str,float]) -> float:
        ref_dist = self.ref_distance_norm(seq, refplus, refminus)
        div_prev = self.diversity_prev_norm(seq, prev_gen)

        pen_fit = max(0.0, fit_base - fit_pred)
        pen_ht  = max(0.0, ht_base  - ht_pred)

        if mode == "sum":
            reward = fit_pred + ht_pred
        elif mode == "min":
            reward = min(fit_pred, ht_pred)
        elif mode == "rank_sum":
           
            reward = fit_pred + ht_pred
        elif mode == "sum+div":
            reward = fit_pred + ht_pred + lambdas.get("w_div",0.2)*div_prev
        elif mode == "sum+ref":
            reward = fit_pred + ht_pred + lambdas.get("w_ref",0.2)*ref_dist
        elif mode == "min+div":
            reward = min(fit_pred, ht_pred) + lambdas.get("w_div",0.2)*div_prev
        elif mode == "min+ref":
            reward = min(fit_pred, ht_pred) + lambdas.get("w_ref",0.2)*ref_dist
        elif mode == "all":
            reward = (fit_pred + ht_pred
                      + lambdas.get("w_div",0.2)*div_prev
                      + lambdas.get("w_ref",0.2)*ref_dist)
        else:
            reward = fit_pred + ht_pred

        penalty_refnear = lambdas.get("w_ref_pen",0.3) * (1.0 - ref_dist)
        penalty_lowfit  = lambdas.get("w_fit_pen",0.3) * pen_fit
        penalty_lowht   = lambdas.get("w_ht_pen",0.3)  * pen_ht

        score = reward - (penalty_refnear + penalty_lowfit + penalty_lowht)
        return float(max(0.0, min(1.0, score)))