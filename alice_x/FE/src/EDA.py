# EDA.py
# Estimation of Distribution Algorithm
import os
import group
import math
import numpy as np
import basic

class EDAGroup(group.Group):
    """
    EDA (Estimation of Distribution Algorithm) Group
    """
    def __init__(self, n: int, length: int):
        super(EDAGroup, self).__init__(n, length)
        self.rbest = 0.5
        self.alpha = 0.6
        self.bestdis = np.zeros((self.length, len(self.lib)), float)

    def setRbest(self, r: float):
        self.rbest = r

    def setAlpha(self, alpha: float):
        self.alpha = alpha

    def calcBestdis(self):
        newdis = np.zeros((self.length, len(self.lib)), float)
        nbest = max(1, math.floor(self.n * self.rbest))
        sorted_idx = np.argsort(self.score)[::-1]
        top_inds = self.inds[sorted_idx[:nbest]]

        c2i = self._char2idx
        L = self.length
        V = len(self.lib)

        for j in range(L):
            cnt = np.zeros(V, dtype=np.float64)
            for s in top_inds:
                if j < len(s):
                    ch = s[j]
                    idx = c2i.get(ch, None)
                    if idx is not None:
                        cnt[idx] += 1.0
            newdis[j] = cnt

        if self.generation == 0:
            self.bestdis = newdis / max(1, nbest)
        else:
            self.bestdis = (1 - self.alpha) * self.bestdis + self.alpha * (newdis / max(1, nbest))

    def sampling(self) -> str:
        seq_idx = []
        V = len(self.lib)
        for i in range(self.length):
            row = self.bestdis[i]
            s = row.sum()
            if s <= 0:
                seq_idx.append(np.random.randint(V))
            else:
                cdf = np.cumsum(row)
                r = np.random.random() * cdf[-1]
                seq_idx.append(int(np.searchsorted(cdf, r)))
        return ''.join(self._idx2char[j] for j in seq_idx)

    def evolution(self):
        self.calcBestdis()
        new = [self.sampling() for _ in range(self.n)]
        self.inds = np.array(new, dtype=f'<U{self.length}')
        self.generation += 1
        if self.auto_eval_after_evolution:
            self.evaluate()