# group.py
import evaluation as eva
import numpy as np
import pandas as pd
from typing import List, Optional
from abc import ABC, abstractmethod

class Group(ABC):
    """
    Represents a group of individuals for genetic algorithms.
    """
    def __init__(self, n: int, length: int):
        self.n = n
        self.length = length
        self.lib = 'ACDEFGHIKLMNPQRSTVWY'
        self._char2idx = {c: i for i, c in enumerate(self.lib)}
        self._idx2char = {i: c for i, c in enumerate(self.lib)}

        self.inds = np.empty(n, dtype=f'<U{length}')
        self.score = np.empty(n, dtype=float)
        self.generation = 0
        self.dis = np.zeros((length, len(self.lib)), dtype=float)

        self._evamet = eva.MSAEva()
        self.evafun = np.vectorize(self._evamet.evaluate)

        self.auto_eval_after_evolution = True

        self.initEqualLen()

    @property
    def evamet(self):
        return self._evamet

    @evamet.setter
    def evamet(self, value):
        self._evamet = value
        if value is not None: self.evafun = np.vectorize(value.evaluate)

    def setMet(self, sampath: str, matpath: str):
        if self._evamet is None: raise ValueError("Evaluation method not initialized")
        self._evamet.loadSample(sampath)
        self._evamet.loadMatrix(matpath)

    def setBasicReg(self, **kwargs):
        if self._evamet is None: raise ValueError("Evaluation method not initialized")
        self._evamet.setBasicReg(**kwargs)

    def set_auto_eval(self, flag: bool):
        self.auto_eval_after_evolution = bool(flag)

    def resetGeneration(self):
        self.generation = 0

    def evaluate(self, loss_type='basic'):
        if self.evafun is None:
            raise ValueError("Evaluation function not set")
        self.score = self.evafun(self.inds, loss_type)

    def showMsg(self, name: Optional[str] = None):
        if name:
            print(f'Group: {name}')
        print(f'Generation: {self.generation}, n: {self.n}, length: {self.length}')
        df = pd.DataFrame({'seq': self.inds, 'score': self.score})
        print(df.sort_values(by='score', ascending=False, ignore_index=True))

    def outputMsg(self, path: str):
        df = pd.DataFrame({'seq': self.inds, 'score': self.score})
        df.to_csv(path, index=False)
        print(f'Message successfully output to file: {path}')

    def initEqualLen(self):
        rng = np.random.default_rng()
        lib_arr = np.array(list(self.lib))
        idx = rng.integers(0, len(self.lib), size=(self.n, self.length))
        self.inds = np.apply_along_axis(lambda row: ''.join(lib_arr[row]), 1, idx).astype(f'<U{self.length}')
        self.evaluate()

    def initFromFile(self, df: pd.DataFrame, col: str = 'seq', update_length: bool = True, cut: bool = False, loss_type='basic'):
        if update_length:
            self.length = int(max(df[col].map(len)))
        if cut:
            self.inds = df[col].head(self.n).values.astype(f'<U{self.length}')
        else:
            self.inds = df[col].values.astype(f'<U{self.length}')
            self.n = len(self.inds)
        self.evaluate(loss_type)

    def calcDis(self):
        lib = np.array(list(self.lib))
        dis = np.zeros((self.length, len(self.lib)), dtype=float)
        for i in range(self.length):
            col = np.array([s[i] for s in self.inds])
            idxs = np.fromiter((self._char2idx[c] for c in col), dtype=np.int32, count=len(col))
            dis[i] = np.bincount(idxs, minlength=len(self.lib))
        self.dis = dis / self.n
        return self.dis

    def sortedScore(self) -> np.ndarray:
        return np.sort(self.score)[::-1]

    @abstractmethod
    def evolution(self):
        pass