import evaluation as eva
import numpy as np
import pandas as pd
from typing import List, Optional
from abc import ABC, abstractmethod

class Group(ABC):
    """
    Represents a group of individuals for genetic algorithms.

    Attributes:
        n (int): Group capacity.
        lib (str): Gene library.
        inds (np.ndarray): Individuals' sequences.
        score (np.ndarray): Scores of individuals.
        generation (int): Current generation, starts from 0.
        length (int): Length of individual sequences.
        evamet: Evaluation method instance.
        evafun: Vectorized evaluation function.
        dis (np.ndarray): Distribution of genes in positions.
    """

    def __init__(self, n: int, length: int):
        self.n = n
        self.length = length
        self.lib = 'ACDEFGHIKLMNPQRSTVWY'
        self.inds = np.empty(n, dtype=f'<U{length}')
        self.score = np.empty(n, dtype=float)
        self.generation = 0
        self.dis = np.zeros((length, len(self.lib)), dtype=float)

        # Default evaluation method
        self.evamet = eva.MSAEva()  # This should be set to an actual evaluation method
        self.evafun = np.vectorize(self.evamet.evaluate)  # This will be set when evamet is set
        self.initEqualLen()

    @property
    def evamet(self):
        return self._evamet

    @evamet.setter
    def evamet(self, value):
        self._evamet = value
        if value is not None:
            self.evafun = np.vectorize(value.evaluate)

    def setMet(self, sampath: str, matpath: str):
        """Set evaluation method and load necessary data."""
        if self.evamet is None:
            raise ValueError("Evaluation method not initialized")
        self.evamet.load_sample(sampath)
        self.evamet.load_matrix(matpath)

    def resetGeneration(self):
        """Reset generation counter to 0."""
        self.generation = 0

    def evaluate(self):
        """Evaluate all individuals in the group."""
        if self.evafun is None:
            raise ValueError("Evaluation function not set")
        self.score = self.evafun(self.inds)

    def showMsg(self, name: Optional[str] = None):
        """Display group information."""
        if name:
            print(f'Group: {name}')
        print(f'Generation: {self.generation}, n: {self.n}, length: {self.length}')
        df = pd.DataFrame({'seq': self.inds, 'score': self.score})
        print(df.sort_values(by='score', ascending=False, ignore_index=True))

    def outputMsg(self, path: str):
        """Output group information to a CSV file."""
        df = pd.DataFrame({'seq': self.inds, 'score': self.score})
        df.to_csv(path, index=False)
        print(f'Message successfully output to file: {path}')

    def initEqualLen(self):
        """Initialize with random equal length sequences."""
        self.inds = np.array([''.join(np.random.choice(list(self.lib), self.length)) for _ in range(self.n)])
        self.evaluate()

    def initFromFile(self, df: pd.DataFrame, col: str = 'seq', update_length: bool = True, cut: bool = False):
        """Initialize from a DataFrame."""
        if update_length:
            self.length = max(len(seq) for seq in df[col])

        if cut:
            self.inds = df[col].head(self.n).values.astype(f'<U{self.length}')
        else:
            self.inds = df[col].values.astype(f'<U{self.length}')
            self.n = len(self.inds)
        self.evaluate()

    def calcDis(self):
        """Calculate distribution of genes in positions."""
        self.dis = np.array([[np.sum(self.inds[:, i] == gene) for gene in self.lib] for i in range(self.length)]) / self.n

    def sortedScore(self) -> np.ndarray:
        """Return sorted scores in descending order."""
        return np.sort(self.score)[::-1]

    @abstractmethod
    def evolution(self):
        """Perform one step of evolution."""
        pass