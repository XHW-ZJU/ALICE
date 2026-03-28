# mixture.py
# Mixed Algorithm EDG
import math
import numpy as np
from typing import Tuple
from EDA import EDAGroup
from GA import GAGroup

class EDGGroup(EDAGroup, GAGroup):
    """
    Co-evolution of Genetic Algorithm and Estimation of Distribution Algorithm
    """
    def __init__(self, n: int, length: int, r_eda: float = 0.3, r_ga_copy: float = 0.2):
        super().__init__(n, length)
        self.alpha = 1.0
        self.r_eda = r_eda
        self.r_ga_copy = r_ga_copy

    def _calculate_population_sizes(self) -> Tuple[int, int, int]:
        n_eda = math.floor(self.n * self.r_eda)
        n_ga = self.n - n_eda
        n_ga_copy = math.floor(n_ga * self.r_ga_copy)
        n_ga_cross = n_ga - n_ga_copy
        return n_eda, n_ga_copy, n_ga_cross

    def _perform_ga_evolution(self, n_ga_cross: int) -> np.ndarray:
        new_inds = self.inds[np.argsort(self.score)]  # ascending
        w = np.maximum(self.score, 0.0)
        s = w.sum()
        probs = None if s <= 0 else (w / s)
        idx_a = np.random.choice(self.n, size=n_ga_cross, p=probs)
        idx_b = np.random.choice(self.n, size=n_ga_cross, p=probs)
        for i in range(n_ga_cross):
            new_ind = self.crossover(self.inds[idx_a[i]], self.inds[idx_b[i]])
            new_ind = self.mutate(new_ind)
            new_inds[i] = new_ind
        return new_inds

    def _perform_eda_evolution(self, new_inds: np.ndarray, n_ga_cross: int, n_eda: int) -> np.ndarray:
        self.calcBestdis()
        samples = [self.sampling() for _ in range(n_eda)]
        new_inds[n_ga_cross:n_ga_cross + n_eda] = samples
        return new_inds

    def evolution(self):
        n_eda, _, n_ga_cross = self._calculate_population_sizes()
        new_inds = self._perform_ga_evolution(n_ga_cross)
        new_inds = self._perform_eda_evolution(new_inds, n_ga_cross, n_eda)
        self.inds = new_inds
        self.generation += 1
        if self.auto_eval_after_evolution:
            self.evaluate()