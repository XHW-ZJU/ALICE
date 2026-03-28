# Mixed Algorithm
import math
import numpy as np
from typing import Tuple
from EDA import EDAGroup
from GA import GAGroup
import basic


class EDGGroup(EDAGroup, GAGroup):
    """
    Co-evolution of Genetic Algorithm and Estimation of Distribution Algorithm

    Attributes:
        alpha (float): Parameter for EDA algorithm.
        r_eda (float): Rate of EDA individuals in the next-generation group.
        r_ga_copy (float): Rate of copy individuals in the GA individuals.
    """

    def __init__(self, n: int, length: int, r_eda: float = 0.3, r_ga_copy: float = 0.2):
        super().__init__(n, length)
        self.alpha = 1
        self.r_eda = r_eda
        self.r_ga_copy = r_ga_copy

    def _calculate_population_sizes(self) -> Tuple[int, int, int]:
        """Calculate the sizes of different subpopulations."""
        n_eda = math.floor(self.n * self.r_eda)
        n_ga = self.n - n_eda
        n_ga_copy = math.floor(n_ga * self.r_ga_copy)
        n_ga_cross = n_ga - n_ga_copy
        return n_eda, n_ga_copy, n_ga_cross

    def _perform_ga_evolution(self, n_ga_cross: int) -> np.ndarray:
        """Perform Genetic Algorithm evolution."""
        new_inds = self.inds[np.argsort(self.score)]  # ascending
        for i in range(n_ga_cross):
            idx_a, idx_b = np.random.choice(self.n, 2, p=self.score/np.sum(self.score))
            new_ind = self.crossover(self.inds[idx_a], self.inds[idx_b])
            new_ind = self.mutate(new_ind)
            new_inds[i] = new_ind
        return new_inds

    def _perform_eda_evolution(self, new_inds: np.ndarray, n_ga_cross: int, n_eda: int) -> np.ndarray:
        """Perform Estimation of Distribution Algorithm evolution."""
        self.calcBestdis()
        new_inds[n_ga_cross:n_ga_cross + n_eda] = [self.sampling() for _ in range(n_eda)]
        return new_inds

    def evolution(self):
        """Perform one step of evolution combining GA and EDA."""
        n_eda, _, n_ga_cross = self._calculate_population_sizes()

        # GA evolution
        new_inds = self._perform_ga_evolution(n_ga_cross)

        # EDA evolution
        new_inds = self._perform_eda_evolution(new_inds, n_ga_cross, n_eda)

        # Update population and evaluate
        self.inds = new_inds
        self.generation += 1
        self.evaluate()