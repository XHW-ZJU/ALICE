# Estimation of Distribution Algorithm
import os
import group
import math
import numpy as np
import basic


class EDAGroup(group.Group):
    """
    EDA (Estimation of Distribution Algorithm) Group

    Attributes:
        rbest (float): Rate of best individuals
        alpha (float): Alpha parameter for PBIL strategy (if using UMDA strategy, alpha = 1)
        bestdis (np.ndarray): Distribution of best individuals
    """

    def __init__(self, n: int, length: int):
        super(EDAGroup, self).__init__(n, length)
        self.rbest = 0.5
        self.alpha = 0.6
        self.bestdis = np.zeros((self.length, len(self.lib)), float)

    def setRbest(self, r: float):
        """Set the rate of best individuals"""
        self.rbest = r

    def setAlpha(self, alpha: float):
        """Set the alpha parameter"""
        self.alpha = alpha

    def calcBestdis(self):
        """Calculate distribution of best individuals"""
        newdis = np.zeros((self.length, len(self.lib)), float)
        nbest = math.floor(self.n * self.rbest)
        sortedind = self.inds[np.argsort(self.score)[::-1]]  # sort by score
        for i in range(nbest):
            for j in range(self.length):
                newdis[j][self.lib.index(sortedind[i][j])] += 1
        if self.generation == 0:
            self.bestdis = newdis / self.n
        else:
            self.bestdis = (1 - self.alpha) * self.bestdis + self.alpha * newdis / self.n

    def sampling(self) -> str:
        """Sample a new individual from the distribution"""
        sample = []
        for i in range(self.length):
            sample.append(self.lib[basic.weighted_choice(self.bestdis[i])])
        return ''.join(sample)

    def evolution(self):
        """Perform one generation of evolution"""
        # Estimation of distribution
        self.calcBestdis()

        # Sampling
        for i in range(self.n):
            self.inds[i] = self.sampling()

        # Evaluate
        self.generation += 1
        self.evaluate()


def create_output_directory(path: str):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(path):
        os.mkdir(path)


def testEDA():
    """Test function for EDA"""
    output_dir = "./output"
    create_output_directory(output_dir)

    group = EDAGroup(20, 10)
    group.initEqualLen()
    group.showMsg()
    for gen in range(100):
        group.evolution()
        group.showMsg()
        # You can add code here to save results to a file in the output directory


# Uncomment the following lines to run the test
# if __name__ == "__main__":
#     testEDA()