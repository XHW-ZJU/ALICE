# Simulated Annealing

import random
import group
import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SAGroup(group.Group):
    '''
    -- MEMBERS --
    rmutate : mutate rate
    rcopy : copy rate ( copy rate + cross rate = 1)
    doublecross : cross strategy

    -- FUNCTIONS --
    evolution() : evolution
    '''
    # Initial Temperature
    t0: float = 1.6
    iteration: int = 500
    coef: float = 2.0
    M: int = 50
    method: str = "classic"

    def __init__(self, n: int, length: int, E: float = 999.99):
        super(SAGroup, self).__init__(n, length)
        self.traj = []
        self.init_score = self.score

    def evolution(self):
        beta: float = 1. / self.t0

        seq_index = np.random.randint(self.length, size=(self.n, ))
        seq_curr = np.copy(self.inds)
        seq_new = np.ndarray(self.n, '<U{}'.format(self.length))
        # Probability Define
        # probability = math.exp(-(res_new-res)/self.t0)

        for i in range(self.n):
            tmp = list(seq_curr[i])
            if sigmoid(self.init_score[i] * beta) > np.random.uniform():
                tmp[seq_index[i]] = random.choice(self.lib)
            # tmp[seq_index[i]] = round(random.uniform(0, len(self.lib)/self.t0)*self.t0)
            seq_new[i] = ''.join(tmp)

        self.inds = np.copy(seq_new)
        self.evaluate()

        # update for next generation
        self.init_score = self.score

        self.generation += 1
        if self.method == "classic":
            self.t0 = self.t0/math.log(1 + self.generation)
        elif self.method == "accelerate":
            self.t0 = self.t0/(1 + self.generation)



