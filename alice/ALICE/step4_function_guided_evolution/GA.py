# Genetic Algorithm

import random
import group
import math
import numpy as np
import basic

'''def testGA():
    group = GAGroup(20, 10)
    group.initEqualLen()
    group.showMsg()
    for gen in range(100):
        group.evolution()
        group.showMsg()'''


class GAGroup(group.Group):
    '''
    -- MEMBERS --
    rmutate : mutate rate
    rcopy : copy rate ( copy rate + cross rate = 1)
    doublecross : cross strategy

    -- FUNCTIONS --
    crossover(str, str) -> str : crossover of two sequence
    mutate(seq) -> str : mutate a specific sequence
    evolution() : evolution
    '''

    def __init__(self, n: int, length: int):
       super(GAGroup, self).__init__(n, length)
       self.rmutate = 0.3
       self.rcopy = 0.2
       self.doublecross = False

    def setRmutate(self, r: float):
        self.rmutate = r
    
    def setRcopy(self, r: float):
        self.rcopy = r
    
    def setDoublecross(self, dc: bool):
        self.doublecross = dc

    def crossover(self, seqa: str, seqb: str) -> str:
        ''' single or double crossover of two sequence '''
        if self.doublecross:
            length = min(len(seqa), len(seqb))
            i, j = random.randint(0, length-1), random.randint(0, length-1)
            i, j = min(i,j), max(i,j)
            return seqa[:i] + seqb[i:j] + seqa[j:]
        else:
            length = min(len(seqa), len(seqb))
            i = random.randint(0, length-1)
            return seqa[:i] + seqb[i:]
    
    def mutate(self, _seq: str) -> str:
        ''' mutate a specific sequence'''
        seq = list(_seq)
        for i in range(len(seq)):
            if random.random() < self.rmutate:
                seq[i] = random.choice(self.lib)
        return ''.join(seq)
        
    def evolution(self):
        ncopy = math.floor(self.n * self.rcopy)
        ncross = self.n - ncopy

        # copy
        newinds = self.inds[np.argsort(self.score)] # ascending
        
        # crossover & mutate
        for i in range(ncross):
            idxa = basic.weighted_choice(self.score)
            idxb = basic.weighted_choice(self.score)
            newind = self.crossover(self.inds[idxa], self.inds[idxb])
            newind = self.mutate(newind)
            newinds[i] = newind
        
        self.inds = newinds
        self.generation += 1
        self.evaluate()



