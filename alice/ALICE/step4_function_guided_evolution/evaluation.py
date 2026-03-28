"""
Evaluation methods and the evaluation function for final sequence generation.
"""

import json
import math
import numpy as np
from abc import ABC, abstractmethod
import basic


class EvaMethod(ABC):
    """
    Abstract Evaluation Method
    """

    @abstractmethod
    def evaluate(self, seq: str) -> float:
        """Evaluate the given sequence"""
        pass


class MSAEva(EvaMethod):
    '''
        -- MEMBERS --
        gap : gap penalty
        extend : extend gap penalty
        local : true for local optimum, false for global optimum

        sampath : sample path
        samples : positive and negative samples

        matpath : matrix path
        matrix : matrix for any pair of characters, else using [mismatch, match] instead
        matseq : matrix index sequence, in upper case, else "NOPE"

        scorebound : average score bound for each aa

        -- FUNCTIONS --
        set functions:
        setGap(gap)
        setExtend(extend)
        setLocal(local)
        loadSample(path)
        loadMatrix(path)

        msa(seqa, seqb) : compare two sequences
        evaluate(seq) : evaluate
    '''

    def __init__(self):
        EvaMethod.__init__(self)
        self.gap = -4
        self.extend = -2
        self.local = True

        self.sampath = ''
        self.samples = ()

        self.matpath = ''
        self.matseq = ''
        self.matrix = []

        self.scorebound = (-1, 1)

        # default parameter
        self.loadSample('seq_LY6A_LY6C1.json')
        self.loadMatrix('BLOSSUM62.txt')

    def setGap(self, gap: float):
        self.gap = gap

    def setExtend(self, extend: float):
        self.extend = extend

    def setLocal(self, local: bool):
        self.local = local

    def loadSample(self, path: str):
        '''
        load sample and return (positive, negative) sample
        '''
        self.sampath = path
        samples = json.load(open(path, 'r'))
        self.samples = samples['Positive'], samples['Negative']
        return

    def loadMatrix(self, path: str):
        '''
        load matrix and matrix sequence and return (sequence, matrix)
        '''
        with open(path, 'r') as f:
            self.matpath = path
            seq = f.readline()[:-1].upper()
            self.matseq = seq
            if seq == 'NOPE':
                mat = list(map(int, f.readline().split()))
                self.scorebound = (mat[0], mat[1])
            else:
                mat = np.zeros((len(seq), len(seq)))
                for i in range(len(seq)):
                    data = f.readline().split()
                    mat[i] = data
                self.scorebound = (self.extend, mat.diagonal().mean())
            self.matrix = mat
        return

    def msa(self, seqa: str, seqb: str) -> float:
        '''
        Calculate the multi-sequence alignment score \n
        return : calculated score for MSA
        '''
        x, y = len(seqa) + 1, len(seqb) + 1
        seqa, seqb = seqa.upper(), seqb.upper()
        mat = np.full((x, y), -math.inf)
        dir = np.full((x, y), -math.inf)  # 0 for local, 1 for x dir, -1 for y dir, 2 for match dir
        score = 0
        mat[0][0] = 0
        dir[0][0] = 0
        for i in range(x):
            for j in range(y):
                # zero if local
                if self.local:
                    mat[i][j] = 0
                    dir[i][j] = 0

                # x dir
                if i > 0:
                    xscore = mat[i - 1][j] + [self.gap, self.extend][dir[i - 1][j] == 1]
                    if xscore > mat[i][j]:
                        mat[i][j] = xscore
                        dir[i][j] = 1

                # y dir
                if j > 0:
                    yscore = mat[i][j - 1] + [self.gap, self.extend][dir[i][j - 1] == -1]
                    if yscore > mat[i][j]:
                        mat[i][j] = yscore
                        dir[i][j] = -1

                # match dir
                if i > 0 and j > 0:
                    if self.matseq == "NOPE":
                        mscore = mat[i - 1][j - 1] + self.matrix[seqa[i - 1] == seqb[j - 1]]
                    else:
                        mscore = mat[i - 1][j - 1] + self.matrix[self.matseq.index(seqa[i - 1])][
                            self.matseq.index(seqb[j - 1])]

                    if mscore > mat[i][j]:
                        mat[i][j] = mscore
                        dir[i][j] = 2

                score = max(mat[i][j], score)

        return score

    def evaluate(self, seq: str, dim=1) -> float:
        '''evaluation function'''
        posSeqs, negSeqs = self.samples
        posScores = [self.msa(seq, seqp) for seqp in posSeqs]
        negScores = [self.msa(seq, seqn) for seqn in negSeqs]
        return (basic.calculate_mean(dim, posScores) - basic.calculate_mean(dim, negScores)) / (
                    len(seq) * (self.scorebound[1] - self.scorebound[0]) * 2) + 0.5


def test_evaluation():
    """Test function for evaluation"""
    method = MSAEva()
    print(method.evaluate('LDEFYQRDQV'))


if __name__ == "__main__":
    test_evaluation()
