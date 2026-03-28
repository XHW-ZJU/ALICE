import random
import numpy as np
import math
import pickle


def get_mask(arr_list):
    a = np.zeros((len(arr_list), MAX_LEN))
    for i, arr in enumerate(arr_list):
        a[i, :len(arr)] = 1
    return a


class MaskSequence(object):
    def __init__(self, mask_p: float = 0.15, msk_stg: list or tuple = (0.8, 0.0, 0.2)):
        """
        :param mask_p: Masked Probability
        :param sub_p: Substitution Probability
        """
        self.mask_p = mask_p
        self.vocab = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.msk, self.sub, self.keep = msk_stg
        self.alphabet = ['<PAD>'] + self.vocab
        # self.alphabet = self.vocab

    def generate_mask(self, sequences: str or list):
        """
        :param sequences: str or list type
        :return: (new_masked_sequences, label(sequences), position_mask)
        """
        if isinstance(sequences, str):
            token = [0]*MAX_LEN
            position_label = [0]*MAX_LEN
            for i, char in enumerate(list(sequences)):
                prob = random.random()
                # Add Mask Token
                if prob < self.mask_p:
                    prob /= self.mask_p
                    if prob < self.msk:
                        token[i] = self.alphabet.index('X')
                    # Substitute Token as xxx
                    # elif self.sub != 0.0 and prob < self.sub + self.msk:
                    elif prob < self.sub + self.msk:
                        token[i] = random.randrange(1, len(self.alphabet))
                    else:
                        try:
                            token[i] = self.alphabet.index(char)
                            continue
                        except ValueError as v:
                            print('Encounter Non-Natural Amino Acids!')
                            return None, None
                    try:
                        position_label[i] = self.alphabet.index(char)
                        continue
                    except ValueError as v:
                        print('Encounter Non-Natural Amino Acids!')
                        return None, None
                else:
                    try:
                        token[i] = self.alphabet.index(char)
                        continue
                    except ValueError as v:
                        print('Encounter Non-Natural Amino Acids!')
                        return None, None

            return token, position_label

        elif isinstance(sequences, (list, tuple)):
            token_seq_lst, output_lst = [], []
            for seq in sequences:
                token_seq, output_label = self.generate_mask(seq)
                if token_seq is None:
                    continue
                token_seq_lst.append(token_seq)
                output_lst.append(output_label)

            return token_seq_lst, output_lst

    def data_enhancement(self, fold: int or float, seqs: list):
        if isinstance(seqs, list):
            return seqs*fold
        else:
            raise TypeError

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        return self.lines[random.randrange(len(self.lines))][1]


def one_hot_repr(x):
    out = np.zeros((MAX_LEN, 22))
    for i, sig in enumerate(x):
       out[i, sig] = 1
    return out


if __name__ == '__main__':
    MAX_LEN = 50
    train_test_split_ratio = 0.8
    seq_lst = []
    with open('../../pre_train_data/uniprot_processed.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            seq_lst.append(line[:-1])
    f.close()
    assert isinstance(seq_lst, (str, list, tuple))

    # Train/Test Index
    index = np.arange(len(seq_lst))
    np.random.shuffle(index)
    train_index = index[:math.floor(len(seq_lst) * train_test_split_ratio)]
    test_index = np.array(list(set(index) - set(train_index)))

    trn_seq = np.array(seq_lst)[train_index].tolist()
    tst_seq = np.array(seq_lst)[test_index].tolist()
    # Mask Sequence
    M = MaskSequence(0.3)
    # Data Enhancement
    trn_seq_lst = M.data_enhancement(fold=1, seqs=trn_seq)
    # Trn data extract
    seq_tokens, mask_label = M.generate_mask(trn_seq_lst)
    # Tst data extract
    tst_seq_tokens, tst_mask_label = M.generate_mask(tst_seq)
    # Get Mask
    batch_mask_lst = get_mask(seq_tokens)
    print('mask token: ', seq_tokens[:10])
    print('position: ', mask_label[:10])
    print('masked seqs (shape: {}): {}'.format(batch_mask_lst.shape, batch_mask_lst[:10]))
    train_pack = [np.array(seq_tokens), np.array(mask_label)]
    test_pack = [np.array(tst_seq_tokens), np.array(tst_mask_label)]
    # data_pack = [mask_token, mask_label]
    print('Train Num: ', len(train_pack[0]))
    print('Test Num: ', len(test_pack[0]))
    np.save('../../pre_train_data/MLM_data_pack_trn_uniprot.npy', train_pack)
    np.save('../../pre_train_data/MLM_data_pack_tst_uniprot.npy', test_pack)
