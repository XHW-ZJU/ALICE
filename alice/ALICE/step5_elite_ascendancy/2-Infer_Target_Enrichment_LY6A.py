"""
author: Alex, Elixir
date: 20230412
"""
import os
import parameters
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from metrics import *
import pandas as pd
import random
import pickle
SEED = 88


def train_and_eval(seqs_desc, seqs, output_pth, models,total_gen_seq):
    random.seed(SEED)
    np.random.seed(SEED)
    for index, model in enumerate(models):
        y_pred = model.predict(pd.DataFrame(seqs_desc).values)
        y_pred = y_pred.tolist()
    total_gen_seq['Target_LY6A_Enri'] = y_pred
    total_gen_seq.to_csv(output_pth + ranking_model_name + '_Prod_Fit_RoBERTa_SeqGAN_RankFiltPro_FE_1000.csv', index=False)


if __name__ == '__main__':

    target_model = pickle.load(file=open('../checkpoint/Property_Evaluation/inference/LY6A_model.sav', 'rb'))
    ranking_model_name = 'LY6A_model'

    models = [
        target_model,
    ]
    path = '../output/EA/'
    total_gen_seq = pd.read_csv(path+'Prod_Fit_RoBERTa_SeqGAN_RankFiltPro_FE_1000.csv')
    pickle.dump(total_gen_seq, file=open(path + 'tar_seq.pkl', 'wb'), protocol=0)
    seqs = pickle.load(file=open(path + 'tar_seq.pkl', 'rb'))
    seq_desc = [parameters.cal_pep(seq) for seq in seqs['AA_sequence'].tolist()]# seq_num x 254
    seqs = seqs['AA_sequence'].tolist()
    output_pth = '../output/EA/'
    train_and_eval(seq_desc, seqs, output_pth, models,total_gen_seq)
