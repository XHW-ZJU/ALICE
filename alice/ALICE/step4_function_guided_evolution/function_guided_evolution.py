#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import os
import group
import GA
import EDA
import mixture
import SA
import argparse
import pandas as pd
import math
import parameters

parser_ = argparse.ArgumentParser(description="Train GAN")
# bert, roberta
parser_.add_argument("--group_type", type=str, default='EDG',
                     choices=['SA', 'GA', 'EDA', 'EDG', 'NONE'])


parser_.add_argument("--gan_date", type=str,
                     default='20230506', )
parser_.add_argument("--pregan_date", type=str,
                     default='20230508', )

parser_.add_argument("--model_name", type=str, default='SeqGAN',
                     choices=['SeqGAN','Random'])#

parser_.add_argument("--pretrain_model", type=str, default='roberta',
                     choices=['bert', 'roberta', 'NONE'])#


args = parser_.parse_args()
model_name = args.model_name
gan_date = args.gan_date
pregan_date = args.pregan_date

pretrain_model = args.pretrain_model

group_type = args.group_type


# different groups of 20 individuals of length 10
if group_type == 'GA':
    group_ty = GA.GAGroup
elif group_type == 'EDA':
    group_ty = EDA.EDAGroup
elif group_type == 'EDG':
    group_ty = mixture.EDGGroup
elif group_type == 'SA':
    group_ty = SA.SAGroup
elif group_type == 'NONE':
    group_ty = group.Group




filepath = '../input/FE/infer_gen_seqs.csv'


df = pd.read_csv(filepath)
seq_num = len(df)
seq_lengths = df['seq'].apply(len)
result_dict = {}
grouped = df.groupby(seq_lengths)
for i, gro in grouped:
    result_dict[i] = gro
seqlen = 7
# for num, len in enumerate(seqlen):
column = 'seq'  # column name in the file
cut = False
group_ty_load = group_ty(seq_num, seqlen)
group_ty_load.initFromFile(result_dict[seqlen], column, cut)
group_ty_load.showMsg('File Group')
# ### Step 3.3 Evolution and Sequence Output
nbest = math.floor(0.25 * result_dict[seqlen].shape[0])  # top 25% scores
score = []
path = '../output/FE/' + group_type + "/" + parameters.RA.date + '/' + pretrain_model + '2' + model_name + '/'
if not os.path.exists(path): os.makedirs(path)
fig_path = '../Figures/FE/' + group_type + "/" + parameters.RA.date + '/' + pretrain_model + '2' + model_name + '/'
if not os.path.exists(fig_path): os.makedirs(fig_path)
print(path)
f = open(path + 'RA_input_filepath.txt', 'a+')
f.write(str(filepath) + '\n')
f.close()
# TODO: change total epoch
total_epoch = 11
for epoch in range(total_epoch):
    group_ty_load.evolution()
    score.append(group_ty_load.sortedScore()[:nbest].mean())
    # TODO: SAVE EPOCH
    path_epoch = path + 'seq-epoch-{}.csv'.format(epoch)
    group_ty_load.outputMsg(path_epoch)

plt.plot([epoch for epoch in range(total_epoch)], score, marker='o', c='black')
plt.xlabel('Generation')
plt.ylabel('Top 25% Score Average')
plt.savefig(fig_path + 'score.png')
plt.savefig(fig_path + 'score.svg')
