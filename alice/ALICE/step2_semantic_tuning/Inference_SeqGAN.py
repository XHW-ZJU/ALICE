'''
date: 20230506
author: Alex, Elixir
project: using roberta to generate aav sequence
'''

import pandas as pd
import tools.parameters
import pandas as pd
import os
import torch
import numpy as np

def generate_samples(G_vs_model, output_file, start_end):
    samples = []
    for i in range(20):
        sample_tensor = G_vs_model.gen_seqs(start_end)
        samples += sample_tensor
    seqlst = []
    chars_to_find = [0, 1]
    for i, sample in enumerate(samples):
        if any(char in sample for char in chars_to_find):
            continue
        else:
            chars = [aa_list[s] for s in sample]
            seq = ''.join(chars)
        seqlst.append(seq)

    df = pd.concat([pd.DataFrame(seqlst,columns=['seq'])], axis=1)
    df.to_csv(output_file + 'infer_gen_seqs.csv', encoding="utf8", index=False)

if __name__ == '__main__':
    GAN_Infer_param = tools.parameters.GAN_Infer
    aa_list = ['<PAD>', 'X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
               'V', 'W', 'Y']
    generator_model = '../checkpoint/pretrain/inference/' + 'G_repeat_0'+ '.pth'
    generator_model = torch.load(open(generator_model, 'rb'))
    print(generator_model)
    output_file = '../output/PreGan/' + str(GAN_Infer_param.date) + "/" + str(GAN_Infer_param.pretrain_model) + "2" + str(GAN_Infer_param.model_name) + "/"
    if not os.path.exists(output_file): os.makedirs(output_file)
    start_end = (7,7)
    print(output_file)
    generate_samples(generator_model, output_file, start_end)
