'''
date: 20230418
author: Alex, Elixir
project: Training roberta to generate aav sequence

'''
import time
import torch.nn
import pandas as pd
import math
import torch.nn
import os
import parameters
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from SeqGAN import *
from minibert import BERT
from utils import *



def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=min(1.0 / math.sqrt(m.weight.data.shape[-1]), 0.1))
        nn.init.constant_(m.bias, 0)

def one_hot_train(y, num_columns):
    """Returns one-hot encoded Variable"""
    seq_len = GAN_param['train_len']
    y_cat = np.zeros((y.shape[0], seq_len, num_columns))
    for i, sig in enumerate(y):
        sig = [int(i) for i in sig]
        y_cat[i, range(y.shape[1]), sig] = 1.0

    return Variable(FloatTensor(y_cat)).cuda()


def train_and_eval(bert: BERT, vs_G_datapack, vs_G_datapack_TEST, start_end, GAN_param, Pretrain_param, columns, aa_list):
    # Loss functions
    adversarial_d_loss = torch.nn.MSELoss()
    adversarial_loss = torch.nn.CrossEntropyLoss().to('cuda')
    generator = Generator(bert,GAN_param, Pretrain_param)
    discriminator = Discriminator(GAN_param, Pretrain_param)
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
    discriminator.apply(weights_init)
    generator.apply(weights_init)
    # Compute the total parameters
    total_G_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    total_D_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print('total G params' + str(GAN_param.date) + str(GAN_param['model_name']), total_G_params)
    print('total D params' + str(GAN_param.date) + str(GAN_param['model_name']), total_D_params)
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=GAN_param['lr'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=GAN_param['lr'], betas=(0.5, 0.999))
    scheduler_GV = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=1.025)
    scheduler_DV = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=1.025)
    # ---------
    #  Training
    # ----------
    batch_size = GAN_param['batch_size']
    shuffle_index = np.arange(len(vs_G_datapack[0]))
    shuffle_index_TEST = np.arange(len(vs_G_TEST_datapack[0]))
    shuffle_index_real_dis = np.arange(len(vs_G_datapack[0]))
    shuffle_index_fake_dis = np.arange(int(len(vs_G_datapack[0]) / batch_size) * batch_size)
    d_real_losses, d_fake_losses, d_losses, grad_penalties = [], [], [],[]
    d_real_losses_TEST, d_fake_losses_TEST, d_losses_TEST, grad_penalties_TEST = [], [], [], []
    G_losses, D_losses, D_acc = [], [], []
    G_losses_TEST, D_losses_TEST, D_acc_TEST = [], [], []
    D_LOSS_LIST, G_LOSS_LIST = [], []
    D_LOSS_LIST_TEST, G_LOSS_LIST_TEST = [], []
    lr_GV, lr_DV = [], []
    for epoch in range(GAN_param['n_epochs']):
        np.random.shuffle(shuffle_index)
        np.random.shuffle(shuffle_index_TEST)
        np.random.shuffle(shuffle_index_real_dis)
        np.random.shuffle(shuffle_index_fake_dis)
        lr_GV.append(optimizer_G.state_dict()['param_groups'][0]['lr'])
        lr_DV.append(optimizer_D.state_dict()['param_groups'][0]['lr'])
        real_valid, true_valid, pred_valid = [], [], []
        pred_valid_TEST, true_valid_TEST, valid_TEST = [], [], []
        # Update the learning rate
        for param_g in optimizer_G.param_groups:
            scheduler_GV.step()
            # print('learning rate_'+str(train_name)+ str(date) +' :', param_g['lr'])
        for param_d in optimizer_D.param_groups:
            scheduler_DV.step()
            # print('learning rate_'+str(train_name)+ str(date) +' :', param_d['lr'])
        generator.train()
        discriminator.train()
        # every batch
        seqs_trn = [vs_G_datapack[0][shuffle_index[i * batch_size:(i + 1) * batch_size]]  for i in range(int(len(vs_G_datapack[0]) / batch_size))]
        tar_trn = [vs_G_datapack[1][shuffle_index[i * batch_size:(i + 1) * batch_size]]  for i in range(int(len(vs_G_datapack[1]) / batch_size))]
        seqs_tst = [vs_G_datapack_TEST[0][shuffle_index_TEST[i * batch_size:(i + 1) * batch_size]]  for i in range(int(len(vs_G_datapack_TEST[0]) / batch_size))]
        tar_tst = [vs_G_datapack_TEST[1][shuffle_index_TEST[i * batch_size:(i + 1) * batch_size]]  for i in range(int(len(vs_G_datapack_TEST[1]) / batch_size))]

        start_time = time.perf_counter()
        for num in range(int(len(vs_G_datapack[0]) / batch_size)):
            # Data Collection for Generator Datasets
            input_seq, target_seq = seqs_trn[num], tar_trn[num]
            input_seq = Variable(FloatTensor(input_seq)).cuda()
            target_seq = Variable(FloatTensor(target_seq)).cuda()
            batch_size = input_seq.shape[0]
            input_seq.cuda()
            target_seq.cuda()
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).cuda()
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).cuda()
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            # Generate a batch of images
            ones = torch.ones((GAN_param['batch_size'], 1)).type(torch.LongTensor).cuda()
            validity_real = discriminator(one_hot_train(input_seq, columns))
            input_seq = Variable(input_seq.long())
            input_seq = Variable(torch.cat([ones.cuda(), input_seq.data], dim=1)[:, :-1].contiguous())
            gen_seq = generator.forward(input_seq).cuda()
            target_seq = target_seq.contiguous().view(-1)
            g_loss_1 = adversarial_loss(gen_seq.float(), target_seq.long())
            # Loss measures generator's ability to fool the discriminator

            neg_seq = generator.gen_sample(GAN_param['train_len'])
            d_real_loss = adversarial_d_loss(validity_real, valid.reshape(-1, 1))

            validity_fake = discriminator(one_hot_train(neg_seq, columns).float().detach().cuda())
            d_fake_loss = adversarial_d_loss(validity_fake, fake.reshape(-1,1))
            # ,(bz,seq_len)
            inputs = generator.gen_sample(GAN_param['train_len'])
            ones = torch.ones((GAN_param['batch_size'], 1)).type(torch.LongTensor).cuda()
            inputseq = Variable(torch.cat([ones.cuda(), inputs.data], dim=1)[:, :-1].contiguous())
            targetseq = Variable(inputs.data).contiguous().view((-1,))
            probseq = generator.forward(inputseq)
            g_loss_2 = adversarial_loss(probseq.float(), targetseq.long())
            validreal = discriminator(one_hot_train(inputs, columns).float().detach().cuda())
            validfake = discriminator(one_hot_train(probseq.argmax(-1).reshape(GAN_param['batch_size'],-1),columns).float().detach().cuda())
            d_fake_loss_1 = adversarial_d_loss(validreal, fake.reshape(-1,1))
            d_fake_loss_2 = adversarial_d_loss(validfake, fake.reshape(-1,1))
            d_loss = (d_real_loss + d_fake_loss + d_fake_loss_1 + d_fake_loss_2)/4
            g_loss = (g_loss_1 + g_loss_2)/2
            g_loss = g_loss.requires_grad_()
            g_loss.backward()
            optimizer_G.step()
            G_LOSS_LIST += g_loss.cpu().detach().numpy().reshape(-1).tolist()
            G_LOSS_LIST = np.array(G_LOSS_LIST)
            d_loss = d_loss.requires_grad_()
            d_loss.backward()
            optimizer_D.step()
            D_LOSS_LIST += d_loss.cpu().detach().numpy().reshape(-1).tolist()
            D_LOSS_LIST = np.array(D_LOSS_LIST)
            pred_valid = validity_real.cpu().detach().reshape(-1).tolist() + validity_fake.cpu().detach().reshape(-1).tolist()
            for i in pred_valid:
                if i < 0.5:
                    true_valid.append(0)
                else:
                    true_valid.append(1)
        real_valid = (valid.cpu().detach().reshape(-1).tolist()+fake.cpu().detach().reshape(-1).tolist())*(int(len(vs_G_datapack[0]) / batch_size))
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"batch:{num:0.6f},Elapsed time: {elapsed_time:0.6f} seconds")
        d_acc = np.mean(np.array(true_valid) == np.array(real_valid))
        start_time = time.perf_counter()
        for num_test in range(int(len(vs_G_datapack_TEST[0]) / batch_size)):
            input_seq_TEST, target_seq_TEST = seqs_tst[num_test], tar_tst[num_test]
            # Adversarial ground truths
            input_seq_TEST = Variable(FloatTensor(input_seq_TEST)).cuda()
            target_seq_TEST = Variable(FloatTensor(target_seq_TEST)).cuda()
            real_valid_TEST = discriminator(one_hot_train(input_seq_TEST,columns))
            input_seq_TEST = Variable(torch.cat([ones.cuda(), input_seq_TEST.data], dim=1)[:, :-1].contiguous())
            gen_seq_TEST = generator.forward(input_seq_TEST.long())
            target_seq_TEST = target_seq_TEST.contiguous().view(-1)
            g_loss_1_TEST = adversarial_loss(gen_seq_TEST.float(), target_seq_TEST.long())
            fake_valid_TEST = discriminator(one_hot_train(gen_seq_TEST.multinomial(1).reshape(batch_size,-1),columns).float())
            # Sample noise and labels as generator input
            d_real_loss_TEST = adversarial_d_loss(real_valid_TEST, valid.reshape(-1, 1))
            d_fake_loss_TEST = adversarial_d_loss(fake_valid_TEST, fake.reshape(-1, 1))
            inputs_TEST = generator.gen_sample(GAN_param['train_len'])
            ones_TEST = torch.ones((batch_size, 1)).type(torch.LongTensor)
            inputseq_TEST = Variable(torch.cat([ones_TEST.cuda(), inputs_TEST.data], dim=1)[:, :-1].contiguous())
            targetseq_TEST = Variable(inputs_TEST.data).contiguous().view((-1,))
            probseq_TEST = generator.forward(inputseq_TEST)
            g_loss_2_TEST = adversarial_loss(probseq_TEST.float(), targetseq_TEST.long())
            validfake1_TEST = discriminator(one_hot_train(inputs_TEST, columns).float().detach())
            validfake2_TEST = discriminator(one_hot_train(probseq_TEST.argmax(-1).reshape(batch_size, -1), columns).float().detach())
            d_fake_loss_TEST1 = adversarial_d_loss(validfake1_TEST, fake.reshape(-1, 1))
            d_fake_loss_TEST2 = adversarial_d_loss(validfake2_TEST, fake.reshape(-1, 1))
            g_loss_TEST = (g_loss_1_TEST + g_loss_2_TEST )/2
            d_loss_TEST = (d_real_loss_TEST + d_fake_loss_TEST + d_fake_loss_TEST1 + d_fake_loss_TEST2)/4
            G_LOSS_LIST_TEST += g_loss_TEST.cpu().detach().numpy().reshape(-1).tolist()
            G_LOSS_LIST_TEST = np.array(G_LOSS_LIST_TEST)
            D_LOSS_LIST_TEST += d_loss_TEST.cpu().detach().numpy().reshape(-1).tolist()
            D_LOSS_LIST_TEST = np.array(D_LOSS_LIST_TEST)
            pred_valid_TEST = real_valid_TEST.cpu().detach().reshape(-1).tolist()+fake_valid_TEST.cpu().detach().reshape(-1).tolist()
            for i in pred_valid_TEST:
                if i < 0.5:
                    true_valid_TEST.append(0)
                else:
                    true_valid_TEST.append(1)
        real_valid_TOTAL_TEST = (valid.cpu().detach().reshape(-1).tolist()+fake.cpu().detach().reshape(-1).tolist())*int(len(vs_G_datapack_TEST[0]) / batch_size)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"test model time: {elapsed_time:0.6f} seconds")
        d_acc_TEST = np.mean(np.array(true_valid_TEST) == np.array(real_valid_TOTAL_TEST))
        D_LOSS_LIST = D_LOSS_LIST/GAN_param['batch_size']
        G_LOSS_LIST = G_LOSS_LIST/GAN_param['batch_size']
        D_LOSS_LIST_TEST = D_LOSS_LIST_TEST / GAN_param['batch_size']
        G_LOSS_LIST_TEST = G_LOSS_LIST_TEST / GAN_param['batch_size']
        G_losses.append((g_loss.data).cpu().numpy())
        D_losses.append((d_loss.data).cpu().numpy())
        d_real_losses.append((d_real_loss.data).cpu().numpy())
        d_fake_losses.append((d_fake_loss.data).cpu().numpy())
        G_losses_TEST.append((g_loss_TEST.data).cpu().numpy())
        D_losses_TEST.append((d_loss_TEST.data).cpu().numpy())
        d_real_losses_TEST.append((d_real_loss_TEST.data).cpu().numpy())
        d_fake_losses_TEST.append((d_fake_loss_TEST.data).cpu().numpy())
        D_acc.append(np.array(d_acc))
        D_acc_TEST.append(np.array(d_acc_TEST))
        print("[data: %s][train_name:%s] [Epoch %d/%d] [D loss_BATCH: %f] [G loss_BATCH: %f] [D_acc: %f]   [lr_DV: %f] [lr_GV: %f]"
              % (GAN_param['date'], GAN_param['model_name'], epoch, GAN_param['n_epochs'], D_LOSS_LIST,  G_LOSS_LIST, d_acc, optimizer_G.state_dict()['param_groups'][0]['lr'],optimizer_D.state_dict()['param_groups'][0]['lr']
                 ))
        print("[data_TEST: %s][train_name_TEST:%s] [Epoch_TEST %d/%d] [D loss_BATCH_TEST: %f] [G loss_BATCH_TEST: %f] [D_acc_TEST: %f] "
              % (GAN_param['date'], GAN_param['model_name'], epoch, GAN_param['n_epochs'], D_LOSS_LIST_TEST, G_LOSS_LIST_TEST, d_acc_TEST))

        with open(Model_Eval_Fig + "LOSS_ACC.txt", 'a+') as f:
            f.writelines(
                "date: {} train_name: {} Epoch :{}  D loss_BATCH:  {}   G loss_BATCH:  {}  d_acc: {}  \n ".format(
                    GAN_param['date'], GAN_param['model_name'], epoch, D_LOSS_LIST, G_LOSS_LIST, d_acc))
        f.close()
        with open(Model_Eval_Fig + "LOSS_ACC_TEST.txt", 'a+') as f:
            f.writelines(
                "date_TEST: {} train_name: {} Epoch_TEST :{}  D loss_BATCH_TEST:  {}   G loss_BATCH_TEST:  {}  d_acc_TEST: {}  \n ".format(
                    GAN_param['date'], GAN_param['model_name'], epoch, D_LOSS_LIST_TEST, G_LOSS_LIST_TEST, d_acc_TEST))
        f.close()
        plot_losses([d_losses, d_real_losses, d_fake_losses, d_losses_TEST, d_real_losses_TEST, d_fake_losses_TEST], ["d_loss", "d_real_loss", 'd_fake_loss', 'd_losses_TEST', 'd_real_losses_TEST','d_fake_losses_TEST'],
                    Model_Eval_Fig + 'd_loss_components'+'_'+str(GAN_param['model_name'])+ str(GAN_param['date']) +'.png')

        plot_losses([D_acc, D_acc_TEST], ["D_acc",'D_acc_TEST'],
                    Model_Eval_Fig + 'D_acc' + '_' + str(GAN_param['model_name']) + str(GAN_param['date']) + '.png')

        plot_losses([D_losses, D_losses_TEST], ["D_losses", 'D_losses_TEST'], Model_Eval_Fig + 'D_losses'+'_'+ str(GAN_param['model_name'])+ str(GAN_param['date']) +'.png')
        plot_losses([G_losses, G_losses_TEST], ["G_losses", 'G_losses_TEST'], Model_Eval_Fig + 'G_losses'+'_'+ str(GAN_param['model_name'])+ str(GAN_param['date']) +'.png')

        plot_losses([lr_GV, lr_DV], ["lr_GV", "lr_DV"], Model_Eval_Fig + 'lr'+'_'+ str(GAN_param['model_name'])+ str(GAN_param['date']) +'.png')
        generator.eval()
        discriminator.eval()
        torch.save(discriminator, checkpoint_dir_D + 'D_repeat_' + str(epoch) + '.pth')
        torch.save(generator, checkpoint_dir_G + 'G_repeat_' + str(epoch) + '.pth')
    print('Finished Training')

if __name__ == '__main__':
    GAN_param = parameters.GAN
    print(GAN_param)
    pretrain_model = 'roberta'
    pretrain_pth = '../checkpoint/pretrain/inference/words_model_ep3.pth'
    Pretrain_param = parameters.roberta
    Model_Eval_Fig = '../Figures/PreGan/' + str(GAN_param.date) + "/" + pretrain_model + "2" + GAN_param.model_name + "/"
    if not os.path.exists(Model_Eval_Fig): os.makedirs(Model_Eval_Fig)
    checkpoint_dir_D = '../checkpoint/pretrain/train/' + str(GAN_param.date) + "/" + pretrain_model + "2" + GAN_param.model_name + "/" + 'D' + "/"
    checkpoint_dir_G = '../checkpoint/pretrain/train/' + str(GAN_param.date) + "/" + pretrain_model + "2" + GAN_param.model_name + "/" + 'G' + "/"
    if not os.path.exists(checkpoint_dir_D): os.makedirs(checkpoint_dir_D)
    if not os.path.exists(checkpoint_dir_G): os.makedirs(checkpoint_dir_G)
    aa_list = ['<PAD>', 'X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                     'V', 'W', 'Y']
    columns = 22
    vs_D_datapack_pth = r'../input/pretrain/' +'train8_D.npy'
    VS_D_XTS, VS_D_YTS = np.load(vs_D_datapack_pth,  allow_pickle=True)
    vs_G_datapack = (VS_D_XTS, VS_D_XTS)
    tst_pth = r'../input/pretrain/' +'test2_D.npy'
    VS_D_XTS_TEST, VS_D_YTS_TEST = np.load(tst_pth, allow_pickle=True)
    vs_G_TEST_datapack = (VS_D_XTS_TEST, VS_D_XTS_TEST)
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    # Load Pretrained Model
    print(len(VS_D_XTS))
    pretrain_pth = torch.load(pretrain_pth)
    for para in pretrain_pth.parameters():
        para.requires_grad = False
    start_end = (7,7)
    train_and_eval(pretrain_pth, vs_G_datapack, vs_G_TEST_datapack, start_end, GAN_param, Pretrain_param, columns, aa_list)
