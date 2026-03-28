import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import json
from minibert import MaskedRoberta, BERT
from optim_schedule import ScheduledOptim
import tqdm
import numpy as np
import pickle
from torch.autograd import Variable


LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction (Deprecation in RoBERTa)

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, param: dict,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = MaskedRoberta(bert, param).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # self.criterion = nn.NLLLoss(ignore_index=0)
        self.criterion = nn.CrossEntropyLoss()
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP_%s:%d" % ("Test", epoch),
                              total=len(self.test_data),
                              bar_format="{l_bar}{r_bar}")
        total_correct = 0
        with torch.no_grad():
            for i, data in data_iter:
                mask_lm_output = self.model.forward(data[0])
                # Masked Words prediction accuracy
                msk_idx = torch.where(data[1] != 0, 1, 0)
                msk_token = torch.where(data[1] != 0, data[1], 0)
                word_acc = (torch.where(torch.argmax(mask_lm_output, dim=-1) == msk_token, 1,
                                        0) * msk_idx).sum() / msk_idx.sum()
                # avg_loss += mask_lm_output
                total_correct += word_acc.item()
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "AA Prediction acc (%)": word_acc.item() * 100
                }
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

            log = "EP%d_%s, avg_loss=" % (epoch, "Test") + "total_acc=" + \
                  str(total_correct * 100.0 / len(data_iter)) + '\n'
            print(log)
            f = open('./msk_roberta_record.txt', 'a+')
            f.write(log)
            f.close()

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0

        for i, data in data_iter:
            # 1. forward the masked_lm model
            mask_lm_output = self.model.forward(data[0])

            # 2. NLLLoss of predicting masked token word
            # label = Variable(FloatTensor(one_hot_repr(i) for i in data[1])).cuda()
            # mask_loss = self.criterion(mask_lm_output, data[1])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data[1])

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                mask_loss.backward()
                self.optim_schedule.step_and_update_lr()

            # Masked Words prediction accuracy
            avg_loss += mask_loss.item()
            msk_idx = torch.where(data[1] != 0, 1, 0)
            msk_token = torch.where(data[1] != 0, data[1], 0)
            word_acc = (torch.where(torch.argmax(mask_lm_output, dim=-1) == msk_token, 1, 0)*msk_idx).sum()/msk_idx.sum()
            # avg_loss += mask_lm_output
            total_correct += word_acc.item()
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": mask_loss.item(),
                "AA Prediction acc (%)": word_acc.item()*100
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        log = "EP%d_%s, avg_loss=" % (epoch, str_code) + ' ' + str(avg_loss / len(data_iter)) + ' ' + "total_acc=" + \
              str(total_correct * 100.0 / len(data_iter)) +'\n'
        print(log)
        f = open('./msk_roberta_record.txt', 'a+')
        f.write(log)
        f.close()

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "words_model_ep%d.pth" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


class AASeqLoader(Dataset):
    def __init__(self, pth):
        super(AASeqLoader, self).__init__()
        self.sequences, self.msk_label = np.load(pth)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequences = torch.tensor(self.sequences[index]).long().cuda()
        msk_label = torch.tensor(self.msk_label[index]).long().cuda()
        return sequences, msk_label


def one_hot_repr(x):
    out = torch.zeros((MAX_LEN, 22))
    for i, sig in enumerate(x):
       out[i, sig] = 1
    return out


if __name__ == '__main__':
    MAX_LEN = 50
    bz = 512
    param = json.load(fp=open('./param.json', 'r'))
    print(param)
    with open('./msk_roberta_record.txt', 'a+') as f:
        f.write(str(param)+'\n')
    f.close()
    trn_dataset = AASeqLoader('../../pre_train_data/MLM_data_pack_trn_uniprot.npy')
    tst_dataset = AASeqLoader('../../pre_train_data/MLM_data_pack_tst_uniprot.npy')
    trn_dataloader = DataLoader(trn_dataset, batch_size=bz)
    tst_dataloader = DataLoader(tst_dataset, batch_size=bz)
    mini_bert = BERT(param)
    for i in range(10):
        train_pro = BERTTrainer(mini_bert, param, trn_dataloader, tst_dataloader)
        for epoch in range(50):
            train_pro.train(epoch=epoch)
            train_pro.save(epoch, file_path='./model/')
        else:
            train_pro.save(epoch, file_path='./model/')
        train_pro.test(epoch=1)
