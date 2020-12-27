import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import math
import os
import sys
import random
import numpy as np

from transformer_lm.lib.util import load_data
from transformer_lm.lib.model.utils import LabelSmoothingLoss
from transformer_lm.lib.language_model import BasicLM
from transformer_lm.lib.data_preprocess import preprocessor, Vocab, TokenMarks
from lib.data_batchify import Corpus, collate_fn
from lib.model.bert import BERT

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


@torch.no_grad()
def accuracy(preds, target):
    preds = [pred.argmax(-1) for pred in preds]
    acc = sum([sum(t == p) for t, p in zip(target, preds)])
    return acc


class BERTEmbedding(BasicLM):
    def __init__(self, *args):
        super(BERTEmbedding, self).__init__(*args)
        self.ko_vocab = Vocab(self.dconf.min_cnt, need_sep=True)

    def train(self):
        ko_corpus = preprocessor(load_data(self.dconf.train_ko_path), lang='ko')
        self.ko_vocab.load(ko_corpus)

        train_set = self.dataset_form(ko_corpus)
        self.dataset = Corpus(train_set)
        self._dataload = DataLoader(self.dataset,
                                    batch_size=self.mconf.batch_size,
                                    num_workers=0, collate_fn=collate_fn)
        self.mconf.ko_size = len(self.ko_vocab) + 1

        self.model = BERT(self.mconf.d_m, self.mconf.ko_size, self.ko_vocab[TokenMarks.SEP])
        self.loss = nn.CrossEntropyLoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=self.mconf.lr)
        self.lrscheder = optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5)

        total_loss = 0
        total_acc = 0
        self.model.train()
        # self.info()
        for epoch in tqdm(range(self.mconf.epoch), desc='epoch'):
            for i, batch in tqdm(enumerate(self._dataload), desc="step", total=len(self._dataload)):
                self.optim.zero_grad()
                # src, trg for Masked LM
                inp, nsp_trgs, lm_trg, lm_posi = batch
                print('inp.shape: ', inp.shape)
                pred_lm, pred_nsp = self.model(inp, lm_posi)

                b_loss_nsp = self.loss(pred_nsp, torch.tensor(nsp_trgs))
                b_loss_mlm = sum([self.loss(pred, trg) for pred, trg in zip(pred_lm, lm_trg)])
                b_loss = b_loss_mlm + b_loss_nsp
                b_loss.backward()

                self.optim.step()

                total_acc += accuracy(pred_lm, lm_trg)
                total_loss += b_loss.item()

            itersize = math.ceil(len(self.dataset) / self.mconf.batch_size)
            ppl = math.exp(total_loss / len(self.dataset))
            print(epoch, total_loss, total_acc / itersize, ppl)
            self.lrscheder.step(total_loss)
            total_loss = 0
        self.ko_vocab.to_idx2word()

    def load(self, fname: str, retrain=False):
        self.model = BERT(self.mconf.d_m, self.mconf.ko_size, self.ko_vocab[TokenMarks.SEP])
        super().load(fname)

    def predict(self, corpus):
        ko_corpus = preprocessor(corpus, lang='ko')
        pred_set = self.dataset_form(ko_corpus)
        inp, nsp_trgs, lm_trg, lm_posi = zip(*pred_set)

        dataset = torch.nn.utils.rnn.pad_sequence(torch.tensor(inp), batch_first=True)
        pred_mlm = self.model.predict_next_word(dataset, lm_posi)
        pred_nsp = self.model.predict_is_next_sent(dataset)
        pred_words = [pred.argmax(-1) for pred in pred_mlm]
        for p, t in zip(pred_words, lm_trg):
            print([self.ko_vocab.idx2word[idx] for idx in t])
            print([self.ko_vocab.idx2word[idx] for idx in p.tolist()])
        print()
        print(pred_nsp.argmax(-1), nsp_trgs)

    def masking_words(self, sent, sep_loc):
        """
        Masking predicted word
        15 % of words in sentence will be TARGET of LM
            80 % of target will be [MASK]
            10 % of target will be random word
            10 % of target will be origin word
        """
        inp = np.array(sent)
        num_mask = round(len(sent) * 0.15)

        mask_candi = [i for i in range(1, len(sent) - 1)]
        mask_candi.remove(sep_loc)
        # target idx of location
        mask_posi = random.sample(mask_candi, num_mask)
        trg = [i for i in inp[mask_posi]]
        to_mask = mask_posi[:int(num_mask * 0.8)]  # index of masking target
        to_rand = mask_posi[int(num_mask * 0.8):int(num_mask * 0.9)]  # index of random target
        inp[to_mask] = self.ko_vocab[TokenMarks.MASK]
        inp[to_rand] = \
            random.sample([i for i in range(len(TokenMarks), len(self.ko_vocab))], len(to_rand))
        return inp, trg, mask_posi

    def dataset_form(self, ko_corpus):
        rst = []
        for i in range(len(ko_corpus)):
            i_trg = ko_corpus[i]
            if random.random() > 0.5 and i != len(ko_corpus) - 1:
                next_trg, nsp_trg = ko_corpus[i + 1], 1
            else:
                next_trg, nsp_trg = ko_corpus[int(random.random() * len(ko_corpus))], 0

            s1_inp = [self.ko_vocab[x] for x in i_trg]
            s2_inp = [self.ko_vocab[x] for x in next_trg]
            sep_loc = len(s1_inp) + 1
            src = [self.ko_vocab[TokenMarks.CLS]] \
                  + s1_inp + [self.ko_vocab[TokenMarks.SEP]] \
                  + s2_inp + [self.ko_vocab[TokenMarks.SEP]]
            inp, lm_trg, posi = self.masking_words(src, sep_loc)

            rst.append((inp, nsp_trg, lm_trg, posi))
        return rst

    def info(self):
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        print("Optimizer's state_dict:")
        for var_name in self.optim.state_dict():
            print(var_name, "\t", self.optim.state_dict()[var_name])
