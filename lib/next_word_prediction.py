import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from lib.util import load_data
from lib.data_batchify import Corpus, collate_fn
from lib.data_preprocess import Vocab, preprocessor
from lib.model.bert import BERT


# TODO Transformer에서 sub-git으로 가져오기

class BERTEmbedding(BasicLM):

    def train(self):
        ko_corpus = preprocessor(load_data(self.dconf.train_ko_path), lang='ko')
        self.ko_vocab.load(ko_corpus)

        train_set = self.dataset_form(ko_corpus, self.ko_vocab)
        self.dataset = Corpus(train_set)

        self._dataload = DataLoader(self.dataset,
                                    batch_size=self.mconf.batch_size,
                                    num_workers=0, collate_fn=collate_fn)
        print(len(self.ko_vocab))
        self.mconf.ko_size = len(self.ko_vocab) + 1

        self.model = BERT(self.mconf.d_m, self.mconf.ko_size, self.mconf.d_ff)
        self.loss = nn.CrossEntropyLoss()
        self.optim = optim.Adam(params=self.model.parameters(), lr=self.mconf.lr)
        self.lrscheder = optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5)

        total_loss = 0
        total_acc = 0
        self.model.train()
        self.info()
        for epoch in tqdm(range(self.mconf.epoch), desc='epoch'):
            for i, batch in tqdm(enumerate(self._dataload), desc="step", total=len(self._dataload)):
                self.optim.zero_grad()
                xs = batch[:, :-1]
                ts = batch[:, 1:]
                pred = self.model(xs, ts)
                pred, ts = pred.view(-1, pred.shape[2]), ts.reshape(1, -1).squeeze(0)
                b_loss = self.loss(pred, ts)
                b_loss.backward()
                self.optim.step()

                total_acc += word_accuracy(pred, ts)
                total_loss += b_loss.item()

            itersize = math.ceil(len(self.dataset) / self.mconf.batch_size)
            ppl = math.exp(total_loss / itersize)
            print(epoch, total_loss, total_acc / itersize, ppl)
            self.lrscheder.step(total_loss)
            total_loss = 0
        self.ko_vocab.to_idx2word()

    def load(self, fname: str, retrain=False):
        self.model = BERT(self.mconf.d_m, self.mconf.ko_size, self.mconf.d_ff)
        BasicLM.load(self, fname)

    def predict(self, corpus):
        ko_corpus = preprocessor(corpus, lang='ko')
        pred_set = self.dataset_form(ko_corpus, self.ko_vocab)
        pred_set = [torch.tensor(data) for data in pred_set]
        dataset = torch.nn.utils.rnn.pad_sequence(pred_set, batch_first=True)
        pred = self.model.predict_next_word(dataset)
        pred = self.model.predict_sent_sequence(dataset)
        return pred

    def dataset_form(self, ko_corpus, ko_vocab):
        rst = []
        # for ko, en in ko_corpus:
        #     ko = [ko_vocab[x] for x in ko]
        #     rst.append(ko)
        return rst

    def info(self):
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        print("Optimizer's state_dict:")
        for var_name in self.optim.state_dict():
            print(var_name, "\t", self.optim.state_dict()[var_name])
