import torch

from .data_preprocess import Vocab


class BasicLM:
    def __init__(self, dconf, mconf):
        self.dconf = dconf
        self.mconf = mconf

        self.ko_vocab = Vocab(self.dconf.min_cnt)
        self.voc_size = 0
        self.dataset = None
        self._dataload = None

        self.model = None
        self.loss = None
        self.perpelexity = None
        self.optim = None
        self.lrscheder = None

    def train(self):
        raise

    def predict(self, corpus):
        raise

    def save(self, fname: str):
        """ save model """

        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'ko_vocab': self.ko_vocab,
        }, 'results/model/' + fname)

    def load(self, fname: str, retrain=False):
        """ load pytorch model """
        if not self.model:
            raise
        checkpoint = torch.load('results/model/' + fname)
        self.model.load_state_dict(checkpoint['model'])
        if self.optim and retrain:
            self.optim.load_state_dict(checkpoint['optim'])
        self.ko_vocab = checkpoint['ko_vocab']
        self.ko_vocab.to_idx2word()
        self.model.eval()
        print(len(self.ko_vocab))

    def test(self, kor: list):
        """ Translate Korean to English """
        pred = self.predict(kor)
        print(pred)
        rst = []
        for sent_idx in pred:
            sent = [self.ko_vocab.get_word(idx) for idx in sent_idx if not 0]
            rst.append(sent)
        return rst


@torch.no_grad()
def accuracy(pred, target):
    trg_idx = target > 0
    trg = target[trg_idx]
    pred = pred.argmax(1)[trg_idx]
    acc = sum(trg == pred).item() / len(target)
    return acc
