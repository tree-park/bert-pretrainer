"""
Transformer
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.layers.layer import Encoder
from lib.model.layers.modules import Affine
from lib.model.layers.embedding import BERTEmbedding


class BERT(nn.Module):
    """ Assemble layers to build Transformer """

    def __init__(self, d_m, vocab_size, d_ff, n=3):
        super(BERT, self).__init__()
        self.inp_emb = BERTEmbedding(vocab_size, d_m)
        self.enc_layers = nn.ModuleList(
            [copy.deepcopy(Encoder(d_m, d_m, d_ff)) for _ in range(n)])

        self.affine_1 = Affine(d_m, vocab_size)
        self.affine_2 = Affine(d_m, 2)
        self.n = n

    def encoder(self, inp_batch, src_mask):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
            src_mask (Tensor): [bsize, 1, maxlen]
        Returns: [batch size, maxlen, d_m]
        """
        # [batch size, maxlen, d_m]
        i_emb = self.inp_emb(inp_batch)
        # Encoder
        enc = i_emb
        for layer in self.enc_layers:
            # [batch size, maxlen, d_m]
            enc = layer(enc, src_mask)
        return enc

    def forward(self, inp_batch):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
            out_batch (Tensor): [batch size, maxlen]
        Returns: [batch size, maxlen, vocab_size]
        """
        # Encoder
        src_mask = get_pad_mask(inp_batch)
        # [batch size, maxlen, d_m]
        enc = self.encoder(inp_batch, src_mask)

        # Next Word Prediction
        # [batch size, maxlen, vocab_size]
        rst_1 = F.log_softmax(self.affine_1(enc))

        # Sentence location Prediction
        # TODO Extract [CLS] token only to go affine_2
        # => [batch size, 1, d_m]

        # [batch size, maxlen, 2]
        rst_2 = F.log_softmax(self.affine_2(enc))
        return rst_1, rst_2

    def predict_next_word(self, inp_batch):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
        Returns: [batch size, maxlen, vocab_size]
        """
        with torch.no_grad:
            src_mask = get_pad_mask(inp_batch)
            # [batch size, maxlen, d_m]
            enc = self.encoder(inp_batch, src_mask)
            # [batch size, maxlen, d_m] @ [d_m, vocab_size]
            # => [batch size, maxlen, vocab_size]
            rst = F.log_softmax(self.affine_1(enc))
        return rst

    def predict_sent_sequence(self, inp_batch):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
        Returns: [batch size, maxlen, vocab_size]
        """
        with torch.no_grad:
            src_mask = get_pad_mask(inp_batch)
            # [batch size, maxlen, d_m]
            enc = self.encoder(inp_batch, src_mask)

            # TODO Extract [CLS] token only to go affine_2
            # => [batch size, 1, d_m]

            # [batch size, maxlen, d_m] @ [d_m, vocab_size]
            # => [batch size, maxlen, vocab_size]
            rst = F.log_softmax(self.affine_2(enc))
        return rst


def get_pad_mask(x):
    """
    Mark True at PAD
    Args:
        x (Tensor): [bsize, maxlen] with word idx
    Returns: [bsize, 1, maxlen] with bool
    """
    return (x <= 0).unsqueeze(1)


def get_dec_mask(x):
    """
    Mark dec right sequence
    Args:
        x (Tensor): [bsize, maxlen] with bool
    Returns: [bsize, maxlen, maxlen] with bool
    """
    # [bsize, 1, maxlen]
    pad_masked = get_pad_mask(x)
    # [maxlen, maxlen]
    seq_masked = torch.tril(torch.ones(x.size(2), x.size(2)))
    # [bsize, maxlen, maxlen]
    seq_masked = seq_masked.repeat(x.size(0))
    # [bsize, maxlen, maxlen]
    masked = seq_masked.masked_fill(pad_masked == 1, 0)
    return masked
