"""
Transformer
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.layer import Encoder, Decoder
from .layers.modules import Affine
from .layers.embedding import WordEmbedding


class Transformer(nn.Module):
    """ Assemble layers to build Transformer """

    def __init__(self, d_m, vocab_size, d_ff, n=3):
        super(Transformer, self).__init__()
        self.inp_emb = WordEmbedding(vocab_size, d_m)
        self.out_emb = WordEmbedding(vocab_size, d_m)
        self.enc_layers = nn.ModuleList(
            [copy.deepcopy(Encoder(d_m, d_ff)) for _ in range(n)])
        self.dec_layers = nn.ModuleList(
            [copy.deepcopy(Decoder(d_m, d_m, d_ff)) for _ in range(n)])
        self.affine = Affine(d_m, vocab_size)
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

    def forward(self, inp_batch, out_batch):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
            out_batch (Tensor): [batch size, maxlen]
        Returns: [batch size, maxlen, vocab_size]
        """
        # Encoder
        src_mask = mask_not_pad(inp_batch)
        # [batch size, maxlen, d_m]
        enc = self.encoder(inp_batch, src_mask)

        # Decoder
        trg_mask = mask_get_dec(out_batch)
        # [batch size, maxlen, d_m]
        o_emb = self.out_emb(out_batch)
        dec = o_emb
        for layer in self.dec_layers:
            # [batch size, maxlen, d_m]
            dec = layer(dec, enc, src_mask, trg_mask)
        # [batch size, maxlen, vocab_size]
        rst = F.log_softmax(self.affine(dec), dim=2)
        return rst

    @torch.no_grad()
    def predict(self, inp_batch):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
        Returns: [batch size, maxlen, vocab_size]
        """
        src_mask = mask_not_pad(inp_batch)
        # [batch size, maxlen, d_m]
        enc = self.encoder(inp_batch, src_mask)
        # [batch size, maxlen, d_m] @ [d_m, vocab_size]
        # => [batch size, maxlen, vocab_size]
        rst = F.log_softmax(self.affine(enc), dim=2)
        rst = torch.argmax(rst, dim=-1).tolist()
        return rst


def mask_not_pad(x):
    """
    Mark True at PAD
    Args:
        x (Tensor): [bsize, maxlen] with word idx
    Returns: [bsize, 1, maxlen] with bool if idx <=0, True
    """
    return (x > 0).unsqueeze(1)


def mask_get_dec(x):
    """
    Mark dec right sequence
    Args:
        x (Tensor): [bsize, maxlen] with bool
    Returns: [bsize, maxlen, maxlen] with bool
    """
    # [bsize, 1, maxlen]
    pad_masked = mask_not_pad(x)
    # [maxlen, maxlen]
    seq_masked = torch.tril(torch.ones(x.size(1), x.size(1)))
    # [bsize, maxlen, maxlen]
    seq_masked = seq_masked.unsqueeze(0).repeat(x.size(0), 1, 1)
    # [bsize, maxlen, maxlen]
    masked = seq_masked.masked_fill(pad_masked == 0, 0)
    return masked
