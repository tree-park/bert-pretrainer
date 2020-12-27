"""
Transformer
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lm.lib.model.layers.modules import Affine
from transformer_lm.lib.model.transformer import mask_not_pad
from lib.model.layers.layer import Encoder
from lib.model.layers.embedding import BERTEmbedding


class BERT(nn.Module):
    """ Assemble layers to build Transformer """

    def __init__(self, vocab_size, sep_idx, d_m=768, attn_heads=12, n=12):
        super(BERT, self).__init__()
        self.inp_emb = BERTEmbedding(vocab_size, d_m, sep_idx)
        self.enc_layers = nn.ModuleList(
            [copy.deepcopy(Encoder(d_m, d_m*4, attn_heads)) for _ in range(n)])

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

    def forward(self, inp_batch, lm_posi):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
            lm_posi (Tensor): [batch size, trg_size(15% of sent)]
        Returns: [batch size, maxlen, vocab_size]
        """
        # Encoder
        src_mask = mask_not_pad(inp_batch)
        # [batch size, maxlen, d_m]
        enc = self.encoder(inp_batch, src_mask)

        # Next Word Prediction
        # [batch size, maxlen, vocab_size]
        lm_enc = self.affine_1(enc)
        # [batch size, num_mask, vocab_size]
        rst_1 = [F.log_softmax(lm_enc[i][posi], dim=-1) for i, posi in enumerate(lm_posi)]

        # Sentence location Prediction
        # [batch size, d_m]
        cls = enc[:, 0]
        # [batch size, 2]
        rst_2 = F.log_softmax(self.affine_2(cls), dim=-1)
        return rst_1, rst_2

    def predict_next_word(self, inp_batch, lm_posi):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
        Returns: [batch size, maxlen, vocab_size]
        """
        with torch.no_grad():
            src_mask = mask_not_pad(inp_batch)
            enc = self.encoder(inp_batch, src_mask)
            lm_enc = self.affine_1(enc)
            rst = [F.log_softmax(lm_enc[i][posi], dim=-1) for i, posi in enumerate(lm_posi)]
        return rst

    def predict_is_next_sent(self, inp_batch):
        """
        Args:
            inp_batch (Tensor): [batch size, maxlen]
        Returns: [batch size, maxlen, vocab_size]
        """
        with torch.no_grad():
            src_mask = mask_not_pad(inp_batch)
            enc = self.encoder(inp_batch, src_mask)
            cls = enc[:, 0]
            rst = F.log_softmax(self.affine_2(cls), dim=-1)
        return rst
