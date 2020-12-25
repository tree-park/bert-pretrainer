"""
Two or Three Sub-layers for Encoder and Decoder
"""
import copy
import torch
import torch.nn as nn

from .modules import Affine, Attention, NormLayer


class MultiHeadAttention(nn.Module):
    """ Multi-head Attention + Add&Norm """

    def __init__(self, d_m, head=4):
        super(MultiHeadAttention, self).__init__()
        self.d_m = d_m
        assert d_m % head == 0
        self.d_k = int(d_m / head)
        self.head = head
        self.Wo = Affine(self.d_m, d_m)
        self.attn_layers = nn.ModuleList(
            [copy.deepcopy(Attention(self.d_m, self.d_k, self.d_k, self.d_k)) for _ in range(head)])
        self.dropout = nn.Dropout(p=0.01)
        self.addnorm = NormLayer(d_m)

    def forward(self, query, key, value, mask):
        """
        Args:
            query (Tensor): [batch size, maxlen, d_m]
            key (Tensor): [batch size, maxlen, d_m]
            value (Tensor): [batch size, maxlen, d_m]
            mask (Tensor): [batch size, ?, maxlen]
        Returns: [batch size, maxlen, d_m]
        """
        heads = []
        for layer in self.attn_layers:  # TODO 이게 맞는지 확인
            # head : [batch size, maxlen, d_k]
            head = layer(query, key, value, mask)
            heads.append(head)
        # [batch size, maxlen, d_k*head]
        multi_attn = self.Wo(torch.cat(heads, dim=2))
        multi_attn = self.dropout(multi_attn)

        # [batch size, maxlen, d_k*head]
        resdl = query + multi_attn
        # [batch size, maxlen, d_k*head]
        out = self.addnorm(resdl)
        return out


class PositionWiseFFLayer(nn.Module):
    """ Position-wise FeedForward + Add&Norm """

    def __init__(self, d_m, d_ff):
        super(PositionWiseFFLayer, self).__init__()
        self.W1 = Affine(d_m, d_ff)
        self.W2 = Affine(d_ff, d_m)
        self.dropout = nn.Dropout(p=0.01)
        self.addnorm = NormLayer(d_m)

    def forward(self, inp):
        """
        Args:
            inp (Tensor): [batch size, maxlen, d_m]
        Returns: [batch size, maxlen, d_m]
        """
        # [batch size, maxlen, d_ff]
        out = torch.relu(self.W1(inp))
        # [batch size, maxlen, d_m]
        out = self.W2(out)
        resdl = inp + self.dropout(out)
        # [batch size, maxlen, d_m]
        out = self.addnorm(resdl)
        return out
