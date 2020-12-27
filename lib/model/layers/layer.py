"""
Encoder and Decoder Layer
"""
import torch
import torch.nn as nn

from transformer_lm.lib.model.layers.sublayer import MultiHeadAttention, PositionWiseFFLayer


class Encoder(nn.Module):
    def __init__(self, inp_dim, d_m, d_ff):
        super(Encoder, self).__init__()
        self.multi_attn = MultiHeadAttention(d_m, inp_dim)
        self.pw_ff = PositionWiseFFLayer(d_m, d_ff)

    def forward(self, inp, mask):
        """
        Args:
            inp (Tensor): [batch size, maxlen, d_m]
            mask (Tensor): [batch size, 1, maxlen]
        Returns: [batch size, maxlen, d_m]
        """
        # Sub-layer 1
        out = self.multi_attn(inp, inp, inp, mask)
        # Sub-layer 2
        out = self.pw_ff(out)
        return out
