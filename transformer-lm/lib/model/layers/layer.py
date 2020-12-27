"""
Encoder and Decoder Layer
"""
import torch.nn as nn

from .sublayer import MultiHeadAttention, PositionWiseFFLayer


class Encoder(nn.Module):
    def __init__(self, d_m, d_ff):
        super(Encoder, self).__init__()
        self.multi_attn = MultiHeadAttention(d_m)
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


class Decoder(nn.Module):
    def __init__(self, inp_dim, d_m, d_ff):
        super(Decoder, self).__init__()
        self.multi_attn = MultiHeadAttention(d_m, inp_dim)
        self.multi_attn = MultiHeadAttention(d_m, inp_dim)
        self.pw_ff = PositionWiseFFLayer(d_m, d_ff)

    def forward(self, inp, enc_out, src_mask, trg_mask):
        """
        Args:
            inp (Tensor): [batch size, maxlen, d_m]
            enc_out (Tensor): [batch size, maxlen, d_m]
            src_mask (Tensor): [batch size, 1, maxlen]
            trg_mask (Tensor): [batch size, maxlen, maxlen]
        Returns:
        """
        # Sub-layer 1
        # [batch size, maxlen, d_m]
        out = self.multi_attn(inp, inp, inp, trg_mask)  # masked self attention
        # Sub-layer 2
        # [batch size, maxlen, d_m]
        out = self.multi_attn(out, enc_out, enc_out, src_mask)  # encoder-decoder attention
        # Sub-layer 3
        # [batch size, maxlen, d_m]
        out = self.pw_ff(out)
        return out
