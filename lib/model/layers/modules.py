"""
Low level layers
    - Linear Layer
    - Attention
    - Multi-head Attention
    - Feed Forward Layer
    - Add & Norm Layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Affine(nn.Module):
    """ Fully Connected Layer """

    def __init__(self, i_dim, o_dim):
        super(Affine, self).__init__()
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(i_dim, o_dim)))
        self.b = nn.Parameter(nn.init.xavier_normal_(torch.empty(o_dim)))

    def forward(self, inp, linear=False):
        """
        Args:
            inp ([Tensor]): [bsize, maxlen, emb_size]
            linear (bool): bool
        Returns: [bsize, maxlen, hid_size]
        """
        # [bsize, maxlen, emb_size] * [emb_size, hid_size]
        if linear:
            (inp * self.W) + self.b
        return F.relu((inp * self.W) + self.b)


class NormLayer(nn.Module):
    def __init__(self, d_inp, eps=1e-05):
        super(NormLayer, self).__init__()
        self.eps = eps
        self.gamma = Affine(d_inp, d_inp)

    def forward(self, x):
        """
        Args:
            x (Tensor): [bsize, maxlen, dim]
        Returns: [bsize, maxlen, dim]
        """
        return self.gamma((x - torch.mean(x)) / torch.sqrt(torch.var(x) + self.eps))


class Attention(nn.Module):
    # 기존에 만들었던거 참조해서 만들기
    """ Scaled Dot-product Attention """

    def __init__(self, d_inp, d_q, d_k, d_v):
        super(Attention, self).__init__()
        self.Wq = Affine(d_inp, d_q)
        self.Wk = Affine(d_inp, d_k)
        self.Wv = Affine(d_inp, d_v)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (): [bsize, maxlen, d_m]
            key (): [bsize, maxlen, d_m]
            value (): [bsize, maxlen, d_m]
            mask (): [bsize, ?, maxlen]
        Returns:  [bsize, maxlen, d_k]
        """
        # [bsize, maxlen, d_k]
        wq = self.Wq(query)
        wk = self.Wk(key)
        wv = self.Wv(value)
        # attention distribution
        # Energy [bsize, maxlen, d_q] @ [bsize, d_k, maxlen] = [bsize, maxlen, maxlen]
        attn_dstr = torch.bmm(wq, torch.transpose(wk, 1, 2)) / torch.sqrt(key.size(-1))
        if mask:
            attn_dstr = attn_dstr.masked_fill(mask == 0, -1e10)
        attn_dstr = F.softmax(attn_dstr)
        # [bsize, maxlen, maxlen] @ [bsize, maxlen, d_v] = [bsize, maxlen, d_v]
        attn = torch.bmm(attn_dstr, wv)
        return attn
