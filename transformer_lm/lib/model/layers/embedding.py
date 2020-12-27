"""
Word Embedding & Positional Embedding
"""
import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Basic Word Embedding
    Let the model learn sequence information with positional-encoding
    """

    def __init__(self, vocab_size, emb_dim):
        super(PositionalEmbedding, self).__init__()
        # self.affine = Affine(vocab_size, emb_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(p=0.1)

    def word_emb(self, inp):
        scale = torch.sqrt(torch.FloatTensor([inp.size(0)]))
        out = self.embedding(inp) / scale
        return out

    @staticmethod
    def posi_emb(bsize, maxlen, d_m):
        out = torch.stack(
            [positional_encodeing(maxlen, d_m)] * bsize
        )
        return out

    def forward(self, inp):
        """
        Args:
            x (Tensor): [bsize, maxlen, emb_dim]
        Returns: [bsize, maxlen, emb_dim]
        """
        """
        임베딩값 dim 값으로 나눠주는거 놓침 
        """
        # [bsize, maxlen, emb_dim]
        out = self.word_emb(inp)
        # [bsize, maxlen, emb_dim]
        pe_rst = self.posi_emb(out.size(0), out.size(1), out.size(2))
        # [bsize, maxlen, emb_dim]
        resdl = out + pe_rst
        return self.dropout(resdl)


def positional_encodeing(maxlen, dim):
    """ Give unique value by position and dimension """
    """  
    블로그 코드 참고함 
    https://catsirup.github.io/ai/2020/04/09/transformer-code.html#Positional-Embedding
    """

    def term(i):
        return 1 / (10000 ** (2 * (i // 2) / dim))

    pos = torch.as_tensor(np.arange(maxlen))
    dims = np.arange(dim)
    dims = torch.tensor(list(map(lambda x: term(x), dims)))
    # [maxlen, dim]
    pe_val = pos.unsqueeze(1) * dims
    # [maxlen, dim]
    pe = torch.zeros(maxlen, dim)
    pe[:, 0::2] = torch.sin(pe_val[:, 0::2])
    pe[:, 1::2] = torch.cos(pe_val[:, 0::2])
    return pe
