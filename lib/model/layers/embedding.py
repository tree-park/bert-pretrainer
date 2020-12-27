"""
Word Embedding & Positional Embedding
"""
import numpy as np
import torch
import torch.nn as nn

from transformer_lm.lib.model.layers.embedding import WordEmbedding, positional_embedding


class BERTEmbedding(nn.Module):

    def __init__(self, vocab_size, emb_dim, sep_idx):
        super(BERTEmbedding, self).__init__()
        self.word_emb = WordEmbedding(vocab_size, emb_dim)
        self.segment_emb = nn.Embedding(2, emb_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.sep_idx = sep_idx

    def forward(self, inp):
        """
        inp [bsize, maxlen]
        """
        # [bsize, maxlen, emb_dim]
        idx_emb = self.word_emb(inp)
        pe_emb = positional_embedding(idx_emb.size(0), idx_emb.size(1), idx_emb.size(2))
        seg_idx = make_seg_idx(inp, self.sep_idx)
        seg_emb = self.segment_emb(seg_idx)
        emb = idx_emb + pe_emb + seg_emb
        return self.dropout(emb)


def make_seg_idx(inp, sep_idx):
    """
    inp [bsize, maxlen]
    """
    out = np.zeros_like(inp)
    dup = []
    for i, posi in [i for i in zip(*np.where(inp == sep_idx))]:
        if i in dup:
            continue
        dup.append(i)
        out[i][:posi+1] = 1
    return torch.tensor(out)

