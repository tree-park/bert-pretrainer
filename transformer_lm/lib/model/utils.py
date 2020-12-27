"""
Batch Norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, num_feature, eps=0.01, momentum=0.9):  # maxlen
        super(BatchNorm, self).__init__()
        shape = 1, 1, num_feature  # (batch, maxlen, hidd), norm target is hidd
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(nn.init.xavier_normal_(torch.empty(shape)))
        self.beta = nn.Parameter(nn.init.xavier_normal_(torch.empty(shape)))

        # The variables that are not model parameters are initialized to 0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def update_movings(self, mean, var):
        self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
        self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var

    def forward(self, batch):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if not torch.is_grad_enabled():
            self.moving_mean = self.moving_mean.to(batch.device)
            self.moving_var = self.moving_var.to(batch.device)
            normed = (batch - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
        else:
            mean = torch.mean(batch, dim=(0, 1), keepdim=True)
            var = torch.var(batch, dim=(0, 1), keepdim=True)
            normed = (batch - mean) / torch.sqrt(var + self.eps)
            self.update_movings(mean, var)
        new_batch = self.gamma * normed + self.beta
        return new_batch


class LabelSmoothingLoss(nn.NLLLoss):
    def __init__(self, a: float = 0.01, reduction='mean', ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.a = a
        self.reduction = reduction
        self.ignore_index = ignore_index

    @torch.no_grad()
    def forward(self, pred, trg):
        K = pred.size(-1)  # class number
        trg_idx = trg != self.ignore_index  # identify not PAD
        trg = trg[trg_idx]

        log_pred = F.log_softmax(pred[trg_idx], dim=-1)
        loss = -torch.sum(log_pred, dim=-1)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        nll_loss = F.nll_loss(log_pred, trg, reduction=self.reduction)
        loss = nll_loss * (1 - self.a) + self.a * (loss / K)
        return loss.mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, a: float = 0.01, reduction='mean', ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.a = a
        self.reduction = reduction
        self.ignore_index = ignore_index

    @torch.no_grad()
    def forward(self, pred, trg):
        pass
