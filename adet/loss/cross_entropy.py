import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, reduction='mean', smoothing=0.1, weight=None):
        assert reduction in ['none', 'mean', 'sum']
        super().__init__()
        self.reduction = reduction
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        c = inputs.size()[-1]
        log_preds = F.log_softmax(inputs, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss * self.smoothing / c + (1 - self.smoothing) * F.nll_loss(log_preds, targets,
                                                                             weight=self.weight,
                                                                             reduction=self.reduction)
