import sys
import torch
from torch import nn
from torch.nn import functional as F

eps = sys.float_info.epsilon


#### POSTER ####

class POSTER_LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(POSTER_LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
    
#### DAN ####
    
class DAN_AffinityLoss(nn.Module):
    def __init__(self, device, num_class=7, feat_dim=512):
        super(DAN_AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))
        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

class DAN_PartitionLoss(nn.Module):
    def __init__(self):
        super(DAN_PartitionLoss, self).__init__()

    def forward(self, x):
        num_head = x.size(1)
        if num_head > 1:
            var = x.var(dim=1).mean()
            loss = torch.log(1 + num_head / (var + eps))
        else:
            loss = 0
        return loss
    
    
#### DDAMFN ####

class DDAMFN_AttentionLoss(nn.Module):
    def __init__(self, ):
        super(DDAMFN_AttentionLoss, self).__init__()
    
    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt+1
                    loss = loss+mse
            loss = cnt/(loss + eps)
        else:
            loss = 0
        return loss     

