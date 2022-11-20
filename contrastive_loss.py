import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

class BiPatchNCE(nn.Module):
    """
    Bidirectional patchwise contrastive loss
    Implemented Based on https://github.com/alexandonian/contrastive-feature-loss/blob/main/models/networks/nce.py
    """
    def __init__(self, N, T, h, w, temperature = 0.07):
        """
        T: number of frames
        N: batch size
        h: feature height
        w: feature width
        temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        
        #mask meaning; 1 for postive pairs, 0 for negative pairs
        mask = torch.eye(h*w).long()
        mask = mask.unsqueeze(0).repeat(N*T, 1, 1).requires_grad_(False) #(N*T, h*w, h*w)
        self.register_buffer('mask', mask)
        self.temperature = temperature

    def forward(self, gt_f, pred_f):
        """
        gt_f: ground truth feature/images, with shape (N, T, C, h, w)
        pred_f: predicted feature/images, with shape (N, T, C, h, w)
        """
        mask = self.mask

        gt_f = rearrange(gt_f, "N T C h w -> (N T) (h w) C")
        pred_f = rearrange(pred_f, "N T C h w -> (N T) (h w) C")

        #direction 1, decompose the matmul to two steps, Stop gradient for the negative pairs
        score1_diag = torch.matmul(gt_f, pred_f.transpose(1, 2)) * mask
        score1_non_diag = torch.matmul(gt_f, pred_f.detach().transpose(1, 2)) * (1.0 - mask)
        score1 = score1_diag + score1_non_diag #(N*T, h*w, h*w)
        score1 = torch.div(score1, self.temperature)
        
        #direction 2
        score2_diag = torch.matmul(pred_f, gt_f.transpose(1, 2)) * mask
        score2_non_diag = torch.matmul(pred_f, gt_f.detach().transpose(1, 2)) * (1.0 - mask)
        score2 = score2_diag + score2_non_diag
        score2 = torch.div(score2, self.temperature)

        target = (mask == 1).int()
        target = target.to(score1.device)
        target.requires_grad = False
        target = target.flatten(0, 1) #(N*T*h*w, h*w)
        target = torch.argmax(target, dim = 1)

        loss1 = nn.CrossEntropyLoss()(score1.flatten(0, 1), target)
        loss2 = nn.CrossEntropyLoss()(score2.flatten(0, 1), target)
        loss = (loss1 + loss2)*0.5

        return loss
