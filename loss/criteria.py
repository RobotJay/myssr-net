'''
@author: gwb
@file: criteria.py
@time: 12/24/18 2:18 PM
@desc:
'''
import torch
import torch.nn as nn


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
    def forward(self, pred, target):
        self.loss = torch.mean((torch.abs(pred - target)))
        return self.loss