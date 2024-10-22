import torch
import torch.nn as nn
import torch.nn.functional as F

class BFLWithLogits(nn.Module):
    def __init__(self, gamma=2):
        """
        Binary Focal Loss.

        Inspired by: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
        """
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        # first = (1-torch.sigmoid(inputs))**self.gamma * targets * torch.log(torch.sigmoid(inputs))
        # second = (torch.sigmoid(inputs))**self.gamma * (1-targets) * torch.log(1-torch.sigmoid(inputs))
        # return torch.mean(-(first + second))
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        bce_exp = torch.exp(-bce)
        focal_loss = (1-bce_exp)**self.gamma * bce
        return focal_loss