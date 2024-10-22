import torch
import torch.nn as nn
import torch.nn.functional as F

class BDLWithLogits(nn.Module):
    def __init__(self, smooth=1):
        """
        Binary Dice Loss.

        Inspired by: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs*targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2*intersection+self.smooth)/(union+self.smooth)
        return 1-dice
