import torch
import torch.nn as nn
import torch.nn.functional as F

class BFLWithLogits(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        first = (1-torch.sigmoid(inputs))**self.gamma * targets * torch.log(torch.sigmoid(inputs))
        second = (torch.sigmoid(inputs))**self.gamma * (1-targets) * torch.log(1-torch.sigmoid(inputs))
        return torch.mean(-(first + second))