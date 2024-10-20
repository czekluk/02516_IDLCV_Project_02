import torch
import torch.nn as nn

class TestCNN(nn.Module):
    """CNN that has no encoder-decoder structure for testing purposes"""
    def __init__(self):
        super(TestCNN, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),

            # Layer 2
            nn.Conv2d(64, 2, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x