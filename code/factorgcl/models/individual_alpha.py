"""Individual Alpha：两次残差后的个股表示；正文式 (7)。"""
from torch import nn


class IndividualAlpha(nn.Module):
    def __init__(self, dimension, negative_slope):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(dimension, dimension), nn.LeakyReLU(negative_slope)
        )

    def forward(self, residual):
        return self.mapping(residual)
