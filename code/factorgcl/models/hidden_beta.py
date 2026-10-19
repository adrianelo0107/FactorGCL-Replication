"""Hidden Beta：原型生成软超边，再提取剩余的公共因子；正文式 (6)。"""
import torch
from torch import nn
from .prior_beta import SymmetricHypergraphConv


class HiddenFactor(nn.Module):
    def __init__(self, dimension, factor_count, negative_slope):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(factor_count, dimension))
        nn.init.xavier_uniform_(self.prototypes)
        self.graph = SymmetricHypergraphConv(dimension, negative_slope)

    def generate_incidence(self, residual):
        return torch.sigmoid(residual @ self.prototypes.T)

    def propagate(self, residual, incidence):
        # 未来接口只传播传入关联矩阵，禁止在这里重新生成未来暴露。
        return self.graph(residual, incidence)

    def forward(self, residual):
        incidence = self.generate_incidence(residual)
        return self.propagate(residual, incidence), incidence
