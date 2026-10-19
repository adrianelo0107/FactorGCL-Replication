"""Prior Beta 模块，正文式 (1)/(5)：完整左右对称归一化，避免构造 N×N 传播矩阵。"""
import torch
from torch import nn
import torch.nn.functional as F


def inverse_degree(degree, power):
    # 对空超边/孤立节点取零；正度数不无条件加 epsilon。
    safe = torch.where(degree > 0, degree, torch.ones_like(degree))
    return torch.where(degree > 0, safe.pow(power), torch.zeros_like(degree))


class SymmetricHypergraphConv(nn.Module):
    def __init__(self, dimension, negative_slope=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dimension, dimension))
        nn.init.xavier_uniform_(self.weight)
        self.negative_slope = negative_slope

    def forward(self, features, incidence):
        node_inverse = inverse_degree(incidence.sum(dim=1), -0.5)
        edge_inverse = inverse_degree(incidence.sum(dim=0), -1.0)
        messages = (features @ self.weight) * node_inverse[:, None]
        edge_messages = (incidence.T @ messages) * edge_inverse[:, None]
        shared = (incidence @ edge_messages) * node_inverse[:, None]
        return F.leaky_relu(shared, self.negative_slope)


class PriorFactor(nn.Module):
    def __init__(self, dimension, negative_slope):
        super().__init__()
        self.graph = SymmetricHypergraphConv(dimension, negative_slope)

    def forward(self, features, beta):
        return self.graph(features, beta)
