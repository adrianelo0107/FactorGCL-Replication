"""Feature Extractor：论文 p.4 的 GRU 时序特征提取；BN 位置采用复现指南明确记录的默认解释。"""
import torch
from torch import nn


class StockEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, bn_eps=1e-5, bn_momentum=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=layers, batch_first=True,
            bias=True, dropout=0.0, bidirectional=False,
        )
        self.normalization = nn.BatchNorm1d(
            hidden_dim, eps=bn_eps, momentum=bn_momentum,
            affine=True, track_running_stats=True,
        )

    def forward(self, x):
        # 每个窗口重置零状态；不把上一日期或另一股票的状态传入。
        _, hidden = self.gru(x)
        return self.normalization(hidden[-1])
