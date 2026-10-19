"""正文式 (5)-(10) 的历史分支和未来残差分支。"""
import torch
from torch import nn
from .feature_extractor import StockEncoder
from .prior_beta import PriorFactor
from .hidden_beta import HiddenFactor
from .individual_alpha import IndividualAlpha
from .temporal_contrastive import ContrastiveProjection


class FactorGCL(nn.Module):
    def __init__(self, config, horizons):
        super().__init__()
        self.config = config
        h = config.embedding_dim
        slope = config.negative_slope
        encoder_args = (
            config.input_dim, h, config.gru_layers,
            config.batchnorm_eps, config.batchnorm_momentum,
        )
        self.encoder = StockEncoder(*encoder_args)
        self.prior = PriorFactor(h, slope) if config.use_prior else None
        self.hidden = HiddenFactor(h, config.hidden_factors, slope) if (
            config.use_hidden and config.hidden_factors > 0
        ) else None
        self.alpha = IndividualAlpha(h, slope) if config.use_alpha else None
        self.cl_enabled = config.use_cl and config.use_alpha
        self.future_encoder = (
            self.encoder if config.share_future_encoder else StockEncoder(*encoder_args)
        ) if self.cl_enabled else None
        self.projection = ContrastiveProjection(
            h, config.projection_hidden_dim, config.projection_output_dim, slope
        ) if self.cl_enabled else None
        # 三路独立无偏置线性映射 + 一个公共偏置，严格对应式 (8)。
        self.heads = nn.ModuleDict()
        if self.prior is not None:
            self.heads["prior"] = nn.Linear(h, horizons, bias=False)
        if self.hidden is not None:
            self.heads["hidden"] = nn.Linear(h, horizons, bias=False)
        if self.alpha is not None:
            self.heads["alpha"] = nn.Linear(h, horizons, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(horizons))

    def forward_history(self, x, beta):
        if beta.shape != (len(x), self.config.prior_factors):
            raise ValueError("beta 维度与股票数或固定 134 行业字典不一致。")
        stock = self.encoder(x)
        prior = self.prior(stock, beta) if self.prior is not None else torch.zeros_like(stock)
        residual = stock - prior
        if self.hidden is not None:
            hidden, beta_h = self.hidden(residual)
        else:
            hidden, beta_h = torch.zeros_like(stock), None
        alpha = self.alpha(residual - hidden) if self.alpha is not None else torch.zeros_like(stock)
        prediction = self.output_bias.expand(len(x), -1)
        for name, representation in [("prior", prior), ("hidden", hidden), ("alpha", alpha)]:
            if name in self.heads:
                prediction = prediction + self.heads[name](representation)
        return {
            "prediction": prediction, "e_s": stock, "e_p": prior,
            "e_r": residual, "beta_h": beta_h, "e_h": hidden, "e_alpha": alpha,
        }

    def forward_future(self, x_future, beta, beta_h_from_history):
        if not self.cl_enabled:
            raise RuntimeError("关闭 CL 的消融不应执行未来编码器。")
        stock = self.future_encoder(x_future)
        prior = self.prior(stock, beta) if self.prior is not None else torch.zeros_like(stock)
        residual = stock - prior
        if self.hidden is not None:
            if beta_h_from_history is None:
                raise ValueError("未来 hidden 必须显式传入历史 beta_h。")
            hidden = self.hidden.propagate(residual, beta_h_from_history)
        else:
            hidden = torch.zeros_like(stock)
        # 式 (10) 字面形式：未来端不再经过历史 alpha 线性层。
        return residual - hidden

    def predict(self, x, beta):
        return self.forward_history(x, beta)["prediction"]

    def forward(self, x, beta):
        return self.predict(x, beta)


class GRUBaseline(nn.Module):
    """同编码器 GRU 基线，未披露参数采用补全设定；不是声称获知论文基线全部参数。"""
    cl_enabled = False

    def __init__(self, config, horizons):
        super().__init__()
        self.encoder = StockEncoder(
            config.input_dim, config.embedding_dim, config.gru_layers,
            config.batchnorm_eps, config.batchnorm_momentum,
        )
        self.head = nn.Linear(config.embedding_dim, horizons)

    def forward_history(self, x, beta):
        return {"prediction": self.head(self.encoder(x))}

    def predict(self, x, beta):
        return self.forward_history(x, beta)["prediction"]

    def forward(self, x, beta):
        return self.predict(x, beta)


def make_model(config, horizons):
    return (FactorGCL if config.name == "factorgcl" else GRUBaseline)(config, horizons)
