from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.models.distilbert import DistilBertModel, DistilBertConfig
from transformers.modeling_outputs import BaseModelOutput


class LpNorm(nn.Module):

    def __init__(self, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> None:
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps
    
    def forward(self, input: Tensor) -> Tensor:
        return F.normalize(input, self.p, self.dim, self.eps)


class AvgPoolSentenceBertModel(nn.Module):

    def __init__(self, bert: DistilBertModel) -> None:
        super().__init__()
        self.bert = bert
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        bert_out: BaseModelOutput = self.bert(input_ids, attention_mask)
        embeddings = bert_out.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).expand_as(embeddings)
        return torch.sum(embeddings * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)


SentenceBertModel = AvgPoolSentenceBertModel


class AddNorm(nn.Module):

    def __init__(self, layer: nn.Module, norm_shape: int | tuple[int, ...], dropout: float = 0.1) -> None:
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(norm_shape)
    
    def forward(self, X: Tensor) -> None:
        Y = self.layer(X)
        return self.norm(self.dropout(Y) + X)


class FeedForwardBlock(nn.Module):

    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, dropout: float = 0.1) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, output_dims)
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return self._net(input)


class ShortTextClusterer(nn.Module):

    def __init__(
        self,
        bert: DistilBertModel,
        n_classes: int,
        contrast_dropout: float = 0.1,
        cluster_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.sbert = SentenceBertModel(bert)

        config: DistilBertConfig = bert.config
        hidden_size = config.dim  # aka hidden_size without IDE prompt

        self.contrast_head = nn.Sequential(
            AddNorm(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(inplace=True),
                ),
                norm_shape=hidden_size,
                dropout=contrast_dropout,
            ),
            nn.Linear(hidden_size, 128),
            LpNorm(2, dim=1)
        )

        self.cluster_head = nn.Sequential(
            nn.Dropout(cluster_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(cluster_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_classes)
        )

        self.init_weights()

    def init_weights(self) -> None:
        # same initialization as the cluster head?
        for m in self.contrast_head.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.trunc_normal_(m.weight, std=0.02)

        for m in self.cluster_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        embeddings = self.sbert(input_ids, attention_mask)
        return self.cluster_head(embeddings)
