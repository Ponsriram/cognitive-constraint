from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(token_ids))
        output, _ = self.rnn(emb)
        masked = output * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled


@dataclass
class CIRAOutput:
    logits: torch.Tensor
    sim_wm: torch.Tensor
    sim_lm: torch.Tensor
    sim_dist: torch.Tensor
    interference: torch.Tensor
    gate: torch.Tensor


class CIRAClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        confidence_wm_prior: float,
        confidence_lm_prior: float,
    ) -> None:
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim, dropout)
        repr_dim = hidden_dim * 2

        self.interference_scorer = nn.Sequential(
            nn.Linear(repr_dim * 4, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, 1),
        )

        self.gate_proj = nn.Linear(4, 1)
        self.classifier = nn.Sequential(
            nn.Linear(repr_dim * 4, repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(repr_dim, num_labels),
        )

        self.confidence_wm_prior = confidence_wm_prior
        self.confidence_lm_prior = confidence_lm_prior

    def forward(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        wm_ids: torch.Tensor,
        wm_mask: torch.Tensor,
        lm_initial_ids: torch.Tensor,
        lm_initial_mask: torch.Tensor,
        lm_distractor_ids: torch.Tensor,
        lm_distractor_mask: torch.Tensor,
    ) -> CIRAOutput:
        q = self.encoder(query_ids, query_mask)
        wm = self.encoder(wm_ids, wm_mask)
        lm_i = self.encoder(lm_initial_ids, lm_initial_mask)
        lm_d = self.encoder(lm_distractor_ids, lm_distractor_mask)

        sim_wm = F.cosine_similarity(q, wm, dim=-1)
        sim_i = F.cosine_similarity(q, lm_i, dim=-1)
        sim_d = F.cosine_similarity(q, lm_d, dim=-1)

        lm_weights = torch.softmax(torch.stack([sim_i, sim_d], dim=1), dim=1)
        lm = lm_weights[:, 0:1] * lm_i + lm_weights[:, 1:2] * lm_d
        sim_lm = F.cosine_similarity(q, lm, dim=-1)

        inter_features = torch.cat([wm, lm, torch.abs(wm - lm), q], dim=-1)
        interference = torch.sigmoid(self.interference_scorer(inter_features)).squeeze(-1)

        wm_conf = torch.sigmoid(sim_wm) * self.confidence_wm_prior
        lm_conf = torch.sigmoid(sim_lm) * self.confidence_lm_prior * (1.0 - lm_weights[:, 1])

        gate_features = torch.stack(
            [wm_conf, lm_conf, interference, sim_wm - sim_d],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_proj(gate_features)).squeeze(-1)

        refined_context = gate.unsqueeze(-1) * wm + (1.0 - gate.unsqueeze(-1)) * lm
        fused = torch.cat([q, refined_context, q * refined_context, torch.abs(q - refined_context)], dim=-1)
        logits = self.classifier(fused)

        return CIRAOutput(
            logits=logits,
            sim_wm=sim_wm,
            sim_lm=sim_lm,
            sim_dist=sim_d,
            interference=interference,
            gate=gate,
        )
