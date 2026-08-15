from __future__ import annotations

import torch
import torch.nn as nn

from radiance.config import ModelConfig

from .block import TransformerBlock
from .core import LoopContext
from .norms import RMSNorm
class MTPHead(nn.Module):
    """One auxiliary multi-token-prediction head (DeepSeek-V3's formulation).

    Predicts a token further ahead than the trunk does, by fusing the previous head's hidden state
    at position t with the embedding of the token the previous head was predicting, then running a
    single transformer block over the result. The unembedding is *shared* with the trunk's lm_head,
    which is what keeps a head's cost to one block rather than another d_model x vocab_size matrix.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_hidden = RMSNorm(cfg.d_model)
        self.norm_embedding = RMSNorm(cfg.d_model)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.block = TransformerBlock(cfg, is_first=True)

    def forward(
        self,
        hidden: torch.Tensor,
        token_embeddings: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.proj(
            torch.cat([self.norm_hidden(hidden), self.norm_embedding(token_embeddings)], dim=-1)
        )
        # Fresh LoopContext: the head is outside the recursion, and passing the trunk's kv_cache
        # would let it claim cache slots that belong to the trunk's blocks.
        out, _, _ = self.block(fused, cos, sin, LoopContext())
        return out


def _shift_left(x: torch.Tensor, positions: int) -> torch.Tensor:
    """Shift a (batch, seq, ...) tensor left along seq, zero-padding the tail.

    Head j reads the embedding of the token j positions ahead; the padded tail positions have no
    real future token, and their loss contribution is masked out by compute_loss's ignore_index.
    """
    if positions == 0:
        return x
    pad = torch.zeros_like(x[:, :positions])
    return torch.cat([x[:, positions:], pad], dim=1)
