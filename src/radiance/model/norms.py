from __future__ import annotations

import torch
import torch.nn as nn

from radiance.config import ModelConfig
class RMSNorm(nn.Module):
    """RMSNorm, optionally with one set of gains per loop iteration.

    `n_variants > 1` gives the norm a `(n_variants, d_model)` gain matrix indexed by which loop
    iteration is executing (cfg.loop_iter_conditioning). This is the cheapest way to tell a
    weight-shared loop body how deep into the recursion it is — the same trick adaLN uses to
    condition a shared block on a timestep — at ~d_model parameters per iteration.

    At `n_variants == 1` the gain keeps its original `(d_model,)` shape exactly, so a model without
    iteration conditioning is bit-identical to one from before this existed and its checkpoints
    load unchanged.
    """

    def __init__(self, d_model: int, eps: float = 1e-6, n_variants: int = 1):
        super().__init__()
        self.n_variants = n_variants
        shape = (d_model,) if n_variants == 1 else (n_variants, d_model)
        self.weight = nn.Parameter(torch.ones(*shape))
        self.eps = eps
        if n_variants > 1:
            self._register_load_state_dict_pre_hook(self._broadcast_legacy_gain, with_module=False)

    def _broadcast_legacy_gain(self, state_dict, prefix, *args) -> None:
        """Load a pre-conditioning checkpoint by copying its single gain into every variant, so
        enabling loop_iter_conditioning on an existing run starts from exactly that run's weights.
        Mirrors MoEFeedForward._upgrade_legacy_expert_keys."""
        weight = state_dict.get(f"{prefix}weight")
        if weight is not None and weight.dim() == 1:
            state_dict[f"{prefix}weight"] = weight.unsqueeze(0).expand(self.n_variants, -1).contiguous()

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        weight = self.weight
        if self.n_variants > 1:
            # Clamped, not wrapped: running more iterations at inference than were trained (see
            # radiance-generate --loops) should reuse the deepest learned gains rather than
            # silently cycling back to the shallow ones.
            weight = weight[min(variant, self.n_variants - 1)]
        return (x * weight.float()).to(dtype)


class IterLoRA(nn.Module):
    """One low-rank adapter per loop iteration, added to a shared base projection.

    The stronger arm of cfg.loop_iter_conditioning: where per-iteration norm gains can only rescale
    channels, this gives each iteration a genuine rank-r update to the projection itself, at
    n_variants * r * (in + out) parameters instead of a full weight matrix per iteration.

    B is zero-initialised (the standard LoRA convention), so the adapter contributes exactly zero
    at init and the model starts identical to the un-adapted one.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, n_variants: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_variants, in_features, rank))
        self.B = nn.Parameter(torch.zeros(n_variants, rank, out_features))
        nn.init.normal_(self.A, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, variant: int = 0) -> torch.Tensor:
        v = min(variant, self.A.size(0) - 1)
        return (x @ self.A[v]) @ self.B[v]


def _make_iter_lora(cfg: ModelConfig, in_features: int, out_features: int, n_variants: int):
    """IterLoRA when cfg selects the "lora" arm and there is more than one iteration, else None."""
    if cfg.loop_iter_conditioning != "lora" or n_variants <= 1:
        return None
    return IterLoRA(in_features, out_features, cfg.loop_lora_rank, n_variants)
