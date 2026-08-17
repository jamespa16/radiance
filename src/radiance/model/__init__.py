"""The model: a looped transformer trunk plus its feature modules.

`DenseTransformer` and the public building blocks are re-exported here so
`from radiance.model import X` works regardless of which submodule owns X.
"""

from .act import ACTRouter
from .attention import CausalSelfAttention, KVCache, RotaryEmbedding, apply_rope, rotate_half
from .block import TransformerBlock
from .core import LoopContext, ModelOutput
from .dpo import sequence_logprob_sum, target_logit_and_logz
from .ffn import BatchedExperts, FeedForward, MoEFeedForward, MoERouter
from .hyper_connections import HyperConnection
from .load import (
    cast_params_to_native_bf16,
    checkpoint_param_bytes,
    checkpoint_vocab_size,
    load_transformer_from_checkpoint,
    mask_vocab_padding,
    padded_vocab_size,
)
from .masking import build_block_mask, document_ids
from .mtp import MTPHead
from .norms import IterLoRA, RMSNorm
from .transformer import DenseTransformer

__all__ = [
    # Consumed outside model/ (train.py, losses.py, batching.py, checkpointing.py, dpo_data.py,
    # generate.py, optim.py, and tests). Changing one of these names' meaning or signature is a
    # cross-package change, not a local one.
    "DenseTransformer",
    "LoopContext",
    "MoEFeedForward",
    "RMSNorm",
    "cast_params_to_native_bf16",
    "checkpoint_param_bytes",
    "checkpoint_vocab_size",
    "document_ids",
    "load_transformer_from_checkpoint",
    "mask_vocab_padding",
    "padded_vocab_size",
    "sequence_logprob_sum",
    "target_logit_and_logz",
    # Re-exported for a uniform `from radiance.model import X` regardless of submodule, but only
    # imported from inside model/ today — free to change without auditing other packages.
    "ACTRouter",
    "BatchedExperts",
    "CausalSelfAttention",
    "FeedForward",
    "HyperConnection",
    "IterLoRA",
    "KVCache",
    "MTPHead",
    "MoERouter",
    "ModelOutput",
    "RotaryEmbedding",
    "TransformerBlock",
    "apply_rope",
    "build_block_mask",
    "rotate_half",
]
