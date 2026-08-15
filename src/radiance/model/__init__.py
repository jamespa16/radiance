"""The model: a looped transformer trunk plus its feature modules.

`DenseTransformer` and the public building blocks are re-exported here so
`from radiance.model import X` works regardless of which submodule owns X.
"""

from .act import ACTRouter
from .attention import CausalSelfAttention, KVCache, RotaryEmbedding, apply_rope, rotate_half
from .block import TransformerBlock
from .core import LoopContext, ModelOutput
from .ffn import BatchedExperts, FeedForward, MoEFeedForward, MoERouter
from .hyper_connections import HyperConnection
from .load import checkpoint_param_bytes, load_transformer_from_checkpoint, padded_vocab_size
from .masking import build_block_mask, document_ids, sequence_logprob_sum
from .mtp import MTPHead
from .norms import IterLoRA, RMSNorm
from .transformer import DenseTransformer

__all__ = [
    "ACTRouter",
    "BatchedExperts",
    "CausalSelfAttention",
    "DenseTransformer",
    "FeedForward",
    "HyperConnection",
    "IterLoRA",
    "KVCache",
    "LoopContext",
    "MTPHead",
    "MoEFeedForward",
    "MoERouter",
    "ModelOutput",
    "RMSNorm",
    "RotaryEmbedding",
    "TransformerBlock",
    "apply_rope",
    "build_block_mask",
    "checkpoint_param_bytes",
    "document_ids",
    "load_transformer_from_checkpoint",
    "padded_vocab_size",
    "rotate_half",
    "sequence_logprob_sum",
]
