from __future__ import annotations

import torch

from radiance.config import Config

from .transformer import DenseTransformer


def cast_params_to_native_bf16(model: "DenseTransformer") -> None:
    """In-place cast every parameter (not buffer) of `model` to bf16, for train.train() and
    load_transformer_from_checkpoint's shared train.native_bf16 handling.

    Parameters only, not buffers: RoPE's cos/sin cache and MoE's expert_bias carry no optimizer
    state, so casting them buys no memory and would cost precision that nothing downstream
    upcasts back (unlike RMSNorm's gain, which forward() already computes in fp32 regardless of
    the gain's own storage dtype). Autograd gives each bf16 parameter a bf16 .grad automatically,
    which is what lets optim.py's state allocators (already dtype-matched to the parameter, see
    _new_state_like) fall out for free.
    """
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)


def checkpoint_vocab_size(ckpt: dict) -> int:
    """The vocab size a saved checkpoint's model was built with.

    `token_emb.weight`'s leading dim is the only place this is recoverable from a raw state dict;
    both DenseTransformer reconstruction and the checkpoint-shape-mismatch check in
    checkpointing.load_pretrained_weights need it.
    """
    return ckpt["model"]["token_emb.weight"].shape[0]


def load_transformer_from_checkpoint(
    path: str, device: str, eos_id: int | None = None
) -> tuple["DenseTransformer", "Config"]:
    """Reconstruct a DenseTransformer + its embedded Config from a train.py checkpoint .pt file.

    Extracted from generate.load_checkpoint (which now calls this) so dpo_data.py's DPO
    reference-logprob precompute (dpo_data.py:153) can reuse the same reconstruction logic without
    a generate.py <-> dpo_data.py <-> model/ import cycle: generate.py already imports
    sft_data.py/data.py, so dpo_data.py must not import generate.py, but both dpo_data.py and
    generate.py already import model/.

    eos_id defaults to None, preserving generate.load_checkpoint's exact prior behavior (doc
    masking off during generation, since a single prompt is one document anyway regardless).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vocab_size = checkpoint_vocab_size(ckpt)
    model = DenseTransformer(cfg.model, vocab_size=vocab_size, eos_id=eos_id)
    if cfg.train.native_bf16:
        # Match training's storage dtype before load_state_dict, not after: load_state_dict's
        # copy_ preserves the *destination* tensor's dtype, so building this model at the default
        # fp32 first would silently upcast a bf16-trained checkpoint back to fp32 on load — twice
        # the VRAM a native_bf16 run was saved specifically to avoid, right where generate/serve
        # cares about it most (inference has no optimizer state to dwarf the parameter memory).
        cast_params_to_native_bf16(model)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg


def checkpoint_param_bytes(path: str) -> int:
    """Total parameter bytes in a checkpoint's state dict, without loading it onto a GPU.

    Used to size how much VRAM a checkpoint will need before deciding to load it — map_location="cpu"
    keeps this a pure host-memory read, so it's safe to call while another model already occupies the
    GPU whose free memory a caller is trying to reserve against.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return sum(t.numel() * t.element_size() for t in ckpt["model"].values())


def padded_vocab_size(vocab_size: int, multiple: int) -> int:
    """Round a tokenizer's vocab up to a multiple of `multiple` for the token_emb/lm_head matmuls.

    A vocab that isn't a multiple of 64/128 leaves the largest matmul in the model (the lm_head,
    d_model x vocab_size) on a ragged tile boundary, so the GPU runs a slow remainder tile. The
    extra rows are pure padding: no tokenizer id ever maps to them, so they never appear as a
    target, and the model simply learns to give them low probability (their embedding rows still
    receive gradient through the softmax denominator). This is the standard nanoGPT trick.

    `multiple <= 1` disables padding and returns vocab_size unchanged.
    """
    if multiple <= 1:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple
