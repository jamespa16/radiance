from __future__ import annotations

import torch

from radiance.config import Config

from .transformer import DenseTransformer
def load_transformer_from_checkpoint(
    path: str, device: str, eos_id: int | None = None
) -> tuple["DenseTransformer", "Config"]:
    """Reconstruct a DenseTransformer + its embedded Config from a train.py checkpoint .pt file.

    Extracted from generate.load_checkpoint (which now calls this) so data.py's DPO
    reference-logprob precompute can reuse the same reconstruction logic without a
    generate.py <-> data.py <-> model.py import cycle: generate.py already imports data.py, so
    data.py must not import generate.py, but both data.py and generate.py already import model.py.

    eos_id defaults to None, preserving generate.load_checkpoint's exact prior behavior (doc
    masking off during generation, since a single prompt is one document anyway regardless).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vocab_size = ckpt["model"]["token_emb.weight"].shape[0]
    model = DenseTransformer(cfg.model, vocab_size=vocab_size, eos_id=eos_id)
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
