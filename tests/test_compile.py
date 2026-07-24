"""torch.compile interactions.

Everything here failed only under a real compiled GPU run — the eager unit tests all passed while
these were broken, which is exactly why they're worth pinning. Both bugs cost a full training run
to discover:

  * document masking + torch.compile: create_block_mask builds a BlockMask from data-dependent
    index tensors. Traced into the enclosing graph those become intermediates with an unresolved
    layout and inductor fails to lower them. DenseTransformer._doc_masks is marked
    @torch._dynamo.disable so it runs eagerly and hands the compiled region a concrete BlockMask.

  * stochastic loop depth + mode="reduce-overhead": CUDA graphs assume a static execution path,
    which a per-step-varying loop count gives up. Replaying a different count overwrote the
    previous graph's gradient tensors. train.py drops to mode=None when loop_count_min is set.

These are slow (each compiles), so they're marked and kept few.
"""

from __future__ import annotations

import pytest
import torch

from radiance.config import ModelConfig
from radiance.model import DenseTransformer
from tests.conftest import TINY_VOCAB


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled paths need CUDA")
slow = pytest.mark.slow


def _build(**overrides) -> DenseTransformer:
    fields = dict(
        # head_dim >= 16: flex_attention's compiled kernel rejects anything smaller.
        d_model=128, head_dim=64, n_layers=3, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=64,
    )
    fields.update(overrides)
    torch.manual_seed(0)
    return DenseTransformer(ModelConfig(**fields), vocab_size=TINY_VOCAB, eos_id=5).cuda()


@slow
@requires_cuda
@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_doc_masking_compiles_for_train_and_eval(grad_checkpoint):
    """Both paths matter: grad_checkpoint is disabled under eval, so eval traces a *different*
    graph — and that was the one that blew up while training compiled fine."""
    model = _build(loop_count=2, grad_checkpoint=grad_checkpoint)
    compiled = torch.compile(model)
    ids = torch.randint(0, TINY_VOCAB, (2, 64), device="cuda")

    compiled.train()
    compiled(ids).logits.square().mean().backward()

    compiled.eval()
    with torch.no_grad():
        assert compiled(ids).logits.shape == (2, 64, TINY_VOCAB)


@slow
@requires_cuda
def test_stochastic_depth_compiles_across_loop_counts():
    """Every sampled loop count traces its own graph; running several in sequence must not corrupt
    each other's gradients."""
    model = _build(loop_count=4, loop_count_min=2, loop_count_max=5)
    # mode=None, matching what train.py selects when loop_count_min is set — "reduce-overhead"
    # captures CUDA graphs, which a varying loop count cannot support.
    compiled = torch.compile(model, mode=None)
    ids = torch.randint(0, TINY_VOCAB, (2, 64), device="cuda")

    compiled.train()
    for loops in (2, 3, 4, 5, 3, 2):
        compiled.zero_grad()
        compiled(ids, loop_count=loops).logits.square().mean().backward()
        assert torch.isfinite(model.token_emb.weight.grad).all()
