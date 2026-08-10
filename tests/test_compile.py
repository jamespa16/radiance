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

  * grad_checkpoint + mode="reduce-overhead": torch.utils.checkpoint recomputes each block during
    backward. AOTAutograd partitions that recompute into its own graph segment, and reduce-overhead
    captures each segment as its own CUDA graph against one shared static pool — so the recompute's
    outputs overwrite a tensor the original forward's backward still needs ("accessing tensor
    output of CUDAGraphs that has been overwritten by a subsequent run"). train.py drops to
    mode=None whenever grad_checkpoint is on, same as it does for stochastic depth. Needs several
    backward() calls sharing one accumulated step (i.e. grad_accum_steps > 1) plus a router/MoE
    layer to actually surface — a plain looped model with grad_checkpoint alone did not reproduce
    it, so the regression test below matches configs/super.yaml's combination that did.

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


@slow
@requires_cuda
def test_grad_checkpoint_compiles_under_grad_accum():
    """Reproduces configs/super.yaml's exact crash (grad_checkpoint + use_router + use_moe, several
    backward() calls per accumulated step, no zero_grad between them) under mode="reduce-overhead",
    then confirms mode=None — what train.py now selects whenever grad_checkpoint is on — is safe
    across repeated accumulated steps. See the module docstring for the mechanism."""
    model = _build(
        loop_count=3, grad_checkpoint=True,
        use_router=True, max_loops=4,
        use_moe=True, n_experts=4, moe_top_k=2,
    )
    compiled = torch.compile(model, mode=None)
    ids = torch.randint(0, TINY_VOCAB, (4, 64), device="cuda")

    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.zeros_like(p)

    compiled.train()
    for _ in range(3):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for _ in range(3):  # grad_accum_steps=3: several backward() calls, no zero_grad between
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = compiled(ids)
                loss = out.logits.float().square().mean() + out.moe_aux_loss + out.ponder_cost
            loss.backward()
        assert torch.isfinite(model.token_emb.weight.grad).all()


@slow
@requires_cuda
@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_nsa_select_mask_compiles_for_train_and_eval(grad_checkpoint):
    """cfg.use_nsa's selection-branch BlockMask is a *second* data-dependent create_block_mask call
    site, distinct from _doc_masks': it's built once per attention call (see
    _nsa_build_select_mask's docstring for why it can't be amortised the way _doc_masks' is), so it
    needs its own @torch._dynamo.disable regression pin rather than assuming _doc_masks' coverage
    extends to it. nsa_block_size must stay 128 here — flex_attention's Triton kernel only accepts
    a BLOCK_SIZE that's a multiple of its own internal tile size, so max_seq_len needs to be large
    enough to have more than one block."""
    model = _build(
        loop_count=2, grad_checkpoint=grad_checkpoint, max_seq_len=256,
        use_nsa=True, nsa_block_size=128, nsa_top_k_blocks=1, doc_attention_mask=False,
    )
    compiled = torch.compile(model)
    ids = torch.randint(0, TINY_VOCAB, (2, 256), device="cuda")

    compiled.train()
    compiled(ids).logits.square().mean().backward()

    compiled.eval()
    with torch.no_grad():
        assert compiled(ids).logits.shape == (2, 256, TINY_VOCAB)
