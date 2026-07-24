"""What is and isn't invariant for a halted ACT position — the constraint on making ACT save
wall-clock.

ACT currently costs the same as running max_loops iterations for every token: halting changes what
gets *accumulated*, not what gets *computed*. The obvious fix is to skip halted positions and reuse
their cached K/V, which is only valid if that K/V doesn't change once a position halts.

It mostly doesn't hold. These tests pin exactly where the line is, because the tempting-but-wrong
version of the argument ("frozen state implies frozen K/V") is very easy to re-derive:

  * K and V are per-position projections of the block's *input*, and a halted position's input to
    the first loop-body block is frozen_x, which by construction stops changing. So loop-body
    block 1's K/V at a halted position genuinely is invariant.

  * Block 1's *output* at that position is not. Attention mixes across positions, and the
    still-running positions it attends to keep evolving. So block 2's input — and therefore block
    2's K/V — changes, and so on down the stack.

Anyone reusing halted K/V beyond the first loop-body block is making an approximation, in the same
family as the one act_ffn_capacity_ratio already documents, not an exact optimisation.
"""

from __future__ import annotations

import torch

from radiance.config import ModelConfig
from radiance.model import DenseTransformer, LoopContext
from tests.conftest import TINY_VOCAB


def _kv_by_block(model, x, cos, sin, frozen, iterations=2):
    """Record each loop-body block's raw K/V per iteration, replaying the ACT freeze by hand."""
    records, state = {}, x
    for iteration in range(1, iterations + 1):
        h = state
        for index, block in enumerate(model.blocks[1:], start=1):
            qkv = block.attn.qkv_proj(block.ln1(h))
            d_model = model.cfg.d_model
            kv_dim = block.attn.n_kv_heads * block.attn.head_dim
            _, k, v = qkv.split([d_model, kv_dim, kv_dim], dim=-1)
            records[(iteration, index)] = (k.detach().clone(), v.detach().clone())
            h, _, _ = block(h, cos, sin, LoopContext(iteration=iteration))
        # The ACT freeze: halted positions keep their state, still-running ones advance.
        state = torch.where(frozen.unsqueeze(-1), state, h)
    return records


def _setup(n_layers=4):
    cfg = ModelConfig(
        d_model=32, head_dim=8, n_layers=n_layers, ffn_mult=2.0, ffn_depth=1,
        dropout=0.0, max_seq_len=32, use_router=True, max_loops=4,
    )
    torch.manual_seed(0)
    model = DenseTransformer(cfg, vocab_size=TINY_VOCAB).eval()
    ids = torch.randint(0, TINY_VOCAB, (1, 12), generator=torch.Generator().manual_seed(1))
    x = model.token_emb(ids)
    cos, sin = model.rope(12)
    x, _, _ = model.blocks[0](x, cos, sin, LoopContext())
    return model, x, cos, sin


def _max_drift(records, block_index, positions):
    k1, v1 = records[(1, block_index)]
    k2, v2 = records[(2, block_index)]
    return max(
        (k1 - k2)[:, positions].abs().max().item(),
        (v1 - v2)[:, positions].abs().max().item(),
    )


def test_first_loop_block_kv_is_invariant_for_halted_positions():
    """K/V are per-position projections of an input that stops changing, so this one is exact."""
    model, x, cos, sin = _setup()
    halted = [0, 2, 5, 7, 9, 11]
    frozen = torch.zeros(1, 12, dtype=torch.bool)
    frozen[0, halted] = True

    records = _kv_by_block(model, x, cos, sin, frozen)
    assert _max_drift(records, 1, halted) == 0.0


def test_later_loop_blocks_kv_is_not_invariant_for_halted_positions():
    """The claim that breaks the 'exact sparsity' argument: block 1's output at a halted position
    depends on still-running positions' K/V, which keep evolving, so block 2+ inputs drift."""
    model, x, cos, sin = _setup()
    halted = [0, 2, 5, 7, 9, 11]
    frozen = torch.zeros(1, 12, dtype=torch.bool)
    frozen[0, halted] = True

    records = _kv_by_block(model, x, cos, sin, frozen)
    assert _max_drift(records, 2, halted) > 1e-5
    assert _max_drift(records, 3, halted) > 1e-5


def test_prefix_halting_is_the_misleading_special_case():
    """Why the wrong conclusion is easy to reach: if the halted set is a causal *prefix*, halted
    positions never attend to a still-running one, so every block looks invariant. Real ACT halting
    is scattered across positions, so this case is not representative."""
    model, x, cos, sin = _setup()
    prefix = list(range(6))
    frozen = torch.zeros(1, 12, dtype=torch.bool)
    frozen[0, prefix] = True

    records = _kv_by_block(model, x, cos, sin, frozen)
    for block_index in (1, 2, 3):
        assert _max_drift(records, block_index, prefix) == 0.0
