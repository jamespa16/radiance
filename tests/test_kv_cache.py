"""KVCache correctness: incremental decoding must equal a single full-sequence forward.

This is the invariant most easily broken by changes to the loop body, because KVCache assigns
slots by *implicit call order* (KVCache.begin_step/write) rather than an explicit index. Anything
that changes how many CausalSelfAttention calls a forward makes — a new loop mode, K/V sharing,
a skipped iteration — silently misaligns every slot after it, and the only symptom is subtly wrong
generation. Testing the whole mode matrix here is what makes those changes safe to attempt.
"""

from __future__ import annotations

import pytest
import torch

from radiance.model import DenseTransformer
from tests.conftest import TINY_VOCAB


MODES = {
    "dense": dict(loop_count=1),
    "looped": dict(loop_count=3),
    "router": dict(use_router=True, max_loops=4),
    "moe": dict(use_moe=True, n_experts=4, moe_top_k=2, loop_count=2),
    "moe_router": dict(use_moe=True, n_experts=4, use_router=True, max_loops=3),
    "gqa": dict(n_kv_heads=2, loop_count=2),
    "hyper": dict(hyper_conn_streams=4, loop_count=2),
    "diff_attn": dict(use_diff_attn=True, loop_count=2),
    "diff_attn_gqa": dict(use_diff_attn=True, n_kv_heads=2, loop_count=2),
}


@pytest.mark.parametrize("mode", list(MODES))
def test_incremental_decode_matches_full_forward(tiny_cfg, tiny_ids, mode):
    cfg = tiny_cfg(**MODES[mode])
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    ids = tiny_ids(batch=1, seq=12)

    with torch.no_grad():
        full = model(ids).logits

        # Prefill on a prompt, then decode the rest one token at a time.
        cache = model.new_kv_cache()
        prompt_len = 5
        stepwise = [model(ids[:, :prompt_len], kv_cache=cache).logits]
        for t in range(prompt_len, ids.size(1)):
            stepwise.append(model(ids[:, t : t + 1], kv_cache=cache).logits)
        stepwise = torch.cat(stepwise, dim=1)

    assert stepwise.shape == full.shape
    # Not bit-exact: SDPA dispatches a different kernel for a 1-row query than for the full
    # triangular case, so this is a numerical-agreement check, not an identity one.
    torch.testing.assert_close(stepwise, full, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mode", list(MODES))
def test_cache_slot_count_matches_attention_calls(tiny_cfg, tiny_ids, mode):
    """new_kv_cache() must size itself to exactly the number of attention calls one forward makes.

    Under-sizing raises IndexError; over-sizing wastes memory silently and means the slot formula
    has drifted from the real execution path, so assert equality rather than sufficiency.
    """
    cfg = tiny_cfg(**MODES[mode])
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()
    cache = model.new_kv_cache()

    with torch.no_grad():
        model(tiny_ids(batch=1, seq=4), kv_cache=cache)

    assert cache._cursor == len(cache._k), (
        f"{mode}: forward made {cache._cursor} attention calls but new_kv_cache() allocated "
        f"{len(cache._k)} slots"
    )


@pytest.mark.parametrize("mode", list(MODES))
def test_key_padding_mask_matches_unpadded_single_row_forward(tiny_cfg, tiny_ids, mode):
    """Two different-length sequences, right-padded into one batch with key_padding_mask, must
    each score the same as running that sequence alone (unpadded, batch=1) — the correctness bar
    for batched generation's padding machinery. Compares full-sequence (no kv_cache) forwards
    since that already exercises every mode's attention path; the incremental-decode case is
    covered separately by generate_tokens_batched's own equivalence test in test_generate.py.
    """
    cfg = tiny_cfg(**MODES[mode])
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()

    short = tiny_ids(batch=1, seq=5, seed=1)
    long = tiny_ids(batch=1, seq=9, seed=2)
    max_len = long.shape[1]

    padded = torch.zeros((2, max_len), dtype=long.dtype)
    padded[0, : short.shape[1]] = short[0]
    padded[1, :] = long[0]
    key_padding_mask = torch.zeros((2, max_len), dtype=torch.bool)
    key_padding_mask[0, : short.shape[1]] = True
    key_padding_mask[1, :] = True
    key_positions = torch.arange(max_len).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        batched = model(padded, key_padding_mask=key_padding_mask, key_positions=key_positions).logits
        expected_short = model(short).logits
        expected_long = model(long).logits

    torch.testing.assert_close(
        batched[0, : short.shape[1]], expected_short[0], rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(batched[1], expected_long[0], rtol=1e-4, atol=1e-4)


def test_key_padding_mask_incremental_decode_matches_unpadded(tiny_cfg, tiny_ids):
    """Prefill with real padding (a short row alongside a longer one, right-padded), then decode
    a few more steps with new real tokens shared by both rows — row 0's output must match running
    its own (unpadded) prompt + those same continuation tokens alone at batch=1. This is the
    decode-time counterpart to test_key_padding_mask_matches_unpadded_single_row_forward: prefill
    and decode take different branches in CausalSelfAttention.forward (is_causal bool vs. the
    padded_causal_mask path once ctx.key_padding_mask is set), and generate_tokens_batched's own
    equivalence test in test_generate.py already covers the full serving path end-to-end, so this
    isolates just the padding-survives-into-decode invariant at the model level.
    """
    cfg = tiny_cfg(loop_count=2)
    model = DenseTransformer(cfg.model, vocab_size=TINY_VOCAB).eval()

    short = tiny_ids(batch=1, seq=4, seed=1)
    long = tiny_ids(batch=1, seq=7, seed=2)
    extra = tiny_ids(batch=1, seq=3, seed=3)  # shared continuation tokens fed to both rows
    max_prompt_len = long.shape[1]

    padded_prompt = torch.zeros((2, max_prompt_len), dtype=short.dtype)
    padded_prompt[0, : short.shape[1]] = short[0]
    padded_prompt[1, :] = long[0]
    key_padding_mask = torch.zeros((2, max_prompt_len), dtype=torch.bool)
    key_padding_mask[0, : short.shape[1]] = True
    key_padding_mask[1, :] = True
    key_positions = torch.arange(max_prompt_len).unsqueeze(0).expand(2, -1)
    real_len = [short.shape[1], long.shape[1]]

    with torch.no_grad():
        cache = model.new_kv_cache()
        model(padded_prompt, kv_cache=cache, key_padding_mask=key_padding_mask, key_positions=key_positions)
        for t in range(extra.shape[1]):
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.ones((2, 1), dtype=torch.bool)], dim=1
            )
            new_positions = torch.tensor([[real_len[0]], [real_len[1]]])
            key_positions = torch.cat([key_positions, new_positions], dim=1)
            real_len = [n + 1 for n in real_len]
            step_input = extra[:, t : t + 1].expand(2, 1)
            last_logits = model(
                step_input, kv_cache=cache, key_padding_mask=key_padding_mask, key_positions=key_positions
            ).logits[:, -1, :]

        row0_unpadded = torch.cat([short, extra], dim=1)
        expected = model(row0_unpadded).logits[:, -1, :]

    torch.testing.assert_close(last_logits[0], expected[0], rtol=1e-4, atol=1e-4)
