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
