"""RadianceLM._loglikelihood_tokens: batched-vs-unbatched equivalence.

The batching trick (right-pad, no attention mask, rely on causal masking to keep real positions
uncontaminated by padding) is the one part of eval_harness.py that isn't a straight port of an
existing code path, so it gets its own equivalence test rather than trusting the reasoning in the
module docstring. Batch size 1 for every request is the reference: it can't have a padding bug by
construction, so any divergence when requests are batched together points at the padding logic.

Parametrized over dense/MoE/router configs: the "padding tokens live past every real position, so
causal masking makes them inert" argument only holds for plain causal attention. A capacity-limited
MoE router could let padding tokens compete for expert slots and change a real token's routing, and
per-token ACT halting could behave differently with trailing padding in the batch — both would be
batch-splitting-dependent bugs that a dense-only test can't see.
"""

from __future__ import annotations

import pytest
import torch

from radiance.eval_harness import RadianceLM
from radiance.model import DenseTransformer
from tests.conftest import TINY_VOCAB


def _make_lm(tiny_cfg, batch_size: int, vocab_size: int = TINY_VOCAB, **model_kwargs) -> RadianceLM:
    cfg = tiny_cfg(loop_count=2, **model_kwargs)
    model = DenseTransformer(cfg.model, vocab_size=vocab_size).eval()

    lm = RadianceLM.__new__(RadianceLM)
    lm.model = model
    lm._device = "cpu"
    lm.batch_size = batch_size
    lm.loops = None
    lm.max_length = cfg.model.max_seq_len
    return lm


class _Eot:
    eos_token_id = 0

    def __init__(self, length: int = TINY_VOCAB):
        self._length = length

    def __len__(self) -> int:
        return self._length


def _requests() -> list[tuple[tuple[None, None], list[int], list[int]]]:
    # Three requests of deliberately different (context, continuation) lengths so a shared batch
    # produces different amounts of right-padding per row.
    return [
        ((None, None), [1, 2, 3], [4, 5]),
        ((None, None), [1, 2, 3, 4, 5, 6], [7]),
        ((None, None), [3, 3], [1, 4, 2]),
    ]


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {},
        {"use_moe": True, "n_experts": 4, "moe_top_k": 2},
        {"use_router": True, "max_loops": 3},
    ],
    ids=["dense", "moe", "router"],
)
def test_batched_loglikelihood_matches_unbatched(tiny_cfg, model_kwargs):
    torch.manual_seed(0)
    lm_batched = _make_lm(tiny_cfg, batch_size=8, **model_kwargs)
    lm_batched.tokenizer = _Eot()
    torch.manual_seed(0)
    lm_single = _make_lm(tiny_cfg, batch_size=1, **model_kwargs)
    lm_single.tokenizer = _Eot()
    lm_single.model.load_state_dict(lm_batched.model.state_dict())

    requests = _requests()
    batched_results = lm_batched._loglikelihood_tokens(requests, disable_tqdm=True)
    single_results = lm_single._loglikelihood_tokens(requests, disable_tqdm=True)

    assert len(batched_results) == len(single_results) == len(requests)
    for (batched_lp, batched_greedy), (single_lp, single_greedy) in zip(batched_results, single_results):
        assert batched_greedy == single_greedy
        assert batched_lp == pytest.approx(single_lp, abs=1e-5)


def test_loglikelihood_matches_direct_forward_for_one_request(tiny_cfg):
    """The batched path must reduce to `logsumexp(logits) - logits[target]`, summed over the
    continuation, for the simplest case: one request, one batch, no padding at all.
    """
    lm = _make_lm(tiny_cfg, batch_size=1)
    lm.tokenizer = _Eot()

    context_enc, continuation_enc = [1, 2, 3], [4, 5]
    requests = [((None, None), context_enc, continuation_enc)]

    (logprob, is_greedy) = lm._loglikelihood_tokens(requests, disable_tqdm=True)[0]

    full_ids = torch.tensor([context_enc + continuation_enc[:-1]], dtype=torch.long)
    with torch.no_grad():
        logits = lm.model(full_ids).logits[0].double()
    log_probs = torch.log_softmax(logits, dim=-1)
    contlen = len(continuation_enc)
    cont_logits = log_probs[-contlen:]
    cont_ids = torch.tensor(continuation_enc, dtype=torch.long)
    expected_logprob = cont_logits.gather(1, cont_ids.unsqueeze(-1)).squeeze(-1).sum().item()
    expected_greedy = bool((cont_logits.argmax(dim=-1) == cont_ids).all())

    assert logprob == pytest.approx(expected_logprob, abs=1e-5)
    assert is_greedy == expected_greedy


def test_loglikelihood_masks_vocab_padding_rows(tiny_cfg):
    """A checkpoint's lm_head is padded to a multiple of 128 (model.padded_vocab_size) for
    tensor-core alignment; those extra rows have no tokenizer id and are never trained toward -inf
    (generate.generate_tokens masks them for the same reason -- see its comment on the vocab-padding
    rows). Without the matching mask in RadianceLM._model_logprobs, an untrained row winning the
    argmax silently flips is_greedy and deflates every logprob's softmax denominator.
    """
    tokenizer_len = TINY_VOCAB
    padded_vocab = TINY_VOCAB + 8

    lm = _make_lm(tiny_cfg, batch_size=1, vocab_size=padded_vocab)
    lm.tokenizer = _Eot(length=tokenizer_len)

    context_enc, continuation_enc = [1, 2, 3], [4, 5]
    requests = [((None, None), context_enc, continuation_enc)]
    (logprob, is_greedy) = lm._loglikelihood_tokens(requests, disable_tqdm=True)[0]

    full_ids = torch.tensor([context_enc + continuation_enc[:-1]], dtype=torch.long)
    with torch.no_grad():
        logits = lm.model(full_ids).logits[0].double()
    logits[:, tokenizer_len:] = float("-inf")
    log_probs = torch.log_softmax(logits, dim=-1)
    contlen = len(continuation_enc)
    cont_logits = log_probs[-contlen:]
    cont_ids = torch.tensor(continuation_enc, dtype=torch.long)
    expected_logprob = cont_logits.gather(1, cont_ids.unsqueeze(-1)).squeeze(-1).sum().item()
    expected_greedy = bool((cont_logits.argmax(dim=-1) == cont_ids).all())

    assert logprob == pytest.approx(expected_logprob, abs=1e-5)
    assert is_greedy == expected_greedy
