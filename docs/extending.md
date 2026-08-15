# Extending

Each entry names the reference example to read before starting.

**New dataset.** Point `data.dataset` at any `user/dataset` HF dataset with `train`/`validation` splits and the right
`text_column`; no code changes unless the schema differs. No `validation` split (e.g. `HuggingFaceFW/fineweb`, see
`configs/fineweb_500m.yaml`): set `data.eval_split_size` to carve one off the front of `train`.

**Dataset too large to tokenize/cache up front.** `data.streaming: true` (see
`configs/tinystories_streaming.yaml`) — trades a full local shuffle and disk cache for a streaming/shuffle-buffer
approximation on both splits; no other changes needed for a standard HF hub dataset. To also avoid re-fetching
already-seen data across repeated short runs, add `data.disk_cache_max_gb`.

**New model variant.** Add config fields to `ModelConfig`, then wire them into `model/`. Keep the
`TransformerBlock` I/O contract so `train.py` and `data.py` stay untouched. `ACTRouter` /
`DenseTransformer._forward_act` is the reference for a variant that changes `forward`'s control flow rather than just
swapping in a different block. `configs/tinystories_gqa.yaml` is a worked example of a simpler one.

**MoE FFN.** `model.use_moe: true` (plus `moe_dense_every` to interleave dense blocks) — see
`configs/tinystories_moe.yaml`. `MoEFeedForward` is the reference for replacing a block's FFN sublayer wholesale
while preserving `FeedForward`'s `(*, d_model) -> (*, d_model)` contract, which is what lets it compose with ACT's
`_sparse_ffn_delta` path with no changes to either.

**A new *training signal*** (an auxiliary gradient, a routing hint, a reweighting): measure whether the signal
carries reproducible structure **before** spending runs on an A/B. Compute it on two disjoint batches at identical
weights and correlate. A signal that doesn't reproduce across batches cannot be learned from, no matter how large it
looks in-sample, and a fixed-step A/B will report "neutral" rather than "this is noise" — which reads as "needs a
longer run" and invites spending the budget to find out. `moe_counterfactual_weight` is the worked example: large
in-sample regret, zero cross-batch correlation, three A/Bs to establish what one forward/backward pair would have.

**Sparse attention.** `model.use_nsa` has been removed; see [model.md](model.md) for the rationale and git history
for the implementation. Not a win at TinyStories scale ([results.md](results.md)); a candidate for a longer-context
follow-up A/B if re-introduced.

**Changing the attention *mechanism* itself** (not just what feeds it, like GQA, or what gates it, like
`attn_out_gate`): `use_diff_attn` is the reference. Two things made it tractable rather than a rewrite — reusing
`qkv_proj`'s existing output width by re-chunking rather than adding projections (check whether a new mechanism's
Q/K/V shapes can be expressed as a different split of the same total width before reaching for new `nn.Linear`s), and
treating `KVCache` as extensible (`write3` alongside `write`) rather than forcing a new cache shape through the
existing two-tensor API. And run the *compiled* forward against eager before trusting either — see the
`torch._dynamo.disable` fix and its regression test in `tests/test_compile.py`, which caught a real inductor bug that
every eager test passed straight through.

**New numeric format / low-precision GEMM.** `nvfp4/` is the reference, and four things it did are worth copying.
**Write a pure-torch reference first and assert the kernel bit-exact against it** — that bar, not "is it close", is
what caught four separate 1-ULP disagreements between Triton and torch, none visible in the dequantized values.
**Quantize inside an `autograd.Function`**, never in autograd's view, or the step-function encode leaves a
nonzero-but-wrong gradient flowing through the scale's `amax`. **Cache anything derived from the weights once per
optimizer step**, invalidated from `train.py` next to `update_expert_bias()` rather than checked inside `forward`
where it becomes a dynamo guard. And prefer **`torch.library.custom_op` over `@torch._dynamo.disable`** for an op
inductor cannot lower: both make it opaque, but only the custom op stays in the graph so inductor can keep fusing
around it. Expect a *structural* inertness test rather than a numerical one — see [config.md](config.md)'s sixth
exception.

**New training behavior** (different scheduler, mixed precision): changes belong in `train.py`; keep the loop
step-based and keep config-driven values in `TrainConfig` rather than hardcoding. A new *optimizer* belongs in
`optim.py` — add it to `build_optimizer` and give it a `build_*_param_groups`; `MuonWithAuxAdam` is the reference for
one that needs different treatment per parameter class. If a new parameter class needs different treatment, classify
it by **what owns it**, not by its rank — see `norm_gain_param_ids` for the bug the `param.dim() < 2` shortcut caused
once a gain grew a variant dimension.

**New parameter *bank*** (one slice per loop iteration, per expert, per stream): check `build_param_groups` and
`build_muon_param_groups` before assuming it lands in the right group. Adding a leading dimension to something that
was 1-D moves it across the `dim() < 2` boundary, which silently changes both its optimizer and whether it is
weight-decayed. `RMSNorm`'s per-iteration gains and `HyperConnection`'s coefficients have both hit this.

**Anything touching the loss or the logits.** The `(batch, seq, vocab_size)` tensor is the largest activation in the
model at small `d_model`, so an extra pass over it is not a rounding error — reductions belong in `_nll_and_logz`
where they can share the one `logsumexp`, and `model.activation_bytes_per_token` has to be updated in step or
`auto_batch_size` will size batches against a loss that no longer exists.

**New per-iteration behavior for the loop body.** Put non-tensor state on `LoopContext` and read it in the block;
`loop_iter_conditioning` is the reference. Anything grad-carrying stays a positional argument (see `LoopContext`'s
docstring), and check `tests/test_loop_identity.py::test_no_dead_parameters` — a parameter bank that some loop mode
never reaches trains at its init value forever, silently.

**Changing the residual stream itself** (rather than what writes into it): `HyperConnection` /
`cfg.model.hyper_conn_streams` is the reference, and the one to read for what such a change touches. Widening the
hidden state changes the *rank* of every tensor between sublayers, so the work is mostly in the places that
reimplement the residual write or index it positionally — `TransformerBlock.forward`, `_run_loop_body`'s sparse
closure, `_run_loop_body_sparse`'s gather/scatter, `_forward_act`'s halting broadcasts, and
`activation_bytes_per_token`. Everything *below* the sublayer boundary (attention, FFN, MoE, the KV cache,
`generate.py`) stays untouched, because the read hands them the same `(batch, seq, d_model)` tensor as before —
preserve that and the blast radius stays small. Watch for blocks built outside the trunk: `MTPHead` constructs its
own `TransformerBlock` and must keep the single-stream path.

**New default-on feature.** Make its parameters inert (zero-init, identity-valued, or range-collapsed — see
[config.md](config.md)), then add the pair of tests that pins it: bit-identical to the feature-off model at init,
*and* demonstrably different once its weights move. `tests/test_inert_defaults.py` is the pattern. The second half
matters as much as the first — an "inert" feature with no second test could be inert forever and nobody would
notice. And check the *cost* before deciding "on": `hyper_conn_streams` is perfectly inert at `n > 1` and still
defaults off, because inert is not the same as free.
