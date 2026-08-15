# `train.py` and `generate.py` — the training loop and inference

The loop is in `train.py`; the loss functions are in `losses.py`, batch sizing and micro-batch splitting in
`batching.py`, checkpoint save/load in `checkpointing.py`, and evaluation in `evaluation.py`.

Plain PyTorch training loop (no HF `Trainer`): the optimizer from `optim.build_optimizer` plus a warmup +
cosine-or-WSD LR schedule (`build_lr_scheduler`), manual loss computation, gradient clipping, periodic W&B logging
(`train/loss`, `train/lm_loss`, `train/z_loss`, `train/mtp_loss`, `train/ponder_cost`, `train/mean_loop_depth`,
`train/expert_bias_spread`, `train/lr`, `val/loss`) plus a matching stdout line, periodic checkpointing to
`cfg.train.output_dir` (raw `torch.save` of state dict + config), and periodic `evaluate()` against validation.

The stdout line matters more than it looks: W&B was previously the only place a loss ever appeared, so a run with
`wandb.mode: disabled` (sweeps, CI, quick A/Bs) produced no visible signal at all. On CUDA it also carries `mem`, the
reserved VRAM.

The loop is step-based (`cfg.train.max_steps`), not epoch-based, and cycles the train `DataLoader` via manual
`StopIteration` handling.

## `resolve_compile_mode(raw_model, cfg, device_type)`

Decides whether the run gets CUDA graphs (`mode="reduce-overhead"`) or plain inductor (`mode=None`). Three things
independently rule CUDA graphs out, because each gives up the static execution path a captured graph assumes:

1. **Stochastic loop depth** — replaying a different loop count overwrites the previous graph's gradient tensors.
2. **`grad_checkpoint`** — the backward recompute is captured as its own graph against the same static pool,
   overwriting a tensor the original backward still needs.
3. **`doc_attention_mask`.**

The third is why this is a named function with tests rather than an `if` in `train()`: it is the only one that fails
*silently*. It rebuilds a `flex_attention` `BlockMask` every step out of that batch's document boundaries, so its
tensors land at fresh addresses; CUDA graph trees treats them as static inputs (they cross the eager `_doc_masks`
graph break as lifted constants), and a changed data pointer forces a re-record. Every re-record instantiates another
`cudaGraphExec` that is never freed. Measured on the 21-executed-block looped config at batch 16 x grad_accum 2:

| | step time | process VRAM |
|---|---|---|
| `reduce-overhead` + doc masking | 282 ms | **+8 MB/step**, exhausts a 32 GB card by step ~1600 |
| `reduce-overhead`, doc masking off | 84 ms | flat |
| `mode=None` (either way) | 87 ms | flat |

Two things to carry forward. **`torch.cuda.memory_reserved()` cannot see it** — it sat flat at 6.43 GB for the entire
run while the process climbed 16 -> 22.5 GB, because the growth is outside the caching allocator. An earlier
investigation measured reserved memory, correctly concluded the model was clean, and spent its time in the
DataLoader. Use `torch.cuda.mem_get_info()` (or `nvidia-smi`); `(total - free) - memory_reserved()` is the
non-allocator footprint and is the number that moved. And **CUDA graphs were never buying anything here anyway**:
84.2 ms vs 82.7 ms in the one configuration where they work correctly. Since `doc_attention_mask` defaults on,
`mode="reduce-overhead"` is now nearly dead code on real configs — it survives for cases that build no per-step mask
(`head_dim < 16`, non-CUDA, doc masking explicitly off, and **DPO runs**, where `resolve_dpo_doc_mask` turns the mask
off because its one-pair-per-row packing makes it provably inert — see [post-training.md](post-training.md)). If a
future change makes CUDA graphs actually pay, the fix
is to give the BlockMask stable addresses (build once, `copy_` into it, keeping `mask_mod`'s `doc_ids` a persistent
buffer so partial blocks stay correct), not to relax this.

`tests/test_compile.py` carries four *fast* tests (no GPU, no compile) pinning `resolve_compile_mode`'s output
directly, because this failure is silent — it raises nothing; it leaks device memory and triples step time until the
card fills up. There is no cheap runtime assertion for that, so what gets tested is the decision. The
`"reduce-overhead"` control case matters as much as the three `None` cases: without it they would all pass against a
function that returned `None` unconditionally, i.e. against "never use CUDA graphs at all" as an accidental fix.

## The loss

`compute_loss` applies the causal-LM one-position shift to the *labels* (padding them with `ignore_index`), not by
slicing `logits[:, :-1]`. The slice is a non-contiguous view whose `.contiguous()`/`.view()` forces a full copy of the
`(batch, seq, vocab_size)` logits — the largest activation in the model — on every forward; shifting labels instead
makes `logits.view(-1, vocab_size)` a free reshape over exactly the same targets. Worth ~7% of step time at
`configs/tinystories.yaml`'s size.

It returns `(lm_loss, z_loss)`: the second is the log-Z regulariser `mean(logsumexp(logits)^2)`
(`cfg.model.z_loss_weight`, default `1e-4`), which keeps logit scale from drifting and matters more here than in a
plain transformer because looping multiplies effective depth without adding parameters. It is applied to the
*training* loss only — `evaluate()` discards it so `val/loss` stays a pure LM number, exactly as `ponder_cost` and
`moe_aux_loss` already are. Note the reduction order: `logsumexp` over the vocab **first**, then mask, rather than
`flat_logits[mask]`, which would copy nearly the whole logits tensor just to drop one row per sequence.

**Both losses come out of one `logsumexp`, and that was the single largest throughput win measured in this repo.**
Cross-entropy is exactly `logsumexp(x) - x[label]`, and z_loss squares that same `logsumexp(x)` — so the old
`F.cross_entropy(...)` plus a separate `torch.logsumexp(...)` reduced the largest tensor in the model twice in the
forward and twice again in the backward. `_nll_and_logz` computes `z` once and gathers the target logit off it;
`compute_loss` and `compute_mtp_loss` (one full-width logits tensor *per head*) both go through it. Two things
compound. `F.cross_entropy` sits on **autocast's fp32 list**, so it silently upcast the whole tensor before reducing
it — nothing on the new path does, so the logits stay at compute width. And `build_loss_fn` wraps the result in
`torch.compile` when `cfg.train.compile` is on (separately from the model's own compile, since the loss consumes the
model's *output*), which lets inductor fuse the whole thing into a single pass and keep the reduction in fp32 without
ever materialising an fp32 copy.

Measured standalone at batch 32 x seq 512 x vocab 50304, fwd+bwd: **22.7 ms and +8.2 GB** for the two-pass version,
17.0 ms eager on the new path, **5.5 ms and +5.0 GB compiled**. End-to-end on `configs/tinystories.yaml` the step went
**82.4 -> 37.5 ms and peak memory 22.5 -> 7.7 GB**. It is also *more* accurate: `F.cross_entropy` under autocast
returned a bf16-rounded loss, so against an fp32 reference the compiled single-pass version is off by 2.5e-5 where
the original was off by 3.4e-3.

The size of this is a property of the *shape*, not of the code being clever: at `d_model: 256` with a 50304-token
vocab the logits are `196 x d_model` wide, so anything that walks them twice costs more than a transformer block.
Expect the win to shrink as `d_model` grows — at `configs/fineweb_500m.yaml`'s `d_model: 1280` it is ~1.5% of step
time and 15% of peak memory rather than 2.2x.

The training loss is `lm_loss + ponder_weight * ponder_cost + moe_aux_loss_weight * moe_aux_loss` (the latter two are
zero unless their feature is on).

## Multi-token prediction

`compute_mtp_loss` handles the auxiliary heads (`cfg.model.mtp_heads`, default `1` = ordinary next-token
prediction). Head *d* predicts the token *d+1* positions ahead, fusing the previous head's hidden state with the
embedding of the token that head was predicting, running one `TransformerBlock`, and reusing the trunk's **shared**
`lm_head` — which keeps a head's cost to one block rather than another `d_model x vocab_size` matrix. `forward()`
returns the heads' *hidden states* rather than logits, so eval and generation (which skip the heads) pay nothing and
training projects one `(batch, seq, vocab_size)` tensor at a time. The heads are excluded from
`num_active_parameters()`: they never run at inference, so counting them would inflate a `tokens_per_param`-derived
`max_steps`. See `configs/tinystories_mtp.yaml`.

## LR schedule

`build_lr_scheduler`'s warmup ramps over `(step + 1)`, because `LambdaLR` evaluates the lambda at step 0 to set the LR
for the *first* `optimizer.step()` — a plain `step / warmup_steps` ramp spends that entire first step at `lr=0`. It
then decays to `cfg.train.min_lr_ratio * lr` (default `0.1`) rather than to 0, since the tail of a run at a ~0 LR
contributes nothing; `min_lr_ratio: 0.0` restores decay-to-zero.

`cfg.train.lr_schedule` selects `"cosine"` (default) or `"wsd"` — warmup, hold at full LR, then decay only over the
final `wsd_decay_ratio`. WSD's advantage is that its stable phase doesn't depend on `max_steps`, so a run can be
extended or branched from a mid-training checkpoint without the earlier steps having been trained on a schedule
shaped for a different horizon; cosine's shape is a function of `max_steps`, so changing it invalidates everything
before the change. It stays off by default because it isn't a quality win and switching it would silently reshape
every config whose `lr` was tuned against cosine.

## Checkpoints and resume

`save_checkpoint` writes the optimizer state, LR-scheduler state and `GradScaler` state alongside the
weights/step/config. Only one checkpoint per run is kept — each `save_every` save deletes the previous `step_*.pt`
before writing the new one — so an output directory holds at most one `.pt` file.

`cfg.train.resume_from` (opt-in, default `null`) restores all saved state. Set it to a checkpoint path, or to the
literal `"auto"` to pick the single `step_*.pt` in `output_dir`, so an interrupted run can be relaunched with its
config unchanged. Without the optimizer moments a "resumed" run restarts AdamW from zero momentum at warmup LR, which
shows up as a loss spike. An explicit `resume_from` path that doesn't exist raises rather than silently starting from
scratch; `"auto"` against an empty `output_dir` is just a fresh run.

What is *not* restored is the DataLoader position and RNG state, which trade off against each other: `train()`
re-seeds off the resumed step so the loader draws a different shuffle order rather than replaying batches already
trained on. A resumed run is therefore statistically equivalent to an uninterrupted one, not bit-identical — with
`dropout: 0.0` it is bit-identical (verified: same weights, same AdamW moments, same LR sequence).

## Steps, batches and memory

`cfg.train.eval_max_batches` (default `50`) caps how many batches each `evaluate()` call consumes. Uncapped — the
previous behavior, restored with `null` — every eval walks the *entire* validation split, which is unbounded for a
streaming one and, at `configs/tinystories.yaml`'s settings, cost more wall-clock than all 1000 training steps
combined. A fixed batch count also keeps `val/loss` comparable across configs whose validation splits differ in size.

`cfg.train.tokens_per_param` (opt-in, default `null`) derives `max_steps` from model size instead of pinning it:
once the model is built, `train()` overwrites `cfg.train.max_steps` with `round(tokens_per_param *
raw_model.num_active_parameters() / (effective_batch_size * the resolved per-row token width))` — pretraining uses
`data.seq_len`, SFT/DPO use their active `seq_len` override, and DPO counts both chosen and rejected rows — e.g. `20`
for a Chinchilla-optimal budget, and prints/logs the resulting step count. So the same config keeps tracking the
"right" number of steps as `model.*` fields change instead of needing `max_steps` hand-recomputed.
`num_active_parameters()` differs from the flat `num_parameters()` only under `use_moe`, excluding each MoE layer's
inactive experts — Chinchilla-style scaling assumes every parameter multiplies against every token, which is false
for MoE, so the flat count would inflate `max_steps` by roughly `n_experts / moe_top_k`. `estimate_batch_size`'s
optimizer-state sizing still uses the flat count, since every expert parameter gets a gradient and optimizer state
regardless. `warmup_ratio` is read as a live property off whatever `max_steps` ends up being, so warmup scales along
with it. See `configs/fineweb_500m.yaml`; leave `tokens_per_param: null` and set `max_steps` directly for pinned runs.

`cfg.train.grad_accum_steps` (default `1`) accumulates gradients over that many `batch_size`-sized micro-batches
before `optimizer.step()`/`scheduler.step()`, so the effective batch (`batch_size * grad_accum_steps`) can exceed what
fits in one forward/backward. Each micro-batch's loss is divided by `grad_accum_steps` before `.backward()` so the
accumulated gradient matches training on one `effective_batch_size`-sized batch, and `step`/W&B
logging/`eval_every`/`save_every` all stay in accumulated-step units.

### `auto_batch_size` (CUDA-only, default `True`)

Overwrites the configured `batch_size`/`grad_accum_steps` at startup with values computed from free VRAM and model
size (`estimate_batch_size`) — a deliberately conservative closed-form estimate (params/gradients/optimizer state
sized exactly, activation memory from `DenseTransformer.activation_bytes_per_token`, `train.optimizer == "muon"`'s
transient Newton-Schulz reserve from `optim.muon_orthogonalize_reserve_bytes` (see docs/optim.md), and memory divided
by `cfg.resolved_row_tokens` rather than `cfg.data.seq_len` so an active SFT/DPO `seq_len` override is reflected)
rather than an expensive live probe.

It defaults `True` — a deliberate behavior change for every existing config, not the usual opt-in convention — since
on CUDA it only ever makes the actual micro-batch *safer* than a hand-picked one, never bigger, and it is what gates
the OOM backoff below. Set it `False` for a manually-chosen or W&B-swept `batch_size` (e.g. a sweep already tuning
`batch_size` itself). On CPU/MPS it is a no-op with a printed note.

`cfg.train.target_effective_batch_size` sets the effective batch the estimate solves `grad_accum_steps` for; left at
`None` it falls back to whatever the configured `batch_size`/`grad_accum_steps` already imply, so an existing
config's effective batch is preserved even as `auto_batch_size` re-splits it. `cfg.train.vram_safety_margin`
(default `0.5`) scales how much of the estimated budget is used.

### Two-tier OOM recovery

Because the estimate is approximate by construction, `auto_batch_size` also enables an escalation in the training
loop, each tier attacking a different memory pool.

**Tier one**: a CUDA OOM halves an internal `micro_chunk_size` (starts equal to `batch_size`, monotonically shrinks,
floor `1`) that further splits each already-fetched micro-batch along the batch dimension and retries the same
accumulated step, rather than ending the run. `batch_size`/`grad_accum_steps`/`effective_batch_size` (and therefore
`tokens_per_param` accounting) are untouched, since backoff never rebuilds the DataLoader — only how many
forward/backward calls it takes to process one already-fetched micro-batch. This only reduces activation memory.

That the *gradient* is untouched too is the part worth stating explicitly, because it is what makes the tier honest
rather than merely survivable, and it is not automatic: a chunk's loss is a mean, so recombining chunks weights each
by its share of that mean's **denominator** — which is scored tokens for `compute_loss`/`compute_sft_loss` and pair
rows for `compute_dpo_loss` (`chunk_reduction_units`). Rows stand in for tokens only when every row contributes the
same number, true for pretrain and DPO and *false for SFT*, where supervised lengths vary per example: weighting SFT
by rows would quietly make a split step optimise a row-weighted average of per-chunk token-means instead of the
micro-batch's token-mean. `tests/test_accumulation.py` pins chunked == un-split for all three modes, with a
deliberately ragged SFT mask so the row-proportional version fails there and only there.

`evaluate()` honors the same ceiling and the same weighting, splitting its batches the way the step loop splits a
micro-batch — so a backoffed run's DPO eval (whose concatenated chosen+rejected forward is 2x the trained width)
doesn't OOM on a forward training no longer attempts.

**Tier two** triggers once `micro_chunk_size` has bottomed out at `1` and a CUDA OOM still hits (so shrinking further
can't help): the AdamW optimizer is swapped for `CPUOffloadAdamW`, keeping `exp_avg`/`exp_avg_sq` in pinned CPU
memory instead of on `device`, permanently freeing ~2x `num_params` fp32 bytes for the rest of the run. Model
parameters, gradients and activations all stay GPU-resident, and only a grad copy (down) and the resulting update
(up) cross PCIe, once per optimizer step rather than once per forward/backward — so this doesn't touch the forward
path or invalidate `torch.compile`'s captured graph. `migrate_optimizer_to_cpu_offload` preserves in-flight momentum
by migrating each param's existing `exp_avg`/`exp_avg_sq`/`step` rather than resetting it, and the LR scheduler is
rebuilt against the new optimizer with `last_epoch` set to the current step so the trajectory continues
uninterrupted. See [optim.md](optim.md) for the `MuonWithAuxAdam` variant of the swap.

Both tiers are sticky, and both are scoped to `auto_batch_size`: with it `False`, a CUDA OOM always ends the run
cleanly, so a manually-chosen or swept `batch_size` behaves exactly as configured.

The handler also resets the `GradScaler`'s per-optimizer bookkeeping via `update(get_scale())` before retrying: an
OOM at or after `scaler.unscale_()` otherwise leaves the optimizer marked as already-unscaled, so the retry's own
`unscale_` raises `"unscale_() has already been called on this optimizer since the last update()"` — an uncaught
crash exactly when the backoff was supposed to save the run. Only reachable under `dtype: fp16`, since bf16/fp32 run
with the scaler disabled; `update(get_scale())` rather than a bare `update()` so the aborted step isn't mistaken for
a successful one and used to grow the scale.

### Precision

`cfg.train.dtype` (`"fp32"`, `"fp16"`, `"bf16"`, resolved via `resolve_dtype`; `"nvfp4"` is sugar, see
[nvfp4.md](nvfp4.md)) runs the forward/loss pass under `torch.autocast` in that dtype while master weights and
optimizer state stay fp32. A `torch.amp.GradScaler` is enabled only for `fp16` — its narrow exponent range can
underflow small gradients, where bf16 has fp32's exponent range and needs no scaling.

---

# `generate.py`

`load_checkpoint(path, device)` reconstructs a `DenseTransformer` + tokenizer via
`model.load_transformer_from_checkpoint` (`torch.load(..., weights_only=False)`, since the checkpoint pickles the
full `Config` object, not just tensors) plus `build_tokenizer`. The reconstruction itself lives in the model
package (`model/load.py`) for import-direction reasons — see [post-training.md](post-training.md).

`generate(...)` runs autoregressive sampling (`--temperature`, `--top-k`; `--temperature 0` for greedy) using a
`KVCache` from `DenseTransformer.new_kv_cache()`: the (possibly truncated) prompt is prefilled once, then each later
step forwards only the newly sampled token against cached past keys/values.

Because `blocks[1:]` is a weight-shared loop body re-invoked `loop_count` (or `max_loops`) times per forward call —
always the full iteration count regardless of per-token halting, since attention is unconditionally dense every
iteration — a single K/V-per-layer cache isn't enough: the same block produces different K/V on each iteration since
it is fed an evolving hidden state. So `KVCache` has one slot per (block, iteration) pair actually executed
(`1 + loop_multiplier * (n_layers - 1)` slots), assigned implicitly by call order (`begin_step()`/`write()`) rather
than an explicit index threaded through every call. `RotaryEmbedding.forward` takes an `offset` so a cached token's
RoPE position reflects its true absolute position rather than always starting from 0.

An overlong prompt is truncated to the trailing `max_seq_len` tokens once at the start; once generation would push
the *total* sequence length past `max_seq_len` it raises a clear error rather than silently sliding the context
window forever.

`--loops N` overrides the iteration count for the whole generation, sizing the KV cache to match. For a model trained
with stochastic loop depth this is test-time compute scaling: the same weights can spend more per token at inference
than any training step did. Per-iteration parameter banks (RMSNorm gains, router biases) clamp at their last entry
rather than wrapping, so running deeper than training reuses the deepest learned parameters instead of cycling back
to the shallow ones.

`load_checkpoint` passes no `eos_id`, so document masking is simply off during generation — a single prompt is one
document, and the mask would be all-ones anyway.
