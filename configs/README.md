# Radiance Config Reference

Configs are plain YAML files that partially override the defaults defined in `src/radiance/config.py`.
Only the fields you need to change — everything else falls back to the dataclass default.

## Top-level

- **`run_name`** — human-readable label for W&B and stdout; does not affect training.

## data

- **`dataset`** — HuggingFace `user/dataset` identifier with `train` (and optionally `validation`) splits.
- **`text_column`** — name of the column containing text in the dataset.
- **`tokenizer`** — HuggingFace tokenizer name or path (e.g. `"gpt2"`).
- **`seq_len`** — number of tokens per packed sequence; should match `model.max_seq_len`.
- **`num_workers`** — DataLoader worker processes; set higher to overlap I/O with training.
- **`cache_dir`** — directory for tokenized+packed cache, keyed by a hash of dataset/tokenizer/text_column/seq_len. Set to `null` to disable caching.
- **`streaming`** — use HF streaming mode instead of full download + disk cache; trades deterministic shuffling for a shuffle-buffer approximation. Ignores `cache_dir` unless `disk_cache_max_gb` is also set.
- **`shuffle_buffer_size`** — size of the shuffle buffer in streaming mode.
- **`disk_cache_max_gb`** — bounded on-disk cache (decimal GB) on top of streaming, so repeated short runs don't re-fetch already-seen data. Ring-buffer per worker, evicts oldest shards first.
- **`disk_cache_shard_size`** — number of packed blocks per shard file flushed to disk; keep well below a short run's block count or nothing gets cached.
- **`prefetch_factor`** — how many batches each DataLoader worker stages ahead; higher values better overlap fetch with the forward/backward pass.
- **`eval_split_size`** — carve this many examples off the front of `train` as validation when no `validation` split exists; `0` disables. No-op when a real validation split is present.

## model

- **`d_model`** — hidden dimension of the transformer.
- **`head_dim`** — attention head size; `n_heads` is derived as `d_model // head_dim`.
- **`n_kv_heads`** — number of K/V heads for grouped-query attention; `None` means standard MHA (one K/V head per query head). Must evenly divide `n_heads`.
- **`qk_norm`** — RMSNorm on each head's Q and K before RoPE, for training stability across loop iterations. Defaults on; inert at init.
- **`value_residual`** — learned mix of each block's values with `blocks[0]`'s (`v = λ·v + (1−λ)·v_first`), giving every loop iteration access to the first block's values. λ starts at 1.0, so inert at init.
- **`attn_out_gate`** — per-head sigmoid gate on attention output before `out_proj`, written `2 * sigmoid(zero_init)` so it is exactly 1.0 at init.
- **`mtp_heads`** — number of multi-token-prediction heads (1 = ordinary next-token); each extra head predicts one token further ahead, reusing the trunk's `lm_head`. Stays 1 by default because each head materialises a full `(batch, seq, vocab_size)` tensor.
- **`mtp_weight`** — coefficient on the averaged auxiliary MTP-head loss.
- **`z_loss_weight`** — coefficient on the log-Z regulariser `mean(logsumexp(logits)²)`; keeps logit scale from drifting, especially important under looping. Applied to training loss only.
- **`n_layers`** — number of transformer blocks; `blocks[0]` runs once, `blocks[1:]` form a weight-shared loop body.
- **`loop_count`** — number of times the shared loop body re-runs; 1 means no looping.
- **`loop_iter_conditioning`** — how the loop body distinguishes iterations: `"norm_gains"` (per-iteration RMSNorm gains), `"lora"` (per-iteration adapters), or `"none"`. Inert when `loop_count: 1`.
- **`loop_lora_rank`** — rank of per-iteration LoRA adapters when `loop_iter_conditioning: "lora"`.
- **`loop_input_injection`** — re-injects `blocks[0]`'s output at the start of each loop iteration via a zero-initialised `W_inj`, stabilising deep recurrences. Required by `loop_bptt_window`.
- **`loop_count_min` / `loop_count_max`** — sample loop count uniformly from this range each training step for stochastic depth; both `None` collapses to `loop_count`. Eval/generation always use `loop_count_max`. Each distinct count compiles a separate graph.
- **`loop_bptt_window`** — backpropagate through only the last N loop iterations, making activation memory O(N). Requires `loop_input_injection` or it raises.
- **`use_router`** — replace fixed `loop_count` with per-token ACT halting via a sigmoid router.
- **`max_loops`** — hard cap on loop iterations in router mode.
- **`ponder_weight`** — coefficient on the ACT ponder-cost loss term (encourages tokens to halt sooner).
- **`halt_epsilon`** — a token halts once cumulative halting probability ≥ `1 - halt_epsilon`.
- **`act_capacity_ratio`** — fraction of positions computed per ACT iteration below 1.0; only the highest-priority still-running positions go through the full block, saving wall-clock and memory. 1.0 is dense. Training-only, incompatible with `grad_checkpoint`.
- **`act_ffn_capacity_ratio`** — older, narrower version that sparsifies only the FFN sublayer. Superseded by `act_capacity_ratio`; setting both below 1.0 raises.
- **`ffn_mult`** — FFN expansion factor; `ffn_dim = round(d_model * ffn_mult)`.
- **`ffn_depth`** — number of hidden `Linear + GELU` layers between FFN up- and down-projections.
- **`use_moe`** — use `MoEFeedForward` instead of `FeedForward` in `blocks[1:]`; `blocks[0]` stays dense.
- **`n_experts`** — number of experts per MoE layer.
- **`moe_top_k`** — number of experts activated per token (Mixtral-style weighted top-k).
- **`moe_capacity_factor`** — per-expert capacity as a multiplier of expected load; excess tokens are dropped.
- **`moe_aux_loss_weight`** — coefficient on the load-balancing auxiliary loss (`n_experts * sum(fᵢ * Pᵢ)`).
- **`moe_balance`** — how load is balanced: `"aux_loss"` (gradient term), `"bias"` (non-learned per-expert bias on routing logits), or `"both"`.
- **`moe_bias_update_rate`** — step size for the `"bias"` / `"both"` balancing update.
- **`moe_n_shared`** — always-on expert(s) added to every token alongside routed ones; absorbs common computation. `0` disables.
- **`moe_shared_ffn_mult`** — shared expert width as a fraction of `ffn_dim`.
- **`moe_expert_ffn_mult`** — each routed expert's width as a fraction of `ffn_dim`; `None` means full `ffn_dim`. Use with many experts for fine-grained MoE.
- **`moe_eval_full_capacity`** — at eval/generation, size capacity to actual load so no token is dropped; prevents batch-dependent outputs.
- **`moe_dense_every`** — keep every Nth block in `blocks[1:]` dense (1-indexed) even when `use_moe: true`. `None` = all MoE.
- **`mup_base_d_model`** — muP base width for hyperparameter scaling across model widths. `None` resolves to `d_model` (inert). Set to the proxy width your LR was tuned at.
- **`dropout`** — attention-weight dropout. Set to `0.0` when using `doc_attention_mask` (flex_attention has no dropout).
- **`max_seq_len`** — maximum context length.
- **`rope_theta`** — RoPE base frequency.
- **`doc_attention_mask`** — mask attention at document boundaries in packed sequences so tokens cannot attend across unrelated documents. Uses `flex_attention` on CUDA; falls back to plain SDPA on CPU and during generation.
- **`loop_attn_windows`** — per-iteration attention window sizes (e.g. `[128, 128, 512, 512]` for local-then-global). Requires `doc_attention_mask` (rides the same flex_attention machinery). `None` = fully global.
- **`use_nsa`** — opt-in DeepSeek NSA-style learned block-sparse attention: a coarse "compression" branch over mean-pooled KV blocks plus a fine "selection" branch over the top-k historical blocks a learned score picks per query token (plus that token's own local block), combined by a per-token gate. Incompatible with `doc_attention_mask`, `loop_attn_windows`, and `act_capacity_ratio`/`act_ffn_capacity_ratio` below 1.0 — see `DenseTransformer.__init__` for why.
- **`nsa_block_size`** — granularity of compression/selection. Must stay `128` (the default) unless you've confirmed a different value's `flex_attention` kernel actually compiles on your GPU.
- **`nsa_top_k_blocks`** — non-local historical blocks the selection branch attends to per query token, per head, beyond that token's own local block.
- **`grad_checkpoint`** — recompute activations during backward instead of storing them; trades ~20-25% throughput for large memory savings, especially under looping. Training-only.
- **`vocab_pad_multiple`** — round vocab size up to this multiple for tensor-core alignment on `lm_head`. `1` disables. Padding rows are unreachable by any tokenizer id.

## train

- **`batch_size`** — micro-batch size per forward/backward pass.
- **`grad_accum_steps`** — micro-batches accumulated per `optimizer.step()`; `effective_batch_size = batch_size * grad_accum_steps`. Raise this to grow effective batch beyond what fits in VRAM.
- **`lr`** — AdamW learning rate; under Muon it governs only embeddings, norms, biases, and routers.
- **`optimizer`** — `"muon"` (default, Newton-Schulz orthogonalisation on hidden weights) or `"adamw"`.
- **`muon_lr`** — learning rate for the Muon group only; ~50x larger than AdamW's since the spectral norm is normalised.
- **`muon_momentum`** — momentum coefficient for the Muon group.
- **`weight_decay`** — weight decay for AdamW groups.
- **`warmup_ratio`** — fraction of `max_steps` spent warming up; `warmup_steps = round(max_steps * warmup_ratio)`.
- **`min_lr_ratio`** — schedule decays to this fraction of `lr` rather than to 0. `0.0` restores decay-to-zero.
- **`lr_schedule`** — `"cosine"` (default) or `"wsd"` (warmup-stable-decay: hold full LR, decay only the final `wsd_decay_ratio`).
- **`wsd_decay_ratio`** — fraction of `max_steps` spent in the final decay phase when `lr_schedule: "wsd"`.
- **`max_steps`** — training step count; ignored if `tokens_per_param` is set.
- **`tokens_per_param`** — derive `max_steps` from model size instead; Chinchilla-optimal is ~20. Overwrites `max_steps` once the model is built.
- **`auto_batch_size`** — compute `batch_size`/`grad_accum_steps` from free VRAM and model size at startup. Defaults on; makes the actual micro-batch safer, never bigger. Also enables OOM backoff. CUDA-only.
- **`target_effective_batch_size`** — effective batch size `auto_batch_size` solves for; `None` preserves the configured `batch_size * grad_accum_steps`.
- **`vram_safety_margin`** — fraction of estimated VRAM budget to use when `auto_batch_size` is on; lower is more conservative.
- **`grad_clip`** — max gradient norm for clipping.
- **`log_every`** — log `train/loss` to stdout and W&B every N steps.
- **`eval_every`** — run evaluation every N steps.
- **`eval_max_batches`** — cap batches per evaluation; `null` walks the full validation split (unbounded for streaming).
- **`save_every`** — checkpoint every N steps.
- **`output_dir`** — directory for checkpoints.
- **`resume_from`** — path to resume from, or `"auto"` for the highest-numbered checkpoint in `output_dir`. Restores weights, optimizer state, LR schedule, and GradScaler.
- **`seed`** — random seed.
- **`device`** — `"auto"` (CUDA > MPS > CPU), `"cuda"`, `"mps"`, or `"cpu"`.
- **`compile`** — enable `torch.compile` on the model.
- **`dtype`** — precision: `"fp32"`, `"fp16"`, or `"bf16"`. Forward runs under `torch.autocast`; master weights and optimizer stay fp32. `GradScaler` is enabled only for `fp16`.

## wandb

- **`project`** — W&B project name.
- **`entity`** — W&B entity/username; `null` logs to your default.
- **`mode`** — `"online"`, `"offline"`, or `"disabled"`.

## radiance-generate CLI args

These are not YAML config knobs but command-line flags for `uv run radiance-generate`:

- **`--checkpoint`** — path to a `.pt` checkpoint (required); the model, tokenizer, and config are rebuilt from the embedded checkpoint data.
- **`--prompt`** — text prompt to generate from; default `"Once upon a time"`.
- **`--max-new-tokens`** — maximum number of tokens to generate; default 200.
- **`--temperature`** — sampling temperature; `0` for greedy decoding. Default 0.8.
- **`--top-k`** — top-k sampling; `0` disables. Default 50.
- **`--loops N`** — override the checkpoint's loop count for this generation; for models trained with stochastic loop depth this enables test-time compute scaling. Per-iteration parameter banks clamp at their last entry rather than wrapping.
- **`--device`** — device to run on; default `"auto"`.
- **`--seed`** — random seed for reproducible generation.

## W&B sweep configs

Sweep configs (e.g. `sweep.yaml`, `sweep_all.yaml`) are W&B sweep definitions, not training configs. They layer sampled values on top of a base config via `src/radiance/sweep.py`.

- **`program`** — entry point; always `src/radiance/sweep.py`.
- **`method`** — W&B sweep method: `"bayes"` (Bayesian optimization).
- **`project`** — W&B project name.
- **`metric.name`** — metric to optimize; always `val/loss`.
- **`metric.goal`** — optimization direction; always `minimize`.
- **`command`** — command template; invokes `sweep.py --config <base-config>` with W&B-injected env vars.
- **`parameters`** — nested under `model.parameters` and `train.parameters`; each key is a config knob with a W&B distribution (`uniform`, `int_uniform`, `log_uniform_values`, or discrete `values`). `loop_iter_conditioning` is encoded as integers (`0="none"`, `1="norm_gains"`, `2="lora"`) since Bayesian method doesn't support string values; `sweep.py` maps them back.
