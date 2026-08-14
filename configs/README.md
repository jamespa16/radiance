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
- **`hyper_conn_streams`** — expansion rate `n` for hyper-connections: the residual stream becomes `n` parallel streams, and each sublayer learns which stream to read, how the streams mix, and how its output is distributed back across them. `1` (default) is exactly one residual stream and allocates nothing. At `n > 1` the model still starts as the plain residual network (one-hot read, identity mix, all-ones write, identical streams), but costs `n` times the residual stream's activation memory — which is why it does not default on. Aimed at the weight-shared loop, where per-iteration variants give each pass its own routing. See `configs/tinystories_hyper.yaml`.
- **`hyper_conn_dynamic`** — additionally condition the connection coefficients on the hidden state (`static + s·tanh(norm(H) W)`, `W` zero-initialised). Bit-identical to static hyper-connections at init; free at `hyper_conn_streams: 1`, where none exist.
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
- **`grad_checkpoint`** — recompute activations during backward instead of storing them; trades ~20-25% throughput for large memory savings, especially under looping. Training-only.
- **`vocab_pad_multiple`** — round vocab size up to this multiple for tensor-core alignment on `lm_head`. `1` disables. Padding rows are unreachable by any tokenizer id.

## train

- **`batch_size`** — micro-batch size per forward/backward pass.
- **`grad_accum_steps`** — micro-batches accumulated per `optimizer.step()`; `effective_batch_size = batch_size * grad_accum_steps`. Raise this to grow effective batch beyond what fits in VRAM.
- **`lr`** — AdamW learning rate; under Muon it governs only embeddings, norms, biases, and routers.
- **`optimizer`** — `"muon"` (default, Newton-Schulz orthogonalisation on hidden weights) or `"adamw"`.
- **`muon_lr`** — learning rate for the Muon group only; ~50x larger than AdamW's since the spectral norm is normalised.
- **`muon_momentum`** — momentum coefficient for the Muon group.
- **`hyper_conn_lr`** — learning rate for the hyper-connection coefficients alone (default `1.0e-3`); only consulted when `model.hyper_conn_streams > 1`, so it is inert otherwise. Not `null`-by-default like `embed_lr`, because AdamW's step is ~`lr` per step regardless of gradient scale and these coefficients are *structural* (one-hot read, identity mix) — at `lr`'s post-Muon `1e-2` a few hundred steps erase that structure instead of refining it.
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
- **`init_from`** — path to a checkpoint to seed *model weights only* from (e.g. starting SFT from a pretrained checkpoint); unlike `resume_from`, the optimizer/scheduler/step are not restored, so this run gets a fresh optimizer/scheduler and starts at step 0. Ignored whenever `resume_from` finds a checkpoint. `None` = fresh init.
- **`seed`** — random seed.
- **`device`** — `"auto"` (CUDA > MPS > CPU), `"cuda"`, `"mps"`, or `"cpu"`.
- **`compile`** — enable `torch.compile` on the model.
- **`dtype`** — precision: `"fp32"`, `"fp16"`, or `"bf16"`. Forward runs under `torch.autocast`; master weights and optimizer stay fp32. `GradScaler` is enabled only for `fp16`.

## sft

Post-training: supervised fine-tuning on chat/instruction data. A mode switch (like `use_moe`/`use_router`), not an inert default-on feature — enabling it swaps `train.py`'s data pipeline and loss function entirely. Pair with `train.init_from` to seed the model from a pretrained checkpoint; see `configs/tinystories_sft.yaml` for a worked example.

- **`enabled`** — route `train.py` through `build_sft_dataloaders`/`compute_sft_loss` instead of the pretrain path. Default `false`.
- **`dataset`** — HF `user/dataset`-style instruction dataset. Required when `enabled`.
- **`messages_column`** — column holding `[{"role", "content"}, ...]` chat turns (the standard shape for chat-formatted HF datasets, e.g. `HuggingFaceH4/no_robots`). Ignored if `instruction_column` is set instead.
- **`instruction_column` / `input_column` / `output_column`** — Alpaca-style fallback for datasets with separate instruction/input/output columns rather than a `messages` column; a 2-turn user/assistant list is built from these instead.
- **`seq_len`** — packed block width for SFT; `None` resolves to `data.seq_len`.
- **`cache_dir`** — tokenized+packed cache for the SFT dataset, separate from `data.cache_dir` so it can never collide with a pretrain cache entry.
- **`eval_split_size`** — same semantics as `data.eval_split_size`, applied to the SFT dataset.
- **`user_prefix` / `assistant_prefix`** — plain text turn markers (e.g. `"\n\nUser: "` / `"\n\nAssistant: "`) joined onto each turn before tokenizing. Not new special tokens — they tokenize through the existing vocab, so no embedding resize is needed. Assistant turns (and the trailing EOS) are what `compute_sft_loss` scores; user/system turns are masked out of the loss but still attended to causally.

## dpo

Post-training: Direct Preference Optimization on `(prompt, chosen, rejected)` triples. A second mode switch alongside `sft` — mutually exclusive with `sft.enabled` (`train.py` raises if both are set) — swapping `train.py`'s data pipeline and loss function the same way `sft.enabled` does, and reusing everything else (optimizer, LR schedule, `auto_batch_size`, checkpointing, `init_from`) identically. Unlike SFT, DPO's loss needs a frozen reference model's log-probabilities on the same sequences; those are precomputed once during data prep and cached to disk alongside the tokenized dataset, so training itself only ever holds the policy model in memory. See `configs/tinystories_dpo.yaml` for a worked example (a three-stage pretrain -> SFT -> DPO chain).

Each `(prompt+chosen)`/`(prompt+rejected)` sequence is packed as its own row, independently padded to `seq_len` with trailing `eos_token_id` (loss-masked) — "packing of one," unlike `sft`'s many-examples-per-block packing — so a pair's two halves can never be separated by `DataLoader` shuffling. This needs no `model.py` changes: attention is strictly causal and the real content always precedes its own padding within a row, so the padding can never influence the scored tokens' logits.

- **`enabled`** — route `train.py` through `build_dpo_dataloaders`/`compute_dpo_loss_from_logits` instead of the pretrain/SFT path. Default `false`.
- **`dataset`** — HF `user/dataset`-style preference dataset. Required when `enabled`.
- **`prompt_column`** — `null` (default): `chosen_column`/`rejected_column` are full `[{"role", "content"}, ...]` message lists that already include the prompt turn (e.g. `argilla/dpo-mix-7k`). Set: `chosen_column`/`rejected_column` are plain completion strings, combined with `prompt_column` (+ optional `system_column`) into a shared 1-turn user prompt (e.g. `Intel/orca_dpo_pairs`, which has separate `system`/`question`/`chosen`/`rejected` string columns — set `prompt_column: question`, `system_column: system`).
- **`system_column`** — only consulted when `prompt_column` is set.
- **`chosen_column` / `rejected_column`** — column names for the preferred/dispreferred side. Default `"chosen"`/`"rejected"`.
- **`seq_len`** — packed row width for DPO; `null` resolves to `data.seq_len`. A pair with either side exceeding this is dropped, not truncated (truncating the completion would score a partial response as fully chosen/rejected; truncating the prompt would silently change what's being conditioned on).
- **`cache_dir`** — tokenized+packed+reference-scored cache for the DPO dataset, separate from `data.cache_dir`/`sft.cache_dir`. The cache key includes the reference checkpoint's path/mtime/size, so changing `reference_checkpoint` invalidates it.
- **`eval_split_size`** — same semantics as `data.eval_split_size`, applied to the DPO dataset.
- **`beta`** — DPO temperature / regularization strength (Rafailov et al. 2023). Default `0.1`.
- **`reference_checkpoint`** — path to a frozen checkpoint whose log-probabilities anchor the DPO loss. Required when `enabled`. Often, but not necessarily, the same checkpoint `train.init_from` seeds the policy from.
- **`reference_batch_size`** — batch size for the one-time reference-logprob precompute pass. Default `32`.
- **`user_prefix` / `assistant_prefix`** — same plain-text turn-marker convention as `sft.user_prefix`/`sft.assistant_prefix`.

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
- **`--chat`** — wrap `--prompt` in the checkpoint's SFT turn template (`sft.user_prefix`/`assistant_prefix`, read back off the checkpoint's saved config) before generating. Requires a checkpoint trained with `sft.enabled: true`.

## W&B sweep configs

Sweep configs (e.g. `sweep.yaml`, `sweep_all.yaml`) are W&B sweep definitions, not training configs. They layer sampled values on top of a base config via `src/radiance/sweep.py`.

- **`program`** — entry point; always `src/radiance/sweep.py`.
- **`method`** — W&B sweep method: `"bayes"` (Bayesian optimization).
- **`project`** — W&B project name.
- **`metric.name`** — metric to optimize; always `val/loss`.
- **`metric.goal`** — optimization direction; always `minimize`.
- **`command`** — command template; invokes `sweep.py --config <base-config>` with W&B-injected env vars.
- **`parameters`** — nested under `model.parameters` and `train.parameters`; each key is a config knob with a W&B distribution (`uniform`, `int_uniform`, `log_uniform_values`, or discrete `values`). `loop_iter_conditioning` is encoded as integers (`0="none"`, `1="norm_gains"`, `2="lora"`) since Bayesian method doesn't support string values; `sweep.py` maps them back.
