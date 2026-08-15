# `config.py` — schema and the defaults convention

Dataclass schema (`DataConfig`, `ModelConfig`, `TrainConfig`, `WandbConfig`, `SFTConfig`, `DPOConfig` nested in
`Config`) plus `load_config(path)`. This is the single source of truth for every tunable; a new hyperparameter is
added here first, then threaded through. Plain dataclasses, not OmegaConf/Hydra — no CLI overrides, no config
composition, one YAML file per run.

A pickled `Config` predating a field is schema-evolved at unpickle time: pickle restores `__dict__` without running
`__init__`, so `Config.__setstate__` backfills each missing field with its *current* default, exactly the way
`load_config` treats an omitted YAML key. A future *required* field raises a clear `ValueError` instead. This is what
keeps old checkpoints loadable everywhere.

## Ratio fields

Several `ModelConfig`/`TrainConfig` fields are stored as ratios and expose the absolute quantity as a read-only
derived property of the same name minus the suffix, so nothing downstream (including `vars(cfg.model)` for W&B
logging) needs to distinguish the two:

| stored | derived |
|---|---|
| `model.head_dim` | `n_heads = d_model // head_dim` |
| `model.ffn_mult` | `ffn_dim = round(d_model * ffn_mult)` |
| `train.warmup_ratio` | `warmup_steps = round(max_steps * warmup_ratio)` |

This keeps those quantities meaningful when sweeping `d_model` or `max_steps` instead of silently decoupling.

## Defaults convention: features default *on*

A new capability ships enabled unless there's a clear reason not to. What makes that safe is that the *parameters*
default to an inert setting, so every existing config keeps training exactly as it did and the feature only engages
once someone configures it or training moves its weights. In practice that means one of:

- **zero-initialised**, so the term contributes exactly nothing at init and can only be learned into —
  `loop_input_injection`'s `W_inj`, the `IterLoRA` adapters' `B`, the router `iter_bias` tensors,
  `hyper_conn_dynamic`'s `dyn_proj`;
- **identity-valued**, chosen so the arithmetic is exact — `value_residual`'s λ starts at exactly 1.0, and
  `attn_out_gate` is written `2 * sigmoid(zero_init)` precisely because a plain `sigmoid` cannot reach 1.0;
- **range-collapsed**, where a `None` (or unit scalar) resolves to the existing quantity — `loop_count_min/max`
  collapse to `loop_count`, `mup_base_d_model` to `d_model` (making every muP correction exactly 1.0),
  `moe_expert_ffn_mult` to `ffn_dim`, `embed_lr` to `lr`, `hyper_conn_streams: 1` back to a single residual stream.

### Six reasons a feature stays off

1. **It costs memory.** `mtp_heads` stays `1` — each extra head materialises a full `(batch, seq, vocab_size)`
   logits tensor, so defaulting to 2 would quietly halve what `auto_batch_size` can fit.
2. **It costs time even while inert.** `hyper_conn_streams` stays `1`: `n` times the residual stream's activation
   memory *and* 30-40% of step time in the looped regime. Inert is not the same as free.
3. **It changes a tuned quantity.** `lr_schedule` stays `"cosine"` — WSD is an operational convenience, not a
   quality win, and switching it would silently reshape the LR trajectory of every config whose `lr` was tuned
   against cosine.
4. **It is an approximation you reach for deliberately.** `loop_bptt_window` stays `None` (truncating the gradient).
5. **The measurement said no.** `moe_counterfactual_weight` (`0.0`) and `moe_balance_signal` (`"count"`) are both
   free and both inert at their defaults, so they would default on under the rule above. They don't, because the A/B
   came out neutral and negative respectively. Keeping them means keeping the recorded result and the diagnostic
   that explains it (see [results.md](results.md)), not keeping a recommendation.
6. **There is no inert setting at all.** `fp4_linear` is the first instance. Everything above assumes a default
   under which the feature is a mathematical no-op; no configuration quantizes every hidden matmul to 4 bits and
   still produces bit-identical logits, so the inertness question doesn't arise. That puts it in `use_diff_attn`'s
   category structurally, and the cost argument then applies on top: FP4 measures **0.34x at `d_model: 256`**, so
   defaulting it on would make `configs/tinystories.yaml` three times *slower*. Consequence for testing:
   `tests/test_inert_defaults.py`'s pattern does not apply, so the contract `tests/test_nvfp4.py` pins is
   *structural* — with the flag off, no `FP4Linear` exists anywhere in the tree.

The distinction in 1-4 is cost and reversibility, not novelty: a feature whose "on" state is free and inert defaults
on; one that spends real memory or changes a tuned quantity doesn't.

### Five settings that change results at their defaults

`doc_attention_mask`, `optimizer: muon`, `z_loss_weight`, MoE's `moe_n_shared`/`moe_balance`, and `train.lr`. Each is
a straightforward improvement rather than an experiment — see [results.md](results.md).

`train.lr` is the odd one out and the one to be careful with: it is the only *tuned quantity* on that list rather
than a feature flag — exactly the category reason 3 above says shouldn't change by default. It changed anyway
because it had become simply wrong (`3.0e-4` -> `1.0e-2`; see [optim.md](optim.md) for the sweep). Two consequences.
It reaches only configs that **omit** `lr`, which today means the `sweep*`/`super*` ones — every worked example in
`configs/` pins `lr: 3.0e-4` and so still trains at the old value until edited. And the 400-step baselines in
[results.md](results.md) predate it, so reproducing them means pinning `lr: 3.0e-4` explicitly.
