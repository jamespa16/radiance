# TODO: native low-precision parameter/optimizer storage

Today, `cfg.train.dtype` only controls `torch.autocast`'s compute dtype (`train.py`'s forward/loss
pass). Parameters, gradients, and AdamW's `exp_avg`/`exp_avg_sq` state always stay fp32 regardless
of the setting — nothing in `model.py` casts dtype, and `.to(device)` in `train.py` doesn't change
it either. This is the standard mixed-precision recipe (fp32 master weights for numerical
stability) and is what CLAUDE.md currently documents as intentional. But it also means a `bf16`
config saves *only* activation memory, not the ~12 bytes/param (grad + 2 Adam buffers, all fp32)
that dominate a large model's static VRAM footprint — likely why `configs/fineweb_500m.yaml`
needed `batch_size` tuned down as far as it did.

Three options, not yet decided:

1. **Keep as-is.** fp32 master weights matches standard practice, is safe for `fp16` (avoids
   underflow-stalled updates), and is already implemented/documented. Con: `bf16` configs don't
   get the VRAM savings a user would reasonably expect from setting `dtype: bf16`.

2. **Add an opt-in native low-precision mode**, e.g. `cfg.train.native_bf16: bool = False` (bf16
   only — fp16's narrow exponent range makes native storage without a master copy meaningfully
   riskier, so `fp16` would keep today's fp32-master behavior unconditionally). When on, build
   `DenseTransformer` directly in bf16 (or `.to(dtype=torch.bfloat16)` after construction) so
   params/grads/AdamW state all end up bf16 — roughly halves static memory (fp32:
   4+4+4+4=16 bytes/param → bf16: 2+2+2+2=8 bytes/param). Con: real numerical-stability question,
   not just a memory optimization — needs empirical validation (loss-curve comparison against the
   fp32-master baseline) before trusting it for real runs; PyTorch's `AdamW` allocates
   `exp_avg`/`exp_avg_sq` matching the parameter's dtype automatically, so no optimizer-side code
   change is needed beyond the model's dtype, but accuracy risk is on the implementer to verify.

3. **Fold into the batch-size auto-sizing work** so the memory formula is precision-mode-aware from
   day one. Rejected for now (scope creep on top of an already-nontrivial plan) — but whichever
   option gets picked later, the auto-sizing formula's "bytes per param" calculation should be
   written as an isolated, easily-extended function so a future dtype-mode decision doesn't require
   re-deriving the whole formula.

No recommendation yet — needs a decision on how much accuracy risk is acceptable for the memory
win before committing to option 2.
