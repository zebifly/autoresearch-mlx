# context-nanochat.md

Domain context for autoresearch on the nanochat MLX fork (Apple Silicon). Read this once at setup. Refer back when stuck, rotating families, or exploring deeper knobs.

**Important**: this fork uses MLX (not PyTorch) and AdamW only (no Muon). All knob names and values below are verified against the actual `train.py` in this repo.

## Baseline reference

Hardware: Mac mini M2 Pro, 32 Go unified memory.
Baseline val_bpb: 2.001726 (depth 4, 176 steps, 11.5M params, 11.5M tokens in 5 min).
Peak memory: ~21 GB on a 32 GB unified-memory system. This suggests some headroom for moderate increases, but unified memory is shared with the OS and other processes — do not treat the remaining ~11 GB as cleanly available to the model.

## Search space map

Five experiment families. Each tagged by depth (cost/risk of changes).

**Depth rule**: explore deep knobs only after surface and medium knobs show diminishing returns, unless a confirmed finding points directly to a deeper knob.

### 1. Optimizer tuning — SURFACE

Highest-density area for early gains. All knobs are top-level constants in `train.py`.

**Learning rates** (per-group, passed to AdamW):
- `MATRIX_LR` = 0.04 — transformer block 2D weights
- `EMBEDDING_LR` = 0.6 — token embedding (scaled by `dmodel_lr_scale`)
- `UNEMBEDDING_LR` = 0.004 — lm_head (scaled by `dmodel_lr_scale`)
- `SCALAR_LR` = 0.5 — resid_lambdas (at 0.01x) and x0_lambdas (at 1x)

**Betas and weight decay**:
- `ADAM_BETAS` = (0.8, 0.95) — shared across all groups except x0_lambdas which use (0.96, 0.95)
- `WEIGHT_DECAY` = 0.2 — applied to transformer block 2D matrices only. All other groups (embeddings, VE, lm_head, scalars) have weight_decay=0.0. Adding weight decay to other groups is an open axis.

**Schedule**:
- `WARMUP_RATIO` = 0.0 — no warmup. Adding warmup is a possible experiment.
- `WARMDOWN_RATIO` = 0.5 — last 50% of training decays the LR linearly.
- `FINAL_LR_FRAC` = 0.0 — LR decays to zero. Non-zero values keep a minimum LR.

**Training regime**:
- `TOTAL_BATCH_SIZE` = 2^16 (65536 tokens) — controls number of optimizer steps in 5 min. Halving doubles the steps.
- `DEVICE_BATCH_SIZE` = 16 — actual batch per forward pass. Gradient accumulation = `TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)`.

### 2. Initialization — SURFACE

All in `init_weights()` method of GPT class.
- Embedding (`wte`) init: normal, std=1.0
- lm_head init: normal, std=0.001
- Attention projections (c_q, c_k, c_v): uniform, bound = sqrt(3) / sqrt(n_embd)
- Output projections (c_proj, mlp.c_proj): zeros
- MLP c_fc: same uniform bound as attention (no 0.5x scale — differs from original)
- `resid_lambdas` init: 1.0
- `x0_lambdas` init: 0.1
- VE gate init: zeros

### 3. Regularization by module — SURFACE

- Weight decay per group — currently only transformer block 2D matrices have WD (0.2). Embeddings, VE, lm_head, and scalars all have WD=0.0. Adding small WD to specific groups is an open axis.
- No logit softcap in this fork (differs from original which uses softcap=15).
- No dropout in the code.

### 4. Attention geometry — MEDIUM

Changes the information flow pattern. Can unlock gains but also destabilize.
- `WINDOW_PATTERN` = "SSSL" — sliding window attention pattern tiled across layers
- RoPE base theta = 10000 (differs significantly from original which uses 100000)
- No QK post-norm scaling in this fork (original uses q,k *= 1.15)
- VE gate channels = 32 (original uses 12)
- VE gate scale = 2x sigmoid (original uses 3x)
- `n_kv_head` = `n_head` — no GQA (heads are equal)

### 5. Architecture sizing — MEDIUM

Trades compute budget allocation.
- `DEPTH` = 4 — primary complexity knob. Depth 5 is a reasonable early test; depth 6 may be possible but should not be assumed safe.
- `ASPECT_RATIO` = 64, `HEAD_DIM` = 128. Model width is derived: `model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM`
- At depth 4: model_dim = 256 (computed: (4*64+127)//128*128 = 256)
- MLP expansion is 4x (hardcoded in MLP class: `4 * config.n_embd`)
- VE gate channels = 32 (hardcoded in CausalSelfAttention)

## Known heuristics

### Confirmed upstream (Karpathy sessions on H100, PyTorch + Muon)

These were observed in a different codebase on much larger compute. They are directional priors, not guaranteed to transfer to this MLX/AdamW setup. Verify before treating as ground truth.

- **VE weight decay helps**: small values (0.001–0.003) upstream. Relevant here since VE has WD=0.0, and VE params are already grouped separately in the AdamW wrapper — changing WD is a one-line edit.
- **Embedding init std < 1.0 helps**: 0.8 outperformed 1.0 upstream. This fork uses 1.0 — potential quick win.
- **Halving batch size helps** when it doubles optimizer steps in fixed time budget.
- **Per-group Adam betas matter** — shared global betas were suboptimal upstream. This fork uses shared betas (except x0).
- **RoPE 100K > 10K** upstream. This fork uses 10K — meaningful transfer candidate.
- **Post-QK-norm scaling improves attention** upstream (q,k *= 1.15). Not present in this fork — adding it is a medium-depth experiment.
- **Weight decay schedule: cosine > linear** upstream. This fork uses linear decay via `get_lr_multiplier()`.

### Confirmed locally (this run)

<!-- Populate as you reproduce or discover findings on this hardware. Promote findings here only after reproduction or clear repeated benefit on this hardware. -->

### Plausible leads

- Warmdown ratio 0.5–0.7 seemed to matter upstream; 0.65 worked in one session.
- Final LR fraction > 0 (e.g. 0.05) may help vs full decay to 0.
- Adding warmup (currently 0.0) may stabilize early training.
- MLP c_fc init at 0.5x scale helped upstream but this fork uses full scale — untested.
- `x0_lambdas` init sensitivity not explored.
- Logit softcap is absent in this fork — adding one may help, but it's a code addition not a knob tweak.

## Risk zones

### OOM patterns
- Peak memory ~21 GB on 32 GB unified memory. Some headroom exists but is shared with OS.
- Increasing depth from 4→6 without reducing batch size — test depth 5 first.
- Increasing `DEVICE_BATCH_SIZE` increases memory directly.

### MLX-specific risks
- MLX compilation behavior differs from PyTorch `@torch.compile`. Be cautious with control flow changes.
- MLX uses unified memory — no hard GPU OOM crash. Instead, the system swaps to disk and gets very slow. Watch for sudden slowdowns in tok/sec.
- `mx.eval()` calls control when computation actually happens. Misplacing them can cause memory buildup.

### Silent degradation
- Schedule changes can cause slow divergence visible only late in training.
- Mixing up which parameter groups get which treatment in the AdamW class.
- Changing gradient accumulation logic incorrectly (must keep `TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0`).

## Suggested opening sequence

After the baseline (val_bpb ~2.00), these 4 experiments give initial signal on sensitivity direction. They are not a reliable map — they tell you where to look next.

1. **Batch size halve**: `TOTAL_BATCH_SIZE = 2**15`. Tests whether more steps beats larger batches on this hardware.
2. **Embedding init std**: 1.0 → 0.8 in `init_weights()`. Tests a confirmed upstream finding.
3. **Depth +1**: `DEPTH = 5`. Tests if the memory headroom allows a bigger model. Watch peak memory and tok/sec.
4. **VE weight decay**: set `weight_decay: 0.001` in the `"value_embeds"` group in the AdamW `__init__`. VE params are already grouped separately — this is a one-line change, a low-friction surface experiment.

After these 4, you have directional signal on which surface knobs respond most on this specific hardware. Use that to prioritize.
