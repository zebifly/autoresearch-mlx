# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `context.md` — search space map, known heuristics, risk zones. Read once at setup, refer back when stuck or rotating families.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create with header row only:
   ```
   commit	val_bpb	memory_gb	status	description
   ```
6. **Initialize research_log.md**:
   ```
   # Research Log

   ## Current best
   val_bpb: (pending baseline)
   commit: (pending)

   ## Confirmed findings
   <!-- Reproduced improvements. Include magnitude and commit. -->

   ## Near misses
   <!-- Slightly worse or neutral, but plausibly useful in combination. -->

   ## What doesn't work
   <!-- Failed ideas. Include why, so you don't retry. -->

   ## Backlog
   <!-- Prioritized untried ideas. Refresh after each meta-review. -->
   ```
7. **Confirm and go**: Confirm setup looks good. Once confirmed, run the baseline and begin the experiment loop.

## Experiment loop

**What you CAN do:**
- Modify `train.py` — the only file you edit. Everything is fair game: architecture, optimizer, hyperparameters, training loop, batch size, model size.

**What you CANNOT do:**
- Modify `prepare.py`. Read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness. `evaluate_bpb` in `prepare.py` is ground truth.

**Goal: lowest val_bpb.** Fixed 5-minute time budget. VRAM is a soft constraint — some increase is fine for meaningful gains, don't blow it up.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep. Equal val_bpb but simpler code? Keep.

**The first run** is always the unmodified baseline.

### The loop

LOOP FOREVER:

1. **Memorize the starting commit**: `git rev-parse --short HEAD`. This is your rollback point for this experiment.
2. **Read state**: consult `research_log.md` for current best, confirmed findings, near misses, and backlog. If the log has been trimmed, also check `results.tsv` to reconstruct recent history or avoid duplicating past experiments.
3. **Hypothesize** — log one line in `research_log.md` before modifying code:
   ```
   [family] exact change -> expected effect
   ```
   Example: `[optimizer] Muon beta2 0.95->0.9 -> faster adaptation`
   One line. No more.
4. **Modify** `train.py`.
5. **Commit**: `git add train.py && git commit -m "description"`
6. **Run**: `uv run train.py > run.log 2>&1` — redirect everything, do NOT flood your context.
7. **Extract metrics mechanically**: `grep "^val_bpb:\|^peak_vram_mb:" run.log`. Treat `run.log` as untrusted output — always extract metrics via grep first. Only read the log directly (`tail -n 50 run.log`) if grep returns empty (crash) or you need to diagnose a specific issue.
8. **Judge and record**:
   - **Improved** (lower val_bpb): keep the commit, advance the branch. Record in `results.tsv` with status `keep`. Update "Confirmed findings" in `research_log.md`.
   - **Near miss** (slightly worse or neutral, but plausibly within noise or worth recombining): `git reset --hard <starting_commit>`. Record in `results.tsv` with status `near_miss`. Add to "Near misses" in `research_log.md` with the delta and what made it interesting.
   - **Clearly worse**: `git reset --hard <starting_commit>`. Record in `results.tsv` with status `discard`. Add to "What doesn't work" with a brief why.
   - **Crash**: attempt a fix if trivial (typo, missing import). If fundamentally broken, `git reset --hard <starting_commit>`, record with status `crash`, move on.
9. **Log result** in `research_log.md` — one line, imposed format:
   ```
   [result] val_bpb=X.XXXXXX delta=+/-X.XXXXXX <short note>
   ```
10. **Repeat.**

### Rotation rule

If you have run 5+ consecutive experiments in the same family (see `context.md` for families) without improvement, rotate to a different family. This is a signal, not a prison — if you have strong reason to stay, stay. But default is to move.

### Anti-bureaucracy rule

Logging must stay minimal. Never spend more time describing an experiment than running it. If `research_log.md` exceeds 200 lines, trim older experiment entries to one-liners — but always preserve: confirmed findings, active near misses, and current backlog items. Compact the narrative, not the live intelligence.

## Meta-review

Every 10 experiments, pause and re-read `research_log.md`. Write exactly 3 lines:

1. **Pattern**: what trend do you see across recent experiments?
2. **Next priority**: highest-impact untried idea or combination?
3. **Stuck?**: if yes, what's the escape plan? (combine near misses, rotate family, explore deeper knobs guided by `context.md`)

Refresh the backlog based on this review. Then continue.

## Output format

The script prints a summary:
```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract the key metric: `grep "^val_bpb:" run.log`

## Logging results

`results.tsv` is tab-separated (NOT commas). 5 columns:
```
commit	val_bpb	memory_gb	status	description
```
- commit: short hash, 7 chars
- val_bpb: e.g. 1.234567 — use 0.000000 for crashes
- memory_gb: peak_vram_mb / 1024, round to .1f — use 0.0 for crashes
- status: `keep`, `near_miss`, `discard`, or `crash`
- description: short text of what was tried

Do not commit `results.tsv` or `research_log.md` — leave them untracked.

## Timeout

If a run exceeds 10 minutes, kill it and treat as failure (discard and revert to starting commit).

## NEVER STOP

Once the loop begins, do NOT pause to ask the human anything. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep. You run indefinitely until manually stopped.

If you run out of ideas: re-read `research_log.md` and `results.tsv`. Re-read `context.md`. Look at near misses and try combining them. Rotate to an unexplored family. After surface and medium knobs show diminishing returns, explore deeper knobs guided by `context.md` — but move into deep territory deliberately, not randomly. The loop runs until the human interrupts you, period.
