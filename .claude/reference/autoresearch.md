# autoresearch Reference

**Location:** `lib/autoresearch/`
**Origin:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
**License:** MIT

Autonomous AI research framework: an agent modifies code, trains for 5 minutes, evaluates, keeps or discards, and repeats — indefinitely. Designed for LLM pretraining experiments on a single GPU.

## Core Idea

The human writes `program.md` (agent instructions). The agent edits `train.py` (model + optimizer + training loop). `prepare.py` is read-only (data, tokenizer, eval). The metric is `val_bpb` (validation bits per byte) — lower is better. Training always runs for exactly 5 minutes wall clock.

## Files

| File | Role | Who edits |
|------|------|-----------|
| `program.md` | Agent instructions, experiment loop protocol | Human |
| `train.py` | GPT model, Muon+AdamW optimizer, training loop | Agent |
| `prepare.py` | Constants, data download, tokenizer, dataloader, `evaluate_bpb` | Nobody (read-only) |

## Experiment Loop (from `program.md`)

```
LOOP FOREVER:
1. Read git state
2. Edit train.py with experimental idea
3. git commit
4. Run: uv run train.py > run.log 2>&1
5. Read results: grep "^val_bpb:" run.log
6. If crash → read traceback, attempt fix or skip
7. Log to results.tsv
8. If val_bpb improved → keep commit (advance branch)
9. If val_bpb equal or worse → git reset to previous
```

Each experiment runs on a dedicated branch (`autoresearch/<tag>`). The agent never stops to ask — it runs autonomously until interrupted.

## Logging Format (`results.tsv`)

Tab-separated, 5 columns:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

Status: `keep` (improved), `discard` (equal/worse), `crash` (failed).

## Key Design Principles

1. **Single file to modify** — only `train.py`. Keeps scope manageable and diffs reviewable.
2. **Fixed time budget** — always 5 minutes. Makes experiments directly comparable regardless of what changed (model size, batch size, architecture).
3. **One metric** — `val_bpb`. Vocab-size-independent, so architectural changes (vocab size, tokenizer) are fairly compared.
4. **Self-contained** — no external dependencies beyond PyTorch + small packages. One GPU, one file, one metric.
5. **Simplicity criterion** — all else equal, simpler is better. Removing code that doesn't help is a win.
6. **Never stop** — the agent runs indefinitely. ~12 experiments/hour, ~100 overnight.

## Model Architecture (`train.py`)

GPT with modern tricks:
- **RMSNorm** (pre-norm)
- **Rotary Position Embeddings** (RoPE)
- **Flash Attention 3** (via `kernels` package)
- **Sliding window attention** (pattern: SSSL — 3 short + 1 long)
- **Value Embeddings** (ResFormer-style, alternating layers, gated)
- **Residual lambdas + x0 lambdas** (per-layer learnable scalars)
- **Squared ReLU** activation in MLP
- **Logit softcapping** (softcap=15)

Default: 8 layers, 768 dim, 6 heads, ~50M params.

## Optimizer: MuonAdamW

Combined optimizer:
- **Muon** for 2D matrix params (attention, MLP projections) — polar express orthogonalization + NorMuon variance reduction + cautious weight decay
- **AdamW** for everything else (embeddings, scalars, lm_head)

Separate LR groups: embedding (0.6), unembedding (0.004), matrices (0.04), scalars (0.5). All scaled by `1/sqrt(d_model/768)`.

## Relevance to PyTorch Connectomics

The autoresearch loop pattern is directly applicable to decoding parameter tuning:

| autoresearch | PyTC decoding |
|-------------|---------------|
| Edit `train.py` | Edit decode params (threshold, merge_function, aff_threshold, ...) |
| Run 5min training | Run decode + evaluate (~seconds) |
| Metric: val_bpb | Metric: adapted_rand |
| Log to results.tsv | Log to experimental log in crackit/decoding.md |
| keep/discard/crash | keep/discard based on ARE improvement |
| git branch per run | Optuna study per sweep |

The key difference: decoding experiments are much faster (~seconds vs 5 minutes), so the agent can run hundreds of experiments per hour instead of 12.
