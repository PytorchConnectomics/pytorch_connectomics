# Binary error correction: corrected baseline (2026-09-20)

## Current direction: endpoint proposals link existing segments

User correction supersedes the voxel-fill experiments below. Train crops at
endpoints of the **predicted segmentation**; keep `[image, owner-fragment mask]`
inputs. Use the output as evidence for links between existing segment IDs.
Do not paint model-predicted foreground into background. `endpoints.py` already
uses predicted skeleton tips and verifies tip/owner membership before GT labeling.
The new `endpoint_links.py` scores overlap with existing fragments, including
disconnected predictions across gaps, and initially accepts reciprocal best pairs.
It exports a whole-fragment mapping and preserves every original zero. Candidate
generation and merge acceptance are separate; this initial selector is not a
replacement for a learned Pathfinder-style assembly scorer. First measure the
A-only perfect-mask link ceiling, then test frozen model proposals. B/seed101 stay
reserved. Asymmetric voxel-fill gate3029139 was stopped following this correction.

## Earlier voxel-fill experiments (historical)

**Probe update:** the first 304-site/64-write oracle was limited by coverage:
regional A NERL **0.701505 -> 0.731612**. All 540 interior trunk tips with
96-cubed writes reach an oracle **0.790666**; unrestricted pure merges plus fill
reach **0.962637**. This led to an explicitly padded variant (still 128-cubed
input), with invalid context ignored by loss/metrics and A reads clipped at 436.
It has 6,119 labelled training tips. See the recorded coverage lesson in
`dw-research/projects/2026_nisb_base/lessons/lesson_binary_endpoint_20260920_coverage.md`.

Eight-example overfit passes its mask diagnostic at a training-selected threshold
of .95 (Dice .9430, missing recall .6274, added wrong/target .0154), but fails at
.5 due to excess growth. The fixed 1,000-step pilot reaches **.707056** A NERL
without additional spanning fragments, a small gain below the +.03 target.
The padded pilot fails at its primary .95 threshold: **.685972**, spanning
fragments 1->2. One false fill causes the new merge; crumb adoption does not.
Its .99 diagnostic reaches .707261 safely at the count gate, still a small gain.

Passing perfect GT masks through the actual conservative decoder gives **.760887**,
not the more permissive union oracle's .790666. Fill alone gives .758500.
The exact decoder therefore still has enough headroom for the +.03 target.

Initializing only the final convolution bias to -3 passes the eight-example
training diagnostic at threshold .7: Dice .963524, missing recall .812744,
added wrong/target .018207. Its matched padded population pilot3025210/decode3025215
failed: primary .577208 NERL, spanning fragments1->4. Its safe .99 diagnostic
is only .704958, below the original initializer. Retain the original initializer.
The 10000-step duration test (3025288/decode3025289) failed: mask Dice .926738,
but primary .99 NERL .491391 and spanning fragments 1->7. Ten wrong zero-node
fills caused six new mixed fragments; adoption alone remained safe. Do not extend
training blindly. Training-only audit 3025570 found both sparse classes in 242/256
crops. Optional `--skeleton-weight 1` adds class-balanced BCE at omitted curated
nodes, retaining the same two model inputs. Eight-example gate 3025578 passes at
.90: Dice .962658, missing recall .807453, wrong additions/target .018713,
210/210 positive skeleton nodes recovered and 0/871 foreign nodes filled.
Explicit original IDs fix the subset mismatch that invalidated cancelled 3025573.
Population pilot 3025585 failed: .397465 NERL at frozen .90, 35 spanning fragments;
.99 still spans 14 and scores .688043. Sparse-node memorization did not transfer.
Do not extend this configuration automatically.
BasicUNet gate 3025650 (full-resolution conv/skip, features [16,32,64,128,128,16],
dense loss and original eight IDs) failed. At .5: Dice .965600 but four foreign
node fills; at .7: one foreign fill and raw Dice .644937; safe thresholds lose
recovery. No population run. Logit/loss contract is correct; confidence remains
low. The 1e-3 learning-rate gate 3025957 improves fit but also fails: four foreign
nodes persist through .85 in calibration refinement 3026209. No population run.
Foreign-node-only BCE gate 3026267 removes all foreign sparse fills. Its original
raw-mask gate fails, but exposes a screening mismatch: the decoder preserves owner
voxels regardless of raw predictions. The explicitly documented corrected screen
uses applied connected growth. At .7: Dice .944091, missing recall .634726,
wrong additions/target .018039, own nodes 161/210, foreign nodes 0/871.
Population job 3026433 fails at frozen .7: NERL .689349, spanning fragments1->2.
Diagnostic .9 gives .706047, spanning1, unchanged frags/GT. Forensics3026488
repeats the single wrong background fill into owner131 at [184,285,291]; all
safe-threshold gain is fill-only. Final +.03 A NERL and merge safety requirements
are unchanged. This is a development-selected configuration, not held-out evidence.
The same foreign-only model at4000steps (3026502,4h14m40s) improves fixed64 training
dense missing recovery to67.3%, but own-skeleton recovery345/986=35.0% fails the
registered50% gate despite foreign0/3836. No A segmentation decode or duration
extension. A CPU audit now checks sparse/dense training-target consistency.
The audit found all986 own/3836 foreign nodes consistent with dense targets.
BasicUNet balanced sparse weight2 at1000steps on fixed64 has75.3% own recovery;
perfect-mask oracle reaches100%. Matching eight-example exposure with4000steps
on fixed64 (3027570) passes at frozen .9:948/986 own,foreign0/3836,dense recall
67.9%,wrong additions/target0.96%. Full-population4000-step job3027899 now tests
generalization, gated on training recovery/safety before any A segmentation decode.
Population3027899 completed but failed training safety: own538/986 at .9,
foreign3/3836,false additions4.83%>2%. No A segmentation decode. The next matched
eight-example gate3029139 doubles the negative sparse coefficient while preserving
the positive coefficient (outerweight4,positive_scale.5);32 focused tests pass.
Its first allocation (3025626) was cancelled because measured speed exceeded the
30-minute estimate and Slurm disallowed extending it; unchanged retry has 45 minutes.
No substantial learned gain has been established. Synthetic suite: **76 tests pass**.

The architecture sweep below did not implement the endpoint task. The actual code still
trained on GT-mined gap/control sites, re-anchored them with GT skeleton nodes, and
jittered the origin at every draw. Calling that one-stratum sampling did not make its
population deployment-shaped. Two-channel augmentation also indexed missing channels
2 and 3. Changing architectures could not repair those contracts.

## Current runnable path

Implementation lives in the existing worktree:
`.claude/worktrees/ccc-ec_model/dev/ec_model/` (this directory is gitignored).

- `endpoints.py`: enumerate **predicted** skeleton tips from `pred_tips.npz`. Keep
  interior tips of fragments with cable >= 1000 nm. Require the tip voxel to belong to
  its stated owner. The original population requires a full 128-cubed crop;
  `--padded` retains boundary-near interior tips with explicit valid bounds.
  Report exclusions explicitly;
  do not move a tip onto some other voxel to make it pass.
- First baseline: **one fixed crop per tip**. No GT gap mining, jitter, movable seeds,
  class strata, purity selection, or architecture sweep. Multi-centre walk-back
  augmentation and its cap remain deferred; this is not an implementation of design
  section 5's multi-centre proposal.
- Inputs: `[EM / 255, seg == owner]`. The mask contains **one broken fragment**, not
  all nonzero segmentation. Inference requires neither affinities nor GT skeletons;
  the optional sparse training loss uses curated skeletons only as supervision.
- Label: `gt == gt[tip]`, the whole neuron inside the crop, including missing pieces.
  GT is used only after prediction-only candidate selection. GT-zero centres cannot
  provide a neuron label and are counted as unlabelled; inference input construction
  does not require GT.
- `binary.py`: small MONAI 3D residual U-Net, widths `[16, 32, 64, 128]`, two inputs,
  one logit, BCE + Dice, AdamW, Lightning training. The dense broken-fragment mask is
  the conditioning signal; no special prompt encoder is needed.
- Checkpoints contain the structured model config and task identity. Evaluation
  reconstructs the model from that metadata. `run.json` records population hashes,
  training IDs, channels, geometry and hyperparameters.
- Validation uses `endpoint_sites_A.npz` for full-context crops or the separate
  `endpoint_sites_A_padded.npz` for padded crops. All input/GT crop reads stay in A;
  synthetic context has target -1 and is ignored in losses and metrics.
  Mining does not rewrite old A/B manifests. Seed101 is rejected.
- Mask diagnostics use the central 64-cubed region, or 96 with `--write 96`:
  model and copy-mask Dice, recovery
  of GT missing from the input, wrong voxels, and newly added wrong voxels. True
  terminals stay in the population. These are **not a segmentation success verdict**.
- `endpoint_decode.py` performs prediction-only connected growth, protects trunks,
  permits only unambiguous crumb adoption, and abstains on conflicting fill claims.
  It reports A-node NERL and spanning-fragment count. GT enters metric computation,
  not the edit rules. `oracle_decoder.py` is a separate GT-aware diagnostic.

Run under Slurm from that worktree, with `EC_MANIFEST` pointing to
`dev/ec_model/manifest_train_all.json`:

```bash
# Once for each training ROI and val_seed100_center:
bash dev/ec_model/run_ec.sh endpoint-mine --roi train_seed0_center
bash dev/ec_model/run_ec.sh endpoint-mine --roi val_seed100_center

# Start with a fixed-subset overfit run. Choose fresh output directories.
bash dev/ec_model/run_ec.sh binary-train --overfit 8 --steps 500 \
  --out dev/ec_model/runs/binary_overfit
bash dev/ec_model/run_ec.sh binary-train --steps 10000 \
  --out dev/ec_model/runs/binary_endpoint
bash dev/ec_model/run_ec.sh binary-eval \
  --checkpoint dev/ec_model/runs/binary_endpoint/last.ckpt \
  --out dev/ec_model/runs/binary_endpoint_eval
```

The old `train.py`, `evaluate.py`, and `decode_merge.py` describe the historical
gap/seed experiment. They are not the entry points for this baseline. In particular,
do not feed its Lightning checkpoint to the historical transitive merge decoder.

## Lessons that determine the scope

1. `lesson_sdt_decode_ceiling.md`: a fixed Gaussian is easy to ignore; a dense owner
   mask must identify the object. High crop Dice does not establish usable correction.
2. The same lesson measured transitive growth spanning 152 GT objects. Endpoint
   matching alone also does not mathematically prevent chains through different tips
   of the same fragment. Decoder safety must be measured, not inferred from its name.
3. `lesson_ec_endpoint_bridge.md`: conservative geometric bridging helped on a regional
   proxy, while the learned local-feature scorer still leaked merges. It does not prove
   a new binary mask model is merge-safe.
4. This experiment's old decoder raised spanning fragments from 4 to 14 despite
   improving frags/GT. The gate is **no increase from the measured baseline spanning
   count**, plus fewer splits and improvement over copying the owner mask. Neither
   "max span must be 1" nor a mask-Dice threshold is the correct gate here.

## Verification

All **68 tests pass** (`conda run -n pytc pytest -q dev/ec_model/test_endpoints.py
dev/ec_model/test_ec_model.py`). New regression tests cover
GT-independent crop locations, trunk/face/owner checks, A-only selection, two-channel
augmentation, copy/recovery/false-growth metrics, and a real U-Net optimizer step and
checkpoint round trip, padded bounds/ignored targets, edit conflict rules and
output-bias initialization. Substantial correction remains an empirical requirement.

Real-volume mining completed under Slurm job **3025040**: **3,951 labelled training
tips** across five ROIs and **304 labelled validation-A tips**. There were 23 unlabelled
training candidates and one unlabelled A candidate. The old A/B manifests were not
rewritten. A two-step GPU integration smoke job **3025041** was submitted with output
`runs/binary_endpoint_smoke_20260920`; it completed successfully.

---

# Historical architecture proposal (superseded; retained for provenance)

**STATUS: CODE WRITTEN, NOTHING VERIFIED.** Written 2026-09-19 while the session's shell
was unavailable (`task output swap refused (tasks dir moved or linked)` on every Bash
call). No test was run, no model was instantiated, no import was checked. Treat every
claim below as a proposal, and run §5 step 0 before anything else.

## 1. What changed

`dataset.py`
- `CHANNELS` module global (4 or 2) and `set_channels(n)`, recorded into `CONFIG`.
- `tensors()` returns `np.stack([em, m])` when `CHANNELS == 2`, else the original
  `np.stack([em, p, m, a])`.
  - 4 = `[EM, point seed, owner mask, affinity advice]`
  - 2 = `[EM, owner mask]` — the binary mask alone is the prompt.

`model.py`
- `strict_inflate` no longer hardcodes a 4-channel stem; it inflates the pretrained
  1-channel EM stem to whatever `in_channels` the variant uses, EM into channel 0, the
  rest zero.
- `build(warm, device, arch, in_channels)` with
  `ARCHITECTURES = (mednext, monai_unet, monai_basic_unet3d, monai_unetr,
  monai_swin_unetr)`. Sets `cfg.model.arch.type`, `cfg.model.input_size = [128,128,128]`,
  and creates empty `cfg.model.monai` / `cfg.model.transformer` nodes under `open_dict`.
- `warm=True` now raises for non-mednext archs (no pretrained checkpoint exists).

`train.py`
- `--arch`, `--channels {2,4}`, `--scratch`, `--tag` (run-directory suffix).
- Warm start is `not --scratch and arch == mednext`.

## 2. Known-unverified risks, highest first

1. **MONAI API drift.** `monai_models.py:289` passes `pos_embed=` to `UNETR`. MONAI
   renamed that argument to `proj_type`, and `SwinUNETR`'s signature also changed
   (`img_size` deprecated/removed in recent versions). If the installed MONAI is recent,
   `monai_unetr` and `monai_swin_unetr` will raise at construction. **Check the installed
   version first**; the fix belongs in `monai_models.py`, not in `dev/ec_model`.
2. **`cfg.model.arch.type` assignment.** The base config may have `arch` as a struct that
   rejects mutation, or may not define `arch` at all. `open_dict` should cover it, untested.
3. **`evaluate.py` and `decode_merge.py` were not wired.** Both still call
   `build(warm=False, device="cuda")`, i.e. mednext/4-channel. Loading a 2-channel or
   UNet checkpoint into them will fail on `load_state_dict(..., strict=True)` — loudly, not
   silently, so results cannot be quietly wrong, but they must be updated before any
   variant can be gated.
4. **`evaluate.py:model_predictor` does `x[2:] = 0` under `--no-advice`.** With 2 channels
   that slice is empty, so the flag silently becomes a no-op instead of erroring.
5. `train.py` prints the arch/channel line but nothing writes them into `result.json`.
   Add them before running the matrix or the runs won't be self-describing.

## 3. Experimental design — two changes are being made at once

Going 4ch → 2ch removes **both** the point seed and the affinity advice. Changing the
architecture at the same time confounds three things. Minimum matrix to keep them
separable (same data sampling throughout):

| # | arch | ch | warm | isolates |
|---|---|---|---|---|
| 0 | mednext | 4 | yes | **already run** — `runs/full_radius_stratified` |
| 1 | mednext | 2 | yes | the channel change alone |
| 2 | mednext | 2 | no | pretraining |
| 3 | monai_unet | 2 | no | UNet vs MedNeXt, matched (no) pretraining |
| 4 | monai_unetr *or* monai_swin_unetr | 2 | no | transformer vs conv |

Rows 2–4 are the honest architecture comparison; only row 2 is a fair partner for rows 3
and 4, because rows 0–1 get a 200k-step head start that the others cannot have.

If only one extra run is affordable, run **row 1** — it answers whether the affinity
advice and point seed were carrying anything, which is cheap and informative regardless
of what happens with architectures.

## 4. "SAM-like in 3D" — what is and isn't on offer

MONAI's UNETR and SwinUNETR are **not promptable** in SAM's sense: there is no prompt
encoder and no cross-attention between a prompt and the image embedding. They are
transformer-encoder / conv-decoder segmentation nets.

Concatenating the binary mask as input channel 1 is the analogue of SAM's **dense mask
prompt** (SAM adds a dense mask embedding to the image embedding), so "the mask can be
prompt like" is satisfied in that weaker sense for every architecture here. What is not
available without new code is sparse point/box prompting or a separate prompt encoder.
A genuine 3D SAM would mean SAM-Med3D or a custom prompt encoder — neither was checked
for, because checking needs a shell.

## 5. Run order once the shell is back

0. **Verify before trusting any of the above:**
   - `python -c "import monai; print(monai.__version__)"` and check `UNETR.__init__`
     for `pos_embed` vs `proj_type`.
   - `python -c "from connectomics.models.architectures import print_available_architectures as p; p()"`
   - `python -m pytest -q dev/ec_model/test_ec_model.py` — 53 tests passed before these
     edits; anything failing now is mine.
   - Instantiate each arch on CPU at `in_channels=2` before submitting a GPU job.
1. Wire `--arch`/`--channels` into `evaluate.py` and `decode_merge.py` (risk 3).
2. Memory: MedNeXt-L at 128³ batch 2 measured **11.7 GB**. SwinUNETR and UNETR at 128³ are
   substantially heavier; expect to need batch 1 or `transformer.use_checkpoint=true`.
   Measure before queueing a long run.
3. **Use ~10k steps, not 30k.** The baseline's thin recovery reached 0.746 by step 7,500
   and gained 0.02 over the remaining 22,500 (`runs/full_radius_stratified/val_A_curves.json`).
   For an architecture screen that is ~3.5 h per MedNeXt run instead of 11 h. Re-check
   saturation per architecture rather than assuming it transfers.
4. Gate on §9 of `design.md`, not on Dice: spanning-fragment count must not rise above the
   cc3d baseline of 4, frags/GT must fall, and the model must beat `b0 = (seg == owner)`.
   Copy-Dice is ≈0.86, so a Dice in the 0.9s means little on its own.

## 6. Relation to the end-point redesign

This is an **architecture/input ablation on the current sampling**, which is what was
asked for ("same data sampling"). It is orthogonal to the tip-centred redesign in
`design.md` and does not supersede it. Note that `design.md` §9.1 found the bottleneck to
be the *decoder* — 0.766 mask recovery already surfaced real joins at 92.5% precision, and
the merge count still tripled. A better architecture improves the mask; on that evidence
it should not be expected, by itself, to produce a safe segmentation gain.
