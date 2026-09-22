## Findings

- [minor] **Previous finding 1 resolved:** removing `mask_background` defines background handling clearly. Computing CCs on valid foreground also prevents masked and ignored voxels from bridging components.

- [minor] **Previous findings 2 and 3 resolved at plan level:** the full-pipeline test now checks exact un-eroded `gt_seg`, affinity/mask equivalence, and thin-process gradients. Make the crop deterministic and place the tested thin process inside all three offsets’ valid region; boundary voxels cannot satisfy the universal mask assertion.

- [major] **Previous finding 4 only partially resolved:** the mixed-precision test adds parameter-gradient checks, but requiring no overflow on the first backward at scale 65536 with approximately ±20 outputs can reject normal AMP scale calibration. Conversely, a final checkpoint scale ≥1024 and within 4× of baseline does not establish that updates were not repeatedly skipped. Specify a bounded calibration period, then measure actual skipped optimizer updates and non-finite gradients during the smoke. Compare loss accuracy using the same quantized predictions; separate fp16/fp32 convolution outputs need not agree within `1e-4`.

- [major] **Previous finding 5 partially resolved:** affinity-only comparisons, numerical thresholds, and automatic cancellation are now concrete. The weight rule still lacks an aggregation rule for the four batches and a definition of whether `g_emb` includes the configured weight. On subsequent reviews, using an already-weighted gradient and multiplying by candidate `w` again would double-count the weight. Define `r` from unweighted losses, its aggregation across batches, zero/non-finite denominator handling, and require a fresh matched smoke at the selected weight.

- [minor] **Previous finding 6 resolved at plan level:** row chunking, a measured forward/backward budget, and checkpointing on budget failure provide an actionable implementation path. Add gradient parity across chunk sizes and checkpointed/uncheckpointed execution; the stated stress comparison checks only loss values.

- [major] **Previous minor batch/denominator finding only partially resolved:** empty-batch policies are now explicit, but the proposed means `0, 0.5, 5.0` make every permitted push term zero. That fixture cannot distinguish division by six from division by four. Use, for example, `0, 0.5, 2.0`, giving a numerator of `6.5` and expected `L_ext = 6.5 / 6`; isolate the external term with `alpha=gamma=0`.

- [minor] **Previous minor connectivity finding resolved:** the limitation and 6-versus-26 fixture are explicit. Correct the fixture’s “sum of two independent pulls” to their **mean**, consistent with the defined object reduction. Also qualify “GT-connected thin processes stay one component” as connectivity within valid foreground.

- [major] **New shape bug in §1:** unconditional `gt_seg[b].squeeze(0)` removes the Z dimension for accepted `[B,Z,Y,X]` inputs when `Z=1`. Normalize the optional channel dimension using the original tensor rank, and test singleton spatial dimensions for both accepted layouts.

- [major] **New precision mismatch in test 6.1:** the implementation always casts to fp32, while the oracle fixture uses fp64 with `atol=1e-6`. That does not specify a reliable parity criterion for accumulated squared L1 distances. Define both absolute and relative tolerances appropriate to fp32, or run both implementations on identical fp32 inputs.

- [major] **Watchdog verification remains incomplete:** continued training scalars can keep the watchdog fresh indefinitely even if the required validation tag or exact gate step never appears. Specify a deadline for missing gate data, reject non-finite values, and pin the monitored directory to the submitted job instead of repeatedly selecting the newest run. Verify `scancel` succeeds before reporting `CANCELLED`; dry-run must report “would cancel.” Add deterministic tests for cancellation, cancellation failure, missing tags, stale data, early termination, and pass/fail combinations. The two proposed dry-runs do not verify these operational contracts.

## Questions

- What aggregation defines the four-batch gradient ratio, and does every selected weight require a fresh smoke?
- What post-calibration skipped-update threshold blocks launch?
- What deadline and action apply when training progresses but a required validation gate never appears?

READY: no