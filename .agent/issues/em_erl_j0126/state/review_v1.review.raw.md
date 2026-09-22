# review_v1 raw reviewer notes (in-session planner review of code_v1)

Reviewed working tree vs run_start_ref c24e685:
- scripts/j0126_workflow.py (396 lines): added load/save/validate LUT helpers,
  score_graph_with_lut / score_skeletons_with_lut, --lut/--mip/--cache-dir plumbing.
- scripts/README.md: documented download-once / save-LUT / reuse workflow + mip tradeoff.
- tests/test_j0126_workflow.py: 7 offline tests.

## All review_v0 findings addressed (code read)

- [major] LUT persistence + zero-GCS reuse: DONE. run_j0126_eval always rebuilds the
  graph from -g. If --lut exists -> load_node_segment_lut() via read_vol (no CloudVolume
  import/open on this branch, lines 284-287); else sample, validate, save via write_h5.
  validate_node_segment_lut() guards len(lut)==graph.num_nodes with a clear error.
- [minor] --mip default 0 -> open_seg_cloudvolume(mip=mip); help + README document the
  measured ~1.5% drift. Faithful default preserved.
- [minor] --cache-dir -> CloudVolume cache=PATH else cache=False (open_seg_cloudvolume
  lines 79-83).
- [confirm] sampling strategy unchanged (occupied-chunk grouping/fetch each once).

Tests are genuine:
- test_lut_round_trip_reuse_scores_without_cloudvolume_access: monkeypatches
  open_seg_cloudvolume to record calls + returns a RaisingCloudVolume; asserts
  opened_cloudvolumes == [] and reuse ERL == direct ERL. Proves the no-GCS property.
- test_lut_length_mismatch_raises_clear_error: mismatch guard.
- test_run_j0126_eval_passes_mip_to_cloudvolume_opener: mip + cache_dir reach opener.
- test_open_seg_cloudvolume_passes_cache_to_constructor: cache=path / cache=False.

## Live end-to-end proof (real skeletons + real CloudVolume), /tmp/verify_reuse.py

    === LEG 1: generate + save LUT ===
    ERL: 96337.92 ; gt ERL: 179065.85
    LEG1 done in 8.3 min | ERL=96337.92 | LUT file 0.32 MB
    === LEG 2: reuse LUT (CloudVolume access forcibly disabled: opener -> raise) ===
    ERL: 96337.92 ; gt ERL: 179065.85
    LEG2 done in 0.1 s | ERL=96337.92
    === SUMMARY ===
    ERL match leg1==leg2: True (96337.92 vs 96337.92)
    code_v0 baseline ERL was 96337.92 -> match: True
    reuse speedup: leg1 8.3 min vs leg2 0.1 s
    LUT file: 315616 bytes

Conclusions:
- Metric unchanged: code_v1 reproduces the code_v0 baseline ERL 96337.92 exactly (the
  LUT/mip/cache refactor did not alter the computation).
- Reuse works with ZERO GCS: leg 2 completed in 0.1 s with the CloudVolume opener wired
  to raise on any access; identical ERL. This is the user's requested "download once,
  reuse/share" behavior, proven.
- Shareable artifact is ~316 KB (gzip-compressed LUT), vs ~3.6 GB one-time egress ->
  ~11,000x smaller to share/reuse than re-querying.
- Offline suite: 48 passed.

No new findings. Registration correctness was verified at review_v0 and is unchanged.

READY: yes
