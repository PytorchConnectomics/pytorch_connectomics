# Raw review notes (planner = claude, in-session), review_v1

Session resumed 2026-09-24 after container restarts killed the benchmark watcher; SLURM job
3039688 (c225) was unaffected and COMPLETED in 2h49m (ended 06:06 EDT).

Inputs examined: artifacts/code_v1.md, artifacts/review_v0.md; `git -C lib/abiss diff -- src/ws/atomic_chunk.cpp`
(full); lib/abiss/tests/ws_gate.py:39-46 (benchmark_thresholds) and :197-198 (stdout regex checks);
tutorials/neuron_liconn_ist/slurm/ws_sizing.py:93-94; build_mem/local_gate_v1.log tail;
build_mem/gate_scratch/benchmark_3039688/{results.json, acceptance.json, memory_table.md}.
Baselines: worktree HEAD 3cbdf39c = run_start_ref; abiss HEAD 92abc91 = abiss_start_ref.
Stock md5 unchanged: build/ws bf4c7343..., build64/ws64 ab83e053...

Prior findings:
- review_v0:1 fixed — benchmark_thresholds casts to float32 and requires finite low<high. The
  benchmark ran with valid thresholds: all 48 runs exit 0, non-trivial outputs.
- review_v0:2 fixed — `num of sv:`/`size of rg:`, agglomeration/writing timing, multi banner and
  per-threshold `(thr) ... in <s> seconds` lines restored; gate regex-checks them; matches
  ws_sizing.py:93-94.
- review_v0:3 fixed — function-try-block on main, `ws runtime error:` + exit 4; gate exercises
  rename failure on both widths.
- review_v0:4 fixed (acceptably) — comment covers high_bit, width memory cost, independent on-disk
  dtype, uint64 boundary/graph ids. It drops the concrete LICONN 98.3%/7.8x numbers; not material.

Step 5 acceptance (plan_v2 Verification Plan step 5), job 3039688, 48 runs, all exit 0:
- acceptance.json: 16/16 checks pass. Max binary RSS ratio new/stock 0.670 (target <=1.0).
  Max wall ratio 0.806 (target <=1.15) — new is faster everywhere.
- Headline target, 5-threshold ws64 binary on 128x3072x3072 (1.21 Gvox core): peak RSS
  72.2 GB -> 39.4 GB, ratio 0.546 (target <=0.70), for both uint64 and uint32 output. Wall 560 s -> 297 s.
- ws binary 1.21 Gvox: 1 thr 59.3 -> 31.9 GB; 5 thr 65.9 -> 31.9 GB. Peak no longer scales with
  threshold count (no seg_copy per threshold).
- uint32 output does not change binary RSS (streamed write) but cuts cgroup aggregate (page cache):
  ws64 t5 binary 80.9 (stock) -> 66.1 (u64) -> 45.7 GB (u32); ws t5 69.8 -> 62.3 -> 39.3 GB.
- Whole Python job: ws t5 65.9 -> 41.9 GB RSS; ws64 t5 cgroup 90.1 -> 84.3 (u64) -> 62.2 GB (u32).

New issues:
- m1 [minor] scripts/run_abiss_volume.py — the Python driver is now the job's peak, not the binary:
  python-job RSS is 41.9 GB on every 1.21 Gvox case vs 31.9 (ws) / 39.4 (ws64) binary, i.e. ~10 GB
  of Python-side arrays remain resident across or after the subprocess. Within plan acceptance
  (new <= stock) but it is the next lever; follow-up, not a blocker.
- m2 [minor] benchmark noise — ws t5 python uint32 cgroup 63.8 GB > uint64 62.8 GB, the only case
  where u32 is not lower. Page-cache writeback timing is not controlled; report only.
No correctness regressions found: local gate 168/168 cases, 19,600 file pairs, only dend_* padding bytes differ.

READY: yes
m1 [minor] scripts/run_abiss_volume.py — Python driver keeps ~10 GB resident, now the job peak (41.9 vs 31.9/39.4 GB binary) — follow-up: free/stream the remaining driver arrays before/after ws subprocess
m2 [minor] ws_gate benchmark — ws t5 python u32 cgroup 63.8 > u64 62.8 GB (page-cache noise) — report only
