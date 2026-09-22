# Phase A handoff

- GPU chain (all 1 L40S, 8 CPUs, 64G, 1h): 3010635 overfit -> afterok 3010636 smoke -> afterok 3010637 A evaluation. Observed at 20:10:59 UTC: overfit PENDING/Priority; smoke/eval PENDING/Dependency. Do not infer execution from submission.
- Expected outputs under worktree dev/ec_model/runs/: overfit/{composition.json,fixed_sites.npz,training.json,last.pt,result.json}; smoke/{training.json,val_A_curves.json,last.pt,result.json}; evaluation_A_3010637.json; frozen_A.json. Logs: logs/overfit_3010635.log, smoke_3010636.log, eval_3010637.log.
- Phase B: inspect sacct/logs without polling faster than 5 minutes. Overfit requires all 2,000 steps, mean Dice >=.90, thin Dice >=.85; confirm actual 32-example composition and pair prerequisites. With 8 trained pairs, require >=6 valid and median valid IoU <=.30; unseen pairs are diagnostic only.
- Smoke: finite/decreasing loss, A curves, full A report, B0 join recall 0, smoke-only label. If upstream fails, afterok jobs remain blocked; check remaining GPU budget before any remedy. Do not run B or full training.
- runs/val_B_access.json must stay absent. Human montage review is pending (49 train0, 55 train1, 50 val-A PNGs; final plots include anchor and seed planes).
- Phase A verified: 50 tests; real model strict load; 1,287,426 aligned nodes, 0 mismatches; deterministic decode on all three ROIs; independent site checks pass; 5,364 thin train and 1,234 thin A sites. Four non-target seed voxels in deployment-style A thin terminal examples were reported, not rejected.
- Curation provenance and resolved failures are in the draft and runs/slurm_phaseA_accounting.txt. Latest final montage jobs: 3010617, 3010618, 3010630, all COMPLETED.
- Finalize/extend artifacts/code_v0.md.tmp after observed GPU results; refresh source inventory/patch only if code changes. Coordinator alone owns .done and run.md; neither was written here.
