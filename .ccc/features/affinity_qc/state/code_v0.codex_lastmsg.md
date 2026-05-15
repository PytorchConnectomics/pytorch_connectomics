Implemented `plan_v2.md` and wrote the CCC artifact at [code_v0.md](/projects/weilab/weidf/lib/pytorch_connectomics/.ccc/features/affinity_qc/artifacts/code_v0.md).

Validation passed:

```text
python -m pytest tests/unit/test_decoding_qc_affinity.py tests/unit/test_v3_guardrails.py tests/unit/test_v2_boundaries.py tests/unit/test_decoding_pipeline.py -q
41 passed, 6 warnings
```

```text
python scripts/validate_tutorial_configs.py
Validated 14 canonical tutorial configs successfully; skipped 2 custom workflow YAMLs.
```

No commit was created, and I did not write any `state/*.done` files. The repo was already broadly dirty before this task; I only edited the requested affinity QC/runtime/test files plus the `code_v0.md` artifact.