Implemented the confident-region decoder in [decode_axon.py](/projects/weilab/weidf/lib/pytorch_connectomics/dev/mit_liconn/decode_axon.py) and finalized [code_v0.md](/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/decode_axon/artifacts/code_v0.md).

Verification completed:

- `decode_axon.py --self-test`: PASS
- All four extraction partition-equality checks: PASS
- `--zslice 0:96 --tag _smoke`: PASS; final SMOKE metric 85.1% valid volume, 6 bumps, 0 parallel merges
- Compilation and focused lint/format checks: PASS
- Full unified 800-slice run intentionally not performed
- No commit created; HEAD remains `e8844b3da0f0a992431c901e7e2034486e7a678b`
- Existing unrelated worktree changes remain untouched

CCC run: `.agent/features/decode_axon`  
Stage completed: `code_v0`  
Next action: `review_v0`