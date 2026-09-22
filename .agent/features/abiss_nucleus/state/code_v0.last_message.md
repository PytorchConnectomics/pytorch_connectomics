Implemented plan_v2 in the isolated `work2/abiss` clone. No commits created; live `lib/abiss` was untouched.

Key results:

- V0: 139,294 global records; representative `72198606672811349` resolves to CONFLICT. 10,827 matching SIDs differ from local records; one local-PROPER SID becomes globally CONFLICT.
- V1: nucleus algebra, single-chunk equivalence, ID-space builder test, and input contracts passed.
- V2: both modes returned `identical=38 differing=0 missing=0`.
- V3: adversarial 70/30 fixture passed; table-enabled hierarchy equals monolithic output.
- V4: NOT RUN because it requires the external SLURM setup and writes outside the isolated clone.

Full implementation and verification details: [code_v0.md](/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/artifacts/code_v0.md)

CCC run: `/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus`  
Stage completed: `code_v0`  
Next action: `review_v0`