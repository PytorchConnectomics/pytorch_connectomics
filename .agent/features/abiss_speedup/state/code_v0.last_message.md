Implemented `code_v0` entirely in the isolated copy.

- Added `ABISS_ARCH`, `-ffp-contract=off`, and explicit allocator selection in [CMakeLists.txt](/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_speedup/work/abiss/CMakeLists.txt).
- Built all 17 targets for system and mimalloc configurations with `Release` and `EXTRACT_SIZE=ON`.
- Verified one `-march` per target, no unsafe FP flags, no allocator linkage for system, and isolated mimalloc linkage.
- Added and smoke-tested the run-local [profiling harness](/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_speedup/state/profiling/README.md).
- Did not modify `work/abiss/scripts/`, install Abseil, run benchmarks, create commits, or target the live checkout.

Deliverable: [code_v0.md](/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_speedup/artifacts/code_v0.md)

Segmentation bit-identity remains explicitly unverified; full repeated-output fidelity testing is the required next step.