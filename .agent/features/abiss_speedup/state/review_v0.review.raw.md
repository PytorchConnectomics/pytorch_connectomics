## Verdict

The CMake diff implements Phase B.1, B.2 and the CMake half of C.2 as approved, and the mechanism is correct in each case. Two things block acceptance under the bit-identity constraint: the **default allocator silently changed** relative to the pre-change build, and the "codegen-identical" argument for `-ffp-contract=off` is asserted, not measured, when a decisive check is cheap. Separately, 6 of the 9 rows in "Files Changed" have no artifact in this review.

## Does the diff implement the plan?

For what is visible: yes. `ABISS_ARCH` is an empty-default cache variable that strips inherited `-march=`/`-mtune=` before appending exactly one `-march` (CMakeLists.txt:11-23) — this is precisely finding 4's remedy, and the coordinator's 17/17 single-`-march` check confirms it. `-ffp-contract=off` is unconditional with no `-ffast-math`/`-Ofast`/`-funsafe-math-optimizations` (audit column `unsafe-fp=0`, direct grep exit 1). `ABISS_ALLOCATOR` replaces the `if(MIMALLOC_FOUND)...elseif(JEMALLOC_FOUND)` fall-through with an explicit, validated selector and `REQUIRED` discovery — a mis-set `PKG_CONFIG_PATH` now fails loudly instead of silently degrading, which is exactly finding 7. Abseil was not enabled. Phase 0.0/0.1 tooling lives outside `scripts/` (`git diff --quiet -- scripts` = 0), resolving finding 10.

## Is "ABISS_ARCH empty == today's behaviour, a no-op" true?

**Partly — and the report's framing invites the wrong reading.** `ABISS_ARCH=""` is a no-op *with respect to `-march`*: two identical `-march=nocona` collapsing to one is order-idempotent and cannot change codegen, and the same holds for the rest of the deduplicated conda flag block (including the duplicated `-O2`, which was and still is overridden by the later `-O3`). But the default build's compile line is **not** today's compile line: it now carries `-ffp-contract=off`, which GCC did not default to (`-std=gnu++20` → `-ffp-contract=fast`). The plan called B.1 a no-op refactor; B.2 was always an intentional flag change. The report's "Empty preserves the inherited architecture selection" is true but incomplete, and the Overview never states that the default build's flags changed in two ways.

## Could `-ffp-contract=off` change generated code here?

**Almost certainly not on this target, but this is unproven.** `-ffp-contract` in GCC gates FMA formation only (`convert_mult_to_fma`, guarded by `direct_internal_fn_supported_p(IFN_FMA, ...)`); with `-march=nocona` there is no FMA instruction, so no contraction can occur and `off` vs `fast` should emit identical code. The coordinator independently confirmed nocona has no FMA. That is a sound argument — but it rests on a compiler-semantics claim the report does not verify, under a constraint that says "proven safe," and the proof costs one command: compile the same tree twice differing only in `-ffp-contract=fast|off` in the same build dir and compare the `.o` files. Do that and this finding closes.

Worth recording in the report's favour: B.2 is the correct hedge. Because GCC will not reorder FP reductions without `-fassociative-math`, `contract=off` is what makes a *later* `-march` raise plausibly bit-exact. It is free today and load-bearing tomorrow.

## Findings

**[major] The default allocator changed from auto-detect to `system`.** Pre-change, `pkg_check_modules(MIMALLOC mimalloc)` + `if(MIMALLOC_FOUND)` meant that if mimalloc (or jemalloc) was discoverable in the build environment, the binaries were built with `-DUSE_MIMALLOC` and linked against it — silently. Post-change the default is `system` with no probe at all (CMakeLists.txt:43-52). If the binaries that produced the measured baseline (`2782527..2782540`) and the provenance replay picked up an allocator, the new default build is *not* the same configuration, and the plan itself withdrew the claim that allocator choice cannot alter results (layout changes can expose address-dependent behaviour or latent UB). The report cannot rule this out: it explicitly did not touch or inspect the live checkout, so no `ldd`/`flags.make` evidence exists for the production build. Resolve by `ldd`-ing the live binaries and grepping the live build's `flags.make` for `USE_MIMALLOC`. If either allocator was present, the fidelity-preserving default is an `auto` mode reproducing the old detection order, with `system` as an explicit opt-in.

**[major] `-ffp-contract=off` is argued safe, not shown safe.** See above; object-file comparison closes it.

**[minor] Six of nine "Files Changed" rows are unreviewable here.** The diff contains only `CMakeLists.txt`. `state/profiling/{capture_baseline_manifest,profile_command,profile_exec,composite_chunk_me,composite_chunk_ws}.sh` and the README are not in any provided artifact, so the Review Focus item "confirm the run-local composite copies cover every requested ABISS executable and cannot default to the live checkout" cannot be answered. `bash -n` is a syntax check, not coverage evidence.

**[minor] Phase C execution without a recorded approval.** The plan gates the isolated dependency prefixes on `AGENTS.md:104`; `build-mimalloc` was built against `<RUN>/deps/lib64/pkgconfig`. No approval is cited. The CMake plumbing itself was ungated, and nothing entered the default build, so the exposure is procedural.

**[minor] The `-DCMAKE_DISABLE_FIND_PACKAGE_absl=TRUE` evidence is weak.** No `find_package(absl)` appears anywhere in the diff; if CMakeLists never probes for absl, that flag proves nothing. The real evidence for "abseil not enabled" is the absence of `USE_ABSL_HASHMAP` from `flags.make`, which the audit table does not actually show a column for — the report asserts it in prose only.

**[minor] The `-march` strip mutates the directory-scope variable, not the cache.** A `-DCMAKE_CXX_FLAGS="-march=native"` will still appear un-stripped in `CMakeCache.txt` while `flags.make` shows the sanitized value. Since the Phase 0.0 manifest captures `CMakeCache.txt`, a future audit reading the cache rather than `flags.make` will be misled. Also note the strip covers `-march`/`-mtune` only, not `-mfma`/`-mavx*`; harmless in the safe direction given B.2, but state it.

**[minor] `add_compile_options(-ffp-contract=off)` is language-agnostic.** Fine for an all-C++ tree; guard with `$<COMPILE_LANGUAGE:CXX>` if a C or CUDA source is ever added. Related: `find_package(PkgConfig REQUIRED)` is now called only inside the non-system branches — empirically fine (system build completed 17/17), but any later `pkg_check_modules` in the file would now fail under the default.

**[minor] Build trees live inside the repo checkout.** `build-system/`, `build-mimalloc/`, `build-arch-sanitize/` are untracked under `work/abiss/`; confirm they are ignored so a later `git add` cannot sweep them in.

## Verification section

Honest overall, and the two most important disclosures are correctly and prominently made: segmentation bit-identity is UNVERIFIED, and no payoff was measured. The `ldd` result is correctly labelled "a linker-resolution check, not a benchmark or a segmentation-fidelity result." One overclaim by omission: the report presents the change as preserving the inherited configuration without stating that the default compile line now differs from the pre-change one in two ways. No accept/reject was published against Phase A's ≥5% / 2–5% / <2% thresholds, which is correct at this stage but means the ranked deliverable does not yet exist.

READY: no
