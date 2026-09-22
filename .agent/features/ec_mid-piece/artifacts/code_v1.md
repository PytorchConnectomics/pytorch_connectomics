# Code v1
## Overview

Revised the focused verification surface for the GT-free mid-piece experiment to resolve all three
accepted review findings. The decisive affinity truth table now exercises the real ABISS producer,
the freeze firewall now tests Stage 3 rejection paths directly, and the GT-free source audit now
reconstructs statically built file-open paths from Python ASTs.

No experiment production code, tracked file, canonical input, dependency, or honest-baseline choice
changed. The raw arm0_96 honest baseline and the existing record of the unselected
`frozen_endpoint_merges.npz` input drift remain exactly as in code v0. The prohibited whole-volume
Stage 0 and Stage 1 passes were not run.

## What Changed

- Replaced the local `_ArrayVolume` affinity oracle in the truth-table test with
  `lib/abiss/scripts/volume_backends.py::_ArrayVolume`. The test loads a real affinity crop, reads
  the producer through its `(X,Y,Z,C)` slicing API, transposes back to `(C,Z,Y,X)`, and verifies
  `R[2-c', q-e_(2-c')]` with bit-exact `maxdiff == 0` on the interior domain.
- Added elementwise agreement between the experiment's `restore_sigmoid` and the producer's
  `_restore_sigmoid`, retained all four negative conventions, and made producer import failure an
  explicit `pytest.skip` with the reason instead of any local fallback.
- Replaced the self-referential freeze-attestation fixture check with a successful call through
  `stage3_evaluate.verify_frozen`, then added Stage 3 rejection tests for a one-byte payload
  mutation, missing `.frozen.json`, `gt_free: false`, and
  `frozen_before_evaluation: false`.
- Extended the GT firewall from literal-token checks to an AST audit of paths passed to `open`,
  `np.load`, `h5py.File`, and zarr open functions, plus `Path.open/read_text/read_bytes`. Static
  string addition, `Path` division, f-strings, `Path(...)`, `str(...)`, and `os.path.join` are
  reconstructed before comparison with forbidden evaluator paths.
- Added a regression test proving that `Path("/tmp") / ("evaluation" + "_gt")` passed to
  `np.load` is rejected even though the forbidden directory name does not occur literally in the
  source.

## Implementation Details

The real-producer affinity test prepends `lib/abiss/scripts` to `sys.path`, imports
`volume_backends`, and verifies that the resolved module file is the expected repository producer.
If import or resolution fails, the test calls `pytest.skip` with the concrete failure reason. A
real `(3,7,8,9)` HDF5 block is wrapped by the producer. The checked query excludes one voxel from
each local face so every `q-e` and negative-control `q+e` sample is supplied by the wrapped block;
the producer result is read as `(X,Y,Z,C)` and transposed to `(C,Z,Y,X)` before comparison.

The freeze tests construct deterministic assignment NPZ bytes and a sibling attestation. The
valid case invokes the exact Stage 3 verifier. Rejection fixtures alter only the condition under
test: one flips a byte after hashing, one omits the sibling JSON, and the two parameterized cases
set each required attestation marker to `False` while leaving the payload hash valid.

The AST audit records simple static name bindings and inspects the path-bearing expression for
known open calls. It retains the original literal checks for forbidden CLI/input tokens, while the
AST layer closes the reviewed string-concatenation gap. It is deliberately a bounded static check,
not general runtime taint analysis; fully opaque values remain represented as dynamic fragments.

## Files Changed

| File | Purpose |
|---|---|
| `dev/zebrafinch/ec_mid_piece/tests/test_affinity_io.py` | Validate affinity indexing and restore-sigmoid against the real ABISS producer with four negative controls |
| `dev/zebrafinch/ec_mid_piece/tests/test_firewall.py` | Audit dynamically constructed opened paths and exercise Stage 3 freeze rejection paths |
| `.agent/features/ec_mid-piece/artifacts/code_v1.md` | CCC code-v1 implementation and verification handoff |

## Git Baseline

run_start_ref: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd
current_head: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd

The pre-existing tracked and staged diffs remain byte-identical to the CCC run-start captures. No
file was staged or committed.

## Verification

Full focused suite, rerun after the final edits:

```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && /usr/bin/time -f 'WALL=%e sec MAXRSS=%M KiB' pytest -q dev/zebrafinch/ec_mid_piece/tests
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /projects/weilab/weidf/lib/pytorch_connectomics
configfile: pyproject.toml
plugins: zarr-3.1.6, cov-7.0.0, anyio-4.12.1
collected 23 items

dev/zebrafinch/ec_mid_piece/tests/test_affinity_io.py ..                 [  8%]
dev/zebrafinch/ec_mid_piece/tests/test_chunk_smoke.py .                  [ 13%]
dev/zebrafinch/ec_mid_piece/tests/test_firewall.py ........              [ 47%]
dev/zebrafinch/ec_mid_piece/tests/test_geometry.py ..                    [ 56%]
dev/zebrafinch/ec_mid_piece/tests/test_lut_evaluation.py ....            [ 73%]
dev/zebrafinch/ec_mid_piece/tests/test_resolver.py ......                [100%]

=============================== warnings summary ===============================
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 23 passed, 3 warnings in 12.05s ========================
WALL=15.62 sec MAXRSS=951740 KiB
```

Syntax compilation:

```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python -m py_compile dev/zebrafinch/ec_mid_piece/tests/test_affinity_io.py dev/zebrafinch/ec_mid_piece/tests/test_firewall.py
```

The command exited 0 with no output.

Tracked mutation guard:

```text
$ sha256sum .agent/features/ec_mid-piece/state/run_start.diff .agent/features/ec_mid-piece/state/run_start_cached.diff && git diff | sha256sum && git diff --cached | sha256sum
b7b755861b3ed581e98374dbda6c679e93143adfc8f6e6d8d3014da8f8256857  .agent/features/ec_mid-piece/state/run_start.diff
0547693ec4d8f72239634f4a168ba05916a891e522419f265ba7615ba6749298  .agent/features/ec_mid-piece/state/run_start_cached.diff
b7b755861b3ed581e98374dbda6c679e93143adfc8f6e6d8d3014da8f8256857  -
0547693ec4d8f72239634f4a168ba05916a891e522419f265ba7615ba6749298  -
```

Black's normal check could not be counted as successful because the environment runs Python 3.11
while repository configuration targets Python 3.12. A `--fast --target-version py311` fallback
reported both files reformatted but the process reached its timeout before terminating cleanly;
the post-format syntax compilation and full suite above are the authoritative checks.

## Review Focus

- Confirm that the affinity test imports the exact repository producer and never falls back to
  `affinity_io._ArrayVolume`.
- Check the producer read-domain construction: the wrapped block supplies one source voxel on each
  low side, and the comparison and all four negatives use identical interior output bounds.
- Confirm `verify_frozen` is the exercised entry point for valid, tampered, missing-attestation,
  and false-marker cases.
- Review the AST resolver's bounded expression forms and the dynamic `evaluation_gt` regression
  fixture. The original literal input/CLI checks remain active alongside it.
- Confirm no raw-baseline, input-manifest drift, Stage 0/1, or experiment implementation behavior
  changed in code v1.

## Risks and Unknowns

The real producer is a checkout-local test dependency. If it becomes genuinely unavailable or
resolves to a different module, the affinity test reports an explicit skip reason, as required,
rather than validating against a local reimplementation.

The AST firewall closes static path concatenation and common path-construction forms but cannot
prove the runtime value of an arbitrary environment variable, function return, or external config.
The GT-free scripts currently expose no such evaluator input surface; the existing literal CLI
audit remains in place.

The scientific unknowns from code v0 remain unmeasured because the prohibited whole-volume Stage 0
and Stage 1 passes were not run: candidate-table scale, band coverage, native policy risk/coverage,
NERL deltas, residual shares, and visual probes. The second-moment geometry approximation and the
recorded unselected frozen-linker input drift are unchanged.

## Changes Since Previous Code Version

- Finding 1: code v0 validated Form 2 against the local `affinity_io._ArrayVolume`, making the
  truth table tautological. Code v1 validates the real
  `lib/abiss/scripts/volume_backends.py::_ArrayVolume` and `_restore_sigmoid`, with bit-exact
  equality, four failing negative controls, and loud skip behavior if the producer is unavailable.
- Finding 2: code v0 only compared a freshly written hash with itself. Code v1 invokes Stage 3's
  `verify_frozen` and proves rejection of a one-byte payload mutation, missing `frozen.json`, and
  each false attestation marker.
- Finding 3: code v0 used literal substring scanning only. Code v1 adds AST inspection and static
  reconstruction of opened-path arguments, plus a regression test for a GT path assembled through
  string concatenation and `Path` division.
- Finding 4 required no code change per maintainer direction. Raw arm0_96 remains the honest
  baseline, and `frozen_endpoint_merges.npz` remains recorded only as unselected input drift.
