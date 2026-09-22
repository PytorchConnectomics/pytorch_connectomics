# review_v1 raw verification transcript

Stage owner is `claude` (planner), so this stage ran in-session. This file is the raw evidence.

## 1. Independent suite re-run, with skips forced to report (`-rs`)

```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc \
  && pytest -q -rs dev/zebrafinch/ec_mid_piece/tests
collected 23 items
dev/zebrafinch/ec_mid_piece/tests/test_affinity_io.py ..                 [  8%]
dev/zebrafinch/ec_mid_piece/tests/test_chunk_smoke.py .                  [ 13%]
dev/zebrafinch/ec_mid_piece/tests/test_firewall.py ........              [ 47%]
dev/zebrafinch/ec_mid_piece/tests/test_geometry.py ..                    [ 56%]
dev/zebrafinch/ec_mid_piece/tests/test_lut_evaluation.py ....            [ 73%]
dev/zebrafinch/ec_mid_piece/tests/test_resolver.py ......                [100%]
======================= 23 passed, 3 warnings in 10.45s ========================
```

`-rs` reports skipped tests explicitly. **Zero skips.** This is the decisive check for finding 1:
the rewritten truth-table test could have "passed" by taking its `pytest.skip` path, and it did not.

## 2. Finding 1 — the truth-table test now compares against the real producer

`tests/test_affinity_io.py:18-31`

```python
def _real_producer():
    producer_dir = REPO / "lib" / "abiss" / "scripts"
    sys.path.insert(0, str(producer_dir))
    try:
        producer = importlib.import_module("volume_backends")
    except Exception as error:
        pytest.skip(f"real ABISS producer unavailable: {type(error).__name__}: {error}")
    expected_path = (producer_dir / "volume_backends.py").resolve()
    if Path(producer.__file__).resolve() != expected_path:
        pytest.skip(... f"volume_backends resolved to {producer.__file__} ...")
    return producer
```

`tests/test_affinity_io.py:50-78`

```python
producer_restored = producer._restore_sigmoid(block, 0.2)
assert np.array_equal(restored, producer_restored)

volume = producer._ArrayVolume(block, convention="banis", restore_sigmoid_scale=0.2)
observed_xyzc = volume[x0:x1, y0:y1, z0:z1]
observed = np.transpose(observed_xyzc, (3, 2, 1, 0))
...
for output_channel in range(3):
    source_channel = 2 - output_channel
    expected[output_channel] = sample(source_channel, source_channel, -1)
assert float(np.max(np.abs(observed - expected))) == 0.0
```

The two sides are now genuinely independent: `observed` is produced by the real ABISS transform,
`expected` by slicing the restored BANIS field per the declared truth table. The query window is the
interior (`z0 = y0 = x0 = 1`), so the one-voxel shift is a real read rather than the zero-padded low
face. The skip path is loud and carries a reason, and the path-identity guard prevents a same-named
module elsewhere on `sys.path` from silently standing in for the producer.

## 3. Finding 2 — Stage 3 rejection is now exercised

```text
$ grep -n "^def test" tests/test_firewall.py
118:def test_gt_free_sources_have_no_evaluator_input_surface():
123:def test_gt_free_audit_rejects_dynamically_constructed_evaluator_path():
156:def test_assignment_attestation_audit(tmp_path):
163:def test_stage3_rejects_tampered_assignment_payload(tmp_path):
172:def test_stage3_rejects_missing_freeze_attestation(tmp_path):
180:def test_stage3_rejects_false_attestation_marker(tmp_path, marker):
190:def test_prepared_manifest_and_any_real_assignments_pass_firewall_audit():
```

The three rejection paths the review asked for exist (tampered bytes, missing attestation, false
marker parametrized over both markers), and they call `verify_frozen` rather than re-asserting a
fixture hash against itself.

## 4. Finding 3 — the firewall audit is now an AST analysis

```text
$ grep -n "ast\.\|import ast" tests/test_firewall.py | head
3:import ast
41:def _qualified_name(node: ast.AST) -> str | None:
50:def _static_path_text(node: ast.AST, bindings: dict[str, str]) -> str:
55:    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
60:    if isinstance(node, ast.JoinedStr):
69:    if isinstance(node, ast.Call):
78:def _opened_paths(source: str) -> list[tuple[int, str]]:
```

It resolves `Name` bindings, `BinOp` for both `+` and `/`, f-strings (`JoinedStr`), and `Call`
nodes, over the argument of `open` / `np.load` / `h5py.File` / `zarr.open{,_array,_group}`, and
`test_gt_free_audit_rejects_dynamically_constructed_evaluator_path` proves the strengthened audit
catches a concatenated GT path.

## 5. Mutation guard

```text
$ git rev-parse HEAD
6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd      # equals run_start_ref
unstaged UNCHANGED
staged   UNCHANGED
```

Verified by diffing `git diff` / `git diff --cached` against the captures taken immediately before
the `code_v1` companion call.
