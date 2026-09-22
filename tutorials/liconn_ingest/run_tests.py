#!/usr/bin/env python3
"""Run the ingest tests without pytest (the image does not carry it, and these
gate the Docker build).

`test_naming` and `test_ingest` are pure and always run. `test_zarr_writer` and
`test_nd2_source` need numpy/zarr/scikit-image and are SKIPPED, loudly, when
those are absent -- so this file is still useful on the host, while the image
build (where they are installed) runs the full set.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MODULES = ["test_naming", "test_ingest", "test_zarr_writer", "test_nd2_source"]

passed = failed = skipped = 0
for name in MODULES:
    try:
        mod = __import__(name)
    except ModuleNotFoundError as exc:
        print(f"SKIP  {name}  (missing {exc.name})")
        skipped += 1
        continue
    print(f"\n{name}")
    for fn_name, fn in sorted(vars(mod).items()):
        if not (fn_name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            print(f"  PASS  {fn_name}")
            passed += 1
        except Exception:
            failed += 1
            print(f"  FAIL  {fn_name}")
            traceback.print_exc()

print(f"\n{passed} passed, {failed} failed, {skipped} module(s) skipped")
sys.exit(1 if failed else 0)
