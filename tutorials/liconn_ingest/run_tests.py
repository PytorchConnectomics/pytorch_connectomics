#!/usr/bin/env python3
"""Run the naming/spacing tests without requiring pytest (the base image may
not carry it, and these gate the Docker build)."""
import sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import test_naming as t

fns = [(n, f) for n, f in vars(t).items() if n.startswith("test_") and callable(f)]
bad = 0
for n, f in sorted(fns):
    try:
        f()
        print(f"  PASS  {n}")
    except Exception:
        bad += 1
        print(f"  FAIL  {n}")
        traceback.print_exc()
print(f"\n{len(fns) - bad}/{len(fns)} passed")
sys.exit(1 if bad else 0)
