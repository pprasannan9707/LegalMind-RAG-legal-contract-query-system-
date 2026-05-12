"""
debug_dataset.py - Run this to inspect the exact CUAD dataset structure on your machine.
"""
from datasets import load_dataset

print("Loading CUAD (this uses cached version)...")
ds = load_dataset("theatticusproject/cuad", split="train")

print(f"\nTotal rows: {len(ds)}")
print(f"\nFeature schema:")
print(ds.features)

print("\n--- First row, all fields ---")
item = ds[0]
for k, v in item.items():
    print(f"\n  KEY: {k!r}")
    print(f"  TYPE: {type(v).__name__}")
    print(f"  VALUE: {repr(v)[:500]}")

print("\n--- Checking 'context' field in first 10 rows ---")
for i in range(10):
    ctx = ds[i].get("context", "MISSING")
    print(f"  Row {i}: context length = {len(ctx) if ctx != 'MISSING' else 'MISSING'}, first 80 chars: {repr(str(ctx)[:80])}")

print("\n--- Checking 'answers' field in first 10 rows ---")
for i in range(10):
    ans = ds[i].get("answers", "MISSING")
    print(f"  Row {i}: answers type={type(ans).__name__}, value={repr(ans)[:200]}")
