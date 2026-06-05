#!/usr/bin/env python3
"""
Quick verification script for prefetch_memory_ on Ascend NPU.

Usage:
    python verify_prefetch_memory.py
"""

import torch
import torch_npu
import time


def green(s):  return f"\033[32m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"

pass_count = 0
fail_count = 0


def check(name, condition, detail=""):
    global pass_count, fail_count
    if condition:
        pass_count += 1
        print(f"  {green('[PASS]')} {name}")
    else:
        fail_count += 1
        print(f"  {red('[FAIL]')} {name}  {detail}")


print("=" * 60)
print("prefetch_memory_ Verification")
print("=" * 60)

# ------------------------------------------------------------------
# Test 1: Basic blocking prefetch, float16
# ------------------------------------------------------------------
print("\n--- Test 1: Basic blocking prefetch (float16) ---")
x = torch.randn(4, 4, dtype=torch.float16, device="npu")
original = x.cpu().clone()
torch_npu.prefetch_memory_(x)
result = x.cpu()
check("data preserved after blocking prefetch", torch.equal(original, result))

# ------------------------------------------------------------------
# Test 2: Basic blocking prefetch, float32
# ------------------------------------------------------------------
print("\n--- Test 2: Basic blocking prefetch (float32) ---")
x = torch.randn(8, 8, dtype=torch.float32, device="npu")
original = x.cpu().clone()
torch_npu.prefetch_memory_(x)
result = x.cpu()
check("float32 data preserved", torch.equal(original, result))

# ------------------------------------------------------------------
# Test 3: Non-blocking prefetch
# ------------------------------------------------------------------
print("\n--- Test 3: Non-blocking prefetch ---")
x = torch.randn(8, 8, dtype=torch.float16, device="npu")
original = x.cpu().clone()
torch_npu.prefetch_memory_(x, non_blocking=True)
torch.npu.synchronize()
result = x.cpu()
check("data preserved after non-blocking prefetch", torch.equal(original, result))

# ------------------------------------------------------------------
# Test 4: Empty tensor (should not crash)
# ------------------------------------------------------------------
print("\n--- Test 4: Empty tensor ---")
try:
    x = torch.empty(0, device="npu", dtype=torch.float16)
    torch_npu.prefetch_memory_(x)
    check("empty tensor handled without error", True)
except Exception as e:
    check("empty tensor handled without error", False, str(e))

# ------------------------------------------------------------------
# Test 5: CPU tensor should raise
# ------------------------------------------------------------------
print("\n--- Test 5: CPU tensor rejection ---")
try:
    x = torch.randn(4, 4)
    torch_npu.prefetch_memory_(x)
    check("CPU tensor raises RuntimeError", False, "no error raised")
except RuntimeError:
    check("CPU tensor raises RuntimeError", True)
except Exception as e:
    check("CPU tensor raises RuntimeError", False, f"unexpected: {type(e).__name__}: {e}")

# ------------------------------------------------------------------
# Test 6: Multiple calls (idempotency)
# ------------------------------------------------------------------
print("\n--- Test 6: Multiple calls ---")
x = torch.randn(16, 16, dtype=torch.float16, device="npu")
original = x.cpu().clone()
for _ in range(10):
    torch_npu.prefetch_memory_(x)
result = x.cpu()
check("data preserved after 10 prefetch calls", torch.equal(original, result))

# ------------------------------------------------------------------
# Test 7: Large tensor
# ------------------------------------------------------------------
print("\n--- Test 7: Large tensor (512x512 float32) ---")
x = torch.randn(512, 512, dtype=torch.float32, device="npu")
original = x.cpu().clone()
t0 = time.time()
torch_npu.prefetch_memory_(x)
elapsed = time.time() - t0
result = x.cpu()
check("large tensor data preserved", torch.equal(original, result))
print(f"  {yellow('[INFO]')} prefetch_memory_ on 512x512 float32 took {elapsed*1000:.2f} ms")

# ------------------------------------------------------------------
# Test 8: 3D tensor
# ------------------------------------------------------------------
print("\n--- Test 8: 3D tensor ---")
x = torch.randn(2, 3, 4, dtype=torch.float16, device="npu")
original = x.cpu().clone()
torch_npu.prefetch_memory_(x)
result = x.cpu()
check("3D tensor data preserved", torch.equal(original, result))

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
total = pass_count + fail_count
print(f"Results: {green(str(pass_count))} passed, {red(str(fail_count))} failed, {total} total")
if fail_count == 0:
    print(green("All tests passed!"))
else:
    print(red(f"{fail_count} test(s) failed!"))
print("=" * 60)
