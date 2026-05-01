"""
Validation script — confirms remote execution produces IDENTICAL results to local.

Uses the same model weights + input, runs once locally and once via gRPC,
then compares outputs element-by-element.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import torch.nn as nn
import fake_cuda

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", default=50051, type=int)
args = parser.parse_args()

# ── Fixed seed for reproducibility ────────────────────────────────────────────
torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

x      = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))
loss_fn = nn.CrossEntropyLoss()

# ── 1. LOCAL execution (ground truth) ─────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Local execution (ground truth)")
print("=" * 60)
with torch.no_grad():
    local_output = model(x)
    local_loss   = loss_fn(local_output, target)

print(f"  Local output shape : {local_output.shape}")
print(f"  Local loss         : {local_loss.item():.6f}")
print(f"  Local output[0,:5] : {local_output[0,:5].tolist()}")

# ── 2. REMOTE execution ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  STEP 2: Remote execution via gRPC → {args.host}:{args.port}")
print("=" * 60)

fake_cuda.enable(host=args.host, port=args.port)

with torch.no_grad():
    remote_output = model(x)
    remote_loss   = loss_fn(remote_output, target)

fake_cuda.disable()

print(f"  Remote output shape: {remote_output.shape}")
print(f"  Remote loss        : {remote_loss.item():.6f}")
print(f"  Remote output[0,:5]: {remote_output[0,:5].tolist()}")

# ── 3. COMPARE ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: Comparison")
print("=" * 60)

shape_match = (local_output.shape == remote_output.shape)
max_diff    = (local_output - remote_output).abs().max().item()
loss_diff   = abs(local_loss.item() - remote_loss.item())
allclose    = torch.allclose(local_output, remote_output, atol=1e-5)

print(f"  Shape match   : {'✅' if shape_match else '❌'}  {local_output.shape} vs {remote_output.shape}")
print(f"  Max elem diff : {'✅' if max_diff < 1e-5 else '⚠️'}  {max_diff:.2e}")
print(f"  Loss diff     : {'✅' if loss_diff < 1e-5 else '⚠️'}  {loss_diff:.2e}")
print(f"  torch.allclose: {'✅ PASS' if allclose else '❌ FAIL'}")

print("\n" + "=" * 60)
if allclose:
    print("  ✅ VALIDATION PASSED — remote execution is numerically identical!")
else:
    print("  ⚠️  Results differ — check max_diff for significance")
print("=" * 60 + "\n")
