"""
examples/demo.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 Demo — Forward pass ops routed to remote GPU over gRPC.

Usage
─────
  Phase 1 (log only, no server needed):
      python examples/demo.py

  Phase 2 (remote GPU execution):
      python examples/demo.py --host 192.168.1.10 --port 50051

Expected Phase 2 output
───────────────────────
  [remote_client] ✔  TCP reachable
  [remote_client] ✔  gRPC channel open → 192.168.1.10:50051
  [LOG]  aten::t                       shape=(128, 784) …
  [LOG]  aten::addmm                   shape=(128,) …
  …
  → loss value: 2.3014   (computed from remote results)

Note on backward pass
─────────────────────
Backward / autograd is Phase 3 work.  Tensors returned from the remote GPU
are plain CPU tensors without a grad_fn, so loss.backward() is intentionally
skipped in Phase 2 mode.  It runs locally when no host is given (Phase 1).
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import fake_cuda

# ─── CLI args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="fake_cuda demo")
parser.add_argument("--host", default="",      help="Remote GPU server IP (Phase 2)")
parser.add_argument("--port", default=50051, type=int, help="gRPC port (default 50051)")
args = parser.parse_args()

REMOTE = bool(args.host)

# ─── Model (same code a friend would write — zero changes) ────────────────────
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

# ─── Enable interception (+ optional remote connection) ───────────────────────
if REMOTE:
    print(f"\n  Mode : Phase 2 — remote GPU at {args.host}:{args.port}")
    fake_cuda.enable(host=args.host, port=args.port)
else:
    print("\n  Mode : Phase 1 — log only (no server)")
    fake_cuda.enable()

# ─── Forward pass ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FORWARD PASS")
print("=" * 60)

try:
    output = model(x)
except Exception:
    import traceback
    with open("error_trace.txt", "w") as f:
        traceback.print_exc(file=f)
    print("\n  !! CRASHED — full traceback written to error_trace.txt\n")
    traceback.print_exc()
    fake_cuda.disable()
    sys.exit(1)

# ─── Loss ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LOSS")
print("=" * 60)

try:
    loss = loss_fn(output, target)
    print(f"\n  → loss value: {loss.item():.4f}\n")
except Exception:
    import traceback
    with open("error_trace.txt", "w") as f:
        traceback.print_exc(file=f)
    print("\n  !! LOSS CRASHED — full traceback written to error_trace.txt\n")
    traceback.print_exc()
    fake_cuda.disable()
    sys.exit(1)

# ─── Backward pass ────────────────────────────────────────────────────────────
if not REMOTE:
    # Phase 1: tensors have full grad_fn, backward works locally.
    print("=" * 60)
    print("  BACKWARD PASS (gradients)")
    print("=" * 60)
    loss.backward()
    print("  → backward complete\n")
else:
    print("  (Backward pass skipped in Phase 2 — remote tensors have no grad_fn)")
    print("  (Phase 3 will add autograd support.)\n")

# ─── Done ─────────────────────────────────────────────────────────────────────
fake_cuda.disable()

print("=" * 60)
if REMOTE:
    print("  Done!  Forward ops executed on the remote GPU.")
else:
    print("  Done!  Every op above would become a network call in Phase 2.")
print("=" * 60 + "\n")
