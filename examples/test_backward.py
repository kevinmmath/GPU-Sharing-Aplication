"""Quick test: does the autograd graph exist on remote tensors?"""
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

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64),  nn.ReLU(),
    nn.Linear(64, 10),
)
x      = torch.randn(32, 784)
target = torch.randint(0, 10, (32,))
loss_fn = nn.CrossEntropyLoss()

fake_cuda.enable(host=args.host, port=args.port)

print("=" * 60)
print("  FORWARD PASS")
print("=" * 60)
output = model(x)
loss   = loss_fn(output, target)

print(f"\n  output.requires_grad = {output.requires_grad}")
print(f"  output.grad_fn       = {output.grad_fn}")
print(f"  loss.requires_grad   = {loss.requires_grad}")
print(f"  loss.grad_fn         = {loss.grad_fn}")

print("\n" + "=" * 60)
print("  BACKWARD PASS")
print("=" * 60)
try:
    loss.backward()
    print("\n  ✅ loss.backward() SUCCEEDED!")

    for i, layer in enumerate(model):
        if hasattr(layer, 'weight'):
            g = layer.weight.grad
            if g is not None:
                print(f"  Layer {i} weight.grad: shape={g.shape}, norm={g.norm():.4f}")
            else:
                print(f"  Layer {i} weight.grad: None")
except Exception as e:
    import traceback
    print(f"\n  ❌ loss.backward() FAILED: {e}")
    with open("backward_error.txt", "w") as f:
        traceback.print_exc(file=f)
    print("  Full traceback written to backward_error.txt")

fake_cuda.disable()
print("=" * 60)
