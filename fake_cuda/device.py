"""
fake_cuda/device.py
────────────────────
Phase 1 stub — registers "remote" as the PrivateUse1 backend name so PyTorch
accepts `tensor.to("remote")` as a valid device string.

Phase 2 will extend this with:
  • A real memory allocator (C++ extension or ctypes bridge)
  • Renaming the backend to "cuda" so user code stays zero-change
  • Registering a stream/event API shim
"""

import torch

DEVICE_NAME = "remote"          # Phase 1 name; renamed to "cuda" in Phase 2
_registered = False


def register() -> None:
    """
    Register our custom PrivateUse1 backend so PyTorch understands
    `torch.device("remote")` without needing a C++ extension.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    global _registered
    if _registered:
        return

    # Tell PyTorch that PrivateUse1 tensors should display/address as "remote".
    # After this call:
    #   tensor.to("remote")  →  PrivateUse1 tensor
    #   tensor.device.type  →  "privateuseone"   (internal)
    #   str(tensor.device)  →  "remote:0"         (user-facing)
    torch.utils.rename_privateuse1_backend(DEVICE_NAME)

    _registered = True
