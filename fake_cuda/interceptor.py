"""
fake_cuda/interceptor.py
──────────────────────────────────────────────────────────────────────────────
TorchDispatchMode that intercepts every ATen op.

Phase 1: Log ops and execute locally.
Phase 2: Serialise → gRPC → remote GPU → return result.

The key to Phase 2 is torch.overrides.enable_reentrant_dispatch() which
allows calling tensor methods from within __torch_dispatch__.  All tensor
serialisation uses ctypes (no ATen ops), and tensor creation uses
template.new_empty() (a tensor method, not a factory).
"""

import sys
import logging
import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from . import remote_client

# ── Logging ────────────────────────────────────────────────────────────────────
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))

log = logging.getLogger("fake_cuda")
log.addHandler(_handler)
log.setLevel(logging.DEBUG)
log.propagate = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tensor_summary(t):
    dtype = str(t.dtype).split(".")[-1]
    dev   = str(t.device)
    return f"shape={tuple(t.shape)} dtype={dtype} dev={dev}"


def _op_label(func):
    parts = str(func).split(".")
    if len(parts) >= 2:
        return f"{parts[0]}::{parts[1]}"
    return str(func)


# ── Dispatch mode ──────────────────────────────────────────────────────────────

class GPUShareMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # ── Log ────────────────────────────────────────────────────────────
        leaves = pytree.tree_leaves(args) + pytree.tree_leaves(list(kwargs.values()))
        summaries = [_tensor_summary(t) for t in leaves if isinstance(t, torch.Tensor)]
        label = _op_label(func)
        info  = "  |  ".join(summaries) if summaries else "(no tensors)"
        log.info(f"[LOG]  {label:<35}  {info}")

        # ── Phase 2: remote execution ─────────────────────────────────────
        if remote_client.is_connected():
            try:
                return remote_client.call_op(str(func), args, kwargs)
            except Exception as exc:
                log.warning(
                    f"[remote] op '{label}' failed ({exc}), "
                    f"falling back to local execution"
                )

        # ── Phase 1 / fallback: local execution ───────────────────────────
        return func(*args, **kwargs)


# ── Install / uninstall ───────────────────────────────────────────────────────

_active_mode = None


def install():
    global _active_mode
    if _active_mode is not None:
        return
    _active_mode = GPUShareMode()
    _active_mode.__enter__()
    log.info("[fake_cuda] ✔  Interception ACTIVE — every ATen op will be logged\n")


def uninstall():
    global _active_mode
    if _active_mode is None:
        return
    _active_mode.__exit__(None, None, None)
    _active_mode = None
    log.info("\n[fake_cuda] ✘  Interception DISABLED")
