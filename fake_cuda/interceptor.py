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
import ctypes
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


# ── In-place op helpers ────────────────────────────────────────────────────────

# ATen in-place ops end with "_" (e.g. aten.copy_.default, aten.add_.Tensor)
_INPLACE_SUFFIXES = {"copy_", "add_", "mul_", "sub_", "div_", "fill_",
                     "zero_", "scatter_", "index_put_", "masked_fill_"}


def _is_inplace(op_name: str) -> bool:
    """Check if an op is in-place (mutates its first argument)."""
    # op_name looks like "aten.copy_.default"
    parts = op_name.split(".")
    if len(parts) >= 2:
        return parts[1] in _INPLACE_SUFFIXES
    return False


def _sync_inplace(dst: torch.Tensor, src: torch.Tensor):
    """
    Copy raw bytes from src into dst using ctypes.memmove.
    Both tensors must have the same shape/dtype.
    No ATen ops called — safe inside __torch_dispatch__.
    """
    nbytes = 1
    for d in dst.shape:
        nbytes *= d
    nbytes *= dst.element_size()
    if nbytes > 0:
        ctypes.memmove(dst.data_ptr(), src.data_ptr(), nbytes)


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
                result = remote_client.call_op(str(func), args, kwargs)

                # Handle in-place ops (copy_, add_, mul_, etc.)
                # In Phase 2.5, data lives on the server. We just attach the
                # server's returned remote_id to our original local tensor.
                op_name = str(func)
                if _is_inplace(op_name) and isinstance(result, torch.Tensor):
                    result_id = getattr(result, "remote_id", "")
                    if not result_id:
                        from rpc_utils import _CLIENT_REGISTRY
                        result_id = _CLIENT_REGISTRY.get(result._cdata, "")
                        
                    if result_id:
                        setattr(args[0], "remote_id", result_id)
                        from rpc_utils import _CLIENT_REGISTRY
                        _CLIENT_REGISTRY[args[0]._cdata] = result_id
                    return args[0]   # return the original (now updated) tensor

                return result
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
