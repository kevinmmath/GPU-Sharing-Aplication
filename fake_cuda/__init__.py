"""
fake_cuda — Phase 1: Intercept & Log
══════════════════════════════════════════════════════════════════════════════

Public API
──────────
    import fake_cuda

    # Option A — persistent global hook (installs for the lifetime of the process)
    fake_cuda.enable()
    ...your torch code...
    fake_cuda.disable()         # optional cleanup

    # Option B — context manager (recommended for scoped interception)
    with fake_cuda.intercept():
        output = model(input)
        loss.backward()

    # Option C — move a specific tensor to the "remote" device
    t = torch.randn(3, 3).to(fake_cuda.DEVICE_NAME)   # "remote"

How it works (one-liner)
────────────────────────
`enable()` pushes a `TorchDispatchMode` onto PyTorch's global dispatch stack.
Every ATen op (matmul, relu, backward, copy_, …) prints a [LOG] line before
executing normally on whatever hardware the tensors are actually stored on.

Phase 2 upgrade
───────────────
Replace the pass-through dispatch call in `interceptor.GPUShareMode` with
serialization + ZeroMQ send/receive.  The rest of the public API stays the same.
"""

from contextlib import contextmanager

from .device        import register, DEVICE_NAME
from .interceptor   import GPUShareMode, install as _install, uninstall as _uninstall
from .              import remote_client

# Register the "remote" PrivateUse1 backend as soon as this package is imported.
register()

__all__ = [
    "enable", "disable", "intercept", "connect", "disconnect",
    "GPUShareMode", "DEVICE_NAME",
]


def enable(host: str = "", port: int = 50051) -> None:
    """
    Install the op-interception hook globally.
    If host is provided, also open a gRPC connection to the remote GPU server
    so every intercepted op is executed there instead of locally.

    Example (Phase 2 — remote GPU):
        fake_cuda.enable(host="192.168.1.10", port=50051)

    Example (Phase 1 — log only, no network):
        fake_cuda.enable()
    """
    if host:
        remote_client.connect(host=host, port=port)
    _install()


def disable() -> None:
    """Remove the op-interception hook and close any gRPC connection."""
    _uninstall()
    remote_client.disconnect()


@contextmanager
def intercept():
    """
    Context manager — intercept and log every PyTorch op for the duration of
    the `with` block.

        with fake_cuda.intercept():
            output = model(input)
    """
    with GPUShareMode():
        yield
