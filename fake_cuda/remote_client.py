"""
fake_cuda/remote_client.py
──────────────────────────────────────────────────────────────────────────────
Persistent gRPC connection used by the interceptor.

Architecture (inside __torch_dispatch__)
────────────────────────────────────────
1. pack_call    — main thread + enable_reentrant_dispatch (ctypes, no ATen ops)
2. gRPC call    — main thread (pure network I/O)
3. unpack_result — background thread (torch.from_numpy needs clean dispatch stack)

Step 3 MUST run in a separate thread because tensor creation methods
(from_numpy, new_empty, empty, etc.) all return NULL when called inside
__torch_dispatch__, even with enable_reentrant_dispatch().
TorchDispatchMode is thread-local, so a worker thread has no active mode.
"""

import os
import sys
import socket
import logging
import threading

import grpc

# ── Make GRPC_Network importable ──────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_GRPC_DIR = os.path.join(_HERE, "..", "GRPC_Network")
if _GRPC_DIR not in sys.path:
    sys.path.insert(0, _GRPC_DIR)

import gpu_service_pb2       as pb2
import gpu_service_pb2_grpc  as pb2_grpc
from rpc_utils import pack_call, unpack_result_normal

log = logging.getLogger("fake_cuda")

# ── Connection state ──────────────────────────────────────────────────────────
_channel = None
_stub    = None
_host    = ""
_port    = 50051


def is_connected():
    return _stub is not None


def connect(host="localhost", port=50051):
    global _channel, _stub, _host, _port
    disconnect()
    _host, _port = host, port

    # Clear proxy env-vars
    for v in ["http_proxy", "https_proxy", "grpc_proxy", "all_proxy"]:
        os.environ.pop(v, None)
        os.environ.pop(v.upper(), None)
    os.environ["no_proxy"] = "*"

    # TCP pre-check
    log.info(f"[remote_client] TCP pre-check → {host}:{port} …")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((host, port))
        s.close()
        log.info("[remote_client] ✔  TCP reachable")
    except OSError as exc:
        raise ConnectionError(
            f"[remote_client] Cannot reach {host}:{port} — "
            f"is the server running?  ({exc})"
        ) from exc

    _channel = grpc.insecure_channel(f"{host}:{port}")
    _stub    = pb2_grpc.GPUServiceStub(_channel)
    log.info(f"[remote_client] ✔  gRPC channel open → {host}:{port}")


def disconnect():
    global _channel, _stub
    if _channel is not None:
        try:
            _channel.close()
        except Exception:
            pass
        _channel = None
        _stub    = None
        log.info("[remote_client] ✘  gRPC channel closed")


# ── Core RPC call ─────────────────────────────────────────────────────────────

def call_op(op_name, args, kwargs):
    """
    Called from __torch_dispatch__ (inside enable_reentrant_dispatch).

    1. pack_call   → main thread (ctypes only, safe inside dispatch)
    2. gRPC call   → main thread (pure network I/O)
    3. unpack      → worker thread (torch.from_numpy needs no dispatch mode)
    """
    if _stub is None:
        raise RuntimeError("remote_client not connected")

    # ── 1. Serialise (ctypes, no tensor creation) ─────────────────────────
    request = pack_call(op_name, args, kwargs, pb2.TensorData, pb2.OpRequest)

    # ── 2. gRPC call (network I/O) ───────────────────────────────────────
    response = _stub.ExecuteOp(request)
    if response.error:
        raise RuntimeError(f"Server error: {response.error}")

    # ── 3. Deserialise in worker thread ──────────────────────────────────
    #    TorchDispatchMode is thread-local.
    #    The worker thread has NO active mode → torch.from_numpy() works.
    result_box = [None, None]  # [result, exception]

    def _unpack_worker():
        try:
            result_box[0] = unpack_result_normal(response)
        except Exception as e:
            result_box[1] = e

    t = threading.Thread(target=_unpack_worker, daemon=True)
    t.start()
    t.join()

    if result_box[1] is not None:
        raise result_box[1]
    return result_box[0]
