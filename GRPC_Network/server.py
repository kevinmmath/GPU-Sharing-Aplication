"""
GRPC_Network/server.py
──────────────────────────────────────────────────────────────────────────────
gRPC server — runs on the GPU machine (WSL2 / Linux / Raspberry Pi).

For every ExecuteOp RPC it:
  1. Deserialises the OpRequest (op name + tensor args + non-tensor args).
  2. Looks up the ATen op via torch.ops.
  3. Runs it on CUDA (or CPU if no GPU is available).
  4. Serialises the result and sends it back.

Run with:
    python server.py              # listens on 0.0.0.0:50051
    python server.py 50052        # custom port
"""

import sys
import os
import pickle
import logging

import grpc
from concurrent import futures
import torch

# ── Make local imports work whether run from the GRPC_Network/ dir or project root
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gpu_service_pb2       as pb2
import gpu_service_pb2_grpc  as pb2_grpc
from rpc_utils import unpack_call, pack_result

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [server] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 50051


# ── Op lookup ─────────────────────────────────────────────────────────────────

def _lookup_op(op_name: str):
    """
    Resolve an ATen op string such as 'aten.mm.default' to a callable.

    str(func) in __torch_dispatch__ gives dot-separated names:
        'aten.mm.default'
        'aten.addmm.default'
        'aten.threshold_backward.default'
    We walk torch.ops attribute-by-attribute.
    """
    parts = op_name.split(".")          # ['aten', 'mm', 'default']
    obj   = torch.ops
    for part in parts:
        obj = getattr(obj, part)
    return obj


# ── Servicer ──────────────────────────────────────────────────────────────────

class GPUServicer(pb2_grpc.GPUServiceServicer):

    def ExecuteOp(self, request, context):
        try:
            # ── 1. Deserialise ────────────────────────────────────────────────
            op_name, args, kwargs = unpack_call(request, device=DEVICE)
            log.info(f"op={op_name!r}  tensors={len(request.tensors)}")

            # ── 2. Look up the ATen op ────────────────────────────────────────
            try:
                op_fn = _lookup_op(op_name)
            except AttributeError as exc:
                err = f"Unknown op '{op_name}': {exc}"
                log.warning(err)
                return pb2.OpResult(error=err)

            # ── 3. Execute on GPU ─────────────────────────────────────────────
            result = op_fn(*args, **kwargs)

            # ── 4. Serialise result ───────────────────────────────────────────
            response = pack_result(result, pb2.TensorData, pb2.OpResult)
            log.info(f"  → OK  result_tensors={len(response.tensors)}")
            return response

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            log.error(f"ExecuteOp failed for '{request.op_name}': {err}")
            return pb2.OpResult(error=err)


# ── Server bootstrap ──────────────────────────────────────────────────────────

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_GPUServiceServicer_to_server(GPUServicer(), server)
    server.add_insecure_port(f"0.0.0.0:{PORT}")
    server.start()
    log.info(f"Device : {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    log.info(f"Listening on 0.0.0.0:{PORT}  (Ctrl-C to stop)")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()