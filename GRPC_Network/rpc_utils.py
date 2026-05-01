"""
GRPC_Network/rpc_utils.py
──────────────────────────────────────────────────────────────────────────────
Shared serialisation helpers for client ↔ server tensor+op exchange.

Two execution contexts
──────────────────────
A) **Client — inside __torch_dispatch__**
   Most tensor methods (detach, numpy, from_numpy, empty) are UNUSABLE here.
   We rely ONLY on:
     • Python properties: t.shape, t.dtype, t.requires_grad
     • C++ metadata accessors: t.stride(), t.data_ptr()
     • Tensor methods that have tensor args: t.new_empty(...)
     • Raw memory via ctypes

B) **Server — normal Python**
   Everything works normally.
"""

import ctypes
import pickle
import numpy as np
import torch
import uuid

_SERVER_REGISTRY = {}
_CLIENT_REGISTRY = {}



# ── dtype helpers ──────────────────────────────────────────────────────────────

_TORCH_TO_NP = {
    "float16": np.float16, "float32": np.float32, "float64": np.float64,
    "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
    "uint8": np.uint8, "bool": np.bool_,
}

_STR_TO_TORCH = {
    "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    "uint8": torch.uint8, "bool": torch.bool,
}

_ELEM_SIZE = {
    "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "bool": 1,
}


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).split(".")[-1]


# ── TensorRef sentinel ────────────────────────────────────────────────────────

class _TensorRef:
    __slots__ = ("idx",)
    def __init__(self, idx: int):
        self.idx = idx


# ── Tree walk ─────────────────────────────────────────────────────────────────

def _extract_tensors(obj):
    """Replace every Tensor with _TensorRef(idx), return (modified_tree, [tensors])."""
    tensors = []
    def _walk(x):
        if isinstance(x, torch.Tensor):
            idx = len(tensors)
            tensors.append(x)
            return _TensorRef(idx)
        if isinstance(x, (tuple, list)):
            cls = type(x)
            return cls(_walk(i) for i in x)
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        return x
    return _walk(obj), tensors


def _restore_tensors(obj, tensors):
    """Replace every _TensorRef(idx) with tensors[idx]."""
    def _walk(x):
        if isinstance(x, _TensorRef):
            return tensors[x.idx]
        if isinstance(x, (tuple, list)):
            cls = type(x)
            return cls(_walk(i) for i in x)
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        return x
    return _walk(obj)


# ── Tensor → Proto (dispatch-safe: ctypes only) ──────────────────────────────

def tensor_to_proto(t: torch.Tensor, pb2_TensorData, is_server=False):
    """
    Serialise tensor using ONLY metadata accessors + ctypes.
    Safe inside __torch_dispatch__.
    """
    shape     = tuple(t.shape)
    dtype_str = _dtype_str(t)
    
    remote_id = getattr(t, "remote_id", "")
    if not remote_id and t._cdata in _CLIENT_REGISTRY:
        remote_id = _CLIENT_REGISTRY[t._cdata]
        
    if is_server:
        # Server always registers the tensor and assigns an ID, skips data sending
        remote_id = str(uuid.uuid4())
        _SERVER_REGISTRY[remote_id] = t
        return pb2_TensorData(
            raw_data=b"", shape=list(shape),
            dtype=dtype_str, requires_grad=t.requires_grad,
            remote_id=remote_id
        )
        
    if remote_id:
        # Client tensor already has an ID, skip data sending
        return pb2_TensorData(
            raw_data=b"", shape=list(shape),
            dtype=dtype_str, requires_grad=t.requires_grad,
            remote_id=remote_id
        )

    elem_sz   = _ELEM_SIZE[dtype_str]
    strides   = tuple(t.stride())

    numel = 1
    for d in shape:
        numel *= d

    if numel == 0:
        raw = b""
    else:
        # Check contiguity manually
        exp = 1
        contig = True
        for i in range(len(shape) - 1, -1, -1):
            if strides[i] != exp:
                contig = False
                break
            exp *= shape[i]

        ptr = t.data_ptr()
        if contig:
            raw = bytes((ctypes.c_char * (numel * elem_sz)).from_address(ptr))
        else:
            # Non-contiguous: read via numpy stride trick
            np_dt = _TORCH_TO_NP[dtype_str]
            bstrides = tuple(s * elem_sz for s in strides)
            max_off = sum((d - 1) * s for d, s in zip(shape, strides))
            buf = (ctypes.c_char * ((max_off + 1) * elem_sz)).from_address(ptr)
            arr = np.ndarray(shape, dtype=np_dt, buffer=buf, strides=bstrides)
            arr = np.ascontiguousarray(arr)
            raw = arr.tobytes()

    return pb2_TensorData(
        raw_data=raw, shape=list(shape),
        dtype=dtype_str, requires_grad=t.requires_grad,
        remote_id=""
    )


# ── Proto → Tensor ────────────────────────────────────────────────────────────

def proto_to_tensor_dispatch_safe(td, template: torch.Tensor) -> torch.Tensor:
    """
    Create tensor inside __torch_dispatch__ context.
    Uses template.new_empty() (a TENSOR METHOD with tensor args) to avoid
    the 'no tensor args?' assertion that factory functions trigger.
    Then copies raw bytes in via ctypes.memmove.
    """
    shape = list(td.shape)
    dtype = _STR_TO_TORCH[td.dtype]
    t = template.new_empty(shape, dtype=dtype)
    nbytes = len(td.raw_data)
    if nbytes > 0:
        ctypes.memmove(t.data_ptr(), td.raw_data, nbytes)
    if td.requires_grad and t.is_floating_point():
        t.requires_grad_(True)
    return t


def proto_to_tensor_normal(td, device: str = "cpu", is_server=False) -> torch.Tensor:
    """
    Create tensor outside dispatch context (server-side or worker thread).
    Uses standard torch.from_numpy.
    NOTE: requires_grad is NOT set — PyTorch's autograd view machinery
    will handle it automatically, and setting it manually causes conflicts
    with differentiable-view ops like aten.t.
    """
    if is_server and td.remote_id:
        if td.remote_id in _SERVER_REGISTRY:
            return _SERVER_REGISTRY[td.remote_id]
        else:
            raise RuntimeError(f"Server cannot find remote_id {td.remote_id}")

    shape = list(td.shape)
    dtype = _STR_TO_TORCH[td.dtype]
    
    if len(td.raw_data) == 0:
        t = torch.empty(shape, dtype=dtype)
    else:
        np_dt = _TORCH_TO_NP[td.dtype]
        arr = np.frombuffer(td.raw_data, dtype=np_dt).reshape(shape).copy()
        t = torch.from_numpy(arr)
        
    if device != "cpu":
        t = t.to(device)
        
    if not is_server and td.remote_id:
        setattr(t, "remote_id", td.remote_id)
        _CLIENT_REGISTRY[t._cdata] = td.remote_id
        # We cannot use weakref.finalize on 't' because PyTorch Python wrappers
        # are short-lived and die immediately after being unwrapped to C++.
            
    return t


# ── High-level pack / unpack ─────────────────────────────────────────────────

def pack_call(op_name, args, kwargs, pb2_TensorData, pb2_OpRequest):
    """Serialise an ATen op call into an OpRequest proto."""
    combined, all_tensors = _extract_tensors((args, kwargs))
    args_pkl = pickle.dumps(combined)
    tensor_protos = [tensor_to_proto(t, pb2_TensorData, is_server=False) for t in all_tensors]
    return pb2_OpRequest(op_name=op_name, args_pkl=args_pkl, tensors=tensor_protos)


def unpack_call(request, device="cpu"):
    """Server-side: deserialise OpRequest → (op_name, args, kwargs)."""
    tensors = [proto_to_tensor_normal(td, device, is_server=True) for td in request.tensors]
    combined = pickle.loads(request.args_pkl)
    args, kwargs = _restore_tensors(combined, tensors)
    return request.op_name, args, kwargs


def pack_result(result, pb2_TensorData, pb2_OpResult):
    """Server-side: serialise op result into OpResult proto."""
    result_mod, tensors = _extract_tensors(result)
    result_pkl = pickle.dumps(result_mod)
    tensor_protos = [tensor_to_proto(t, pb2_TensorData, is_server=True) for t in tensors]
    return pb2_OpResult(result_pkl=result_pkl, tensors=tensor_protos)


def unpack_result_dispatch_safe(response, template: torch.Tensor):
    """Client-side inside __torch_dispatch__: uses dispatch-safe tensor creation."""
    # NOT USED ANYMORE IN PHASE 2
    pass


def unpack_result_normal(response, device="cpu"):
    """Normal context: uses torch.from_numpy."""
    tensors = [proto_to_tensor_normal(td, device, is_server=False) for td in response.tensors]
    result_mod = pickle.loads(response.result_pkl)
    return _restore_tensors(result_mod, tensors)
