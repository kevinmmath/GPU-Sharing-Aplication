import io
import pickle
import numpy as np
import torch
import grpc
import os
import socket

# Clear any system proxies that cause 'tcp handshaker shutdown' in gRPC
for proxy_var in ['http_proxy', 'https_proxy', 'grpc_proxy', 'all_proxy']:
    os.environ.pop(proxy_var, None)
    os.environ.pop(proxy_var.upper(), None)
os.environ['no_proxy'] = '*'

import gpu_service_pb2
import gpu_service_pb2_grpc
from model import SimpleModel

import sys

# Default IP, can be overridden by passing the IP as a command-line argument
HOST_IP = "192.168.112.1"
PORT    = 50051          # <-- must match server.py


def get_dummy_batch(batch_size=32):
    """Swap this for your real DataLoader in Stage 3."""
    data   = torch.randn(batch_size, 784)
    labels = torch.randint(0, 10, (batch_size,), dtype=torch.int64)
    return data, labels


def main():
    # Build model on client CPU
    model = SimpleModel()
    state_dict = model.state_dict()

    # Serialise model state dict to bytes
    model_bytes = pickle.dumps(state_dict)

    # Get a batch
    batch_data, batch_labels = get_dummy_batch(batch_size=32)

    # Serialise tensors to raw bytes (no pickle — safer for large tensors)
    data_bytes   = batch_data.numpy().astype(np.float32).tobytes()
    labels_bytes = batch_labels.numpy().astype(np.int64).tobytes()

    print(f"[client] Testing raw TCP connection to {HOST_IP}:{PORT}...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((HOST_IP, PORT))
        s.close()
        print(f"[client] TCP connection successful! Port is open.")
    except Exception as e:
        print(f"[client] TCP connection FAILED: {e}")
        print(f"         This means the Server isn't running, the IP is wrong, or a Firewall/Antivirus on the Server is still blocking the connection.")
        import sys
        sys.exit(1)

    print(f"[client] Connecting to host at {HOST_IP}:{PORT} via protocol...")
    print(f"[client] Sending model ({len(state_dict)} layers) + batch {list(batch_data.shape)}")

    # ⚠ This is the gRPC channel — HOST_IP:PORT must be reachable
    with grpc.insecure_channel(f"{HOST_IP}:{PORT}") as channel:
        stub = gpu_service_pb2_grpc.GPUServiceStub(channel)

        request = gpu_service_pb2.DataRequest(
            model_state  = model_bytes,
            batch_data   = data_bytes,
            batch_labels = labels_bytes,
            data_shape   = list(batch_data.shape),
            label_shape  = list(batch_labels.shape),
        )

        response = stub.SendData(request)

    print(f"[client] Response from host:")
    print(f"         status  : {response.status}")
    print(f"         device  : {response.device}")
    print(f"         message : {response.message}")


if __name__ == "__main__":
    main()