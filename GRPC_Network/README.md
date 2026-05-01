# GPU Sharing via gRPC (Windows ↔ WSL2)

A minimal proof-of-concept that sends a PyTorch model state-dict and a data batch
from a **Windows client** to a **WSL2 server** (or any remote Linux machine with a
GPU) over gRPC, runs the tensors on the remote device, and returns an acknowledgement.

---

## Architecture

```
┌─────────────────────────────┐          gRPC / TCP          ┌──────────────────────────────┐
│  Windows Machine (Client)   │  ─────────────────────────►  │  WSL2 / Linux Server (GPU)   │
│                             │                              │                              │
│  client.py                  │   DataRequest (proto)        │  server.py                   │
│  • builds SimpleModel       │ ─────────────────────────►  │  • deserialises model        │
│  • serialises state_dict    │                              │  • loads tensors onto CUDA   │
│  • sends batch data         │   DataAck (proto)            │  • returns status + device   │
│                             │ ◄─────────────────────────  │                              │
└─────────────────────────────┘    PORT 50051                └──────────────────────────────┘
```

### Files

| File | Role |
|------|------|
| `gpu_service.proto` | Protobuf service & message definitions |
| `gpu_service_pb2.py` | Auto-generated message classes |
| `gpu_service_pb2_grpc.py` | Auto-generated gRPC stubs |
| `model.py` | `SimpleModel` — a 3-layer MLP (784 → 256 → 10) |
| `server.py` | gRPC server — runs on the GPU machine (WSL2) |
| `client.py` | gRPC client — runs on Windows |

---

## Prerequisites

### Both machines
```
Python >= 3.9
pip install torch grpcio grpcio-tools numpy
```

### Quick dependency install
```bash
pip install torch grpcio grpcio-tools numpy
```

> **Tip:** If you are using WSL2, create a separate virtual environment there so
> packages do not conflict with Windows packages.

---

## Step 1 — Find the WSL2 IP Address

The Windows client needs the IP address of the WSL2 instance.

**Inside WSL2, run:**
```bash
hostname -I
```
Example output:
```
172.24.176.5
```
Copy this IP — you will need it in the next step.

> **Note:** The WSL2 IP changes every time WSL2 restarts. Always re-run `hostname -I`
> before starting a new session.

---

## Step 2 — Configure the Client

Open `client.py` and replace the placeholder IP with the one from Step 1:

```python
# client.py  line 13
HOST_IP = "172.24.176.5"   # ← paste your WSL2 IP here
PORT    = 50051             # ← must match server.py
```

---

## Step 3 — Start the Server (WSL2 / Linux)

Inside WSL2 (or any Linux terminal), navigate to the project directory and run:

```bash
python server.py
```

Expected output:
```
[host] gRPC server listening on port 50051
[host] Run `hostname -I` in WSL2 to get the IP for the client
```

The server will block and wait for incoming connections.

---

## Step 4 — Run the Client (Windows)

Open a **Windows** PowerShell or Command Prompt, navigate to the project directory
and run:

```powershell
python client.py
```

Expected client output:
```
[client] Connecting to host at 172.24.176.5:50051
[client] Sending model (4 layers) + batch [32, 784]
[client] Response from host:
         status  : received
         device  : cuda:0        ← "cpu" if no GPU is available on WSL2
         message : Model (4 layers) and batch [32, 784] received OK
```

Expected server output (WSL2 terminal):
```
[host] Device available: cuda
[host] Model loaded onto cuda: 4 layers
[host] Batch received — data: torch.Size([32, 784]), labels: torch.Size([32])
[host] Both tensors on cuda:0
```

---

## Verifying the Connection

### Quick connectivity check (before running Python)

From Windows PowerShell, test whether the port is reachable:
```powershell
Test-NetConnection -ComputerName 172.24.176.5 -Port 50051
```
Look for `TcpTestSucceeded : True`.

Alternatively, use `curl` or `telnet`:
```powershell
# telnet (may need to be enabled via Windows Features)
telnet 172.24.176.5 50051
```

### WSL2 firewall — allow port 50051 (if connection is refused)

Inside WSL2:
```bash
sudo ufw allow 50051/tcp   # if ufw is active
```

On Windows (run PowerShell as Administrator):
```powershell
# Add inbound rule for WSL2 → Windows communication on port 50051
New-NetFirewallRule -DisplayName "gRPC WSL2" `
    -Direction Inbound -Protocol TCP -LocalPort 50051 -Action Allow
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `StatusCode.UNAVAILABLE` on client | Wrong IP or server not started | Re-run `hostname -I` in WSL2, update `client.py` |
| `Connection refused` | Server not listening / firewall | Start `server.py` first, open port 50051 |
| `ModuleNotFoundError: gpu_service_pb2` | Missing generated files | Run `python -m grpc_tools.protoc` (see below) |
| `CUDA not available` on server | No GPU in WSL2 or drivers missing | Install CUDA-enabled WSL2 drivers from NVIDIA |
| IP changes after WSL restart | WSL2 dynamic IP | Always re-run `hostname -I` and update `client.py` |

### Re-generate protobuf files (if you edit the `.proto`)

From the project root (can be run on either machine):
```bash
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    gpu_service.proto
```

This regenerates `gpu_service_pb2.py` and `gpu_service_pb2_grpc.py`.

---

## gRPC Message Reference

### `DataRequest` (client → server)

| Field | Type | Description |
|-------|------|-------------|
| `model_state` | `bytes` | `pickle`-serialised `state_dict` |
| `batch_data` | `bytes` | Raw `float32` bytes of the input tensor |
| `batch_labels` | `bytes` | Raw `int64` bytes of the label tensor |
| `data_shape` | `repeated int32` | Shape of the data tensor, e.g. `[32, 784]` |
| `label_shape` | `repeated int32` | Shape of the labels tensor, e.g. `[32]` |

### `DataAck` (server → client)

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string` | `"received"` on success |
| `device` | `string` | Device used, e.g. `"cuda:0"` or `"cpu"` |
| `message` | `string` | Human-readable summary |

---

## Next Steps

- **Stage 3**: Replace `get_dummy_batch()` in `client.py` with your real `DataLoader`.
- **Security**: Switch from `insecure_channel` to TLS (`grpc.ssl_channel_credentials`).
- **Scalability**: Increase `max_workers` in `server.py` for concurrent requests.
- **Static IP**: Set a fixed WSL2 IP via `.wslconfig` to avoid updating `HOST_IP` every restart.
