# GPU Backend (NVIDIA CUDA / AMD ROCm)

> **Prerequisites:** Read the [Architecture Deep Dive](architecture.md),
> [Compilation Pipeline](compilation_pipeline.md),
> [Execution Pipeline](execution_pipeline.md), and
> [Buffer Management](buffer_management.md) for the common PJRT concepts this
> backend implements.

This document covers the GPU-specific PJRT implementation, supporting both
NVIDIA CUDA and AMD ROCm platforms.

## Table of Contents

- [Overview](#overview)
- [Client Initialization](#client-initialization)
- [Memory Allocators](#memory-allocators)
- [Stream Management](#stream-management)
- [Execution Flow](#execution-flow)
- [Multi-GPU and Distributed](#multi-gpu-and-distributed)
- [GPU Extensions](#gpu-extensions)
- [C API Entry Points](#c-api-entry-points)
- [Further Resources](#further-resources)

---

## Overview

The GPU backend provides **two runtime paths**:

```mermaid
classDiagram
    class PjRtClient {
        <<abstract>>
    }

    class CommonPjRtClient {
        <<abstract>>
    }

    class PjRtStreamExecutorClient {
        +StreamExecutor integration
        +LocalDeviceState per device
    }

    class StreamExecutorGpuClient {
        +RunAsync()
        +Multi-GPU topology
        +NCCL/RCCL collectives
    }

    class TfrtGpuClient {
        +TFRT runtime
        +Alternative execution path
    }

    PjRtClient <|-- CommonPjRtClient
    PjRtClient <|-- TfrtGpuClient
    CommonPjRtClient <|-- PjRtStreamExecutorClient
    PjRtStreamExecutorClient <|-- StreamExecutorGpuClient
```

| Runtime | Class | Description |
|---------|-------|-------------|
| **StreamExecutor** | `StreamExecutorGpuClient` | Primary path. Uses StreamExecutor for device abstraction. Thunk-based execution. |
| **TFRT** | `TfrtGpuClient` | Alternative path using TensorFlow Runtime. Independent implementation. |

### CUDA vs ROCm

The same client classes serve both platforms via conditional compilation:

```cpp
#if GOOGLE_CUDA
  // NVIDIA CUDA path
  platform_id = stream_executor::cuda::kCudaPlatformId;
#elif TENSORFLOW_USE_ROCM
  // AMD ROCm path
  platform_id = stream_executor::rocm::kROCmPlatformId;
#endif
```

Platform detection at runtime:
```cpp
StreamExecutorGpuClient::platform_version()
  // Returns "cuda 12040" or "rocm 60100"
```

> **Source:** [`xla/pjrt/gpu/se_gpu_pjrt_client.h`](../../xla/pjrt/gpu/se_gpu_pjrt_client.h)

---

## Client Initialization

```mermaid
flowchart TD
    Entry["GetStreamExecutorGpuClient(GpuClientOptions)"]

    Entry --> Platform["PlatformManager::GetPlatform('gpu')"]
    Platform --> Devices["Enumerate GPU devices"]

    Devices --> SingleNode{Multi-node?}
    SingleNode -->|No| LocalDevices["BuildLocalDevices()"]
    SingleNode -->|Yes| DistDevices["BuildDistributedDevices()"]

    LocalDevices --> Allocator["Create memory allocators<br/>(per device)"]
    DistDevices --> Allocator

    Allocator --> CreateClient["Create StreamExecutorGpuClient"]

    DistDevices --> TopoExchange["Exchange topology via<br/>KeyValueStore"]
    TopoExchange --> GlobalIDs["Assign global device IDs"]
    GlobalIDs --> Allocator
```

### GpuClientOptions

Key configuration:

| Option | Default | Description |
|--------|---------|-------------|
| `node_id` | 0 | This node's ID in multi-node setup |
| `num_nodes` | 1 | Total node count |
| `allowed_devices` | all | Restrict to specific GPU ordinals |
| `platform_name` | auto | Force "cuda" or "rocm" |
| `allocator_config` | BFC | Memory allocator selection |
| `enable_mock_nccl` | false | Use mock collectives for testing |

> **Source:** [`xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h`](../../xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h)

---

## Memory Allocators

The GPU backend uses the `kComputeSynchronized` allocation model (see
[Buffer Management: Allocation Models](buffer_management.md#memory-allocation-models)).

### Available Allocators

```mermaid
flowchart TD
    Config["GpuAllocatorConfig::Kind"]

    Config --> BFC["kBFC<br/>Best-Fit with Coalescing"]
    Config --> Async["kCudaAsync<br/>CUDA Async Allocator"]
    Config --> VMM["kVmm<br/>Virtual Memory Management"]
    Config --> Default["kDefault<br/>(picks best available)"]
    Config --> PlatformAlloc["kPlatform<br/>Platform default"]

    BFC --> Both["CUDA + ROCm"]
    Async --> CUDAOnly["CUDA 11.2+ only"]
    VMM --> CUDAOnly2["CUDA 11.2+ only"]
```

| Allocator | Platforms | Description | Best For |
|-----------|-----------|-------------|----------|
| **BFC** | CUDA, ROCm | Best-Fit with Coalescing. Sub-allocates from a large reserved pool. | Default, general purpose |
| **CUDA Async** | CUDA 11.2+ | Uses `cudaMallocAsync`. Lower fragmentation. | Workloads with varying memory needs |
| **VMM** | CUDA 11.2+ | Virtual Memory Management. Fine-grained allocation with oversubscription support. | Memory-constrained workloads |

### BFC Configuration

```cpp
GpuAllocatorConfig config;
config.kind = GpuAllocatorConfig::Kind::kBFC;
config.memory_fraction = 0.75;    // Reserve 75% of GPU memory
config.preallocate = true;        // Allocate pool at startup
```

### Per-Device Memory Spaces

Each GPU device has multiple memory spaces managed by a `MultiDeviceAdapter`:

| Memory Space | Purpose |
|-------------|---------|
| Default (device) | Standard GPU HBM for computation |
| Collective | Dedicated memory for NCCL/RCCL operations |
| Host (pinned) | Host memory pinned for fast DMA |
| TempBuffer | Persistent temp space (CUDA 11.2+) |

> **Source:** [`xla/pjrt/gpu/gpu_helpers.h`](../../xla/pjrt/gpu/gpu_helpers.h),
> [`xla/pjrt/gpu/se_gpu_pjrt_client.cc`](../../xla/pjrt/gpu/se_gpu_pjrt_client.cc) -- `GetStreamExecutorGpuDeviceAllocator`

---

## Stream Management

The GPU backend uses **dedicated streams** to overlap computation with data
transfers:

```mermaid
flowchart LR
    subgraph "Per-Device Streams"
        CS["Compute Stream<br/>(1, main execution)"]
        H2D["H2D Stream<br/>(1, host → device)"]
        D2H["D2H Streams<br/>(4, round-robin)"]
        D2D["D2D Streams<br/>(4, round-robin)"]
        CB["Callback Stream<br/>(1, host callbacks)"]
    end

    subgraph Operations
        Exec["Kernel execution"] --> CS
        Upload["Host data upload"] --> H2D
        Download["Device data download"] --> D2H
        Copy["Device-to-device copy"] --> D2D
        Callback["Host callback"] --> CB
    end
```

### Stream Assignment

| Stream | Count | Selection | Purpose |
|--------|-------|-----------|---------|
| `compute_stream` | 1 | Fixed | Kernel execution, main computation |
| `host_to_device_stream` | 1 | Fixed | H2D transfers |
| `device_to_host_stream` | 4 | Round-robin | D2H transfers (overlapped) |
| `device_to_device_stream` | 4 | Round-robin | D2D transfers (overlapped) |
| `callback_stream` | 1 | Fixed | Host callback invocation |

Multiple D2H and D2D streams enable **overlapping** multiple transfers with
computation. The round-robin selection ensures even distribution.

> **Source:** [`xla/pjrt/local_device_state.h`](../../xla/pjrt/local_device_state.h)

---

## Execution Flow

The GPU backend uses `StreamExecutorGpuClient::RunAsync` for execution:

```mermaid
sequenceDiagram
    participant Client as StreamExecutorGpuClient
    participant SE as StreamExecutor
    participant GpuExec as GpuExecutable

    Client->>SE: executor->Activate()<br/>(bind GPU context)

    Client->>GpuExec: ResolveConstantGlobals(stream)
    Note over GpuExec: Pre-allocated constant buffers

    Client->>Client: Build BufferAllocations
    Note over Client: Map parameters, constants,<br/>and temp buffers

    Client->>Client: Allocate temporary buffers
    Note over Client: With retry on OOM

    Client->>GpuExec: ExecuteThunks(buffer_allocations)
    Note over GpuExec: Each thunk is a kernel launch,<br/>memcpy, or collective op

    GpuExec->>SE: Enqueue on compute_stream

    Client->>Client: TearDown temporary allocations
    Note over Client: Free temps not in result set
```

### Thunk-Based Execution

The GPU compiler generates a sequence of **thunks** -- small units of device
work:

| Thunk Type | Purpose |
|-----------|---------|
| Kernel launch thunk | Launch a CUDA/HIP kernel |
| Memcpy thunk | Device-to-device memory copy |
| Collective thunk | NCCL/RCCL all-reduce, all-gather, etc. |
| Synchronization thunk | Stream synchronization barrier |
| Conditional thunk | Conditional execution |

Thunks are dispatched **sequentially** on the compute stream. The stream itself
provides ordering guarantees.

### Buffer Alignment

| Buffer Type | Required Alignment |
|------------|-------------------|
| Parameters | 256 bytes |
| Constants | 128 bytes |
| Temporaries | 256 bytes |

> **Source:** [`xla/pjrt/gpu/se_gpu_pjrt_client.cc`](../../xla/pjrt/gpu/se_gpu_pjrt_client.cc) -- `StreamExecutorGpuClient::RunAsync`

---

## Multi-GPU and Distributed

### Local Multi-GPU

Within a single host, multiple GPUs are managed by the same client:

```mermaid
flowchart TD
    Client["StreamExecutorGpuClient"]

    Client --> Dev0["GPU 0<br/>LocalDeviceState"]
    Client --> Dev1["GPU 1<br/>LocalDeviceState"]
    Client --> Dev2["GPU 2<br/>LocalDeviceState"]

    Dev0 --> Alloc0["Allocator 0"]
    Dev0 --> Streams0["Streams 0"]
    Dev1 --> Alloc1["Allocator 1"]
    Dev1 --> Streams1["Streams 1"]
    Dev2 --> Alloc2["Allocator 2"]
    Dev2 --> Streams2["Streams 2"]

    subgraph "Collectives"
        NCCL["NCCL/RCCL Communicators"]
    end

    Dev0 --- NCCL
    Dev1 --- NCCL
    Dev2 --- NCCL
```

### Multi-Host Topology Exchange

For distributed training across multiple hosts:

```mermaid
sequenceDiagram
    participant Node0 as Node 0
    participant KV as KeyValueStore
    participant Node1 as Node 1

    Node0->>Node0: Collect local GPU info<br/>(compute capability, memory, fabric UUID)
    Node1->>Node1: Collect local GPU info

    Node0->>KV: Put("topology/0", local_topology_proto)
    Node1->>KV: Put("topology/1", local_topology_proto)

    Node0->>KV: Get("topology/1")
    Node1->>KV: Get("topology/0")

    Note over Node0,Node1: Timeout: 5 min for global exchange

    Node0->>Node0: BuildGlobalTopology()<br/>Assign global device IDs
    Node1->>Node1: BuildGlobalTopology()<br/>Assign global device IDs

    Node0->>Node0: Set up NCCL communicators
    Node1->>Node1: Set up NCCL communicators
```

**Exchange protocol:**
1. Each node collects local device metadata (compute capability, memory size,
   fabric UUID for NVLink/NVSwitch detection)
2. Topology protos exchanged via `KeyValueStore` (backed by coordination service)
3. Global device IDs assigned: `global_id = process_index * max_devices_per_process + local_ordinal`
4. Partitions determined by boot IDs (same boot ID = same physical host)
5. Device interconnect maps built from fabric UUIDs

### Collectives

| Platform | Library | Description |
|----------|---------|-------------|
| NVIDIA | NCCL | NVIDIA Collective Communications Library |
| AMD | RCCL | ROCm Collective Communications Library |

Collectives are managed through `GpuCliques` -- groups of communicators keyed by
`GpuCliqueKey` (set of participating device IDs + operation type).

> **Source:** [`xla/pjrt/gpu/se_gpu_pjrt_client.cc`](../../xla/pjrt/gpu/se_gpu_pjrt_client.cc) -- `BuildDistributedDevices`

---

## GPU Extensions

The GPU backend supports several PJRT extensions:

| Extension | Header | Purpose |
|-----------|--------|---------|
| `Gpu_Custom_Call` | `pjrt_c_api_gpu_extension.h` | Register custom CUDA/HIP kernels callable from HLO |
| `Stream` | `pjrt_c_api_stream_extension.h` | Get the underlying CUDA/HIP stream for a device (for interop) |
| `Triton` | `pjrt_c_api_triton_extension.h` | Integrate Triton-compiled kernels |
| `CrossHostTransfers` | `pjrt_c_api_cross_host_transfers_extension.h` | Cross-host buffer send/receive |

### Stream Extension Example

```c
// Get the CUDA stream for a device (for use with cuBLAS, cuDNN, etc.)
PJRT_Stream_Extension* stream_ext = FindExtension(api, PJRT_Extension_Type_Stream);
PJRT_Stream_GetStream_Args args = {.device = device};
stream_ext->get_stream(&args);
cudaStream_t cuda_stream = (cudaStream_t)args.stream;
```

> **Source:** Extension headers in [`xla/pjrt/c/`](../../xla/pjrt/c)

---

## C API Entry Points

The GPU plugin exports the C API via:

```c
// xla/pjrt/c/pjrt_c_api_gpu.h
const PJRT_Api* GetPjrtApi();
```

**GPU-specific `PJRT_Client_Create` options:**

| Key | Type | Description |
|-----|------|-------------|
| `"platform_name"` | string | Force "cuda" or "rocm" |
| `"allowed_devices"` | int list | Restrict to specific GPU ordinals |
| `"node_id"` | int | This node's ID in distributed setup |
| `"num_nodes"` | int | Total number of nodes |
| `"allocator"` | string | Allocator type: "default", "bfc", "cuda_async", "vmm" |
| `"memory_fraction"` | float | Fraction of GPU memory to reserve |
| `"preallocate"` | bool | Pre-allocate memory pool at creation |

**Platform registration:**
- CUDA: `se_gpu_pjrt_compiler_cuda_registration.cc` registers compiler with
  `kCudaPlatformId`
- ROCm: `se_gpu_pjrt_compiler_rocm_registration.cc` registers compiler with
  `kROCmPlatformId`

> **Source:**
> - [`xla/pjrt/c/pjrt_c_api_gpu.h`](../../xla/pjrt/c/pjrt_c_api_gpu.h)
> - [`xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h`](../../xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h)

---

## Further Resources

- [Architecture Deep Dive](architecture.md) -- overall PJRT structure
- [Compilation Pipeline](compilation_pipeline.md#gpu-compilation) -- GPU compilation details
- [Execution Pipeline](execution_pipeline.md) -- common execution model
- [Buffer Management](buffer_management.md) -- buffer lifecycle and events
- Other backends: [CPU](backend_cpu.md) | [TPU](backend_tpu.md)
- [PJRT Plugin Tutorial (video)](https://www.youtube.com/watch?v=2GlMqaNxP_w)
- [OpenXLA DevLab playlist](https://www.youtube.com/playlist?list=PLlFotmaRrOzv2OIEpijqiHGmY7rpscFcj)
