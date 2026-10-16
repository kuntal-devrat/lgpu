# Architecture, Safety & Synchronization Model

This document outlines the architectural philosophy, safety invariants, synchronization rules, and ecosystem positioning of **`lgpu`**.

---

## 1. Ecosystem Positioning: How `lgpu` Compares

Developers often ask how `lgpu` relates to existing graphics layers in the Rust ecosystem like **`wgpu`** and **`vulkano`**:

| Dimension | `wgpu` | `vulkano` | `lgpu` |
| :--- | :--- | :--- | :--- |
| **API Target** | WebGPU Specification | Vulkan 1.0 – 1.2+ | **Vulkan 1.3+ Native** |
| **Backends** | Vulkan, Metal, D3D12, WebGL, WebGPU | Vulkan only | **Vulkan 1.3 only** (via native drivers / MoltenVK) |
| **Shader Model** | WGSL (runtime translation to SPIR-V/MSL/HLSL via Naga) | Offline or macro-compiled SPIR-V | **Native SPIR-V** (runtime WGSL optional via Naga) |
| **Render Pass Model** | RenderPass objects (`wgpu::RenderPass`) | RenderPass state objects (`vulkano::render_pass`) | **Dynamic Rendering only** (`VK_KHR_dynamic_rendering`) |
| **Descriptor Model** | Explicit Bind Groups (emulated/restricted) | Explicit Descriptor Sets | **Global Bindless Heaps** (`update_after_bind`) |
| **GPU Hazard Tracking**| Heavy runtime CPU hazard/barrier tracking | Static / runtime type-state tracking | **Explicit / Manual** (Timeline Semaphores & Barriers) |
| **Clean Compile Time** | 25 – 45 seconds | 10 – 20 seconds | **~1.5 seconds** |
| **Dependencies** | Naga, spirv, wgpu-core, wgpu-hal, etc. | Custom proc-macros, large type hierarchy | `ash`, `bitflags`, `thiserror`, `raw-window-handle` |
| **C / FFI Support** | Separate `wgpu-native` project | Rust only | **Built-in C99 FFI (`include/lgpu.h`)** |

### Summary of Philosophy:
- **Use `wgpu`** if you need cross-platform portability targeting the browser, Metal, DirectX 12, and Vulkan from a single codebase, and you want safe runtime GPU state-tracking.
- **Use `vulkano`** if you want deep, type-safe compile-time checking of traditional Vulkan 1.0–1.2 pipelines and descriptor sets.
- **Use `lgpu`** if you are targeting modern desktop hardware, want **instant compile times (~1.5s)**, prefer modern Vulkan 1.3 features (**Dynamic Rendering** without `VkRenderPass` boilerplate, and **global bindless descriptor heaps**), and want an optional universal C99 FFI.

---

## 2. Safety, Soundness & Rust Invariants

### Can you trigger Undefined Behavior (UB) in Safe Rust?
- **Host (CPU) Memory**: **No.** Safe Rust code cannot trigger CPU-side memory safety violations or data races. Buffer reads and writes enforce slice bounds checks before copying bytes. Mapped memory pointers are guarded by mutexes.
- **Device (GPU) Execution**: Safe Rust wraps Vulkan handles in RAII types, but GPU shaders are inherently arbitrary bytecode. If a shader indexes past the end of an unbounded bindless descriptor heap or accesses invalid GPU memory addresses, the hardware MMU / driver will fault, typically producing `VK_ERROR_DEVICE_LOST` or a TDR (Timeout Detection and Recovery) at the OS level.

### CPU Mappings of Buffers
- Each `VulkanBuffer` holds a host-mapped pointer (when allocated in `CpuToGpu` or `GpuToCpu` memory) and a `sync_lock: std::sync::Mutex<()>`.
- `buffer.write_at()` and `buffer.read_at()` acquire this mutex before performing memory copies, preventing concurrent host read/write data races across threads.
- If the selected Vulkan memory type is **non-coherent** (`!buffer.is_coherent`), `lgpu` calculates non-coherent atom-aligned ranges (`non_coherent_atom_size`) and calls `vkFlushMappedMemoryRanges` (on write) or `vkInvalidateMappedMemoryRanges` (on read).

### Parent Object Lifetimes
- In Vulkan, destroying a `VkInstance` while a child `VkDevice` is active produces undefined behavior.
- In `lgpu`, both `Device` and `Queue` hold `Arc<lgpu_hal::vulkan::VulkanInstance>`. The parent Vulkan instance cannot be destroyed until all devices, queues, and resources have been dropped.

---

## 3. Thread Safety & Concurrency (`Send` / `Sync`)

| Type | `Send` | `Sync` | Concurrency Model |
| :--- | :---: | :---: | :--- |
| **`Instance`** | ✅ | ✅ | Thread-safe adapter enumeration and surface creation. |
| **`Device`** | ✅ | ✅ | Thread-safe resource allocation (buffers, textures, pipelines). |
| **`Queue`** | ✅ | ✅ | Queue submissions are internally synchronized via `Mutex<vk::Queue>`. |
| **`Buffer`** | ✅ | ✅ | Mapped memory read/write operations guarded by internal mutex. |
| **`Texture`** | ✅ | ✅ | Read-only handle after creation; layout transitions synchronized. |
| **`Sampler`** | ✅ | ✅ | Immutable Vulkan sampler handle. |
| **`CommandEncoder`** | ✅ | ❌ | **Not `Sync`**. Follows Vulkan command buffer rules: an encoder may be recorded on any thread, but multiple threads cannot record into the same encoder concurrently without external synchronization. |
| **`CommandBuffer`** | ✅ | ❌ | Produced by `encoder.finish()`. Can be transferred to another thread for queue submission. |
| **`TimelineSemaphore`** | ✅ | ✅ | Thread-safe monotonic counter signaling and waiting across CPU and GPU threads. |

---

## 4. Synchronization & Resource Lifecycle

### Does `lgpu` track GPU hazards automatically?
**No.** `lgpu` does not maintain a runtime directed acyclic graph (DAG) or track per-command-buffer resource hazards on the CPU.
- Developers synchronize execution between dependent GPU passes using **Timeline Semaphores** (`queue.submit_async`) or pipeline execution barriers.
- For swapchain rendering, `lgpu` provides explicit presentation semaphores via `queue.submit_surface_work` and `SurfaceTexture::present`.

### Command Recorder Lifecycle & Pooling
- Call `device.create_command_encoder()` to acquire a recorder for a frame, render pass, or worker thread.
- Recorders are backed by pooled `vk::CommandPool` and `vk::CommandBuffer` pairs.
- When an encoder finishes and is submitted, or if an encoder is dropped unsubmitted, its underlying recorder is automatically returned to the device cache for zero-allocation reuse in subsequent frames.

---

## 5. Ergonomic Shortcuts

For test suites, quick prototyping, and one-shot utilities, `lgpu` provides one-line convenience helpers:

### One-Line Device Initialization: `lgpu::default_device()`
```rust,no_run
// Automatically initializes an Instance, selects the primary GPU, and requests the Device & Queue:
let (device, queue) = lgpu::default_device()?;
```

### One-Line Command Execution: `queue.run()` or `device.run()`
```rust,no_run
# use lgpu::{Device, Queue};
# fn run_job(device: &Device, queue: &Queue) -> Result<(), lgpu::Error> {
let mut encoder = device.create_command_encoder()?;
// ... record commands ...
queue.run(encoder)?; // Finishes, submits, and waits synchronously for completion
# Ok(())
# }
```
