# Architecture, Safety and Synchronization

This document describes the design of **`lgpu`**, the guarantees it makes, and what is
left to the caller.

## 1. Relationship to other graphics layers

| | `wgpu` | `vulkano` | `kvark/blade` | `lgpu` |
| :--- | :--- | :--- | :--- | :--- |
| API target | WebGPU | Vulkan 1.0 to 1.2+ | Minimal modern GPU layer | Vulkan 1.3 |
| Backends | Vulkan, Metal, D3D12, WebGL, WebGPU | Vulkan | Vulkan, Metal | Vulkan 1.3 (native drivers, MoltenVK) |
| Shaders | WGSL, translated at runtime | Offline or macro SPIR-V | SLANG or WGSL via Naga | SPIR-V, optional runtime WGSL via Naga |
| Render passes | `wgpu::RenderPass` objects | Render pass state objects | Encoder scoped | Dynamic rendering only |
| Descriptors | Explicit bind groups | Explicit descriptor sets | Resource tables | Global bindless heaps |
| Hazard tracking | Runtime tracking on the CPU | Type state, partly runtime | Explicit | Explicit, caller managed |
| Clean build | 25 to 45 s | 10 to 20 s | 8 to 15 s | 1 to 2 s |
| C / FFI | Separate `wgpu-native` project | Rust only | Rust only | C99 header in `lgpu-c` |

Pick `wgpu` when you need browser, Metal and D3D12 coverage from one codebase. Pick
`vulkano` when compile-time checked Vulkan 1.0 to 1.2 state is what you want. Pick `blade`
for a minimal Rust API that also runs on Metal. Pick `lgpu` for modern desktop Vulkan
with fast builds, dynamic rendering, bindless descriptors and an optional C API.

## 2. Safety on the host and on the device

### Host memory

Safe Rust cannot corrupt CPU memory through `lgpu`. Buffer reads and writes check the
slice bounds against the buffer size before copying, and every mapped access goes through
a mutex. Cloning a `Buffer` or `Texture` shares that mutex and the image layout state with
the original, so handles to the same resource cannot race each other.

### Device execution

Vulkan handles are RAII types, but a shader is arbitrary bytecode. Indexing past a
bindless heap or reading an invalid address faults at the MMU or driver level, usually as
`VK_ERROR_DEVICE_LOST`.

### Mapped buffers

Each `VulkanBuffer` keeps the host pointer for `CpuToGpu` and `GpuToCpu` memory plus a
`sync_lock`. `write_at` and `read_at` take that lock for the duration of the copy. When
the selected memory type is not host coherent, the range is expanded to
`non_coherent_atom_size` boundaries and flushed or invalidated with
`vkFlushMappedMemoryRanges` / `vkInvalidateMappedMemoryRanges`.

### Parent objects

Destroying a `VkInstance` while a `VkDevice` is alive is undefined behavior, so `Device`
and `Queue` each hold an `Arc<VulkanInstance>` and keep the instance alive until the last
resource is dropped.

## 3. Thread safety

| Type | `Send` | `Sync` | Model |
| :--- | :---: | :---: | :--- |
| `Instance` | yes | yes | Adapter enumeration and surface creation are thread safe. |
| `Device` | yes | yes | Resource creation is thread safe. |
| `Queue` | yes | yes | Submissions are serialized by a queue mutex. |
| `Buffer` | yes | yes | Mapped access is serialized by a shared mutex. |
| `Texture` | yes | yes | Layout transitions are serialized by a shared mutex. |
| `Sampler` | yes | yes | Immutable Vulkan handle. |
| `CommandEncoder` | yes | no | One thread records at a time, as Vulkan requires. |
| `CommandBuffer` | yes | no | Can move to another thread before submission. |
| `TimelineSemaphore` | yes | yes | Monotonic counter, usable across threads. |

## 4. Synchronization and resource lifetime

`lgpu` does not track GPU hazards. There is no CPU side dependency graph and no per
command buffer resource tracking. Dependent passes are ordered with timeline semaphores
(`queue.submit_async`) or pipeline barriers, and swapchain work is ordered through the
semaphores used by `queue.submit_surface_work` and `SurfaceTexture::present`.

Command recorders come from a pool of command pool and command buffer pairs. A recorder
returned by `finish` is recycled after submission, and an encoder dropped without being
submitted is recycled straight away. The idle pool is capped at
`MAX_CACHED_COMMAND_RECORDERS`; anything above the cap destroys its command pool.

## 5. Shortcuts

For tests and one shot jobs:

```rust,no_run
// Create an instance, pick the first GPU and get a device and queue.
let (device, queue) = lgpu::default_device()?;
```

```rust,no_run
# use lgpu::{Device, Queue};
# fn run_job(device: &Device, queue: &Queue) -> Result<(), lgpu::Error> {
let mut encoder = device.create_command_encoder()?;
// ... record commands ...
queue.run(encoder)?; // finishes, submits and waits
# Ok(())
# }
```

## 6. Lock order

To avoid deadlocks between recording threads, async uploads and queue submissions, locks
are always taken in this order and never in reverse:

1. Object locks: `Buffer::sync_lock` and `Texture::layout`
2. Staging ring: `VulkanDevice::staging_ring`
3. Queue: `VulkanDevice::queue`
4. In flight async recorders: `VulkanDevice::pending_async_recorders`
5. Caches: `VulkanDevice::command_recorder_cache` and `VulkanDevice::fence_cache`
6. Suballocator: `VulkanDevice::allocator`
7. Bindless slot pools: `BindlessDescriptorHeap` slot mutexes

Code in `lgpu` and `lgpu-hal` must not take a higher ranked lock while holding a lower
ranked one. Poisoned mutexes are recovered with `into_inner()`.

## 7. Mobile and web targets

Android runs Vulkan 1.3 natively from Android 10 (API 29) with the NDK. iOS and macOS run
the Vulkan portability subset through MoltenVK with dynamic rendering enabled. WebAssembly
and WebGPU are not implemented; the design maps to WebGPU, and experiments are tracked in
the issue tracker.
