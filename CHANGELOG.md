# Changelog

All notable changes to the `lgpu` workspace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2026-10-28

### Added
- **Automated C99 Bindings Generation**: Added `cbindgen.toml` configuration and `scripts/generate_c_bindings.sh` for generating universal C99 / Zig FFI headers (`crates/lgpu-c/include/lgpu.h`) (#33).
- **Headless CI Test Resilience**: Added `require_gpu!` test harness across all integration test suites to gracefully skip tests on headless CI runners without physical Vulkan devices or ICDs instead of panicking (#19, #36, #37).
- **Lock Ordering & Platform Roadmap Documentation**: Added comprehensive lock hierarchy documentation and cross-platform mobile (Android/iOS) and WebGPU (wasm32) roadmaps to `docs/ARCHITECTURE_AND_SAFETY.md` (#22, #23, #32, #42, #57).

### Changed
- **Command Recorder Cache Capping**: Capped idle command recorder pooling to `MAX_CACHED_COMMAND_RECORDERS = 32`. Recycled command recorders exceeding the cap immediately destroy their command pools to eliminate memory bloat during burst operations (#21, #34, #54).
- **Device Memory Trimming**: `VulkanDevice::trim_memory()` now drains and destroys all cached idle command pools in addition to freeing unused suballocator memory blocks.

### Fixed
- **Zero Panics / `.unwrap()` in HAL & C FFI**: Replaced all panicking `.unwrap()` and `.expect()` calls in `crates/lgpu-hal` and `crates/lgpu-c` with robust error propagation returning `HalError` or `LgpuResult` (#17, #18, #20, #27, #28, #30, #31, #35, #40, #43, #45, #46, #47, #49, #50, #52, #53, #55, #56, #58, #59).
- **Suballocator OOM & Exhaustion Handling**: Handled block exhaustion and memory allocation failures gracefully by returning `HalError::OutOfMemory` rather than panicking (#25, #29, #48).
- **Swapchain Safety**: Validated physical device surface format list is non-empty before format querying to prevent indexing out of bounds (#18, #26, #39, #40).
- **Poison-Safe Mutexes**: Replaced unsafe mutex locks in bindless managers, command caches, and device cleanup with poison recovery (`.into_inner()`) (#24, #31, #43, #49, #56).
- **Strict Lock Acquisition Order**: Enforced deadlock-free lock hierarchy (`sync_lock` -> `staging_ring` -> `queue` -> `pending_recorders` -> `allocator`) across buffer and texture lifecycles (#22, #42, #57).

---

## [0.1.4] - 2026-10-27

### Added
- **Asynchronous Ring Staging Buffer**: 64 MiB host-visible, host-coherent persistent staging memory with linear suballocation and timeline semaphore synchronization, eliminating blocking `vkQueueWaitIdle()` on texture uploads.
- **Vulkan 1.3 Synchronization2**: Replaced legacy pipeline barriers across command encoders with `vkCmdPipelineBarrier2` and `VkDependencyInfo`.
- **Runtime WGSL Compilation**: Optional `Device::create_shader_wgsl()` powered by Naga under the `wgsl` Cargo feature flag (#3).
- **Burst Upload & Resize Stress Testing**: Integration test validating 100 burst uploads/frame across rapid window resize cycles with zero driver crashes or memory leaks.

### Changed
- **Robust Frames-in-Flight Buffer**: Implemented triple buffering (`MAX_FRAMES_IN_FLIGHT = 3`) with CPU fence waits before command recording and per-swapchain-image render semaphores to strictly adhere to Vulkan presentation requirements.
- **Buddy Memory Allocator**: Replaced naive vector sorting with a 64 MiB block power-of-two Buddy Allocator with $O(1)$ coalescing and warm block pooling, preventing driver `maxMemoryAllocationCount` limit crashes (#1, #13).
- **Bulletproof Swapchain Reconfiguration**: Intercepts `VK_ERROR_OUT_OF_DATE_KHR`, `VK_SUBOPTIMAL_KHR`, `VK_NOT_READY`, passes `old_swapchain` during re-creation, and gracefully pauses rendering on zero-size minimized windows.

### Fixed
- **Bindless Descriptor Slot Guards**: Implemented `BufferSlotGuard`, `TextureSlotGuard`, and `SamplerSlotGuard` to ensure descriptor heap slots are recycled safely without leaks or double-frees (#14, #15).
- **CI & Headless Test Stability**: Gracefully handle environments without physical Vulkan devices in bindless heap tests (#11, #12).
- **Concurrent Submission Safety**: Enforced monotonic timeline semaphore signal ordering under concurrent multi-threaded queue submissions.

---

## [0.1.3] - 2026-10-18

### Added
- Real-time voxel raymarching engine (`example-voxel-renderer`) with DDA traversal, procedural mountains, islands, and ancient citadel.
- Minecraft-style volumetric translucent water with Beer-Lambert depth absorption, animated surface waves, Fresnel reflection, and seabed caustics.
- Polyglot multi-language examples:
  - C99 (`examples/c_example`): Device query, buffer initialization, texture rendering, and timeline semaphores.
  - Modern C++17 (`examples/cpp_example`): RAII handles, GPU-to-GPU buffer copies, and `std::vector` readbacks.
  - Zig 0.14 (`examples/zig_example`): `@cImport` integration, buffer allocation, and memory verification.
- Convenience ergonomic helpers `lgpu::default_device()` / `Device::default_device()` and `Queue::run(encoder)` / `Device::run(&queue, encoder)`.
- Architectural and safety documentation (`docs/ARCHITECTURE_AND_SAFETY.md`) comparing `lgpu` with `wgpu`, `vulkano`, and `kvark/blade`.
- Dedicated CI release pipeline with pre-flight crates.io idempotency checks and maintainer release automation.

### Fixed
- Soundness fix: `Arc<VulkanInstance>` retained in `Device` and `Queue` ensuring parent Vulkan instance outlives child devices during default device creation.
- Suballocator memory reclamation: Automatically free completely empty memory blocks back to Vulkan (`vkFreeMemory`), keeping at most 1 empty block per memory type as a warm cache (#6).
- Exposed `trim_memory()` and `suballocator_block_count()` on `Device`, `VulkanDevice`, and C FFI.
- Surface swapchain deadlock fix: Implemented `Drop` for `SurfaceTexture` to cleanly signal in-flight frame fences and release acquired images back to the presentation engine if dropped without calling `present()` (#7).
- Voxel renderer mouse look jumping on click by resetting last cursor position.
- Smooth camera controls with sprint speed boost (`Shift`), vertical navigation (`Space` / `C`), and clamped frame deltas.
- Early vertical bounds culling in voxel DDA to eliminate wasted steps into empty sky or void.

---

## [0.1.2] - 2026-10-16

### Changed
- Cleaned and organized integration test suite names.
- Streamlined documentation comments across `Buffer` and memory suballocator.
- Updated documentation and architecture guides to clearly specify project scope, safety invariants, and synchronization guarantees.

---

## [0.1.1] - 2026-10-16

### Added
- Pipeline cache serialization and loading via `device.get_pipeline_cache_data()`.
- Flexible blend state and component configuration (`BlendState`, `BlendComponent`, `BlendFactor`, `BlendOp`).
- Configurable primitive topologies (`PointList`, `LineList`, `LineStrip`, `TriangleList`, `TriangleStrip`).
- Rasterizer cull modes (`None`, `Front`, `Back`) and front-face winding (`Ccw`, `Cw`).
- `GraphicsPipelineDesc` for structured pipeline creation.
- Multi-pass compositing: `LoadOp<T>`, `ColorLoadOp`, `DepthLoadOp`, and scoped `encoder.begin_render_pass`.
- Indexed and indirect drawing commands (`draw_indexed`, `draw_indirect`).
- Buffer-to-buffer and texture-to-buffer memory copies on `CommandEncoder`.
- Mipmapped and multi-sampled textures (MSAA) with sample count validation.
- C99 FFI functions for device idling, native window surfaces, and surface presentation.

### Fixed
- Fixed surface swapchain texture ownership so dropping surface textures does not prematurely destroy swapchain images.
- `CommandEncoder` rendering methods now return descriptive errors on parameter or descriptor overflows.
- Automatically recycle unsubmitted command recorders back to the device pool on drop.
- Replaced boolean fields in `LgpuAdapterInfo` with ABI-stable `uint8_t`.
- Memory suballocator chunk size tracking during deallocation and chunk coalescing.
- Ensured host-visible and device-local suballocations use separate memory blocks.
- Thread-safe guards on mapped buffer writes and concurrent texture transitions.

---

## [0.1.0] - 2026-10-14

### Added
- Initial release of `lgpu`.
- Dynamic Vulkan 1.3 loader via `ash`.
- Global bindless descriptor indexing heap (`update_after_bind`) for buffers, textures, and samplers.
- Dynamic rendering support (`VK_KHR_dynamic_rendering`).
- C99 FFI headers (`lgpu.h`).
- Headless compute, dynamic clear, bindless triangle, and particle simulation examples.
