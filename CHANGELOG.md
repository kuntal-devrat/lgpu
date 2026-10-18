# Changelog

All notable changes to the `lgpu` workspace are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Voxel renderer mouse look jumping on click by resetting last cursor position.
- Smooth camera controls with sprint speed boost (`Shift`), vertical navigation (`Space` / `C`), and clamped frame deltas.
- Early vertical bounds culling in voxel DDA to eliminate wasted steps into empty sky or void.

---

## [0.1.2] - 2026-10-16

### Changed
- Cleaned and organized integration test suite names.
- Streamlined documentation comments across `Buffer` and memory suballocator.
- Updated documentation and README to clearly specify project scope and transparently describe AI-assisted boilerplate scaffolding.

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
