# Changelog

All notable changes to the `lgpu` workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - 2026-10-15

### Added
- **Core API & Architecture**:
  - `HalDevice`, `HalBuffer`, `HalTexture` trait abstractions in `lgpu-hal` with modular submodule exports (`allocator`, `buffer`, `command`, `device`, `instance`, `pipeline`, `swapchain`, `sync`, `texture`).
  - Pipeline cache serialization and reuse via `device.get_pipeline_cache_data()`.
  - Non-pow2 suballocation support exceeding 4,096 block allocations without memory fragmentation.
- **Rendering & Pipeline Flexibility**:
  - Configurable blend modes: `BlendState`, `BlendComponent`, `BlendFactor`, `BlendOp` (including `REPLACE`, `ALPHA_BLENDING`, `ADDITIVE`).
  - Configurable topologies: `PrimitiveTopology` (`PointList`, `LineList`, `LineStrip`, `TriangleList`, `TriangleStrip`).
  - Rasterizer cull modes (`None`, `Front`, `Back`) and front face winding (`Ccw`, `Cw`).
  - `GraphicsPipelineDesc` for unified, type-safe graphics pipeline creation.
  - Multi-pass compositing: `LoadOp<T>`, `ColorLoadOp`, `DepthLoadOp`, and scoped `encoder.begin_render_pass(desc)`.
  - Advanced drawing commands: `draw_indexed`, `draw_indexed_with_depth`, `draw_indexed_with_ops`, `draw_indirect`, and `draw_indirect_with_depth`.
- **Resource Streaming & Features**:
  - `encoder.copy_buffer_to_buffer` for high-throughput GPU memory transfers.
  - `queue.write_texture` and `encoder.copy_texture_to_buffer` for texture staging and readbacks.
  - Mipmapped and multi-sampled textures (MSAA) with sample count validation.
  - `queue.write_buffer` convenience helper on `Queue`.
  - `Texture::is_owned(&self) -> bool` and `Texture::clone_unowned(&self) -> Texture`.
  - `std::fmt::Debug` implementations for `Buffer`, `Texture`, `Sampler`, `Shader`, `ComputePipeline`, and `GraphicsPipeline`.
- **C FFI & Cross-Language Bindings (`lgpu-c`)**:
  - `lgpu_device_wait_idle` to wait for all in-flight GPU operations to complete.
  - Complete native window and swapchain presentation API: `lgpu_create_surface_headless`, `lgpu_create_surface_from_xlib`, `lgpu_create_surface_from_xcb`, `lgpu_create_surface_from_wayland`, `lgpu_create_surface_from_win32`.
  - Swapchain configuration and acquisition: `lgpu_surface_configure`, `lgpu_surface_get_current_texture`, `lgpu_surface_texture_get_texture`, `lgpu_queue_submit_surface`, `lgpu_surface_texture_destroy`, `lgpu_destroy_surface`.
  - Explicit resource destruction: `lgpu_destroy_command_encoder`, `lgpu_destroy_command_buffer`.
  - Full `# Safety` rustdoc annotations for every `extern "C"` function.
- **Testing**:
  - Integration tests for Section 1 through Section 9 audit items across 15 dedicated test suites.
  - Headless swapchain resize and multi-frame in flight synchronization tests.

### Fixed
- **API-01 / BUG-02**: Surface swapchain textures are marked `owned: false` so that dropping surface textures does not destroy swapchain-managed images.
- **API-02**: `CommandEncoder` rendering commands now return `Result<(), Error>` rather than silently ignoring push constant overflows or missing bindless heaps.
- **API-03 / LEAK-01**: `CommandEncoder` now implements `Drop` to automatically recycle abandoned command recorders back to the device cache, preventing command pool and buffer leaks.
- **FFI-06**: `LgpuAdapterInfo` boolean fields replaced with ABI-stable `uint8_t` across GCC, Clang, MSVC, and Zig.
- **BUG-01**: `create_buffer_init` and `create_buffer_init_slice` now correctly copy user data to host-mapped memory without truncation.
- **BUG-06 / BUG-07**: `request_device` checks actual hardware capabilities and features rather than hardcoding feature enablement.
- **SAFETY-01 / SAFETY-02**: Enforced `Send + Sync` invariants with thread-safe mutex guards protecting mapped buffer pointers and concurrent texture layout transitions.
- **LEAK-02 / BUG-04**: Swapchain recreation preserves old swapchains until new swapchains are initialized and properly synchronizes multi-frame in-flight fences.
- **DOC-05**: Removed `Cargo.lock` from `.gitignore` for reproducible builds.

---

## [0.1.0] - 2026-10-14

### Added
- Initial release of `lgpu` (Lean GPU).
- Zero-overhead dynamic Vulkan 1.2+ loader (`lgpu-hal`).
- Global bindless descriptor indexing heap for unbounded texture and buffer access.
- Dynamic rendering support (`VK_KHR_dynamic_rendering`).
- Initial C99 / Zig FFI headers (`lgpu.h`).
- Showcase examples:
  - `01_headless_compute`
  - `02_clear_screen`
  - `03_bindless_triangle`
  - `04_bindless_100k`
  - `05_window_shader`
