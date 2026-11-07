# Changelog

All notable changes to the `lgpu` workspace are recorded in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2026-11-07

### Added
- Example `09_greedy_mesher`: 3D voxel chunk greedy mesher comparing naive 6-face meshing against binary slice greedy meshing with quad merging (>85% to >94% polygon reduction), dynamic rendering, depth testing, and indexed draws via `SurfaceFrames`.
- Real-time voxel world storage buffer mode in `06_voxel_renderer` with interactive DDA raycasting for block mining (`[E]` / Right Mouse Button) and placement (`[Q]` / Middle Mouse Button).
- `register_bindless_uniform_buffer` and `unregister_bindless_uniform_buffer` methods added to the `HalDevice` trait.
- Integration tests in `test_p0_hal_hardening` covering multi-queue command recycling, multi-layer image transitions, suballocation failure cleanup, and uniform descriptor recycling.

### Fixed
- Queue command recorder caches are separated per queue family (`graphics_recorder_cache` and `compute_recorder_cache`), preventing command pool contamination and reset issues on multi-queue devices.
- `compute_recorder_cache` pools and command buffers are properly drained and freed on `trim_memory` and device drop.
- Vulkan device memory is freed if suballocation memory block creation fails during fresh block allocation.
- `Depth16Unorm` format size corrected from 4 bytes per pixel to 2 bytes per pixel.
- Multi-layer 2D-array, cubemap, and 3D textures now use correct `base_array_layer` and `layer_count` in barrier transitions and copy descriptors.
- Mipmap generation loop enforces level-to-level pipeline barriers between source read and destination write to prevent Read-After-Write hazards.

## [0.1.6] - 2026-11-01

### Added
- Texture dimensions: 1D, 3D, cube, cube-array and 2D-array textures through
  `TextureDimension` and `TextureDesc::array_layers`, with per-layer and per-slice uploads,
  readbacks and mip generation.
- `TextureDesc` convenience constructors: `new_1d`, `new_3d`, `new_2d_array`, `new_cube`,
  `new_cube_array`, `with_mips` and `with_samples`.
- Depth formats `Depth16Unorm`, `Depth24Stencil8` and `Depth32FloatStencil8`, including
  stencil aspect barriers and depth/stencil attachment selection. Pipelines and render
  passes now accept every depth format instead of only `D32_SFLOAT`.
- `FrameRenderer`/`FrameConfig`: MSAA and depth attachments that follow the surface size,
  and `FrameAttachments::begin` to start a pass with the correct load ops and resolve target.
- `SurfaceFrames`/`SurfaceFrame`: acquire, reconfigure once on `OutOfDate`, prepare
  attachments, submit and present in one flow.
- Separate async compute queue support: adapters with a second compute-capable family get a
  dedicated queue (`Device::compute_queue`, `Device::create_compute_command_encoder`,
  `QueueFamily`), and buffers/textures are created with concurrent sharing so both queues can
  use them without ownership transfers.
- Vertex buffer layouts and attributes in `GraphicsPipelineDesc`, plus
  `CommandEncoder::bind_vertex_buffer`, so meshes can be drawn from vertex and index buffers.
- A bindless uniform buffer binding (binding 4), `Buffer::uniform_id()` and
  `Device::write_uniform_buffer_descriptors_batch()`.
- MSAA resolve targets through `draw_with_ops_resolve`, `draw_with_depth_and_ops_resolve`,
  `draw_indexed_with_ops_resolve`, `draw_indirect_with_ops_resolve` and
  `begin_render_pass_with_resolve`.
- Pipeline cache persistence with `Device::pipeline_cache_data()` and
  `Adapter::request_device_with_cache()`.
- Sampler depth comparison (`SamplerDesc::compare_enable` and `compare_op`) for shadow maps.
- Debug labels for GPU debuggers through `CommandEncoder::begin_debug_label` and
  `end_debug_label`.
- `LGPU_PANIC_ON_VALIDATION=1` to turn validation errors into panics; they are logged
  otherwise.
- C API parity for texture dimensions, vertex buffers, MSAA resolve, uniform-buffer slots,
  GPU timestamps, staged texture uploads, device-local buffers, debug labels, comparison
  samplers and async compute queues (`lgpu_create_texture_ex`,
  `lgpu_encoder_draw_with_resolve`, `lgpu_device_has_async_compute_queue`, ...).
- Examples: `07_textured_mesh` (indexed textured quad with uniform buffer, mips, depth test
  and 4x MSAA resolve) and `08_windowed_mesh` (winit window rendering through `SurfaceFrames`).

### Fixed
- The queue mutex is now held across `vkQueueSubmit` and timeline value allocation, so
  concurrent submissions no longer race and timeline signals stay ordered.
- Cloned buffers and textures now share their host-access mutex and image layout state,
  so reads and writes through different handles are serialized and barriers always see
  the layout the image is actually in.
- The buddy allocator ignores frees of untracked offsets instead of inserting a duplicate
  chunk, which could hand the same memory to two live allocations.
- Dedicated and block allocations free their `VkDeviceMemory` when `vkMapMemory` fails,
  and buffer bind failures unmap before freeing.
- Texture creation releases the image and its memory when image view creation fails, and
  pipeline and bindless heap creation release their layouts and pools on failure paths.
- Staging ring creation releases the buffer and memory if binding, mapping or semaphore
  creation fails.
- Freeing a dedicated allocation no longer frees memory that is still mapped.
- `create_texture` rejects `sample_count` values that are not 1, 2, 4, 8, 16, 32 or 64,
  rejects more mip levels than the texture extent allows, and rejects cube textures that are
  not square or whose layer count is not a multiple of six.
- `write_texture` rejects uploads smaller than one packed mip level, multisampled
  targets and depth formats.
- Bindless slot recycling ignores duplicate and out of range frees.
- `acquire_next_image` reports `VK_SUBOPTIMAL_KHR` as an error instead of returning image
  index 0.

### Changed
- Command recorders, command pools and the recorder cache are split per queue family, and
  recycled recorders return to the cache they belong to.
- Surfaces can only be submitted on the graphics queue; submitting a surface frame on the
  compute queue is rejected with a descriptive error.
- `Surface::configure` destroys a newly created swapchain and any sync objects it
  created when a later step fails, instead of leaking them.
- The buddy allocator keeps free chunks in hash sets, so buddy lookup during coalescing is
  O(1) rather than a linear scan.

### Breaking
- `TextureDesc` gained required `dimension` and `array_layers` fields; use the
  `TextureDesc::new_*` constructors or set them explicitly.

## [0.1.5] - 2026-10-28

### Added
- Automated C99 binding generation: `cbindgen.toml` and
  `scripts/generate_c_bindings.sh` produce `crates/lgpu-c/include/lgpu.h` (#33).
- A `require_gpu!` harness in the integration tests, so suites skip instead of panicking
  on headless machines without a Vulkan device (#19, #36, #37).
- Lock ordering rules and a mobile and web roadmap in
  `docs/ARCHITECTURE_AND_SAFETY.md` (#22, #23, #32, #42, #57).

### Changed
- Idle command recorder pooling is capped at `MAX_CACHED_COMMAND_RECORDERS = 32`.
  Recorders past the cap destroy their command pool immediately (#21, #34, #54).
- `VulkanDevice::trim_memory()` drains and destroys cached idle command pools as well as
  freeing unused suballocator blocks.

### Fixed
- Panicking `.unwrap()` and `.expect()` calls in `lgpu-hal` and `lgpu-c` were replaced
  with error returns (#17, #18, #20, #27, #28, #30, #31, #35, #40, #43, #45, #46, #47,
  #49, #50, #52, #53, #55, #56, #58, #59).
- Suballocator exhaustion and allocation failure return `HalError::OutOfMemory` instead
  of panicking (#25, #29, #48).
- Swapchain creation checks that the surface reports at least one format before reading
  the list (#18, #26, #39, #40).
- Mutexes in bindless managers, command caches and device cleanup recover from poisoning
  with `into_inner()` (#24, #31, #43, #49, #56).
- Lock acquisition follows the documented order across buffer and texture lifetimes
  (#22, #42, #57).

## [0.1.4] - 2026-10-27

### Added
- Asynchronous ring staging buffer: 64 MiB of persistent host visible, host coherent
  memory with linear suballocation and timeline semaphore tracking, so texture uploads no
  longer call `vkQueueWaitIdle()`.
- Vulkan 1.3 synchronization2: command encoders use `vkCmdPipelineBarrier2` and
  `VkDependencyInfo` instead of legacy pipeline barriers.
- Runtime WGSL compilation through `Device::create_shader_wgsl()` behind the `wgsl`
  feature (#3).
- An integration test that runs 100 uploads per frame across window resize cycles.

### Changed
- Frames in flight: triple buffering (`MAX_FRAMES_IN_FLIGHT = 3`), a CPU fence wait
  before recording and one render semaphore per swapchain image.
- Memory allocation uses a 64 MiB power-of-two buddy allocator with O(1) coalescing and
  a warm block pool, which avoids the driver `maxMemoryAllocationCount` limit (#1, #13).
- Swapchain reconfiguration handles `VK_ERROR_OUT_OF_DATE_KHR`, `VK_SUBOPTIMAL_KHR` and
  `VK_NOT_READY`, passes `old_swapchain` on re-creation, and pauses rendering for
  zero-size windows.

### Fixed
- `BufferSlotGuard`, `TextureSlotGuard` and `SamplerSlotGuard` return descriptor heap
  slots without leaks or double frees (#14, #15).
- CI and headless test runs handle machines without a Vulkan device (#11, #12).
- Timeline semaphore signals stay monotonic under concurrent queue submissions.

## [0.1.3] - 2026-10-18

### Added
- Voxel raymarching example (`example-voxel-renderer`) with DDA traversal, procedural
  mountains, islands and a citadel.
- Translucent water with Beer-Lambert absorption, animated waves, Fresnel reflection and
  seabed caustics.
- C99 (`examples/c_example`), C++17 (`examples/cpp_example`) and Zig 0.14
  (`examples/zig_example`) examples using the shared header.
- `lgpu::default_device()`, `Device::default_device()`, `Queue::run(encoder)` and
  `Device::run(&queue, encoder)` helpers.
- `docs/ARCHITECTURE_AND_SAFETY.md`.
- A release workflow with pre-flight crates.io checks.

### Fixed
- `Device` and `Queue` keep an `Arc<VulkanInstance>`, so the parent instance outlives its
  devices.
- Empty suballocator blocks are returned to Vulkan, keeping one warm block per memory
  type (#6).
- `trim_memory()` and `suballocator_block_count()` on `Device`, `VulkanDevice` and the C
  API.
- `SurfaceTexture` implements `Drop`, so an acquired image is returned to the presentation
  engine if it is dropped without `present()` (#7).
- Voxel example mouse look no longer jumps on click.
- Camera movement in the voxel example: sprint on `Shift`, vertical movement on `Space`
  and `C`, and clamped frame deltas.
- The voxel DDA skips empty sky and void early.

## [0.1.2] - 2026-10-16

### Changed
- Integration test suites were renamed and organized.
- Documentation comments on `Buffer` and the memory suballocator were rewritten.
- The architecture guide now states project scope, safety invariants and
  synchronization guarantees more precisely.

## [0.1.1] - 2026-10-16

### Added
- Pipeline cache serialization through `device.get_pipeline_cache_data()`.
- `BlendState`, `BlendComponent`, `BlendFactor` and `BlendOp`.
- Configurable primitive topologies.
- Rasterizer cull modes and front face winding.
- `GraphicsPipelineDesc` for structured pipeline creation.
- Multi-pass compositing with `LoadOp<T>`, `ColorLoadOp`, `DepthLoadOp` and
  `encoder.begin_render_pass`.
- Indexed and indirect draws.
- Buffer to buffer and texture to buffer copies on `CommandEncoder`.
- Mipmapped and multisampled textures with sample count validation.
- C99 functions for device idling, native window surfaces and presentation.

### Fixed
- Dropping a surface texture no longer destroys swapchain images.
- `CommandEncoder` methods return errors on parameter and descriptor overflows.
- Unsubmitted command recorders return to the device pool on drop.
- `LgpuAdapterInfo` uses `uint8_t` instead of `bool` for ABI stability.
- The memory suballocator tracks chunk sizes during deallocation and coalesces chunks.
- Host visible and device local suballocations use separate memory blocks.
- Mapped buffer writes and texture transitions are guarded for concurrent use.

## [0.1.0] - 2026-10-14

### Added
- Initial release of `lgpu`.
- Dynamic Vulkan 1.3 loading through `ash`.
- Global bindless descriptor heap (`update_after_bind`) for buffers, textures and
  samplers.
- Dynamic rendering (`VK_KHR_dynamic_rendering`).
- C99 FFI header (`lgpu.h`).
- Headless compute, clear, bindless triangle and particle examples.
