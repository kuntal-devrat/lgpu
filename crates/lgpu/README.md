# lgpu

A small Vulkan 1.3 wrapper for Rust, built on [`ash`](https://github.com/ash-rs/ash).

It targets desktop programs that want dynamic rendering and global bindless descriptor
indexing without the boilerplate of raw Vulkan or the build cost of a multi-backend
framework.

## Features

- Vulkan 1.3 dynamic rendering, without `VkRenderPass` or `VkFramebuffer` state.
- Bindless descriptor heaps using `update_after_bind` for storage buffers, sampled
  textures, storage images and samplers.
- A block suballocator for buffers and textures.
- Optional runtime WGSL compilation through Naga, behind the `wgsl` feature.
- A C API in the companion crate `lgpu-c`.

Clean builds take a second or two, since the dependency set is `ash`, `bitflags`,
`thiserror` and `raw-window-handle`.

## Example

```rust,no_run
use lgpu::{BufferDesc, BufferUsage, MemoryLocation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (device, _queue) = lgpu::default_device()?;

    let desc = BufferDesc {
        label: Some("ComputeBuffer"),
        size: 256,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };

    let buffer = device.create_buffer_init_slice(&desc, &[1.0f32, 2.0, 3.0, 4.0])?;
    println!("Buffer allocated with bindless ID: {:?}", buffer.id());

    Ok(())
}
```

Full documentation and examples are in the
[GitHub repository](https://github.com/kuntal-devrat/lgpu).
