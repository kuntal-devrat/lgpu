# lgpu

A lightweight, bindless-first Vulkan 1.3 wrapper for Rust, built on [`ash`](https://github.com/ash-rs/ash).

Designed for desktop experiments where you want dynamic rendering and global bindless descriptor indexing without the verbosity of raw Vulkan boilerplate or the overhead of large multi-backend frameworks.

## Key Features

- **Vulkan 1.3 Dynamic Rendering**: Uses dynamic rendering directly—no `VkRenderPass` or `VkFramebuffer` state boilerplate.
- **Bindless Descriptor Heaps**: Unbounded descriptor indexing (`update_after_bind`) for storage buffers, sampled textures, storage images, and samplers.
- **Fast Clean Builds**: Minimal dependency footprint (`ash`, `bitflags`, `thiserror`, `raw-window-handle`), compiling in 1–2 seconds.
- **C99 FFI**: Universal C API included via `lgpu-c`.

## Quick Example

```rust,no_run
use lgpu::{BufferDesc, BufferUsage, MemoryLocation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One-line GPU initialization
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

For full documentation and examples, visit the [GitHub Repository](https://github.com/kuntal-devrat/lgpu).
