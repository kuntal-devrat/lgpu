# lgpu (Lean GPU)

An ultra-lightweight, bindless-first cross-platform GPU abstraction for Rust.

Designed for sub-second compile times, minimal binary footprint, and maximum performance.

## Key Highlights

- **Bindless by Design**: Unbounded descriptor indexing (`VK_EXT_descriptor_indexing`) for buffers, sampled textures, storage images, and samplers.
- **Pure Dynamic Rendering**: Native Vulkan 1.3 dynamic rendering without legacy `VkRenderPass` or `VkFramebuffer` boilerplate.
- **Fine-Grained Synchronization**: Native timeline semaphores and asynchronous multi-queue batch submission.
- **Cross-Platform Native**: Tested across Linux (Wayland, X11, Headless), Windows (Win32), and macOS/iOS (via MoltenVK).
- **Zero Mock Policy**: 100% real hardware execution with zero simulated RAM mocks or dummy stubs.

## Quick Example

```rust,no_run
use lgpu::{BufferDesc, BufferUsage, Instance, MemoryLocation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = Instance::new()?;
    let adapters = instance.enumerate_adapters()?;
    let (device, queue) = adapters[0].request_device(&instance)?;

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
