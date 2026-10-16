# lgpu-hal

Micro Hardware Abstraction Layer for `lgpu` (Lean GPU).

Provides zero-overhead, dynamic-loading access to Vulkan 1.3+ dynamic rendering, bindless descriptor indexing (`VK_EXT_descriptor_indexing`), and timeline semaphores.

## Features

- **Dynamic Loading**: Loads the platform Vulkan library at runtime without static SDK linkage.
- **Bindless Descriptor Heaps**: Pre-configured unbounded heaps for storage buffers, textures, samplers, and storage images.
- **Dynamic Rendering**: Direct dynamic render passes with modern dynamic pipeline states.
- **Cross-Platform**: Tested and verified across Linux (Wayland/X11), Windows (Win32), and macOS/iOS (via MoltenVK).

For the high-level, safe Rust API, see [`lgpu`](https://crates.io/crates/lgpu).
