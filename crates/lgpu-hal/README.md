# lgpu-hal

Hardware abstraction layer for `lgpu`.

It loads the platform Vulkan library at runtime and wraps the Vulkan 1.3 pieces the
higher level crate needs: dynamic rendering, bindless descriptor indexing
(`VK_EXT_descriptor_indexing`), timeline semaphores, a memory suballocator and swapchain
presentation.

## Features

- Runtime loading of the Vulkan library, with no SDK linkage at build time.
- Bindless descriptor heaps for storage buffers, textures, samplers and storage images.
- Dynamic render passes with dynamic viewport and scissor state.
- Surfaces for Linux (Wayland, X11), Windows (Win32) and macOS (MoltenVK).

For the safe Rust API, use [`lgpu`](https://crates.io/crates/lgpu).
