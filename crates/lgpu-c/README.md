# lgpu-c

Universal C99 / C++ / Zig / Odin FFI bindings for `lgpu` (Lean GPU).

Provides a clean, header-driven C API (`lgpu.h`) wrapping the modern Vulkan 1.3 GPU abstraction layer.

## Features

- **Standard C99 Interface**: Header-only bindings (`include/lgpu.h`) suitable for C, C++, Zig, Nim, and Odin.
- **Cross-Platform Surfaces**: Native surface creation for X11 (Xlib/XCB), Wayland, Win32, and macOS (AppKit/Metal).
- **Safe ABI Design**: Typed opaque pointer handles, thread-local error messaging, and explicit resource lifecycle management.

For build instructions and examples, visit the [GitHub Repository](https://github.com/kuntal-devrat/lgpu).
