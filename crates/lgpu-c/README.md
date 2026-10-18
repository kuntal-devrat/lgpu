# lgpu-c

Universal C99 / C++ / Zig / Odin FFI bindings for `lgpu` (Lean GPU).

Provides a clean, header-driven C API (`lgpu.h`) wrapping the modern Vulkan 1.3 GPU abstraction layer.

## Features

- **Standard C99 Interface**: Header-only bindings (`include/lgpu.h`) suitable for C, C++, Zig, Nim, and Odin.
- **Cross-Platform Surfaces**: Native surface creation for X11 (Xlib/XCB), Wayland, Win32, and macOS (AppKit/Metal).
- **Safe ABI Design**: Typed opaque pointer handles, thread-local error messaging, and explicit resource lifecycle management.

## Examples

Runnable polyglot examples using `include/lgpu.h`:

- **C99**: [`examples/c_example`](../../examples/c_example) — Vulkan instance setup, GPU enumeration, memory-mapped buffer write, and texture rendering.
- **Modern C++17**: [`examples/cpp_example`](../../examples/cpp_example) — RAII handle wrappers, GPU-to-GPU buffer copy, and result verification.
- **Zig 0.14**: [`examples/zig_example`](../../examples/zig_example) — Direct `@cImport`, buffer allocation, and memory safety checks.

For build instructions and full workspace documentation, visit the [GitHub Repository](https://github.com/kuntal-devrat/lgpu).
