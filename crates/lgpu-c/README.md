# lgpu-c

C99, C++, Zig and Odin bindings for `lgpu`.

The crate exposes a C ABI over the Vulkan 1.3 layer in `lgpu` and ships a single header,
[`include/lgpu.h`](include/lgpu.h).

## Features

- A C99 header usable from C, C++, Zig, Nim and Odin.
- Surface creation for X11 (Xlib and XCB), Wayland, Win32 and macOS through Metal.
- Opaque pointer handles, an error message per thread, and explicit resource destruction.

## Examples

The examples use `include/lgpu.h`:

- C99: [`examples/c_example`](../../examples/c_example) creates an instance, enumerates
  GPUs, writes a mapped buffer and renders a texture.
- C++17: [`examples/cpp_example`](../../examples/cpp_example) wraps handles with RAII and
  copies a buffer on the GPU.
- Zig 0.14: [`examples/zig_example`](../../examples/zig_example) imports the header with
  `@cImport` and allocates and verifies a buffer.

Build instructions and the rest of the workspace are in the
[GitHub repository](https://github.com/kuntal-devrat/lgpu).
