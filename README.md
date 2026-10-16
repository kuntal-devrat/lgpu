<p align="center">
  <img src="assets/logo_banner.png" alt="lgpu Logo Banner" width="560" />
</p>

<p align="center">
  <strong>Ultra-lightweight, bindless-first, modern Vulkan 1.3 GPU abstraction for Rust & C99.</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/lgpu"><img src="https://img.shields.io/crates/v/lgpu.svg?style=flat-square" alt="Crates.io" /></a>
  <a href="https://docs.rs/lgpu"><img src="https://docs.rs/lgpu/badge.svg?style=flat-square" alt="Docs.rs" /></a>
  <a href="https://github.com/kuntal-devrat/lgpu/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/kuntal-devrat/lgpu/ci.yml?branch=main&style=flat-square&label=CI" alt="CI Status" /></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue.svg?style=flat-square" alt="License" /></a>
  <img src="https://img.shields.io/badge/platforms-Linux%20%7C%20Windows%20%7C%20macOS-informational?style=flat-square" alt="Platforms" />
  <img src="https://img.shields.io/badge/Vulkan-1.3%2B%20Dynamic%20Rendering-red.svg?style=flat-square" alt="Vulkan 1.3" />
</p>

---

## Overview

`lgpu` (Lean GPU) departs from the compile-time bloat and restrictive legacy models of WebGPU/W3C by embracing the native hardware reality of modern GPUs:

* **Pure Vulkan 1.3 Dynamic Rendering**: Completely eliminates legacy `VkRenderPass` and `VkFramebuffer` state object boilerplate.
* **Bindless by Design**: Unbounded descriptor indexing (`VK_EXT_descriptor_indexing` / `update_after_bind`) for uniform & storage buffers, sampled textures, storage images, and samplers.
* **Sub-Second Compile Times**: Minimal dependency footprint (`ash`, `bitflags`, `thiserror`, `raw-window-handle`) compiling in ~1.5s from clean cache.
* **Universal C99 FFI**: First-class, zero-cost C99 / C++ / Zig / Odin bindings via [`include/lgpu.h`](crates/lgpu-c/include/lgpu.h).
* **Zero Mock Policy**: Every single function, pipeline, draw call, and compute dispatch executes against physical GPU hardware drivers.

---

## Interactive Showcase

<p align="center">
  <img src="assets/demo.gif" alt="lgpu Interactive 3D Window Demo" width="700" />
</p>

<p align="center">
  <em>Live 3D rotating polygon geometries, volumetric raymarched Torus SDF, and dynamic procedural shaders rendered via <code>lgpu</code> native swapchain (<code>cargo run -p example-window-shader</code>).</em>
</p>

---

## Architectural Comparison

| Dimension | `wgpu` | Raw Vulkan (`ash`) | `lgpu` (Lean GPU) |
| :--- | :---: | :---: | :---: |
| **Clean Compile Time** | 25 – 45s | 1 – 3s | **~1.5s** |
| **Binary Overhead** | 15 – 35 MB | Minimal | **< 1.8 MB** |
| **API Paradigm** | WebGPU (Pre-2018 Model) | Explicit Driver C-API | **Modern Vulkan 1.3+ Native** |
| **Descriptor Indexing** | Restricted / Emulated | Manual Vulkan Bookkeeping | **First-Class Bindless Heaps** |
| **Dynamic Rendering** | No (RenderPass Objects) | Raw Vulkan Structs | **Built-in Dynamic Rendering** |
| **C / C++ Bindings** | Separate C Wrapper | Native C | **Header-Driven C99 (`lgpu.h`)** |
| **Dynamic Loading** | Yes | Manual / Entry | **Automatic Dynamic Loading** |

---

## Workspace Structure

The repository is modularized into three focused crates:

```text
lgpu/
├── crates/
│   ├── lgpu-hal/   # Micro Hardware Abstraction Layer (Dynamic Vulkan 1.3 loading, heaps, suballocator)
│   ├── lgpu/       # High-level, ergonomic, safe Rust API with RAII resource lifecycles
│   └── lgpu-c/     # Universal C99 / Zig / Odin FFI bindings and shared library exports
├── examples/       # 5 Real-world hardware examples (Compute, Clear, Triangle, 100k Particles, Window)
└── assets/         # Project banners, media, and interactive demo assets
```

---

## Platform Support

* **Linux (100% Native)**: Wayland, X11 (Xlib & XCB), and Headless offscreen CI targets via Mesa (`RADV`, `ANV`, `NVK`) and NVIDIA proprietary drivers.
* **Windows (100% Native)**: Win32 surface presentation (`HWND` / `HINSTANCE`) with standard NVIDIA, AMD, and Intel GPU drivers dynamically loading `vulkan-1.dll`.
* **macOS / iOS (Hardware-Accelerated via MoltenVK)**: Supports `VK_EXT_metal_surface` for AppKit (`NSView` / `CAMetalLayer`) and UIKit with automatic `VK_KHR_portability_enumeration` and `VK_KHR_portability_subset` negotiation.

---

## Quick Start

### 1. Rust API

Add `lgpu` to your `Cargo.toml`:

```toml
[dependencies]
lgpu = "0.1.1"
bytemuck = { version = "1.19", features = ["derive"] }
```

#### Headless GPU Compute Dispatch:

```rust,no_run
use lgpu::{BufferDesc, BufferUsage, Instance, MemoryLocation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize instance and acquire hardware device
    let instance = Instance::new()?;
    let adapter = &instance.enumerate_adapters()?[0];
    let (device, queue) = adapter.request_device(&instance)?;
    println!("GPU: {}", adapter.info().name);

    // 2. Allocate GPU buffers (with automatic bindless IDs)
    let desc = BufferDesc {
        label: Some("ComputeBuffer"),
        size: 256,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };
    let buf_a = device.create_buffer_init_slice(&desc, &[1.0f32, 2.0, 3.0, 4.0])?;
    let buf_b = device.create_buffer_init_slice(&desc, &[10.0f32, 20.0, 30.0, 40.0])?;
    let buf_out = device.create_buffer(&desc)?;

    // 3. Compile SPIR-V and create compute pipeline
    let shader = device.create_shader_spirv(VECTOR_ADD_SPV)?;
    let pipeline = device.create_compute_pipeline(&shader, 3)?;

    // 4. Record and submit compute dispatch
    let mut encoder = device.create_command_encoder()?;
    encoder.dispatch_compute(&pipeline, &[&buf_a, &buf_b, &buf_out], 1, 1, 1);
    queue.submit(encoder.finish())?;

    // 5. Read back results
    let mut results = [0.0f32; 4];
    buf_out.read(bytemuck::cast_slice_mut(&mut results))?;
    println!("GPU Results: {:?}", results);

    Ok(())
}
```

---

### 2. C99 / C++ API

Include [`lgpu.h`](crates/lgpu-c/include/lgpu.h) in your C application:

```c
#include <stdio.h>
#include <lgpu.h>

int main(void) {
    lgpu_instance_t instance = NULL;
    lgpu_create_instance(&instance);

    uint32_t count = lgpu_get_adapter_count(instance);
    printf("Found %u physical GPU(s)\n", count);

    lgpu_adapter_info_t info;
    lgpu_get_adapter_info(instance, 0, &info);
    printf("Primary GPU: %s\n", info.name);

    lgpu_device_t device = NULL;
    lgpu_queue_t queue = NULL;
    lgpu_create_device(instance, 0, &device, &queue);

    // Clean up
    lgpu_destroy_device(device);
    lgpu_destroy_queue(queue);
    lgpu_destroy_instance(instance);
    return 0;
}
```

Compile and link against `lgpu_c`:

```bash
cargo build --release -p lgpu-c
gcc -std=c99 main.c -Icrates/lgpu-c/include -Ltarget/release -llgpu_c -Wl,-rpath,target/release -o app
./app
```

---

## Interactive Examples & Tutorials

The workspace includes 5 runnable hardware demonstrations:

```bash
# 1. Headless Compute (Vector addition verified on GPU)
cargo run -p example-headless-compute

# 2. Modern Dynamic Rendering Clear Screen (Verification of 4,096 pixels)
cargo run -p example-clear-screen

# 3. Procedural Bindless Triangle (Zero vertex buffer; ASCII rasterized terminal view)
cargo run -p example-bindless-triangle

# 4. 100,000 Particle Physics Simulation (49 Million particles/sec on GPU)
cargo run -p example-bindless-100k

# 5. Interactive 3D Windowed Demo (Rotating Polygons, Raymarched Torus, Plasma)
cargo run --release -p example-window-shader
```

---

## Verification & Testing

Run the integration and documentation test suites across the workspace:

```bash
# Run all 41 integration tests against live GPU hardware
cargo test --workspace

# Run all 17 documentation tests
cargo test --doc

# Verify strict zero-warning compiler and clippy standards
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

---

## License

`lgpu` is dual-licensed under:

* **MIT License** ([LICENSE-MIT](LICENSE-MIT))
* **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))

You may choose either license at your option.
