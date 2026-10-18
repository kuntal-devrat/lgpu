# Real-Time Voxel Renderer (`example-voxel-renderer`)

A real-time voxel raymarching engine built with **`lgpu`** modern Vulkan 1.3 dynamic rendering, Push Constants, and Amanatides & Woo Fast Voxel DDA (Digital Differential Analyzer).

---

## Features

- **Digital Differential Analyzer (DDA)**: Pixel-perfect voxel raymarching executed directly in the fragment shader with up to 140 traversal steps per ray.
- **Minecraft-Style Translucent Water**:
  - **Beer-Lambert Light Absorption**: Realistic underwater depth attenuation for realistic aquatic clarity and deep blues.
  - **Animated Surface Waves**: Multi-frequency sine perturbation simulating fluid surface rippling.
  - **Fresnel Reflection**: Schlick approximation reflecting the atmospheric skybox based on viewing angles.
  - **Seabed Caustics**: Animated light patterns projecting onto underwater sand and stone.
  - **Underwater Camera Fog**: Dynamic tint and atmospheric fog when the camera dives below the water line.
- **Multiple Voxel Worlds**:
  - **1: Mountain Highlands & Lakes**: Rolling alpine terrain with pine trees, sandy beaches, underwater depth, and mineral veins.
  - **2: Floating Sky Islands**: Overhanging islands with gravity-defying terrain and vertical drops.
  - **3: Ancient Citadel**: Step pyramid ruins with battlements and river plains.
- **Dynamic Lighting & Voxel Shadows**: Secondary DDA shadow rays cast towards an animated sun light source.
- **Voxel Ambient Occlusion (AO)**: Corner darkening based on neighboring solid blocks.
- **Shader in Clean WGSL**: The shader is written in plain-text modern WGSL ([`shaders/voxel.wgsl`](shaders/voxel.wgsl)) and compiled to SPIR-V at runtime via Naga in ~2ms.

---

## Controls

| Key / Input | Action |
| :--- | :--- |
| **`W`, `A`, `S`, `D`** | Fly forward, left, backward, right |
| **`Space`** | Ascend (fly up) |
| **`C` / `Ctrl`** | Descend (fly down) |
| **`Shift` (Hold)** | Sprint boost (2.5x fly speed) |
| **Mouse Left-Drag** | Rotate camera (smooth look around) |
| **Arrow Keys** | Alternative camera rotation |
| **`1`, `2`, `3`** | Switch voxel world mode |
| **`Tab`** | Cycle render modes (Beauty, Ambient Occlusion, Normals, Depth, Voxel IDs) |
| **`T`** | Toggle animated day/night sun cycle |
| **`R`** | Reset camera to initial spawn position |
| **`Esc`** | Exit application |

---

## Running

```bash
cargo run --release -p example-voxel-renderer
```
