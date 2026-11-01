# Voxel renderer (`example-voxel-renderer`)

A voxel raymarcher written with `lgpu`, Vulkan 1.3 dynamic rendering, push constants and
the Amanatides and Woo voxel DDA.

## Features

- Fragment shader voxel traversal with up to 140 DDA steps per ray.
- Translucent water:
  - Beer-Lambert absorption for depth dependent colour.
  - Animated surface waves from summed sine functions.
  - Schlick Fresnel reflection of the sky.
  - Animated caustics on the seabed.
  - Underwater fog and tint when the camera is below the water line.
- Three worlds: alpine highlands with lakes and trees, floating sky islands, and a
  step pyramid citadel.
- Shadow rays traced towards an animated sun.
- Corner based ambient occlusion from neighbouring solid voxels.
- The shader is plain WGSL in [`shaders/voxel.wgsl`](shaders/voxel.wgsl) and is compiled
  to SPIR-V at startup with Naga.

## Controls

| Input | Action |
| :--- | :--- |
| `W`, `A`, `S`, `D` | Fly forward, left, backward, right |
| `Space` | Fly up |
| `C` or `Ctrl` | Fly down |
| `Shift` (hold) | Sprint, 2.5x fly speed |
| Left mouse drag | Rotate the camera |
| Arrow keys | Rotate the camera |
| `1`, `2`, `3` | Switch world |
| `Tab` | Cycle render mode: beauty, ambient occlusion, normals, depth, voxel IDs |
| `T` | Toggle the day and night cycle |
| `R` | Reset the camera to the spawn point |
| `Esc` | Exit |

## Running

```bash
cargo run --release -p example-voxel-renderer
```
