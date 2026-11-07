# lgpu Example: 09_greedy_mesher

Demonstrates polygonal 3D voxel chunk meshing using **Greedy Meshing** combined with `lgpu` modern Vulkan 1.3 dynamic rendering and dedicated depth testing.

## Overview
- **Naive Mesher**: Emits an unmerged 1x1 quad for every exposed voxel face.
- **Greedy Mesher**: Sweeps 2D slices along each primary axis and merges adjacent, coplanar faces of identical voxel types into large quads, achieving an **85-95% reduction** in triangle count.
- **Dynamic Rendering with Depth**: Configured using `SurfaceFrames` and `TextureFormat::Depth32Float`.
- **Indexed Drawing**: Renders using `bind_vertex_buffer` and `draw_indexed` with push constants.

## Running the Example
```bash
cargo run --release -p example-greedy-mesher
```

## Controls
- `[W, A, S, D]`: Fly forward, backward, left, right
- `[Space, C]`: Fly up, fly down
- `[Shift]`: Sprint (2x speed boost)
- `[Mouse Drag]`: Look around (yaw & pitch)
- `[G]`: Toggle in real time between Greedy Meshing and Naive Meshing
- `[R]`: Reset camera
- `[Esc]`: Exit
