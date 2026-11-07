//! Voxel Chunk representation and Naive vs Greedy Meshing algorithms.

pub const CHUNK_SIZE: usize = 32;

pub const VOXEL_AIR: u8 = 0;
pub const VOXEL_GRASS: u8 = 1;
pub const VOXEL_DIRT: u8 = 2;
pub const VOXEL_STONE: u8 = 3;
pub const VOXEL_SAND: u8 = 4;
pub const VOXEL_WOOD: u8 = 5;
pub const VOXEL_LEAVES: u8 = 6;
pub const VOXEL_GOLD: u8 = 7;
pub const VOXEL_BRICK: u8 = 8;

pub fn voxel_color(v: u8) -> [f32; 4] {
    match v {
        VOXEL_GRASS => [0.22, 0.65, 0.25, 1.0],
        VOXEL_DIRT => [0.45, 0.30, 0.18, 1.0],
        VOXEL_STONE => [0.55, 0.55, 0.58, 1.0],
        VOXEL_SAND => [0.85, 0.78, 0.52, 1.0],
        VOXEL_WOOD => [0.50, 0.35, 0.20, 1.0],
        VOXEL_LEAVES => [0.18, 0.52, 0.18, 1.0],
        VOXEL_GOLD => [0.95, 0.80, 0.15, 1.0],
        VOXEL_BRICK => [0.72, 0.28, 0.22, 1.0],
        _ => [0.8, 0.8, 0.8, 1.0],
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Default, Clone)]
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

pub struct VoxelChunk {
    pub data: Box<[u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]>,
}

impl Default for VoxelChunk {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelChunk {
    pub fn new() -> Self {
        Self {
            data: vec![0u8; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }

    #[inline(always)]
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE {
            self.data[Self::index(x, y, z)]
        } else {
            VOXEL_AIR
        }
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: u8) {
        if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE {
            let idx = Self::index(x, y, z);
            self.data[idx] = val;
        }
    }

    pub fn generate_sample_terrain() -> Self {
        let mut chunk = Self::new();

        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let fx = x as f32 * 0.15;
                let fz = z as f32 * 0.15;
                let ground_h =
                    ((fx.sin() * 0.5 + 0.5) * (fz.cos() * 0.5 + 0.5) * 8.0 + 4.0) as usize;

                for y in 0..=ground_h.min(CHUNK_SIZE - 1) {
                    if y == ground_h {
                        chunk.set(x, y, z, VOXEL_GRASS);
                    } else if y >= ground_h.saturating_sub(2) {
                        chunk.set(x, y, z, VOXEL_DIRT);
                    } else {
                        chunk.set(x, y, z, VOXEL_STONE);
                    }
                }
            }
        }

        // Add a castle tower with windows in the center
        let cx = 16i32;
        let cz = 16i32;
        let base_y = 10;
        let top_y = 22;

        for y in base_y..=top_y {
            for dz in -4i32..=4 {
                for dx in -4i32..=4 {
                    let is_wall = dx.abs() == 4 || dz.abs() == 4;
                    let is_window =
                        y == 16 && (dx.abs() == 4 && dz == 0 || dz.abs() == 4 && dx == 0);
                    let is_battlement = y == top_y && (dx + dz).abs() % 2 == 1;

                    if is_wall && !is_window && !is_battlement {
                        chunk.set((cx + dx) as usize, y, (cz + dz) as usize, VOXEL_BRICK);
                    } else if y == base_y || y == top_y - 2 {
                        chunk.set((cx + dx) as usize, y, (cz + dz) as usize, VOXEL_WOOD);
                    }
                }
            }
        }

        // Add small gold deposits
        for y in 2..5 {
            for z in 4..7 {
                for x in 4..7 {
                    chunk.set(x, y, z, VOXEL_GOLD);
                }
            }
        }

        chunk
    }
}

type FaceDef = ([i32; 3], [f32; 3], [[f32; 3]; 4]);

/// Naive mesher: emits an unmerged 1x1 quad for every exposed voxel face.
pub fn mesh_naive(chunk: &VoxelChunk) -> MeshData {
    let mut mesh = MeshData::default();

    let dirs: [FaceDef; 6] = [
        // +X
        (
            [1, 0, 0],
            [1.0, 0.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
        ),
        // -X
        (
            [-1, 0, 0],
            [-1.0, 0.0, 0.0],
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ),
        // +Y
        (
            [0, 1, 0],
            [0.0, 1.0, 0.0],
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
        ),
        // -Y
        (
            [0, -1, 0],
            [0.0, -1.0, 0.0],
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
        ),
        // +Z
        (
            [0, 0, 1],
            [0.0, 0.0, 1.0],
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        // -Z
        (
            [0, 0, -1],
            [0.0, 0.0, -1.0],
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ),
    ];

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let v = chunk.get(x, y, z);
                if v == VOXEL_AIR {
                    continue;
                }
                let color = voxel_color(v);

                for (d, normal, quad_offsets) in &dirs {
                    let nx = x as i32 + d[0];
                    let ny = y as i32 + d[1];
                    let nz = z as i32 + d[2];

                    let neighbor = if nx >= 0
                        && nx < CHUNK_SIZE as i32
                        && ny >= 0
                        && ny < CHUNK_SIZE as i32
                        && nz >= 0
                        && nz < CHUNK_SIZE as i32
                    {
                        chunk.get(nx as usize, ny as usize, nz as usize)
                    } else {
                        VOXEL_AIR
                    };

                    if neighbor == VOXEL_AIR {
                        let base_idx = mesh.vertices.len() as u32;
                        for off in quad_offsets {
                            mesh.vertices.push(MeshVertex {
                                position: [x as f32 + off[0], y as f32 + off[1], z as f32 + off[2]],
                                normal: *normal,
                                color,
                            });
                        }
                        mesh.indices.extend_from_slice(&[
                            base_idx,
                            base_idx + 1,
                            base_idx + 2,
                            base_idx,
                            base_idx + 2,
                            base_idx + 3,
                        ]);
                    }
                }
            }
        }
    }

    mesh
}

/// Greedy mesher: combines coplanar, adjacent exposed faces of the same voxel type
/// into large single quads across each axis.
pub fn mesh_greedy(chunk: &VoxelChunk) -> MeshData {
    let mut mesh = MeshData::default();
    let dims = [CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE];

    // Sweep along each primary axis: 0 = X, 1 = Y, 2 = Z
    for d in 0..3 {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;

        let mut x = [0usize; 3];
        let mut q = [0usize; 3];
        q[d] = 1;

        // mask stores (voxel_type, normal_sign: +1 or -1) as i16
        // 0 means no exposed face
        let mut mask = vec![0i16; dims[u] * dims[v]];

        // Sweep slices from -1 to dims[d] - 1
        x[d] = 0;
        while x[d] <= dims[d] {
            let mut n = 0;

            // Generate mask for the current slice boundary
            for j in 0..dims[v] {
                x[v] = j;
                for i in 0..dims[u] {
                    x[u] = i;

                    let a = if x[d] > 0 {
                        chunk.get(x[0] - q[0], x[1] - q[1], x[2] - q[2])
                    } else {
                        VOXEL_AIR
                    };

                    let b = if x[d] < dims[d] {
                        chunk.get(x[0], x[1], x[2])
                    } else {
                        VOXEL_AIR
                    };

                    if (a != VOXEL_AIR) == (b != VOXEL_AIR) && a == b {
                        mask[n] = 0;
                    } else if a != VOXEL_AIR {
                        mask[n] = a as i16; // Face pointing in +d direction
                    } else {
                        mask[n] = -(b as i16); // Face pointing in -d direction
                    }
                    n += 1;
                }
            }

            // Greedy merge faces in mask
            n = 0;
            for j in 0..dims[v] {
                let mut i = 0;
                while i < dims[u] {
                    let val = mask[n];
                    if val != 0 {
                        // Compute width of identical faces
                        let mut w = 1;
                        while i + w < dims[u] && mask[n + w] == val {
                            w += 1;
                        }

                        // Compute height of identical rows of width w
                        let mut h = 1;
                        'outer: while j + h < dims[v] {
                            for k in 0..w {
                                if mask[n + k + h * dims[u]] != val {
                                    break 'outer;
                                }
                            }
                            h += 1;
                        }

                        // Construct the merged quad
                        x[u] = i;
                        x[v] = j;

                        let mut du = [0.0f32; 3];
                        let mut dv = [0.0f32; 3];
                        du[u] = w as f32;
                        dv[v] = h as f32;

                        let pos = [x[0] as f32, x[1] as f32, x[2] as f32];
                        let voxel_id = val.unsigned_abs() as u8;
                        let color = voxel_color(voxel_id);

                        let mut normal = [0.0f32; 3];
                        let base_idx = mesh.vertices.len() as u32;

                        if val > 0 {
                            // Positive normal direction (+d)
                            normal[d] = 1.0;
                            mesh.vertices.push(MeshVertex {
                                position: pos,
                                normal,
                                color,
                            });
                            mesh.vertices.push(MeshVertex {
                                position: [pos[0] + du[0], pos[1] + du[1], pos[2] + du[2]],
                                normal,
                                color,
                            });
                            mesh.vertices.push(MeshVertex {
                                position: [
                                    pos[0] + du[0] + dv[0],
                                    pos[1] + du[1] + dv[1],
                                    pos[2] + du[2] + dv[2],
                                ],
                                normal,
                                color,
                            });
                            mesh.vertices.push(MeshVertex {
                                position: [pos[0] + dv[0], pos[1] + dv[1], pos[2] + dv[2]],
                                normal,
                                color,
                            });

                            mesh.indices.extend_from_slice(&[
                                base_idx,
                                base_idx + 1,
                                base_idx + 2,
                                base_idx,
                                base_idx + 2,
                                base_idx + 3,
                            ]);
                        } else {
                            // Negative normal direction (-d)
                            normal[d] = -1.0;
                            mesh.vertices.push(MeshVertex {
                                position: pos,
                                normal,
                                color,
                            });
                            mesh.vertices.push(MeshVertex {
                                position: [pos[0] + dv[0], pos[1] + dv[1], pos[2] + dv[2]],
                                normal,
                                color,
                            });
                            mesh.vertices.push(MeshVertex {
                                position: [
                                    pos[0] + du[0] + dv[0],
                                    pos[1] + du[1] + dv[1],
                                    pos[2] + du[2] + dv[2],
                                ],
                                normal,
                                color,
                            });
                            mesh.vertices.push(MeshVertex {
                                position: [pos[0] + du[0], pos[1] + du[1], pos[2] + du[2]],
                                normal,
                                color,
                            });

                            mesh.indices.extend_from_slice(&[
                                base_idx,
                                base_idx + 1,
                                base_idx + 2,
                                base_idx,
                                base_idx + 2,
                                base_idx + 3,
                            ]);
                        }

                        // Clear merged cells in mask
                        for l in 0..h {
                            for k in 0..w {
                                mask[n + k + l * dims[u]] = 0;
                            }
                        }

                        i += w;
                        n += w;
                    } else {
                        i += 1;
                        n += 1;
                    }
                }
            }

            x[d] += 1;
        }
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_chunk_mesh() {
        let chunk = VoxelChunk::new();
        let naive = mesh_naive(&chunk);
        let greedy = mesh_greedy(&chunk);
        assert_eq!(naive.vertices.len(), 0);
        assert_eq!(greedy.vertices.len(), 0);
    }

    #[test]
    fn test_cube_greedy_reduction() {
        let mut chunk = VoxelChunk::new();
        // Place a solid 4x4x4 cube of stone
        for z in 2..6 {
            for y in 2..6 {
                for x in 2..6 {
                    chunk.set(x, y, z, VOXEL_STONE);
                }
            }
        }

        let naive = mesh_naive(&chunk);
        let greedy = mesh_greedy(&chunk);

        // Naive mesher emits 16 quads per face * 6 faces = 96 quads (192 triangles)
        assert_eq!(naive.quad_count(), 96);
        assert_eq!(naive.triangle_count(), 192);

        // Greedy mesher must merge each 4x4 face into a single 4x4 quad = 6 quads (12 triangles)
        assert_eq!(greedy.quad_count(), 6);
        assert_eq!(greedy.triangle_count(), 12);
        assert_eq!(greedy.vertices.len(), 24);
        assert_eq!(greedy.indices.len(), 36);

        let reduction = 1.0 - (greedy.triangle_count() as f32 / naive.triangle_count() as f32);
        assert!(
            reduction > 0.93,
            "Greedy reduction must be >93% for solid cube"
        );
    }

    #[test]
    fn test_sample_terrain_mesh_efficiency() {
        let chunk = VoxelChunk::generate_sample_terrain();
        let naive = mesh_naive(&chunk);
        let greedy = mesh_greedy(&chunk);

        assert!(naive.triangle_count() > 0);
        assert!(greedy.triangle_count() > 0);
        assert!(
            greedy.triangle_count() < naive.triangle_count(),
            "Greedy mesher must have fewer triangles than naive: {} vs {}",
            greedy.triangle_count(),
            naive.triangle_count()
        );
    }
}
