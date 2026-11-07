//! Polygonal Voxel Chunk Mesher with Greedy Meshing
//!
//! Demonstrates:
//! - Naive vs. Greedy Meshing on a 3D voxel chunk (>85% polygon reduction).
//! - Dedicated depth testing with [`TextureFormat::Depth32Float`].
//! - Indexed rendering using [`lgpu::CommandEncoder::bind_vertex_buffer`] and [`lgpu::SurfaceFrames`].
//! - Real-time camera navigation and mesh toggling.

mod mesher;

use bytemuck::{Pod, Zeroable};
use lgpu::{
    BufferDesc, BufferUsage, ColorLoadOp, FrameConfig, GraphicsPipelineDesc, MemoryLocation,
    PresentMode, SurfaceConfiguration, SurfaceFrames, TextureFormat, TextureUsage, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexStepMode,
};
use mesher::{mesh_greedy, mesh_naive, MeshVertex, VoxelChunk, CHUNK_SIZE};
use std::collections::HashSet;
use std::time::Instant;
use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::WindowBuilder,
};

const SHADER_SRC: &str = r#"
struct PushConstants {
    mvp: mat4x4<f32>,
    light_dir: vec3<f32>,
    ambient: f32,
};

var<push_constant> pc: PushConstants;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = pc.mvp * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(pc.light_dir);
    let diff = max(dot(n, l), 0.0);
    let lighting = pc.ambient + diff * (1.0 - pc.ambient);
    return vec4<f32>(in.color.rgb * lighting, in.color.a);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstants {
    mvp: [f32; 16],
    light_dir: [f32; 3],
    ambient: f32,
}

struct Camera {
    pos: [f32; 3],
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(pos: [f32; 3], yaw: f32, pitch: f32) -> Self {
        Self { pos, yaw, pitch }
    }

    fn forward(&self) -> [f32; 3] {
        let y = self.yaw.to_radians();
        let p = self.pitch.to_radians();
        let cp = p.cos();
        [y.cos() * cp, p.sin(), y.sin() * cp]
    }

    fn right(&self) -> [f32; 3] {
        let y = self.yaw.to_radians();
        [-y.sin(), 0.0, y.cos()]
    }

    fn view_matrix(&self) -> [f32; 16] {
        let f = self.forward();
        let r = self.right();
        let u = [
            r[1] * f[2] - r[2] * f[1],
            r[2] * f[0] - r[0] * f[2],
            r[0] * f[1] - r[1] * f[0],
        ];

        let p = self.pos;
        [
            r[0],
            u[0],
            -f[0],
            0.0,
            r[1],
            u[1],
            -f[1],
            0.0,
            r[2],
            u[2],
            -f[2],
            0.0,
            -(r[0] * p[0] + r[1] * p[1] + r[2] * p[2]),
            -(u[0] * p[0] + u[1] * p[1] + u[2] * p[2]),
            f[0] * p[0] + f[1] * p[1] + f[2] * p[2],
            1.0,
        ]
    }
}

fn perspective(fov_y_rad: f32, aspect: f32, z_near: f32, z_far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y_rad * 0.5).tan();
    [
        f / aspect,
        0.0,
        0.0,
        0.0,
        0.0,
        -f, // Invert Y for Vulkan coordinate space
        0.0,
        0.0,
        0.0,
        0.0,
        z_far / (z_near - z_far),
        -1.0,
        0.0,
        0.0,
        (z_near * z_far) / (z_near - z_far),
        0.0,
    ]
}

fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  lgpu Polygonal Voxel Mesher | Greedy Meshing Demo");
    println!("  Demonstrating dynamic rendering with Depth32Float depth buffer");
    println!("============================================================");
    println!("Controls:");
    println!("  [W, A, S, D]       Fly forward / left / back / right");
    println!("  [Space, C]         Fly up / fly down");
    println!("  [Shift]            Sprint (2x speed boost)");
    println!("  [Mouse Drag]       Look around (yaw / pitch)");
    println!("  [G]                Toggle between Greedy Meshing and Naive Meshing");
    println!("  [R]                Reset Camera");
    println!("  [Esc]              Exit");
    println!("============================================================");

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("lgpu - Greedy Mesher Demo")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let instance = lgpu::Instance::new()?;
    let adapters = instance.enumerate_adapters()?;
    let adapter = adapters.first().ok_or("no Vulkan adapter found")?;
    let (device, queue) = adapter.request_device(&instance)?;
    println!(
        "GPU: {} ({:?})",
        adapter.info().name,
        adapter.info().device_type
    );

    let mut surface = instance.create_surface(&window)?;
    let size = window.inner_size();
    let mut config = SurfaceConfiguration {
        usage: TextureUsage::RENDER_ATTACHMENT,
        format: TextureFormat::Bgra8Unorm,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: PresentMode::Mailbox,
    };
    surface.configure(&device, &config)?;

    // 1. Generate chunk and build both meshes
    println!("Generating sample 32x32x32 voxel chunk with castle & terrain...");
    let chunk = VoxelChunk::generate_sample_terrain();

    let naive_mesh = mesh_naive(&chunk);
    let greedy_mesh = mesh_greedy(&chunk);

    let naive_triangles = naive_mesh.triangle_count();
    let greedy_triangles = greedy_mesh.triangle_count();
    let reduction = (1.0 - (greedy_triangles as f32 / naive_triangles as f32)) * 100.0;

    println!("------------------------------------------------------------");
    println!(
        "  Naive Mesher:  {:>6} quads | {:>6} vertices | {:>6} triangles",
        naive_mesh.quad_count(),
        naive_mesh.vertices.len(),
        naive_triangles
    );
    println!(
        "  Greedy Mesher: {:>6} quads | {:>6} vertices | {:>6} triangles",
        greedy_mesh.quad_count(),
        greedy_mesh.vertices.len(),
        greedy_triangles
    );
    println!("  Optimization:  {:.1}% polygon reduction!", reduction);
    println!("------------------------------------------------------------");

    // 2. Upload vertex and index buffers to GPU
    let naive_vb = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("NaiveVB"),
            size: std::mem::size_of_val(&naive_mesh.vertices[..]) as u64,
            usage: BufferUsage::VERTEX,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &naive_mesh.vertices,
    )?;
    let naive_ib = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("NaiveIB"),
            size: std::mem::size_of_val(&naive_mesh.indices[..]) as u64,
            usage: BufferUsage::INDEX,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &naive_mesh.indices,
    )?;

    let greedy_vb = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("GreedyVB"),
            size: std::mem::size_of_val(&greedy_mesh.vertices[..]) as u64,
            usage: BufferUsage::VERTEX,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &greedy_mesh.vertices,
    )?;
    let greedy_ib = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("GreedyIB"),
            size: std::mem::size_of_val(&greedy_mesh.indices[..]) as u64,
            usage: BufferUsage::INDEX,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &greedy_mesh.indices,
    )?;

    // 3. Create graphics pipeline with Depth32Float depth testing
    let shader = device.create_shader_wgsl(SHADER_SRC)?;
    let mut pipeline_desc =
        GraphicsPipelineDesc::new("vs_main", "fs_main", TextureFormat::Bgra8Unorm);
    pipeline_desc.label = Some("VoxelMesherPipeline");
    pipeline_desc.depth_format = Some(TextureFormat::Depth32Float);
    pipeline_desc.depth_test = true;
    pipeline_desc.depth_write = true;
    pipeline_desc.vertex_buffers = vec![VertexBufferLayout::new(
        std::mem::size_of::<MeshVertex>() as u32,
        VertexStepMode::Vertex,
        vec![
            VertexAttribute::new(0, VertexFormat::Float32x3, 0),
            VertexAttribute::new(1, VertexFormat::Float32x3, 12),
            VertexAttribute::new(2, VertexFormat::Float32x4, 24),
        ],
    )];

    let pipeline = device.create_graphics_pipeline_with_desc(&shader, &pipeline_desc)?;

    // 4. Setup SurfaceFrames with dedicated Depth32Float depth buffer
    let mut frames = SurfaceFrames::new(
        config.clone(),
        FrameConfig::with_depth(TextureFormat::Depth32Float),
    );

    let mut camera = Camera::new(
        [
            CHUNK_SIZE as f32 * 0.5,
            CHUNK_SIZE as f32 * 0.8,
            CHUNK_SIZE as f32 * 1.8,
        ],
        -90.0,
        -20.0,
    );
    let mut use_greedy = true;
    let mut held_keys: HashSet<KeyCode> = HashSet::new();
    let mut mouse_dragging = false;
    let mut last_mouse_pos: Option<(f64, f64)> = None;

    let mut last_frame = Instant::now();
    let mut frame_count = 0u32;
    let mut last_fps_time = Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(new_size) => {
                    config.width = new_size.width.max(1);
                    config.height = new_size.height.max(1);
                    frames.set_config(config.clone());
                    if let Err(e) = surface.configure(&device, &config) {
                        eprintln!("Failed to reconfigure surface: {e}");
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key,
                            state,
                            logical_key,
                            ..
                        },
                    ..
                } => {
                    if let PhysicalKey::Code(code) = physical_key {
                        match state {
                            ElementState::Pressed => {
                                held_keys.insert(code);
                            }
                            ElementState::Released => {
                                held_keys.remove(&code);
                            }
                        }
                    }

                    if state == ElementState::Pressed {
                        match logical_key {
                            Key::Named(NamedKey::Escape) => elwt.exit(),
                            Key::Character(ref ch) => match ch.as_str() {
                                "g" | "G" => {
                                    use_greedy = !use_greedy;
                                    println!(
                                        "Mesher mode: {}",
                                        if use_greedy {
                                            "Greedy Meshing (merged quads)"
                                        } else {
                                            "Naive Meshing (1x1 quads)"
                                        }
                                    );
                                }
                                "r" | "R" => {
                                    camera.pos = [
                                        CHUNK_SIZE as f32 * 0.5,
                                        CHUNK_SIZE as f32 * 0.8,
                                        CHUNK_SIZE as f32 * 1.8,
                                    ];
                                    camera.yaw = -90.0;
                                    camera.pitch = -20.0;
                                    println!("Camera reset");
                                }
                                _ => {}
                            },
                            _ => {}
                        }
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        mouse_dragging = state == ElementState::Pressed;
                        if mouse_dragging {
                            last_mouse_pos = None;
                        }
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if mouse_dragging {
                        if let Some((lx, ly)) = last_mouse_pos {
                            let dx = (position.x - lx) as f32;
                            let dy = (position.y - ly) as f32;
                            camera.yaw += dx * 0.15;
                            camera.pitch = (camera.pitch - dy * 0.15).clamp(-88.0, 88.0);
                        }
                        last_mouse_pos = Some((position.x, position.y));
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = now.duration_since(last_frame).as_secs_f32().clamp(0.001, 0.05);
                    last_frame = now;

                    let fwd = camera.forward();
                    let right = camera.right();
                    let is_sprint = held_keys.contains(&KeyCode::ShiftLeft)
                        || held_keys.contains(&KeyCode::ShiftRight);
                    let speed = if is_sprint { 30.0 } else { 15.0 } * dt;

                    if held_keys.contains(&KeyCode::KeyW) {
                        camera.pos[0] += fwd[0] * speed;
                        camera.pos[1] += fwd[1] * speed;
                        camera.pos[2] += fwd[2] * speed;
                    }
                    if held_keys.contains(&KeyCode::KeyS) {
                        camera.pos[0] -= fwd[0] * speed;
                        camera.pos[1] -= fwd[1] * speed;
                        camera.pos[2] -= fwd[2] * speed;
                    }
                    if held_keys.contains(&KeyCode::KeyD) {
                        camera.pos[0] += right[0] * speed;
                        camera.pos[1] += right[1] * speed;
                        camera.pos[2] += right[2] * speed;
                    }
                    if held_keys.contains(&KeyCode::KeyA) {
                        camera.pos[0] -= right[0] * speed;
                        camera.pos[1] -= right[1] * speed;
                        camera.pos[2] -= right[2] * speed;
                    }
                    if held_keys.contains(&KeyCode::Space) {
                        camera.pos[1] += speed;
                    }
                    if held_keys.contains(&KeyCode::KeyC) || held_keys.contains(&KeyCode::ControlLeft)
                    {
                        camera.pos[1] -= speed;
                    }

                    // Compute MVP matrix
                    let aspect = config.width as f32 / config.height as f32;
                    let proj = perspective(70.0f32.to_radians(), aspect, 0.1, 500.0);
                    let view = camera.view_matrix();
                    let mvp = mat4_mul(&proj, &view);

                    let pc = PushConstants {
                        mvp,
                        light_dir: [0.6, 0.8, 0.4],
                        ambient: 0.35,
                    };

                    let frame = match frames.acquire(&device, &mut surface) {
                        Ok(frame) => frame,
                        Err(lgpu::Error::Hal(lgpu::HalError::NotReady(_))) => return,
                        Err(e) => {
                            eprintln!("Acquire failed: {e}");
                            return;
                        }
                    };

                    let mut encoder = match device.create_command_encoder() {
                        Ok(enc) => enc,
                        Err(e) => {
                            eprintln!("Failed to create encoder: {e}");
                            return;
                        }
                    };

                    let (active_vb, active_ib, active_indices) = if use_greedy {
                        (&greedy_vb, &greedy_ib, greedy_mesh.indices.len() as u32)
                    } else {
                        (&naive_vb, &naive_ib, naive_mesh.indices.len() as u32)
                    };

                    encoder.bind_vertex_buffer(0, active_vb, 0);

                    {
                        let mut pass = frame.begin(
                            &mut encoder,
                            ColorLoadOp::Clear([0.08, 0.12, 0.18, 1.0]),
                            1.0,
                        );
                        if let Err(e) = pass.draw_indexed(
                            &pipeline,
                            active_ib,
                            active_indices,
                            1,
                            0,
                            0,
                            0,
                            bytemuck::bytes_of(&pc),
                        ) {
                            eprintln!("Draw failed: {e}");
                        }
                    }

                    if let Err(e) = frame.present(&queue, encoder) {
                        eprintln!("Present failed: {e}");
                    }

                    frame_count += 1;
                    if last_fps_time.elapsed().as_secs_f32() >= 1.0 {
                        let fps = frame_count;
                        frame_count = 0;
                        last_fps_time = Instant::now();
                        let title = format!(
                            "lgpu - Greedy Mesher | FPS: {} | Mode: {} | Triangles: {} | Press [G] to toggle",
                            fps,
                            if use_greedy { "Greedy (Merged)" } else { "Naive (1x1)" },
                            if use_greedy { greedy_triangles } else { naive_triangles }
                        );
                        window.set_title(&title);
                    }

                    window.request_redraw();
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}
