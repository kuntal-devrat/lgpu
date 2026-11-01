//! Windowed textured mesh using [`lgpu::SurfaceFrames`].
//!
//! The frame helper acquires the swapchain image, rebuilds the depth and 4x MSAA attachments on
//! resize, submits the recorded draw and presents it, so the event loop only records draws.

use bytemuck::{Pod, Zeroable};
use lgpu::{
    BufferDesc, BufferUsage, ColorLoadOp, FrameConfig, GraphicsPipelineDesc, MemoryLocation,
    PresentMode, SamplerDesc, SurfaceConfiguration, SurfaceFrames, TextureDesc, TextureDimension,
    TextureFormat, TextureUsage, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const SHADER: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
    tint: vec4<f32>,
};

@group(0) @binding(4) var<uniform> ubos: array<Uniforms, 8>;
@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(0) @binding(2) var albedo_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.pos = ubos[0].mvp * vec4<f32>(position, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    return textureSample(albedo, albedo_sampler, input.uv) * ubos[0].tint;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    mvp: [f32; 16],
    tint: [f32; 4],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("lgpu - Windowed Textured Mesh")
        .with_inner_size(LogicalSize::new(960, 600))
        .build(&event_loop)?;

    let instance = lgpu::Instance::new()?;
    let adapters = instance.enumerate_adapters()?;
    let adapter = adapters.first().ok_or("no Vulkan adapter found")?;
    let (device, queue) = adapter.request_device(&instance)?;
    println!(
        "Renderer: {} ({:?}), async compute queue: {}",
        adapter.info().name,
        adapter.info().device_type,
        device.has_async_compute_queue()
    );

    let mut surface = instance.create_surface(&window)?;
    let size = window.inner_size();
    let mut config = SurfaceConfiguration {
        usage: TextureUsage::RENDER_ATTACHMENT,
        format: TextureFormat::Bgra8Unorm,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: PresentMode::Fifo,
    };
    surface.configure(&device, &config)?;

    // Bindless slots are allocated in creation order: the first uniform buffer, texture and
    // sampler occupy slot 0 of their descriptor arrays.
    let uniform_buffer = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("MeshUniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: BufferUsage::UNIFORM,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &[Uniforms {
            mvp: rotation_matrix(0.0),
            tint: [1.0, 1.0, 1.0, 1.0],
        }],
    )?;

    let texture = device.create_texture(&TextureDesc {
        label: Some("Checker"),
        width: 8,
        height: 8,
        depth: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        mip_levels: 4,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
    })?;
    let mut texels = vec![0u8; 8 * 8 * 4];
    for y in 0..8u32 {
        for x in 0..8u32 {
            let idx = ((y * 8 + x) * 4) as usize;
            let color = if (x + y) % 2 == 0 {
                [235, 180, 70, 255]
            } else {
                [55, 35, 120, 255]
            };
            texels[idx..idx + 4].copy_from_slice(&color);
        }
    }
    device.write_texture_mip(&texture, 0, &texels, None)?;
    device.generate_mipmaps(&texture)?;

    let _sampler = device.create_sampler(&SamplerDesc {
        label: Some("MeshSampler"),
        min_filter: lgpu::FilterMode::Linear,
        mag_filter: lgpu::FilterMode::Linear,
        mipmap_filter: lgpu::FilterMode::Linear,
        ..Default::default()
    })?;

    let vertices = [
        Vertex {
            position: [-0.9, -0.9, 0.0],
            uv: [0.0, 1.0],
        },
        Vertex {
            position: [0.9, -0.9, 0.0],
            uv: [1.0, 1.0],
        },
        Vertex {
            position: [0.9, 0.9, 0.0],
            uv: [1.0, 0.0],
        },
        Vertex {
            position: [-0.9, 0.9, 0.0],
            uv: [0.0, 0.0],
        },
    ];
    let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

    let vertex_buffer = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("MeshVertices"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: BufferUsage::VERTEX,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &vertices,
    )?;
    let index_buffer = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("MeshIndices"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: BufferUsage::INDEX,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &indices,
    )?;

    let shader = device.create_shader_wgsl(SHADER)?;
    let mut pipeline_desc =
        GraphicsPipelineDesc::new("vs_main", "fs_main", TextureFormat::Bgra8Unorm);
    pipeline_desc.label = Some("WindowedMesh");
    pipeline_desc.sample_count = 4;
    pipeline_desc.depth_format = Some(TextureFormat::Depth32Float);
    pipeline_desc.depth_test = true;
    pipeline_desc.depth_write = true;
    pipeline_desc.vertex_buffers = vec![VertexBufferLayout::new(
        std::mem::size_of::<Vertex>() as u32,
        VertexStepMode::Vertex,
        vec![
            VertexAttribute::new(0, VertexFormat::Float32x3, 0),
            VertexAttribute::new(1, VertexFormat::Float32x2, 12),
        ],
    )];
    let pipeline = device.create_graphics_pipeline_with_desc(&shader, &pipeline_desc)?;

    let mut frames = SurfaceFrames::new(
        config.clone(),
        FrameConfig::multisampled(4, Some(TextureFormat::Depth32Float)),
    );
    let mut angle = 0.0f32;

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => elwt.exit(),
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                config.width = size.width.max(1);
                config.height = size.height.max(1);
                frames.set_config(config.clone());
                if let Err(e) = surface.configure(&device, &config) {
                    eprintln!("Failed to reconfigure the surface: {e}");
                }
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                angle += 0.01;
                if let Err(e) = queue.write_buffer(
                    &uniform_buffer,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        mvp: rotation_matrix(angle),
                        tint: [1.0, 1.0, 1.0, 1.0],
                    }),
                ) {
                    eprintln!("Failed to update uniforms: {e}");
                    return;
                }

                let frame = match frames.acquire(&device, &mut surface) {
                    Ok(frame) => frame,
                    Err(lgpu::Error::Hal(lgpu::HalError::NotReady(message))) => {
                        println!("{message}");
                        return;
                    }
                    Err(e) => {
                        eprintln!("Failed to acquire a frame: {e}");
                        return;
                    }
                };

                let mut encoder = match device.create_command_encoder() {
                    Ok(encoder) => encoder,
                    Err(e) => {
                        eprintln!("Failed to create an encoder: {e}");
                        return;
                    }
                };
                encoder.bind_vertex_buffer(0, &vertex_buffer, 0);
                {
                    let mut pass = frame.begin(
                        &mut encoder,
                        ColorLoadOp::Clear([0.02, 0.02, 0.04, 1.0]),
                        1.0,
                    );
                    if let Err(e) = pass.draw_indexed(&pipeline, &index_buffer, 6, 1, 0, 0, 0, &[])
                    {
                        eprintln!("Draw failed: {e}");
                    }
                }
                if let Err(e) = frame.present(&queue, encoder) {
                    eprintln!("Present failed: {e}");
                }
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;

    Ok(())
}

fn rotation_matrix(angle: f32) -> [f32; 16] {
    let (sin, cos) = angle.sin_cos();
    let scale = 0.9;
    [
        cos * scale,
        -sin * scale,
        0.0,
        0.0,
        sin * scale,
        cos * scale,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}
