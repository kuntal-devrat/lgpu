//! Textured mesh with vertex input, uniform buffers, mipmaps, depth testing and MSAA.
//!
//! Draws an indexed, textured quad at 4x MSAA with a depth attachment, resolves the image to a
//! single-sampled target and reads it back to verify the sampled color reached the framebuffer.

use bytemuck::{Pod, Zeroable};
use lgpu::{
    BufferDesc, BufferUsage, ColorLoadOp, DepthLoadOp, GraphicsPipelineDesc, MemoryLocation,
    SamplerDesc, TextureDesc, TextureDimension, TextureFormat, TextureUsage, VertexAttribute,
    VertexBufferLayout, VertexFormat, VertexStepMode,
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

const SIZE: u32 = 64;
const TEX_SIZE: u32 = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== lgpu: Textured Mesh ===");

    let instance = lgpu::Instance::new()?;
    let adapters = instance.enumerate_adapters()?;
    let adapter = adapters.first().ok_or("no Vulkan adapter found")?;
    let (device, queue) = adapter.request_device(&instance)?;
    println!(
        "Using GPU: {} ({:?})",
        adapter.info().name,
        adapter.info().device_type
    );

    // Bindless slots are allocated in creation order, so the first uniform buffer, texture and
    // sampler occupy slot 0 of their respective descriptor arrays.
    let uniforms = Uniforms {
        mvp: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        tint: [1.0, 1.0, 1.0, 1.0],
    };
    let uniform_buffer = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("MeshUniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: BufferUsage::UNIFORM,
            memory_location: MemoryLocation::CpuToGpu,
        },
        std::slice::from_ref(&uniforms),
    )?;
    assert_eq!(
        uniform_buffer.uniform_id().map(|id| id.0),
        Some(0),
        "the first uniform buffer must occupy bindless slot 0"
    );

    // A 4x4 checkerboard with a generated mip chain.
    let texture = device.create_texture(&TextureDesc {
        label: Some("Checker"),
        width: TEX_SIZE,
        height: TEX_SIZE,
        depth: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        mip_levels: 3,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
    })?;
    let mut texels = vec![0u8; (TEX_SIZE * TEX_SIZE * 4) as usize];
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let idx = ((y * TEX_SIZE + x) * 4) as usize;
            let bright = (x + y) % 2 == 0;
            let color = if bright {
                [255, 210, 60, 255]
            } else {
                [40, 20, 90, 255]
            };
            texels[idx..idx + 4].copy_from_slice(&color);
        }
    }
    device.write_texture_mip(&texture, 0, &texels, None)?;
    device.generate_mipmaps(&texture)?;

    let _sampler = device.create_sampler(&SamplerDesc {
        label: Some("NearestSampler"),
        min_filter: lgpu::FilterMode::Nearest,
        mag_filter: lgpu::FilterMode::Nearest,
        mipmap_filter: lgpu::FilterMode::Nearest,
        ..Default::default()
    })?;

    // Quad with UVs, drawn through an index buffer.
    let vertices = [
        Vertex {
            position: [-0.8, -0.8, 0.0],
            uv: [0.0, 1.0],
        },
        Vertex {
            position: [0.8, -0.8, 0.0],
            uv: [1.0, 1.0],
        },
        Vertex {
            position: [0.8, 0.8, 0.0],
            uv: [1.0, 0.0],
        },
        Vertex {
            position: [-0.8, 0.8, 0.0],
            uv: [0.0, 0.0],
        },
    ];
    let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

    let vertex_buffer = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("QuadVertices"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: BufferUsage::VERTEX,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &vertices,
    )?;
    let index_buffer = device.create_buffer_init_slice(
        &BufferDesc {
            label: Some("QuadIndices"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: BufferUsage::INDEX,
            memory_location: MemoryLocation::CpuToGpu,
        },
        &indices,
    )?;

    let shader = device.create_shader_wgsl(SHADER)?;
    let mut pipeline_desc =
        GraphicsPipelineDesc::new("vs_main", "fs_main", TextureFormat::Rgba8Unorm);
    pipeline_desc.label = Some("TexturedMesh");
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

    let msaa_color = device.create_texture(&TextureDesc {
        label: Some("MsaaColor"),
        width: SIZE,
        height: SIZE,
        depth: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        mip_levels: 1,
        sample_count: 4,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT,
    })?;
    let depth = device.create_texture(&TextureDesc {
        label: Some("Depth"),
        width: SIZE,
        height: SIZE,
        depth: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        mip_levels: 1,
        sample_count: 4,
        format: TextureFormat::Depth32Float,
        usage: TextureUsage::RENDER_ATTACHMENT,
    })?;
    let resolve = device.create_texture(&TextureDesc {
        label: Some("Resolve"),
        width: SIZE,
        height: SIZE,
        depth: 1,
        array_layers: 1,
        dimension: TextureDimension::D2,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT
            | TextureUsage::COPY_SRC
            | TextureUsage::TEXTURE_BINDING,
    })?;
    let readback = device.create_buffer(&BufferDesc {
        label: Some("Readback"),
        size: (SIZE * SIZE * 4) as u64,
        usage: BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::GpuToCpu,
    })?;

    let mut encoder = device.create_command_encoder()?;
    encoder.bind_vertex_buffer(0, &vertex_buffer, 0);
    encoder.begin_debug_label("textured mesh", [0.2, 0.8, 0.4, 1.0]);
    encoder.draw_indexed_with_ops_resolve(
        &msaa_color,
        Some(&resolve),
        Some((&depth, DepthLoadOp::Clear(1.0))),
        &pipeline,
        &index_buffer,
        6,
        1,
        0,
        0,
        0,
        &[],
        ColorLoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
    )?;
    encoder.end_debug_label();
    encoder.copy_texture_to_buffer(&resolve, &readback);
    queue.submit(encoder.finish())?;

    let mut pixels = vec![0u8; (SIZE * SIZE * 4) as usize];
    readback.read(&mut pixels)?;

    let center = (((SIZE / 2) * SIZE + (SIZE / 2)) * 4) as usize;
    let center_pixel = &pixels[center..center + 4];
    assert_eq!(
        center_pixel[3], 255,
        "center must be opaque: {center_pixel:?}"
    );
    assert!(
        center_pixel[0] > 30 || center_pixel[1] > 30 || center_pixel[2] > 30,
        "center must show the sampled checker texel, got {center_pixel:?}"
    );
    assert_eq!(
        &pixels[0..4],
        &[0, 0, 0, 255],
        "the corner must keep the clear color"
    );

    println!("Resolved center pixel: {center_pixel:?}");
    println!("SUCCESS: textured mesh rendered with vertex input, mips, depth and MSAA resolve.");
    Ok(())
}
