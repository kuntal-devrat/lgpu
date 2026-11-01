//! Integration test covering vertex attributes, bindless uniform buffers and MSAA resolve.

use lgpu::{
    BufferDesc, BufferUsage, ColorLoadOp, GraphicsPipelineDesc, MemoryLocation, TextureDesc,
    TextureDimension, TextureFormat, TextureUsage, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexStepMode,
};

macro_rules! require_gpu {
    () => {
        match (|| {
            let instance = lgpu::Instance::new().ok()?;
            let adapters = instance.enumerate_adapters().ok()?;
            if adapters.is_empty() {
                return None;
            }
            let (device, queue) = adapters[0].request_device(&instance).ok()?;
            Some((instance, device, queue))
        })() {
            Some(tuple) => tuple,
            None => {
                eprintln!("Skipping test: Vulkan 1.3 physical device not available");
                return;
            }
        }
    };
}

const VERTEX_SHADER: &str = r#"
struct Uniforms {
    scale: vec4<f32>,
};

@group(0) @binding(4) var<uniform> ubos: array<Uniforms, 8>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> VsOut {
    var out: VsOut;
    let scale = ubos[0].scale.x;
    out.pos = vec4<f32>(position.xy * scale, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(input: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[cfg(feature = "wgsl")]
#[test]
fn test_vertex_input_uniform_buffer_and_msaa_resolve() {
    let (_instance, device, queue) = require_gpu!();

    const SIZE: u32 = 64;

    let shader = device
        .create_shader_wgsl(VERTEX_SHADER)
        .expect("compile WGSL vertex and fragment shaders");

    let mut pipeline_desc =
        GraphicsPipelineDesc::new("vs_main", "fs_main", TextureFormat::Rgba8Unorm);
    pipeline_desc.sample_count = 4;
    pipeline_desc.vertex_buffers = vec![VertexBufferLayout::new(
        std::mem::size_of::<Vertex>() as u32,
        VertexStepMode::Vertex,
        vec![
            VertexAttribute::new(0, VertexFormat::Float32x3, 0),
            VertexAttribute::new(1, VertexFormat::Float32x3, 12),
        ],
    )];

    let pipeline = device
        .create_graphics_pipeline_with_desc(&shader, &pipeline_desc)
        .expect("create pipeline with vertex attributes");

    let vertices = [
        Vertex {
            position: [-0.8, -0.8, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.8, -0.8, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.0, 0.8, 0.0],
            color: [0.0, 0.0, 1.0],
        },
    ];

    let vertex_buffer = device
        .create_buffer_init_slice(
            &BufferDesc {
                label: Some("VertexBuffer"),
                size: std::mem::size_of_val(&vertices) as u64,
                usage: BufferUsage::VERTEX,
                memory_location: MemoryLocation::CpuToGpu,
            },
            &vertices,
        )
        .expect("create vertex buffer");

    let uniform_buffer = device
        .create_buffer_init_slice(
            &BufferDesc {
                label: Some("UniformBuffer"),
                size: 16,
                usage: BufferUsage::UNIFORM,
                memory_location: MemoryLocation::CpuToGpu,
            },
            &[1.0f32, 0.0, 0.0, 0.0],
        )
        .expect("create uniform buffer");
    assert!(
        uniform_buffer.uniform_id().is_some(),
        "a UNIFORM buffer must receive a bindless uniform slot"
    );

    let msaa_texture = device
        .create_texture(&TextureDesc {
            label: Some("MsaaColor"),
            width: SIZE,
            height: SIZE,
            depth: 1,
            dimension: TextureDimension::D2,
            array_layers: 1,
            mip_levels: 1,
            sample_count: 4,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::RENDER_ATTACHMENT,
        })
        .expect("create multisampled attachment");

    let resolve_texture = device
        .create_texture(&TextureDesc {
            label: Some("MsaaResolve"),
            width: SIZE,
            height: SIZE,
            depth: 1,
            dimension: TextureDimension::D2,
            array_layers: 1,
            mip_levels: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::RENDER_ATTACHMENT
                | TextureUsage::COPY_SRC
                | TextureUsage::TEXTURE_BINDING,
        })
        .expect("create resolve target");

    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("MsaaReadback"),
            size: (SIZE * SIZE * 4) as u64,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            memory_location: MemoryLocation::GpuToCpu,
        })
        .expect("create readback buffer");

    let mut encoder = device.create_command_encoder().expect("create encoder");
    encoder.bind_vertex_buffer(0, &vertex_buffer, 0);
    encoder
        .draw_with_ops_resolve(
            &msaa_texture,
            Some(&resolve_texture),
            &pipeline,
            3,
            &[],
            ColorLoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
        )
        .expect("draw with MSAA resolve");
    encoder.copy_texture_to_buffer(&resolve_texture, &readback);
    queue.submit(encoder.finish()).expect("submit");

    let mut pixels = vec![0u8; (SIZE * SIZE * 4) as usize];
    readback
        .read(&mut pixels)
        .expect("read back resolved image");

    let center = (((SIZE / 2) * SIZE + (SIZE / 2)) * 4) as usize;
    let center_pixel = &pixels[center..center + 4];
    assert_eq!(
        center_pixel[3], 255,
        "resolved center pixel must be opaque, got {center_pixel:?}"
    );
    assert!(
        center_pixel[0] as u32 + center_pixel[1] as u32 + center_pixel[2] as u32 > 0,
        "center pixel must be covered by the triangle, got {center_pixel:?}"
    );

    assert_eq!(
        &pixels[0..4],
        &[0, 0, 0, 255],
        "corner pixel must keep the clear color"
    );
}

#[cfg(feature = "wgsl")]
#[test]
fn test_vertex_attribute_validation() {
    let (_instance, device, _queue) = require_gpu!();

    let shader = device
        .create_shader_wgsl(VERTEX_SHADER)
        .expect("compile WGSL shaders");

    // An attribute that runs past its stride must be rejected.
    let mut bad = GraphicsPipelineDesc::new("vs_main", "fs_main", TextureFormat::Rgba8Unorm);
    bad.vertex_buffers = vec![VertexBufferLayout::new(
        8,
        VertexStepMode::Vertex,
        vec![VertexAttribute::new(0, VertexFormat::Float32x3, 0)],
    )];
    assert!(
        device
            .create_graphics_pipeline_with_desc(&shader, &bad)
            .is_err(),
        "attribute larger than the stride must be rejected"
    );

    // Duplicate locations must be rejected.
    let mut duplicate = GraphicsPipelineDesc::new("vs_main", "fs_main", TextureFormat::Rgba8Unorm);
    duplicate.vertex_buffers = vec![VertexBufferLayout::new(
        24,
        VertexStepMode::Vertex,
        vec![
            VertexAttribute::new(0, VertexFormat::Float32x3, 0),
            VertexAttribute::new(0, VertexFormat::Float32x3, 12),
        ],
    )];
    assert!(
        device
            .create_graphics_pipeline_with_desc(&shader, &duplicate)
            .is_err(),
        "duplicate attribute locations must be rejected"
    );
}
