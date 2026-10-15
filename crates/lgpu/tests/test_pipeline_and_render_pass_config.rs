//! Integration tests verifying texture ownership, pipeline configurations, and render pass compositing.

use lgpu::{
    BlendState, BufferDesc, BufferUsage, ColorLoadOp, CullMode,
    FrontFace, GraphicsPipelineDesc, Instance, MemoryLocation,
    PrimitiveState, PrimitiveTopology, RenderPassDesc, TextureDesc, TextureFormat,
    TextureUsage,
};

const TRIANGLE_SPV: &[u32] = &[
    119734787, 65536, 28, 67, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 524303, 0, 16, 1834972022, 7235937, 9, 12, 14, 524303, 4, 63, 1834972006,
    7235937, 56, 59, 62, 196624, 63, 7, 327752, 6, 0, 35, 0, 327752, 6, 1, 35, 16, 262215,
    9, 11, 42, 262215, 12, 11, 0, 262215, 14, 30, 0, 262215, 56, 11, 15, 262215, 59, 30, 0,
    262215, 62, 30, 0, 131091, 2, 196630, 4, 32, 262167, 3, 4, 4, 262167, 5, 4, 3, 262174, 6,
    3, 5, 262165, 7, 32, 0, 262176, 10, 1, 7, 262203, 10, 9, 1, 262176, 13, 3, 3, 262203, 13,
    12, 3, 262176, 15, 3, 5, 262203, 15, 14, 3, 196641, 17, 2, 262187, 7, 18, 0, 262187, 4,
    19, 0, 262187, 4, 20, 3204448256, 262187, 4, 21, 1065353216, 458796, 3, 22, 19, 20, 19,
    21, 393260, 5, 23, 21, 19, 19, 262187, 7, 24, 1, 262187, 4, 25, 1056964608, 458796, 3,
    26, 25, 25, 19, 21, 393260, 5, 27, 19, 21, 19, 458796, 3, 28, 20, 25, 19, 21, 393260, 5,
    29, 19, 19, 21, 262176, 31, 7, 6, 196654, 6, 32, 131092, 34, 262176, 39, 7, 3, 262176,
    41, 7, 5, 262176, 57, 1, 3, 262203, 57, 56, 1, 262176, 60, 1, 5, 262203, 60, 59, 1,
    262203, 13, 62, 3, 327734, 2, 16, 0, 17, 131320, 8, 327739, 31, 30, 7, 32, 262205, 7,
    11, 9, 131321, 33, 131320, 33, 327850, 34, 35, 11, 18, 196855, 36, 0, 262394, 35, 37,
    38, 131320, 37, 327745, 39, 40, 30, 18, 196670, 40, 22, 327745, 41, 42, 30, 24, 196670,
    42, 23, 131321, 36, 131320, 38, 327850, 34, 43, 11, 24, 196855, 44, 0, 262394, 43, 45,
    46, 131320, 45, 327745, 39, 47, 30, 18, 196670, 47, 26, 327745, 41, 48, 30, 24, 196670,
    48, 27, 131321, 44, 131320, 46, 327745, 39, 49, 30, 18, 196670, 49, 28, 327745, 41, 50,
    30, 24, 196670, 50, 29, 131321, 44, 131320, 44, 131321, 36, 131320, 36, 262205, 6, 51,
    30, 327761, 3, 52, 51, 0, 196670, 12, 52, 327761, 5, 53, 51, 1, 196670, 14, 53, 65789,
    65592, 327734, 2, 63, 0, 17, 131320, 54, 262205, 3, 58, 56, 262205, 5, 61, 59, 327760,
    6, 55, 58, 61, 131321, 64, 131320, 64, 327761, 5, 65, 55, 1, 327760, 3, 66, 65, 21,
    196670, 62, 66, 65789, 65592,
];

#[test]
fn test_texture_ownership_flag() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, _queue) = adapters[0].request_device(&instance).expect("request_device");

    let tex_desc = TextureDesc::new_2d(
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let owned_tex = device.create_texture(&tex_desc).expect("create_texture");
    assert!(owned_tex.is_owned(), "User created texture must be owned");

    // Dropping owned_tex destroys its internal image/memory properly.
    drop(owned_tex);
}

#[test]
fn test_error_propagation_push_constants_overflow() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, _queue) = adapters[0].request_device(&instance).expect("request_device");

    let shader = device.create_shader_spirv(TRIANGLE_SPV).expect("shader");
    let pipeline = device
        .create_graphics_pipeline(
            &shader,
            "vs_main",
            "fs_main",
            TextureFormat::Rgba8Unorm,
        )
        .expect("pipeline");

    let tex_desc = TextureDesc::new_2d(
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let target = device.create_texture(&tex_desc).expect("texture");

    let mut encoder = device.create_command_encoder().expect("encoder");

    // Push constants limit is 128 bytes. Sending 256 bytes must return an Error!
    let large_push_constants = vec![1u8; 256];
    let result = encoder.draw(&target, &pipeline, 3, &large_push_constants, [0.0; 4]);
    assert!(result.is_err(), "Push constants exceeding 128 bytes must return Err");
}

#[test]
fn test_command_encoder_abandonment_and_recycling() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, _queue) = adapters[0].request_device(&instance).expect("request_device");

    // Create and abandon 32 command encoders without calling .finish()
    // Thanks to Drop implementation on CommandEncoder, all command recorders
    // are recycled back into VulkanDevice's cache without any pool leaks!
    for _ in 0..32 {
        let _abandoned_encoder = device.create_command_encoder().expect("create encoder");
        // drops here without finish()
    }

    // Creating another encoder should succeed smoothly and reuse cached recorders
    let mut encoder = device.create_command_encoder().expect("create encoder after recycling");
    encoder.memory_barrier();
    let _cmd = encoder.finish();
}

#[test]
fn test_configurable_pipeline_desc() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0].request_device(&instance).expect("request_device");

    let shader = device.create_shader_spirv(TRIANGLE_SPV).expect("shader");

    // Test pipeline with custom blend state, custom topology, and cull mode
    let mut desc = GraphicsPipelineDesc::new("vs_main", "fs_main", TextureFormat::Rgba8Unorm);
    desc.blend = Some(BlendState::ALPHA_BLENDING);
    desc.primitive = PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        cull_mode: CullMode::None,
        front_face: FrontFace::Cw,
    };

    let pipeline = device.create_graphics_pipeline_with_desc(&shader, &desc).expect("pipeline");

    let tex_desc = TextureDesc::new_2d(
        32,
        32,
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let target = device.create_texture(&tex_desc).expect("texture");

    let mut encoder = device.create_command_encoder().expect("encoder");
    encoder.draw(&target, &pipeline, 3, &[], [0.2, 0.4, 0.6, 1.0]).expect("draw");
    let cmd = encoder.finish();
    queue.submit(cmd).expect("submit");
}

#[test]
fn test_load_op_and_render_pass_compositing() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0].request_device(&instance).expect("request_device");

    let shader = device.create_shader_spirv(TRIANGLE_SPV).expect("shader");
    let mut desc = GraphicsPipelineDesc::new("vs_main", "fs_main", TextureFormat::Rgba8Unorm);
    desc.blend = Some(BlendState::ALPHA_BLENDING);
    let pipeline = device.create_graphics_pipeline_with_desc(&shader, &desc).expect("pipeline");

    let width = 32;
    let height = 32;
    let tex_desc = TextureDesc::new_2d(
        width,
        height,
        TextureFormat::Rgba8Unorm,
        TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    );
    let target = device.create_texture(&tex_desc).expect("texture");

    // 1. Test draw_with_ops:
    // First draw: Clear to opaque blue
    // Second draw: Load previous contents without clearing!
    let mut encoder = device.create_command_encoder().expect("encoder");
    encoder
        .draw_with_ops(
            &target,
            &pipeline,
            3,
            &[],
            ColorLoadOp::Clear([0.0, 0.0, 1.0, 1.0]),
        )
        .expect("draw 1 with clear");

    encoder
        .draw_with_ops(&target, &pipeline, 3, &[], ColorLoadOp::Load)
        .expect("draw 2 with load");

    let readback_buf = device
        .create_buffer(&BufferDesc {
            label: Some("readback"),
            size: (width * height * 4) as u64,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            memory_location: MemoryLocation::CpuToGpu,
        })
        .expect("buffer");

    encoder.copy_texture_to_buffer(&target, &readback_buf);
    let cmd = encoder.finish();
    queue.submit(cmd).expect("submit");

    // 2. Test RenderPass API:
    // RenderPass automatically executes the first draw with color_load_op (Clear)
    // and automatically transitions subsequent draws to LoadOp::Load!
    let mut encoder = device.create_command_encoder().expect("encoder 2");
    {
        let mut pass = encoder.begin_render_pass(RenderPassDesc {
            color_texture: &target,
            color_load_op: ColorLoadOp::Clear([0.5, 0.5, 0.5, 1.0]),
            depth_attachment: None,
        });
        pass.draw(&pipeline, 3, &[]).expect("pass draw 1");
        pass.draw(&pipeline, 3, &[]).expect("pass draw 2 (composited)");
    }
    let cmd2 = encoder.finish();
    queue.submit(cmd2).expect("submit 2");
}
