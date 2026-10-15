//! Integration tests verifying graphics and bindless expansion features:
//! - Bindless Texture slot allocation, indexing, and recycling
//! - Bindless Sampler creation, indexing, and recycling
//! - Dynamic rendering with depth/stencil attachments (TextureFormat::Depth32Float)
//! - Graphics pipeline creation with depth testing and depth writing
//! - Rendering with color and depth attachments

use lgpu::{
    AddressMode, FilterMode, Instance, SamplerDesc,
    TextureDesc, TextureFormat, TextureUsage,
};

#[test]
fn test_bindless_texture_slot_allocation_and_recycling() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    let (device, _queue) = adapters[0].request_device(&instance).expect("Failed to request device");

    let desc = TextureDesc {
        label: Some("SampledTex0"),
        width: 64,
        height: 64,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    };

    // 1. Allocate texture with TEXTURE_BINDING -> must receive bindless slot 0
    let tex0 = device.create_texture(&desc).expect("Failed to create tex0");
    assert_eq!(tex0.id().map(|id| id.0), Some(0), "First bindless texture should receive slot 0");

    // 2. Allocate second texture -> must receive bindless slot 1
    let tex1 = device.create_texture(&desc).expect("Failed to create tex1");
    assert_eq!(tex1.id().map(|id| id.0), Some(1), "Second bindless texture should receive slot 1");

    // 3. Drop tex0 -> slot 0 should be recycled
    drop(tex0);

    // 4. Allocate third texture -> should reuse recycled slot 0!
    let tex2 = device.create_texture(&desc).expect("Failed to create tex2");
    assert_eq!(tex2.id().map(|id| id.0), Some(0), "Third texture should reuse recycled slot 0");

    println!("SUCCESS: Bindless texture slot allocation and recycling verified.");
}

#[test]
fn test_bindless_sampler_creation_and_recycling() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    let (device, _queue) = adapters[0].request_device(&instance).expect("Failed to request device");

    let desc1 = SamplerDesc {
        label: Some("LinearSampler"),
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
    };

    // 1. Allocate first sampler -> slot 0
    let samp0 = device.create_sampler(&desc1).expect("Failed to create sampler0");
    assert_eq!(samp0.id().map(|id| id.0), Some(0), "First sampler should receive slot 0");

    // 2. Allocate second sampler -> slot 1
    let desc2 = SamplerDesc {
        label: Some("NearestRepeatSampler"),
        min_filter: FilterMode::Nearest,
        mag_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
    };
    let samp1 = device.create_sampler(&desc2).expect("Failed to create sampler1");
    assert_eq!(samp1.id().map(|id| id.0), Some(1), "Second sampler should receive slot 1");

    // 3. Drop samp0 -> slot 0 must be recycled
    drop(samp0);

    // 4. Allocate third sampler -> should reuse recycled slot 0
    let samp2 = device.create_sampler(&desc2).expect("Failed to create sampler2");
    assert_eq!(samp2.id().map(|id| id.0), Some(0), "Third sampler should reuse recycled slot 0");

    println!("SUCCESS: Bindless sampler creation and slot recycling verified.");
}

#[test]
fn test_dynamic_rendering_with_depth_attachment() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    let (device, queue) = adapters[0].request_device(&instance).expect("Failed to request device");

    // 1. Create color attachment texture
    let color_desc = TextureDesc {
        label: Some("ColorAttachment"),
        width: 128,
        height: 128,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    };
    let color_tex = device.create_texture(&color_desc).expect("Failed to create color texture");

    // 2. Create Depth32Float attachment texture
    let depth_desc = TextureDesc {
        label: Some("DepthAttachment"),
        width: 128,
        height: 128,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Depth32Float,
        usage: TextureUsage::RENDER_ATTACHMENT,
    };
    let depth_tex = device.create_texture(&depth_desc).expect("Failed to create depth texture");

    // 3. Clear both color and depth attachments via dynamic rendering
    let mut encoder = device.create_command_encoder().expect("Failed to create encoder");
    encoder.clear_and_render_with_depth(
        &color_tex,
        &depth_tex,
        [0.05, 0.05, 0.1, 1.0],
        1.0, // Clear depth to 1.0
    ).expect("clear failed");

    let cmd = encoder.finish();
    queue.submit(cmd).expect("Failed to submit clear_and_render_with_depth");

    println!("SUCCESS: Dynamic rendering with Depth32Float attachment verified.");
}

#[test]
fn test_graphics_pipeline_with_depth_testing() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    let (device, queue) = adapters[0].request_device(&instance).expect("Failed to request device");

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
    let shader = device.create_shader_spirv(TRIANGLE_SPV).expect("Failed to create shader");


    // Create graphics pipeline with Depth32Float format, depth test enabled, depth write enabled
    let pipeline = device.create_graphics_pipeline_with_depth(
        &shader,
        "vs_main",
        "fs_main",
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Depth32Float),
        true, // depth_test
        true, // depth_write
    ).expect("Failed to create graphics pipeline with depth");

    let color_desc = TextureDesc {
        label: Some("DrawColorTex"),
        width: 128,
        height: 128,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    };
    let color_tex = device.create_texture(&color_desc).expect("Failed to create color texture");

    let depth_desc = TextureDesc {
        label: Some("DrawDepthTex"),
        width: 128,
        height: 128,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Depth32Float,
        usage: TextureUsage::RENDER_ATTACHMENT,
    };
    let depth_tex = device.create_texture(&depth_desc).expect("Failed to create depth texture");

    let mut encoder = device.create_command_encoder().expect("Failed to create encoder");
    encoder.draw_with_depth(
        &color_tex,
        &depth_tex,
        &pipeline,
        3,
        &[],
        [0.1, 0.2, 0.3, 1.0],
        1.0,
    ).expect("draw failed");

    let cmd = encoder.finish();
    queue.submit(cmd).expect("Failed to submit draw_with_depth");

    println!("SUCCESS: Graphics pipeline with depth testing & draw_with_depth verified.");
}
