//! Integration tests verifying error paths, bounds checks, swapchain resizing, frames in flight, and multi-threading.

use lgpu::{
    BufferDesc, BufferUsage, Error, HalError, Instance, MemoryLocation, PresentMode,
    SurfaceConfiguration, TextureDesc, TextureFormat, TextureUsage,
};
use std::sync::Arc;
use std::thread;

#[test]
fn test_resource_creation_error_paths() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, _queue) = adapters[0]
        .request_device(&instance)
        .expect("request_device");

    // 1. Buffer creation with size: 0 must return InvalidParameter
    let zero_buffer_desc = BufferDesc {
        label: Some("ZeroBuffer"),
        size: 0,
        usage: BufferUsage::STORAGE,
        memory_location: MemoryLocation::CpuToGpu,
    };
    let err_buf = device.create_buffer(&zero_buffer_desc);
    assert!(
        matches!(err_buf, Err(Error::Hal(HalError::InvalidParameter(_)))),
        "Creating a buffer with size 0 must return HalError::InvalidParameter, got: {err_buf:?}"
    );

    // 2. Texture creation with width: 0 must return InvalidParameter
    let zero_width_tex = TextureDesc {
        label: Some("ZeroWidthTex"),
        width: 0,
        height: 64,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT,
    };
    let err_tex1 = device.create_texture(&zero_width_tex);
    assert!(
        matches!(err_tex1, Err(Error::Hal(HalError::InvalidParameter(_)))),
        "Creating a texture with width 0 must return HalError::InvalidParameter, got: {err_tex1:?}"
    );

    // 3. Texture creation with height: 0 must return InvalidParameter
    let zero_height_tex = TextureDesc {
        label: Some("ZeroHeightTex"),
        width: 64,
        height: 0,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT,
    };
    let err_tex2 = device.create_texture(&zero_height_tex);
    assert!(
        matches!(err_tex2, Err(Error::Hal(HalError::InvalidParameter(_)))),
        "Creating a texture with height 0 must return HalError::InvalidParameter, got: {err_tex2:?}"
    );

    // 4. Shader creation with invalid SPIR-V bytecode must return ShaderError
    let invalid_spirv = [0x12345678u32, 0x00000001u32];
    let err_shader = device.create_shader_spirv(&invalid_spirv);
    assert!(
        matches!(err_shader, Err(Error::Hal(HalError::ShaderError(_)))),
        "Invalid SPIR-V bytecode must return HalError::ShaderError, got: {err_shader:?}"
    );
}

#[test]
fn test_buffer_overflow_checks() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0]
        .request_device(&instance)
        .expect("request_device");

    let desc = BufferDesc {
        label: Some("OverflowTestBuffer"),
        size: 64,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };
    let buffer = device.create_buffer(&desc).expect("create_buffer");

    // In-bounds write should succeed
    let valid_data = [7u8; 64];
    assert!(buffer.write_at(0, &valid_data).is_ok());

    // In-bounds read should succeed and match
    let mut readback = [0u8; 64];
    assert!(buffer.read_at(0, &mut readback).is_ok());
    assert_eq!(readback, valid_data);

    // Overflow write 1: start at 60, write 10 bytes (ends at 70 > 64)
    let overflow_data = [9u8; 10];
    let err_write1 = buffer.write_at(60, &overflow_data);
    assert!(
        matches!(err_write1, Err(Error::Hal(HalError::CommandError(_)))),
        "Write exceeding buffer size must return CommandError, got: {err_write1:?}"
    );

    // Overflow write 2: offset beyond buffer size
    let err_write2 = buffer.write_at(65, &[1u8]);
    assert!(
        matches!(err_write2, Err(Error::Hal(HalError::CommandError(_)))),
        "Write at offset > buffer size must return CommandError, got: {err_write2:?}"
    );

    // Overflow read: start at 50, read 20 bytes (ends at 70 > 64)
    let mut overflow_readback = [0u8; 20];
    let err_read = buffer.read_at(50, &mut overflow_readback);
    assert!(
        matches!(err_read, Err(Error::Hal(HalError::CommandError(_)))),
        "Read exceeding buffer size must return CommandError, got: {err_read:?}"
    );

    // Queue::write_buffer overflow check
    let err_queue_write = queue.write_buffer(&buffer, 60, &overflow_data);
    assert!(
        matches!(err_queue_write, Err(Error::Hal(HalError::CommandError(_)))),
        "Queue::write_buffer overflow must return CommandError, got: {err_queue_write:?}"
    );
}

#[test]
fn test_swapchain_resize_and_reconfiguration() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0]
        .request_device(&instance)
        .expect("request_device");

    // Try creating headless surface
    let surface_res = instance.create_headless_surface();
    let mut surface = match surface_res {
        Ok(s) => s,
        Err(_) => {
            println!("VK_EXT_headless_surface not supported on this driver, skipping live swapchain test.");
            return;
        }
    };

    // 1. Calling get_current_texture on unconfigured surface must return Error
    let err_tex = surface.get_current_texture();
    assert!(
        err_tex.is_err(),
        "Calling get_current_texture before configure must return error"
    );

    // 2. Initial configuration: 640x480
    let config_initial = SurfaceConfiguration {
        usage: TextureUsage::RENDER_ATTACHMENT,
        format: TextureFormat::Bgra8Unorm,
        width: 640,
        height: 480,
        present_mode: PresentMode::Fifo,
    };
    surface
        .configure(&device, &config_initial)
        .expect("Initial surface configure");

    let surface_tex = surface
        .get_current_texture()
        .expect("get_current_texture at 640x480");
    assert_eq!(surface_tex.texture.width, 640);
    assert_eq!(surface_tex.texture.height, 480);
    assert!(
        !surface_tex.texture.is_owned(),
        "Swapchain textures must not be owned"
    );

    let mut encoder = device
        .create_command_encoder()
        .expect("create_command_encoder");
    encoder
        .clear_and_render_texture(&surface_tex.texture, [0.1, 0.2, 0.3, 1.0])
        .expect("clear");
    let cmd = encoder.finish();
    queue
        .submit_surface(cmd, surface_tex)
        .expect("submit_surface");

    // 3. Swapchain Resize & Reconfiguration: 800x600
    let config_resized = SurfaceConfiguration {
        usage: TextureUsage::RENDER_ATTACHMENT,
        format: TextureFormat::Bgra8Unorm,
        width: 800,
        height: 600,
        present_mode: PresentMode::Fifo,
    };
    surface
        .configure(&device, &config_resized)
        .expect("Resize surface configure");

    let resized_tex = surface
        .get_current_texture()
        .expect("get_current_texture at 800x600");
    assert_eq!(resized_tex.texture.width, 800);
    assert_eq!(resized_tex.texture.height, 600);
    assert!(!resized_tex.texture.is_owned());

    let mut encoder2 = device
        .create_command_encoder()
        .expect("create_command_encoder");
    encoder2
        .clear_and_render_texture(&resized_tex.texture, [0.4, 0.5, 0.6, 1.0])
        .expect("clear");
    let cmd2 = encoder2.finish();
    queue
        .submit_surface(cmd2, resized_tex)
        .expect("submit_surface resized");

    device.wait_idle().expect("wait_idle");
}

#[test]
fn test_multiple_frames_in_flight() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0]
        .request_device(&instance)
        .expect("request_device");

    let surface_res = instance.create_headless_surface();
    if let Ok(mut surface) = surface_res {
        let config = SurfaceConfiguration {
            usage: TextureUsage::RENDER_ATTACHMENT,
            format: TextureFormat::Bgra8Unorm,
            width: 256,
            height: 256,
            present_mode: PresentMode::Fifo,
        };
        surface
            .configure(&device, &config)
            .expect("surface.configure");

        // Render 6 consecutive frames (cycling through the MAX_FRAMES_IN_FLIGHT = 2 double buffer slots 3 times)
        for i in 0..6 {
            let surface_tex = surface.get_current_texture().expect("get_current_texture");
            let mut encoder = device
                .create_command_encoder()
                .expect("create_command_encoder");
            let color = (i as f32 + 1.0) / 7.0;
            encoder
                .clear_and_render_texture(&surface_tex.texture, [color, color, color, 1.0])
                .expect("clear");
            let cmd = encoder.finish();
            queue
                .submit_surface(cmd, surface_tex)
                .expect("submit_surface");
        }

        device.wait_idle().expect("wait_idle");
    }

    // Also test multiple frames in flight using timeline semaphores pipelining
    let timeline = device
        .create_timeline_semaphore(0)
        .expect("timeline semaphore");
    for frame in 1..=4 {
        let mut encoder = device.create_command_encoder().expect("encoder");
        encoder.memory_barrier();
        let cmd = encoder.finish();

        let wait_val = frame - 1;
        let signal_val = frame;
        queue
            .submit_async(cmd, &[(&timeline, wait_val)], &[(&timeline, signal_val)])
            .expect("submit_async frame in flight");
    }

    timeline
        .wait(4, 5_000_000_000)
        .expect("wait for all 4 pipelined frames");
    assert_eq!(timeline.value().expect("read timeline"), 4);
}

#[test]
fn test_bindless_descriptor_update_after_bind() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0]
        .request_device(&instance)
        .expect("request_device");

    let tex_desc = TextureDesc::new_2d(
        64,
        64,
        TextureFormat::Rgba8Unorm,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    );

    // 1. Allocate initial bindless textures
    let tex0 = device.create_texture(&tex_desc).expect("tex0");
    let tex1 = device.create_texture(&tex_desc).expect("tex1");
    assert_eq!(tex0.id().map(|id| id.0), Some(0));
    assert_eq!(tex1.id().map(|id| id.0), Some(1));

    // 2. Begin recording command encoder
    let mut encoder = device.create_command_encoder().expect("encoder");
    encoder.memory_barrier();

    // 3. Update descriptors while command recording is active!
    // Allocate tex2 (slot 2)
    let tex2 = device.create_texture(&tex_desc).expect("tex2");
    assert_eq!(tex2.id().map(|id| id.0), Some(2));

    // Drop tex0 (slot 0 recycled)
    drop(tex0);

    // Allocate tex3 (reuses slot 0!)
    let tex3 = device.create_texture(&tex_desc).expect("tex3");
    assert_eq!(tex3.id().map(|id| id.0), Some(0));

    // 4. Finish recording and submit
    encoder.memory_barrier();
    let cmd = encoder.finish();
    queue
        .submit(cmd)
        .expect("submit command recorded with concurrent descriptor updates");

    device.wait_idle().expect("wait_idle");
}

#[test]
fn test_concurrent_multithreaded_safety() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0]
        .request_device(&instance)
        .expect("request_device");

    let device_arc = Arc::new(device);
    let queue_arc = Arc::new(queue);
    let mut handles = Vec::new();

    for t_idx in 0..6 {
        let dev = Arc::clone(&device_arc);
        let q = Arc::clone(&queue_arc);

        let h = thread::spawn(move || {
            // Each thread allocates a buffer and reads/writes
            let desc = BufferDesc {
                label: Some("ThreadBuffer"),
                size: 128,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory_location: MemoryLocation::CpuToGpu,
            };
            let buf = dev.create_buffer(&desc).expect("thread create_buffer");

            let data = vec![(t_idx as u8) + 1; 128];
            buf.write(&data).expect("thread buf.write");

            let mut read_data = vec![0u8; 128];
            buf.read(&mut read_data).expect("thread buf.read");
            assert_eq!(read_data, data);

            // Record a command encoder and submit
            let mut enc = dev
                .create_command_encoder()
                .expect("thread create_command_encoder");
            enc.memory_barrier();
            let cmd = enc.finish();
            q.submit(cmd).expect("thread q.submit");
        });
        handles.push(h);
    }

    for h in handles {
        h.join().expect("thread panicked");
    }
}

#[cfg(feature = "wgsl")]
#[test]
fn test_create_shader_wgsl() {
    let instance = Instance::new().expect("Instance::new");
    let adapters = instance.enumerate_adapters().expect("enumerate_adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, _queue) = adapters[0]
        .request_device(&instance)
        .expect("request_device");

    let wgsl_src = r#"
        @vertex
        fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
            var pos = array<vec2<f32>, 3>(
                vec2<f32>(0.0, 0.5),
                vec2<f32>(-0.5, -0.5),
                vec2<f32>(0.5, -0.5)
            );
            return vec4<f32>(pos[in_vertex_index], 0.0, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
    "#;

    let shader = device.create_shader_wgsl(wgsl_src);
    assert!(
        shader.is_ok(),
        "create_shader_wgsl should succeed: {shader:?}"
    );

    let invalid_wgsl = "invalid wgsl syntax !!!";
    let err_shader = device.create_shader_wgsl(invalid_wgsl);
    assert!(
        err_shader.is_err(),
        "create_shader_wgsl with invalid syntax should fail"
    );
}
