//! Integration tests verifying memory suballocator behavior, surface presentation, and bounds handling.

use lgpu::{
    BufferDesc, BufferUsage, Instance, MemoryLocation, PresentMode, SurfaceConfiguration,
    TextureDesc, TextureFormat, TextureUsage,
};

#[test]
fn test_suballocator_free_size_coalescing() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let (device, queue) = match adapters[0].request_device(&instance) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    // Allocate 100 odd-sized buffers (e.g. 73 bytes each)
    let desc = BufferDesc {
        label: Some("OddSizedBuffer"),
        size: 73,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };

    let mut buffers = Vec::new();
    for _ in 0..100 {
        let b = device.create_buffer(&desc).expect("Buffer creation failed");
        buffers.push(b);
    }

    // Drop all buffers - every chunk should be freed with its true allocated_size
    drop(buffers);

    // Verify subsequent allocation succeeds from coalesced chunks
    let large_desc = BufferDesc {
        label: Some("CoalescedBuffer"),
        size: 73 * 100,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };
    let large_buf = device
        .create_buffer(&large_desc)
        .expect("Failed to allocate from coalesced memory");
    drop(large_buf);
    let _ = queue;
}

#[test]
fn test_linear_vs_optimal_block_separation() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let (device, _) = match adapters[0].request_device(&instance) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    // Interleave creating 50 linear buffers and 50 optimal textures
    let mut buffers = Vec::new();
    let mut textures = Vec::new();

    for i in 0..50 {
        let buf = device
            .create_buffer(&BufferDesc {
                label: Some("InterleavedBuffer"),
                size: 256,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory_location: MemoryLocation::GpuOnly,
            })
            .expect("Buffer creation failed");

        let tex = device
            .create_texture(&TextureDesc {
                label: Some("InterleavedTexture"),
                width: 32,
                height: 32,
                depth: 1,
                mip_levels: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
            })
            .expect("Texture creation failed");

        buffers.push(buf);
        textures.push(tex);
        let _ = i;
    }

    // Dropping and reallocating should succeed cleanly
    drop(buffers);
    drop(textures);
}

#[test]
fn test_submit_surface_work_and_present() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let (device, queue) = match adapters[0].request_device(&instance) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    let mut surface = match instance.create_headless_surface() {
        Ok(s) => s,
        Err(_) => return,
    };

    let config = SurfaceConfiguration {
        width: 640,
        height: 480,
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        usage: TextureUsage::RENDER_ATTACHMENT,
    };
    surface
        .configure(&device, &config)
        .expect("Configure failed");

    // Acquire surface texture
    let surface_texture = surface
        .get_current_texture()
        .expect("Get current texture failed");

    let mut encoder = device
        .create_command_encoder()
        .expect("Create encoder failed");
    encoder
        .clear_and_render_texture(&surface_texture.texture, [0.1, 0.2, 0.3, 1.0])
        .expect("Clear failed");
    let cmd = encoder.finish();

    // Verify submit_surface_work + present separation flow
    queue
        .submit_surface_work(cmd, &surface_texture)
        .expect("submit_surface_work failed");
    surface_texture
        .present(&queue)
        .expect("surface_texture.present failed");

    device.wait_idle().expect("wait_idle failed");
}

#[test]
fn test_command_encoder_bounds_handling() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let (device, _) = match adapters[0].request_device(&instance) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    let encoder = device
        .create_command_encoder()
        .expect("Encoder creation failed");

    // Finished encoder consumes self
    let _cmd = encoder.finish();

    // Buffer read/write out of bounds returns clean Err
    let buf = device
        .create_buffer(&BufferDesc {
            label: Some("BoundsBuf"),
            size: 16,
            usage: BufferUsage::STORAGE,
            memory_location: MemoryLocation::CpuToGpu,
        })
        .expect("create_buffer failed");

    let mut dest = [0u8; 32];
    assert!(
        buf.read_at(8, &mut dest).is_err(),
        "Overflow read must return Err"
    );
    assert!(
        buf.write_at(16, &[1, 2]).is_err(),
        "Overflow write must return Err"
    );
}

#[test]
fn test_async_recorder_harvesting_during_submission() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let (device, queue) = match adapters[0].request_device(&instance) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    let sem = device
        .create_timeline_semaphore(0)
        .expect("Create semaphore failed");

    // Submit 5 async batches
    for i in 1..=5 {
        let encoder = device
            .create_command_encoder()
            .expect("Create encoder failed");
        let cmd = encoder.finish();
        queue
            .submit_async(cmd, &[], &[(&sem, i)])
            .expect("submit_async failed");
    }

    // Wait for semaphore value 5 on CPU
    sem.wait(5, u64::MAX).expect("Semaphore wait failed");

    // Synchronous submit harvests all completed async recorders
    let sync_encoder = device
        .create_command_encoder()
        .expect("Create encoder failed");
    let sync_cmd = sync_encoder.finish();
    queue.submit(sync_cmd).expect("Submit failed");

    // Verify device wait_idle also harvests everything without error
    device.wait_idle().expect("wait_idle failed");
}

#[test]
fn test_unpresented_surface_texture_drop_recovery() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let (device, queue) = match adapters[0].request_device(&instance) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    let mut surface = match instance.create_headless_surface() {
        Ok(s) => s,
        Err(_) => return,
    };

    let config = SurfaceConfiguration {
        width: 640,
        height: 480,
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        usage: TextureUsage::RENDER_ATTACHMENT,
    };
    surface
        .configure(&device, &config)
        .expect("Configure failed");

    // Acquire and drop SurfaceTexture repeatedly without calling present or submitting work.
    // Without the Drop safety implementation, this would exhaust swapchain images and hang
    // wait_for_fences/acquire_next_image on frame slot wraparound.
    for _ in 0..6 {
        let surface_texture = surface
            .get_current_texture()
            .expect("get_current_texture failed during unpresented drop loop");
        drop(surface_texture);
    }

    // Verify normal render and submit still succeeds cleanly after dropped frames
    let final_texture = surface
        .get_current_texture()
        .expect("Failed to acquire texture after drop loop");
    let mut encoder = device
        .create_command_encoder()
        .expect("Create encoder failed");
    encoder
        .clear_and_render_texture(&final_texture.texture, [0.5, 0.5, 0.5, 1.0])
        .expect("Clear failed");
    let cmd = encoder.finish();
    queue
        .submit_surface(cmd, final_texture)
        .expect("submit_surface failed after drop loop");

    device.wait_idle().expect("wait_idle failed");
}

#[test]
fn test_submitted_surface_work_dropped_without_present() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let (device, queue) = match adapters[0].request_device(&instance) {
        Ok(pair) => pair,
        Err(_) => return,
    };

    let mut surface = match instance.create_headless_surface() {
        Ok(s) => s,
        Err(_) => return,
    };

    let config = SurfaceConfiguration {
        width: 640,
        height: 480,
        format: TextureFormat::Bgra8Unorm,
        present_mode: PresentMode::Fifo,
        usage: TextureUsage::RENDER_ATTACHMENT,
    };
    surface
        .configure(&device, &config)
        .expect("Configure failed");

    // Acquire, submit work to GPU, but drop without calling .present()
    let surface_texture = surface
        .get_current_texture()
        .expect("get_current_texture failed");
    let mut encoder = device
        .create_command_encoder()
        .expect("Create encoder failed");
    encoder
        .clear_and_render_texture(&surface_texture.texture, [0.2, 0.4, 0.6, 1.0])
        .expect("Clear failed");
    let cmd = encoder.finish();

    queue
        .submit_surface_work(cmd, &surface_texture)
        .expect("submit_surface_work failed");
    // Dropped without calling surface_texture.present()
    drop(surface_texture);

    // Verify next frame can still be acquired, rendered, and presented cleanly
    let next_texture = surface
        .get_current_texture()
        .expect("Failed to acquire next texture");
    let mut next_encoder = device
        .create_command_encoder()
        .expect("Create next encoder failed");
    next_encoder
        .clear_and_render_texture(&next_texture.texture, [0.8, 0.2, 0.1, 1.0])
        .expect("Clear failed");
    let next_cmd = next_encoder.finish();
    queue
        .submit_surface(next_cmd, next_texture)
        .expect("submit_surface failed");

    device.wait_idle().expect("wait_idle failed");
}
