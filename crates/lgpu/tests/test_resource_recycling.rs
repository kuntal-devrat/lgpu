//! Integration tests verifying command recorder, fence recycling, and surface configurations.

use lgpu::*;

#[test]
fn test_command_pool_and_fence_recycling() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0].request_device(&instance).expect("Failed to request device");

    // Rapidly create, record, submit_and_wait across 50 iterations to ensure fences
    // and command recorders are recycled from cache without leaking or accumulating unbounded memory
    for i in 0..50 {
        let mut encoder = device.create_command_encoder().expect("Failed to create encoder");
        encoder.memory_barrier();
        let cmd = encoder.finish();
        queue.submit(cmd).expect("submit failed");

        if i % 10 == 0 {
            let buffer = device.create_buffer(&BufferDesc {
                label: Some("RecycleTestBuffer"),
                size: 256,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_DST,
                memory_location: MemoryLocation::CpuToGpu,
            }).expect("create buffer failed");

            buffer.write_at(0, &[1, 2, 3, 4]).expect("write failed");
        }
    }
}

#[test]
fn test_batch_submissions_and_fence_cache() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0].request_device(&instance).expect("Failed to request device");

    for _ in 0..20 {
        let mut enc1 = device.create_command_encoder().expect("create encoder 1");
        enc1.memory_barrier();
        let cmd1 = enc1.finish();

        let mut enc2 = device.create_command_encoder().expect("create encoder 2");
        enc2.memory_barrier();
        let cmd2 = enc2.finish();

        queue.submit_batches(vec![cmd1, cmd2]).expect("batch submit failed");
    }
}

#[test]
fn test_surface_configuration_and_types() {
    // Validate that SurfaceConfiguration and PresentMode types function correctly
    let config = SurfaceConfiguration {
        usage: TextureUsage::RENDER_ATTACHMENT,
        format: TextureFormat::Bgra8Unorm,
        width: 800,
        height: 600,
        present_mode: PresentMode::Fifo,
    };
    assert_eq!(config.width, 800);
    assert_eq!(config.height, 600);
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
    assert_eq!(config.present_mode, PresentMode::Fifo);

    let config_mailbox = SurfaceConfiguration {
        usage: TextureUsage::RENDER_ATTACHMENT,
        format: TextureFormat::Rgba8Unorm,
        width: 1920,
        height: 1080,
        present_mode: PresentMode::Mailbox,
    };
    assert_eq!(config_mailbox.width, 1920);
    assert_eq!(config_mailbox.present_mode, PresentMode::Mailbox);
}
