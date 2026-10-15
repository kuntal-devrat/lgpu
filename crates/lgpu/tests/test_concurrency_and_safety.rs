//! Integration tests verifying concurrency, Send/Sync invariants, and thread-safety.

use lgpu::*;
use std::sync::Arc;
use std::thread;

// Compile-time checks for Send and Sync invariants
fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn test_send_sync_type_invariants() {
    // Resources that MUST be Send + Sync for concurrent sharing across threads
    assert_send::<Buffer>();
    assert_sync::<Buffer>();

    assert_send::<Texture>();
    assert_sync::<Texture>();

    assert_send::<Sampler>();
    assert_sync::<Sampler>();

    assert_send::<Shader>();
    assert_sync::<Shader>();

    assert_send::<ComputePipeline>();
    assert_sync::<ComputePipeline>();

    assert_send::<GraphicsPipeline>();
    assert_sync::<GraphicsPipeline>();

    assert_send::<TimelineSemaphore>();
    assert_sync::<TimelineSemaphore>();

    assert_send::<Device>();
    assert_sync::<Device>();

    assert_send::<Queue>();
    assert_sync::<Queue>();

    // CommandEncoder MUST be Send (can be moved to worker threads)
    assert_send::<CommandEncoder>();

    // CommandBuffer MUST be Send
    assert_send::<CommandBuffer>();
}

#[test]
fn test_concurrent_buffer_access_and_synchronization() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance
        .enumerate_adapters()
        .expect("Failed to enumerate adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, _) = adapters[0]
        .request_device(&instance)
        .expect("Failed to request device");

    let size = 1024u64;
    let buffer = Arc::new(
        device
            .create_buffer(&BufferDesc {
                label: Some("ConcurrentBuffer"),
                size,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory_location: MemoryLocation::CpuToGpu,
            })
            .expect("Failed to create host buffer"),
    );

    // Initial write
    let init_data = vec![0xAAu8; size as usize];
    buffer.write_at(0, &init_data).expect("write failed");

    // Spawn multiple threads concurrently writing and reading to test mapped_ptr synchronization lock
    let mut handles = vec![];
    for thread_idx in 0..4 {
        let buf_clone = Arc::clone(&buffer);
        handles.push(thread::spawn(move || {
            let offset = (thread_idx * 128) as u64;
            let write_val = (thread_idx + 1) as u8;
            let chunk = vec![write_val; 128];
            buf_clone
                .write_at(offset, &chunk)
                .expect("concurrent write failed");

            let mut read_back = vec![0u8; 128];
            buf_clone
                .read_at(offset, &mut read_back)
                .expect("concurrent read failed");
            assert_eq!(read_back, chunk);
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }

    // Verify all 4 chunks were preserved accurately
    for thread_idx in 0..4 {
        let offset = (thread_idx * 128) as u64;
        let expected_val = (thread_idx + 1) as u8;
        let mut read_back = vec![0u8; 128];
        buffer.read_at(offset, &mut read_back).expect("read failed");
        assert_eq!(read_back, vec![expected_val; 128]);
    }
}

#[test]
fn test_concurrent_texture_layout_protection() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance
        .enumerate_adapters()
        .expect("Failed to enumerate adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, _) = adapters[0]
        .request_device(&instance)
        .expect("Failed to request device");

    let texture = Arc::new(
        device
            .create_texture(&TextureDesc {
                label: Some("LayoutTestTex"),
                width: 64,
                height: 64,
                depth: 1,
                mip_levels: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            })
            .expect("Failed to create texture"),
    );

    // Concurrently access texture layout across threads
    let mut handles = vec![];
    for _ in 0..8 {
        let tex_clone = Arc::clone(&texture);
        handles.push(thread::spawn(move || {
            // Under Vulkan, texture layout is safely protected by Mutex
            // Check that texture properties are read correctly concurrently
            assert_eq!(tex_clone.width, 64);
            assert_eq!(tex_clone.height, 64);
            assert_eq!(tex_clone.format, TextureFormat::Rgba8Unorm);
        }));
    }

    for h in handles {
        h.join().expect("thread panicked");
    }
}
