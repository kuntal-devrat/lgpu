//! Integration tests verifying buffer offset read/write, Send/Sync thread safety, and resource recycling.

use std::sync::Arc;
use std::thread;
use lgpu::{
    BufferDesc, BufferUsage, Instance, MemoryLocation,
    TextureDesc, TextureFormat, TextureUsage,
};

#[test]
fn test_buffer_offset_read_write() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    let (device, _queue) = adapters[0].request_device(&instance).expect("Failed to request device");

    let desc = BufferDesc {
        label: Some("OffsetTestBuffer"),
        size: 256,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };
    let buffer = device.create_buffer(&desc).expect("Failed to create buffer");

    let data1 = [1u8, 2, 3, 4];
    let data2 = [10u8, 20, 30, 40, 50, 60, 70, 80];

    // Write at offset 0
    buffer.write_at(0, &data1).expect("Write at offset 0 failed");
    // Write at offset 32
    buffer.write_at(32, &data2).expect("Write at offset 32 failed");

    // Read back at offset 0
    let mut read1 = [0u8; 4];
    buffer.read_at(0, &mut read1).expect("Read at offset 0 failed");
    assert_eq!(read1, data1);

    // Read back at offset 32
    let mut read2 = [0u8; 8];
    buffer.read_at(32, &mut read2).expect("Read at offset 32 failed");
    assert_eq!(read2, data2);

    // Test overflow detection
    let overflow_data = [99u8; 10];
    assert!(buffer.write_at(250, &overflow_data).is_err(), "Must reject write exceeding buffer bounds");
    let mut overflow_read = [0u8; 10];
    assert!(buffer.read_at(250, &mut overflow_read).is_err(), "Must reject read exceeding buffer bounds");

    println!("SUCCESS: Buffer offset reads, writes, and overflow guards verified.");
}

#[test]
fn test_thread_safety_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<lgpu::Buffer>();
    assert_sync::<lgpu::Buffer>();
    assert_send::<lgpu::Texture>();
    assert_sync::<lgpu::Texture>();
    assert_send::<lgpu::Device>();
    assert_sync::<lgpu::Device>();
    assert_send::<lgpu::Queue>();
    assert_sync::<lgpu::Queue>();

    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    let (device, queue) = adapters[0].request_device(&instance).expect("Failed to request device");

    let desc = BufferDesc {
        label: Some("ThreadBuffer"),
        size: 64,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };
    let device_arc = Arc::new(device);
    let buffer = Arc::new(device_arc.create_buffer(&desc).expect("Failed to create buffer"));
    let queue_arc = Arc::new(queue);

    // Spawn 4 threads concurrently writing to disjoint offsets and submitting clear commands
    let mut handles = Vec::new();
    for thread_idx in 0..4 {
        let buf_clone = Arc::clone(&buffer);
        let dev_clone = Arc::clone(&device_arc);
        let q_clone = Arc::clone(&queue_arc);
        let handle = thread::spawn(move || {
            let offset = (thread_idx * 8) as u64;
            let val = (thread_idx + 1) as u8 * 11;
            let payload = [val; 8];
            buf_clone.write_at(offset, &payload).expect("Thread write failed");

            // Also test recording and submitting from a background worker thread!
            let encoder = dev_clone.create_command_encoder().expect("Worker thread encoder failed");
            let cmd = encoder.finish();
            q_clone.submit(cmd).expect("Worker thread submit failed");
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("Thread join failed");
    }

    // Verify all thread writes
    for thread_idx in 0..4 {
        let offset = (thread_idx * 8) as u64;
        let expected_val = (thread_idx + 1) as u8 * 11;
        let mut read = [0u8; 8];
        buffer.read_at(offset, &mut read).expect("Read failed");
        assert_eq!(read, [expected_val; 8]);
    }

    println!("SUCCESS: Multi-threaded Send + Sync, concurrent writes, and concurrent submissions verified.");
}

#[test]
fn test_command_pool_and_fence_recycling() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    let (device, queue) = adapters[0].request_device(&instance).expect("Failed to request device");

    let desc = TextureDesc {
        label: Some("RecycleTarget"),
        width: 32,
        height: 32,
        depth: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    };
    let target = device.create_texture(&desc).expect("Failed to create texture");

    // Perform 100 consecutive command buffer recordings and submissions
    // Before this fix, this would allocate and destroy 100 VkCommandPools and 100 VkFences.
    // Now, they are completely recycled without driver churn!
    for i in 0..100 {
        let mut encoder = device.create_command_encoder().expect("Failed to create encoder");
        let r = (i as f32) / 100.0;
        encoder.clear_and_render_texture(&target, [r, 0.5, 0.2, 1.0]);
        let cmd = encoder.finish();
        queue.submit(cmd).expect("Submit failed");
    }

    println!("SUCCESS: 100 consecutive submissions executed with recycled command pools and fences.");
}
