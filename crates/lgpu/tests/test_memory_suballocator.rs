//! Integration tests verifying GPU memory suballocation and pipeline caching.

use lgpu::*;

#[test]
fn test_suballocator_exceeding_4096_allocations() {
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

    // Allocate 5,000 small buffers to verify suballocation.
    // Without a suballocator, this would exceed the Vulkan maxMemoryAllocationCount
    // limit (typically 4096) and fail with VK_ERROR_TOO_MANY_OBJECTS.
    const BUFFER_COUNT: usize = 5000;
    let mut buffers = Vec::with_capacity(BUFFER_COUNT);

    for i in 0..BUFFER_COUNT {
        let buf = device
            .create_buffer(&BufferDesc {
                label: Some("SuballocatedBuffer"),
                size: 256,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory_location: MemoryLocation::CpuToGpu,
            })
            .expect("create_buffer failed under suballocation");

        // Write unique pattern
        let pattern = [(i & 0xFF) as u8; 4];
        buf.write_at(0, &pattern).expect("write failed");
        buffers.push(buf);
    }

    assert_eq!(buffers.len(), BUFFER_COUNT);

    // Verify samples across the entire allocated range
    for i in (0..BUFFER_COUNT).step_by(250) {
        let mut read_back = [0u8; 4];
        buffers[i].read_at(0, &mut read_back).expect("read failed");
        assert_eq!(read_back, [(i & 0xFF) as u8; 4]);
    }

    // Drop all buffers to verify clean free list recycling
    drop(buffers);

    // Reallocate 1,000 buffers into the recycled space
    let mut recycled_buffers = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let buf = device
            .create_buffer(&BufferDesc {
                label: Some("RecycledBuffer"),
                size: 512,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory_location: MemoryLocation::CpuToGpu,
            })
            .expect("reallocation in recycled suballocator block failed");
        recycled_buffers.push(buf);
    }
    assert_eq!(recycled_buffers.len(), 1000);
}

#[test]
fn test_pipeline_cache_reuse() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance
        .enumerate_adapters()
        .expect("Failed to enumerate adapters");
    if adapters.is_empty() {
        return;
    }
    let (device, queue) = adapters[0]
        .request_device(&instance)
        .expect("Failed to request device");

    pub const VECTOR_ADD_SPV: &[u32] = &[
        119734787, 65536, 28, 36, 0, 131089, 1, 720906, 1599492179, 1599227979, 1919906931,
        1600481121, 1717990754, 1935635045, 1634889588, 1667196263, 1936941420, 0, 393227, 1,
        1280527431, 1685353262, 808793134, 0, 196622, 0, 1, 393231, 5, 20, 1852399981, 0, 17,
        393232, 20, 17, 64, 1, 1, 262215, 4, 6, 4, 262215, 7, 34, 0, 262215, 7, 33, 0, 196679, 8,
        2, 327752, 8, 0, 35, 0, 262215, 10, 34, 0, 262215, 10, 33, 1, 196679, 11, 2, 327752, 11, 0,
        35, 0, 262215, 13, 34, 0, 262215, 13, 33, 2, 196679, 14, 2, 327752, 14, 0, 35, 0, 262215,
        17, 11, 28, 131091, 2, 196630, 3, 32, 196637, 4, 3, 262165, 6, 32, 0, 262167, 5, 6, 3,
        196638, 8, 4, 262176, 9, 12, 8, 262203, 9, 7, 12, 196638, 11, 4, 262176, 12, 12, 11,
        262203, 12, 10, 12, 196638, 14, 4, 262176, 15, 12, 14, 262203, 15, 13, 12, 262176, 18, 1,
        5, 262203, 18, 17, 1, 196641, 21, 2, 262176, 22, 12, 4, 262187, 6, 23, 0, 262176, 29, 12,
        3, 327734, 2, 20, 0, 21, 131320, 16, 262205, 5, 19, 17, 327745, 22, 24, 7, 23, 327745, 22,
        25, 10, 23, 327745, 22, 26, 13, 23, 131321, 27, 131320, 27, 327761, 6, 28, 19, 0, 327745,
        29, 30, 24, 28, 262205, 3, 31, 30, 327745, 29, 32, 25, 28, 262205, 3, 33, 32, 327809, 3,
        34, 31, 33, 327745, 29, 35, 26, 28, 196670, 35, 34, 65789, 65592,
    ];

    let shader = device
        .create_shader_spirv(VECTOR_ADD_SPV)
        .expect("shader creation failed");

    let buffer_desc = BufferDesc {
        label: Some("PipelineTestBuffer"),
        size: 256,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };
    let buf_a = device
        .create_buffer(&buffer_desc)
        .expect("create buffer A failed");
    let buf_b = device
        .create_buffer(&buffer_desc)
        .expect("create buffer B failed");
    let buf_c = device
        .create_buffer(&buffer_desc)
        .expect("create buffer C failed");

    // Create 10 compute pipelines consecutively using the same shared pipeline cache
    for _ in 0..10 {
        let pipeline = device
            .create_compute_pipeline(&shader, 3)
            .expect("create_compute_pipeline failed");

        let mut encoder = device
            .create_command_encoder()
            .expect("create encoder failed");
        encoder.dispatch_compute(&pipeline, &[&buf_a, &buf_b, &buf_c], 1, 1, 1);
        encoder.memory_barrier();
        let cmd = encoder.finish();
        queue.submit(cmd).expect("submit failed");
    }
}

#[test]
fn test_suballocator_block_reclamation_and_trim() {
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

    assert_eq!(device.suballocator_block_count(), 0);

    // Each buffer is 10 MiB (< SUBALLOCATION_THRESHOLD of 32 MiB), so it will be suballocated.
    // In a 64 MiB DEFAULT_BLOCK_SIZE, 12 buffers force at least 3 distinct 64 MiB blocks.
    const CHUNK_SIZE: u64 = 10 * 1024 * 1024;
    let desc = BufferDesc {
        label: Some("LargeSuballocBuffer"),
        size: CHUNK_SIZE,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };

    let mut buffers = Vec::new();
    for i in 0..12 {
        let b = device.create_buffer(&desc).expect("buffer creation failed");
        b.write_at(0, &[(i as u8) + 1]).expect("write failed");
        buffers.push(b);
    }

    let initial_blocks = device.suballocator_block_count();
    assert!(
        initial_blocks >= 3,
        "Expected at least 3 suballocated blocks, got {}",
        initial_blocks
    );

    // Drop all buffers. The suballocator should automatically reclaim redundant empty blocks,
    // retaining at most 1 empty block per memory configuration as a warm pool.
    drop(buffers);

    let blocks_after_drop = device.suballocator_block_count();
    assert_eq!(
        blocks_after_drop, 1,
        "Suballocator should have automatically reclaimed redundant empty blocks down to 1 warm block"
    );

    // Explicitly trim all empty memory blocks back to Vulkan
    let reclaimed_count = device.trim_memory();
    assert_eq!(reclaimed_count, 1);
    assert_eq!(
        device.suballocator_block_count(),
        0,
        "All blocks should be released after trim_memory"
    );

    // Verify suballocator recovers and allocates cleanly after full trim
    let fresh_buf = device
        .create_buffer(&BufferDesc {
            label: Some("FreshBufferAfterTrim"),
            size: 1024,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            memory_location: MemoryLocation::CpuToGpu,
        })
        .expect("Failed to allocate buffer after trim");

    assert_eq!(device.suballocator_block_count(), 1);
    fresh_buf
        .write_at(0, &[0xDE, 0xAD, 0xBE, 0xEF])
        .expect("write failed");
    let mut read_back = [0u8; 4];
    fresh_buf.read_at(0, &mut read_back).expect("read failed");
    assert_eq!(read_back, [0xDE, 0xAD, 0xBE, 0xEF]);
}
