//! Integration tests verifying ergonomic helpers: `default_device` and `Queue::run`.

use lgpu::{BufferDesc, BufferUsage, MemoryLocation};

#[test]
fn test_default_device_and_run() {
    let (device, queue) = match lgpu::default_device() {
        Ok(pair) => pair,
        Err(_) => return, // Headless environment without Vulkan 1.3 physical GPU
    };

    let desc = BufferDesc {
        label: Some("ErgoTestBuffer"),
        size: 64,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };

    let buffer = device
        .create_buffer_init_slice(&desc, &[42u32; 16])
        .expect("create_buffer_init_slice");

    // Test queue.run(encoder)
    let mut encoder = device
        .create_command_encoder()
        .expect("create_command_encoder");
    encoder.copy_buffer_to_buffer(&buffer, 0, &buffer, 32, 16);
    queue.run(encoder).expect("queue.run failed");

    let mut read_back = [0u32; 4];
    buffer
        .read_at(32, bytemuck::cast_slice_mut(&mut read_back))
        .expect("read_at");
    assert_eq!(read_back, [42u32; 4]);

    // Test device.run(&queue, encoder)
    let encoder2 = device
        .create_command_encoder()
        .expect("create_command_encoder");
    device.run(&queue, encoder2).expect("device.run failed");
}
