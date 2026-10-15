//! Integration tests verifying resource initialization and device capabilities.

use lgpu::{BufferDesc, BufferUsage, Instance, MemoryLocation, TextureFormat};

#[test]
fn test_buffer_initialization_slice_and_bytes() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance
        .enumerate_adapters()
        .expect("Failed to enumerate adapters");
    let (device, _queue) = adapters[0]
        .request_device(&instance)
        .expect("Failed to request device");

    let desc = BufferDesc {
        label: Some("InitTestBuffer"),
        size: 64,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    };

    // 1. Test create_buffer_init_slice directly with typed float slice &[f32]
    let float_data = [1.0f32, 2.0, 3.0, 4.0];
    let buf_slice = device
        .create_buffer_init_slice(&desc, &float_data)
        .expect("Failed to create buffer with typed slice");
    let mut readback_floats = [0.0f32; 4];
    buf_slice
        .read(bytemuck::cast_slice_mut(&mut readback_floats))
        .expect("Failed to readback buffer initialized via slice");
    assert_eq!(readback_floats, float_data);

    // 2. Test create_buffer_init with bytemuck::cast_slice
    let buf_bytes = device
        .create_buffer_init(&desc, bytemuck::cast_slice(&float_data))
        .expect("Failed to create buffer with byte slice");
    let mut readback_bytes = [0.0f32; 4];
    buf_bytes
        .read(bytemuck::cast_slice_mut(&mut readback_bytes))
        .expect("Failed to readback buffer initialized via bytes");
    assert_eq!(readback_bytes, float_data);
}

#[test]
fn test_graphics_pipeline_lifecycle() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance
        .enumerate_adapters()
        .expect("Failed to enumerate adapters");
    let (device, _queue) = adapters[0]
        .request_device(&instance)
        .expect("Failed to request device");

    const TRIANGLE_SPV: &[u32] = &[
        119734787, 65536, 28, 67, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
        196622, 0, 1, 524303, 0, 16, 1834972022, 7235937, 9, 12, 14, 524303, 4, 63, 1834972006,
        7235937, 56, 59, 62, 196624, 63, 7, 327752, 6, 0, 35, 0, 327752, 6, 1, 35, 16, 262215, 9,
        11, 42, 262215, 12, 11, 0, 262215, 14, 30, 0, 262215, 56, 11, 15, 262215, 59, 30, 0,
        262215, 62, 30, 0, 131091, 2, 196630, 4, 32, 262167, 3, 4, 4, 262167, 5, 4, 3, 262174, 6,
        3, 5, 262165, 7, 32, 0, 262176, 10, 1, 7, 262203, 10, 9, 1, 262176, 13, 3, 3, 262203, 13,
        12, 3, 262176, 15, 3, 5, 262203, 15, 14, 3, 196641, 17, 2, 262187, 7, 18, 0, 262187, 4, 19,
        0, 262187, 4, 20, 3204448256, 262187, 4, 21, 1065353216, 458796, 3, 22, 19, 20, 19, 21,
        393260, 5, 23, 21, 19, 19, 262187, 7, 24, 1, 262187, 4, 25, 1056964608, 458796, 3, 26, 25,
        25, 19, 21, 393260, 5, 27, 19, 21, 19, 458796, 3, 28, 20, 25, 19, 21, 393260, 5, 29, 19,
        19, 21, 262176, 31, 7, 6, 196654, 6, 32, 131092, 34, 262176, 39, 7, 3, 262176, 41, 7, 5,
        262176, 57, 1, 3, 262203, 57, 56, 1, 262176, 60, 1, 5, 262203, 60, 59, 1, 262203, 13, 62,
        3, 327734, 2, 16, 0, 17, 131320, 8, 327739, 31, 30, 7, 32, 262205, 7, 11, 9, 131321, 33,
        131320, 33, 327850, 34, 35, 11, 18, 196855, 36, 0, 262394, 35, 37, 38, 131320, 37, 327745,
        39, 40, 30, 18, 196670, 40, 22, 327745, 41, 42, 30, 24, 196670, 42, 23, 131321, 36, 131320,
        38, 327850, 34, 43, 11, 24, 196855, 44, 0, 262394, 43, 45, 46, 131320, 45, 327745, 39, 47,
        30, 18, 196670, 47, 26, 327745, 41, 48, 30, 24, 196670, 48, 27, 131321, 44, 131320, 46,
        327745, 39, 49, 30, 18, 196670, 49, 28, 327745, 41, 50, 30, 24, 196670, 50, 29, 131321, 44,
        131320, 44, 131321, 36, 131320, 36, 262205, 6, 51, 30, 327761, 3, 52, 51, 0, 196670, 12,
        52, 327761, 5, 53, 51, 1, 196670, 14, 53, 65789, 65592, 327734, 2, 63, 0, 17, 131320, 54,
        262205, 3, 58, 56, 262205, 5, 61, 59, 327760, 6, 55, 58, 61, 131321, 64, 131320, 64,
        327761, 5, 65, 55, 1, 327760, 3, 66, 65, 21, 196670, 62, 66, 65789, 65592,
    ];
    let shader = device
        .create_shader_spirv(TRIANGLE_SPV)
        .expect("Failed to create shader");

    // Create graphics pipeline
    let pipeline = device
        .create_graphics_pipeline(&shader, "vs_main", "fs_main", TextureFormat::Rgba8Unorm)
        .expect("Failed to create graphics pipeline");

    // Drop cleanly without double-free or dangling references
    drop(pipeline);
}

#[test]
fn test_adapter_feature_detection_and_device_creation() {
    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance
        .enumerate_adapters()
        .expect("Failed to enumerate adapters");
    assert!(!adapters.is_empty());

    for adapter in &adapters {
        let info = adapter.info();
        println!("Checking adapter: {}", info.name);
        println!(
            "  supports_dynamic_rendering: {}",
            info.supports_dynamic_rendering
        );
        println!(
            "  supports_timeline_semaphores: {}",
            info.supports_timeline_semaphores
        );
        println!(
            "  supports_descriptor_indexing: {}",
            info.supports_descriptor_indexing
        );

        // Modern Vulkan 1.3 systems must support these features
        assert!(
            info.supports_dynamic_rendering,
            "Vulkan 1.3 device should support dynamic rendering"
        );
        assert!(
            info.supports_timeline_semaphores,
            "Vulkan 1.3 device should support timeline semaphores"
        );
        assert!(
            info.supports_descriptor_indexing,
            "Vulkan 1.3 device should support descriptor indexing"
        );

        // Request device should succeed because all required features are supported
        let res = adapter.request_device(&instance);
        assert!(
            res.is_ok(),
            "Request device should succeed on capable adapter"
        );
    }
}
