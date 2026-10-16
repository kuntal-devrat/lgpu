//! Integration tests verifying bindless storage image allocation, MRT draw operations, and device limits.

use lgpu::{
    BufferDesc, BufferUsage, Instance, MemoryLocation, TextureDesc, TextureFormat, TextureUsage,
};

#[test]
fn test_bindless_storage_image_registration_and_slot_recycling() {
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

    // 1. Create a texture with STORAGE_BINDING
    let desc = TextureDesc {
        label: Some("StorageImageTest"),
        width: 128,
        height: 128,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::STORAGE_BINDING | TextureUsage::TEXTURE_BINDING,
    };

    let mut textures = Vec::new();
    for i in 0..10 {
        let tex = device
            .create_texture(&desc)
            .expect("Failed to create storage texture");
        let storage_id = tex.storage_id().expect("Texture must have storage_id");
        let sampled_id = tex.id().expect("Texture must have sampled texture id");
        assert_eq!(storage_id.0, i as u32);
        assert_eq!(sampled_id.0, i as u32);
        textures.push(tex);
    }

    // 2. Drop slots 3 and 7
    let dropped_storage_3 = textures.remove(3).storage_id().unwrap();
    let dropped_storage_7 = textures.remove(6).storage_id().unwrap(); // after removing 3, index 6 was slot 7
    assert_eq!(dropped_storage_3.0, 3);
    assert_eq!(dropped_storage_7.0, 7);

    // 3. Allocate 2 new textures and verify slot recycling
    let recycled_1 = device
        .create_texture(&desc)
        .expect("Failed to create recycled texture 1");
    let recycled_2 = device
        .create_texture(&desc)
        .expect("Failed to create recycled texture 2");

    let rec_id_1 = recycled_1.storage_id().unwrap();
    let rec_id_2 = recycled_2.storage_id().unwrap();

    assert!(rec_id_1.0 == 3 || rec_id_1.0 == 7);
    assert!(rec_id_2.0 == 3 || rec_id_2.0 == 7);
    assert_ne!(rec_id_1, rec_id_2);
}

#[test]
fn test_batched_descriptor_set_writes() {
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

    #[cfg(feature = "vulkan")]
    {
        let desc = BufferDesc {
            label: Some("BatchBuf"),
            size: 256,
            usage: BufferUsage::STORAGE,
            memory_location: MemoryLocation::CpuToGpu,
        };

        let mut bufs = Vec::new();
        for _ in 0..8 {
            bufs.push(device.create_buffer(&desc).expect("create_buffer failed"));
        }

        let writes: Vec<(u32, &lgpu::Buffer)> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b))
            .collect();

        // Perform batch descriptor set update
        let res = device.write_buffer_descriptors_batch(&writes);
        assert!(res.is_ok(), "Batch descriptor set write failed");
    }
    let _ = device;
}

#[test]
fn test_adapter_device_limits_query() {
    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match instance.enumerate_adapters() {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };

    let limits = &adapters[0].info().limits;
    assert!(limits.max_buffer_size > 0, "max_buffer_size must be > 0");
    assert!(
        limits.max_texture_dimension_2d >= 1024,
        "max_texture_dimension_2d must be >= 1024"
    );
    assert!(
        limits.max_compute_workgroup_invocations >= 64,
        "compute invocations must be >= 64"
    );
}

#[test]
fn test_multiple_render_targets_mrt() {
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

    let pipeline = device
        .create_graphics_pipeline(&shader, "vs_main", "fs_main", TextureFormat::Rgba8Unorm)
        .expect("Failed to create graphics pipeline");

    let desc = TextureDesc {
        label: Some("MrtColor"),
        width: 64,
        height: 64,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
    };

    let target_albedo = device
        .create_texture(&desc)
        .expect("Albedo tex creation failed");
    let target_normal = device
        .create_texture(&desc)
        .expect("Normal tex creation failed");

    let mut encoder = device
        .create_command_encoder()
        .expect("create encoder failed");
    encoder
        .draw_mrt(
            &[&target_albedo, &target_normal],
            None,
            &pipeline,
            3,
            1,
            0,
            0,
            &[],
            [0.1, 0.2, 0.3, 1.0],
            None,
        )
        .expect("draw_mrt failed");

    let cmd = encoder.finish();
    queue.submit(cmd).expect("queue submit failed");
    device.wait_idle().expect("device wait_idle failed");
}
