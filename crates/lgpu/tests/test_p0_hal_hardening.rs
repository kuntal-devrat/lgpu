//! Regression tests for HAL fixes:
//! 1. Depth16Unorm bytes_per_pixel == 2
//! 2. 3D Volume texture write_texture and readback across all depth slices
//! 3. 2D Array & Cube texture write_texture and multi-layer mipmap generation
//! 4. HalDevice trait uniform buffer registration and unregistration
//! 5. Queue family separation, recorder cache routing, and memory trimming

use lgpu::{
    BufferDesc, BufferUsage, MemoryLocation, TextureDesc, TextureDimension, TextureFormat,
    TextureUsage,
};
use lgpu_hal::{HalBackend, HalDevice, HalInstance, VulkanBackend};

macro_rules! require_gpu {
    () => {
        match (|| {
            let instance = lgpu::Instance::new().ok()?;
            let adapters = instance.enumerate_adapters().ok()?;
            if adapters.is_empty() {
                return None;
            }
            let (device, queue) = adapters[0].request_device(&instance).ok()?;
            Some((instance, device, queue))
        })() {
            Some(tuple) => tuple,
            None => {
                eprintln!("Skipping test: Vulkan 1.3 physical device not available");
                return;
            }
        }
    };
}

#[test]
fn test_depth16_unorm_bytes_per_pixel() {
    assert_eq!(
        TextureFormat::Depth16Unorm.bytes_per_pixel(),
        2,
        "Depth16Unorm is a 16-bit depth format and must be exactly 2 bytes per pixel"
    );
    assert_eq!(TextureFormat::Depth32Float.bytes_per_pixel(), 4);
    assert_eq!(TextureFormat::Rgba8Unorm.bytes_per_pixel(), 4);
    assert_eq!(TextureFormat::Rgba16Float.bytes_per_pixel(), 8);
    assert_eq!(TextureFormat::Rgba32Float.bytes_per_pixel(), 16);
}

#[test]
fn test_volume_texture_queue_write_texture() {
    let (_instance, device, queue) = require_gpu!();

    const W: u32 = 4;
    const H: u32 = 4;
    const D: u32 = 3;
    let texture = match device.create_texture(&TextureDesc {
        label: Some("test_volume"),
        width: W,
        height: H,
        depth: D,
        array_layers: 1,
        dimension: TextureDimension::D3,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
    }) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("3D texture creation not supported on this adapter: {e}");
            return;
        }
    };

    let mut texels = Vec::with_capacity((W * H * D * 4) as usize);
    for slice in 0..D {
        for _ in 0..(W * H) {
            texels.extend_from_slice(&[slice as u8 * 50 + 10, 42, 99, 255]);
        }
    }

    // Exercises write_texture for 3D textures (verifying depth > 1 is respected in staging and copy extents)
    queue
        .write_texture(&texture, &texels)
        .expect("queue.write_texture must succeed for 3D volume texture");

    let total_bytes = (W * H * D * 4) as u64;
    let readback_buf = device
        .create_buffer(&BufferDesc {
            label: Some("VolumeReadback"),
            size: total_bytes,
            usage: BufferUsage::TRANSFER_DST,
            memory_location: MemoryLocation::GpuToCpu,
        })
        .expect("create readback buffer");

    let mut encoder = device.create_command_encoder().expect("create encoder");
    encoder.copy_texture_to_buffer(&texture, &readback_buf);
    queue
        .submit(encoder.finish())
        .expect("submit readback command");

    let mut out_data = vec![0u8; total_bytes as usize];
    readback_buf
        .read(&mut out_data)
        .expect("read back 3D volume texture");

    let slice_bytes = (W * H * 4) as usize;
    for slice in 0..D as usize {
        let offset = slice * slice_bytes;
        let expected_val = slice as u8 * 50 + 10;
        assert_eq!(
            out_data[offset], expected_val,
            "Slice {slice} texel byte 0 mismatch in 3D volume texture"
        );
        assert_eq!(out_data[offset + 1], 42);
        assert_eq!(out_data[offset + 2], 99);
        assert_eq!(out_data[offset + 3], 255);
    }
}

#[test]
fn test_array_and_cube_write_texture() {
    let (_instance, device, queue) = require_gpu!();

    const W: u32 = 4;
    const H: u32 = 4;
    const LAYERS: u32 = 4;
    let array_tex = device
        .create_texture(&TextureDesc {
            label: Some("test_array"),
            width: W,
            height: H,
            depth: 1,
            array_layers: LAYERS,
            dimension: TextureDimension::D2Array,
            mip_levels: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        })
        .expect("create 2D array texture");

    let mut array_texels = Vec::with_capacity((W * H * LAYERS * 4) as usize);
    for layer in 0..LAYERS {
        for _ in 0..(W * H) {
            array_texels.extend_from_slice(&[layer as u8 * 30 + 5, 88, 120, 255]);
        }
    }

    queue
        .write_texture(&array_tex, &array_texels)
        .expect("queue.write_texture must succeed for 2D array texture");

    // Test Cube texture with mipmap generation
    let cube_tex = device
        .create_texture(&TextureDesc {
            label: Some("test_cube"),
            width: 8,
            height: 8,
            depth: 1,
            array_layers: 6,
            dimension: TextureDimension::Cube,
            mip_levels: 3,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        })
        .expect("create cube texture");

    let cube_texels = vec![170u8; (8 * 8 * 6 * 4) as usize];
    device
        .write_texture_mip(&cube_tex, 0, &cube_texels, None)
        .expect("write cube level 0");

    // Exercises generate_mipmaps with the new level-by-level RAW barrier synchronization
    device
        .generate_mipmaps(&cube_tex)
        .expect("generate_mipmaps for cube map must execute without validation errors");
}

#[test]
fn test_hal_device_uniform_buffer_registration() {
    let instance = match <VulkanBackend as HalBackend>::Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let adapters = match HalInstance::enumerate_adapters(&instance) {
        Ok(a) if !a.is_empty() => a,
        _ => return,
    };
    let hal_device = match HalInstance::create_device(&instance, &adapters[0]) {
        Ok(d) => d,
        Err(_) => return,
    };

    let ubuf = HalDevice::create_buffer(
        &hal_device,
        &lgpu_hal::BufferDesc {
            label: Some("hal_uniform_buf"),
            size: 256,
            usage: BufferUsage::UNIFORM | BufferUsage::TRANSFER_DST,
            memory_location: MemoryLocation::CpuToGpu,
        },
    )
    .expect("create uniform buffer on HalDevice");

    if let Ok(slot) = HalDevice::register_bindless_uniform_buffer(&hal_device, &ubuf) {
        // If bindless heap is available, unregister must succeed without panic
        HalDevice::unregister_bindless_uniform_buffer(&hal_device, slot);
    }
    HalDevice::destroy_buffer(&hal_device, ubuf);
}

#[test]
fn test_cross_queue_command_recorder_separation_and_trim() {
    let (_instance, device, queue) = require_gpu!();

    // If an async compute queue is available, exercise both queues
    if let Some(compute_queue) = device.compute_queue() {
        let enc = device
            .create_command_encoder()
            .expect("create graphics encoder");
        queue.submit(enc.finish()).expect("submit graphics queue");

        let comp_enc = device
            .create_command_encoder()
            .expect("create compute encoder");
        compute_queue
            .submit(comp_enc.finish())
            .expect("submit compute queue");
    }

    // Wait idle and trim memory - exercises the separate command pool harvesting and compute cache destruction
    device.wait_idle().expect("device.wait_idle must succeed");
    let _freed = device.trim_memory();
}
