//! Integration tests for mip uploads, mipmap generation, device local uploads and timestamps.

use lgpu::{
    BufferDesc, BufferUsage, MemoryLocation, TextureDesc, TextureDimension, TextureFormat,
    TextureUsage,
};

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
fn test_mip_upload_with_row_pitch_and_mipmap_generation() {
    let (_instance, device, queue) = require_gpu!();

    const SIZE: u32 = 16;
    const MIPS: u32 = 5;

    let texture = device
        .create_texture(&TextureDesc {
            label: Some("MipChain"),
            width: SIZE,
            height: SIZE,
            depth: 1,
            dimension: TextureDimension::D2,
            array_layers: 1,
            mip_levels: MIPS,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        })
        .expect("create mipped texture");

    // Upload mip 0 with 8 bytes of row padding to exercise the row pitch path.
    let packed_row = (SIZE * 4) as usize;
    let padded_row = packed_row + 8;
    let mut padded = vec![0u8; padded_row * SIZE as usize];
    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let base = y * padded_row + x * 4;
            padded[base] = (x * 16) as u8;
            padded[base + 1] = (y * 16) as u8;
            padded[base + 2] = 64;
            padded[base + 3] = 255;
        }
    }

    device
        .write_texture_mip(&texture, 0, &padded, Some(padded_row as u32))
        .expect("upload mip 0 with row pitch");

    device
        .generate_mipmaps(&texture)
        .expect("generate mip chain");

    // Copy mip 0 back and verify the upload survived both operations.
    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("MipReadback"),
            size: (SIZE * SIZE * 4) as u64,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            memory_location: MemoryLocation::GpuToCpu,
        })
        .expect("create readback buffer");

    let mut encoder = device.create_command_encoder().expect("create encoder");
    encoder.copy_texture_to_buffer(&texture, &readback);
    queue.submit(encoder.finish()).expect("submit copy");

    let mut pixels = vec![0u8; (SIZE * SIZE * 4) as usize];
    readback.read(&mut pixels).expect("read back mip 0");

    for y in 0..SIZE as usize {
        for x in 0..SIZE as usize {
            let base = (y * SIZE as usize + x) * 4;
            assert_eq!(
                &pixels[base..base + 4],
                &[(x * 16) as u8, (y * 16) as u8, 64, 255],
                "mip 0 pixel ({x}, {y}) should match the padded upload"
            );
        }
    }
}

#[test]
fn test_device_local_upload_round_trip() {
    let (_instance, device, queue) = require_gpu!();

    let source = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    let local = device
        .create_buffer_init_device_local(
            &BufferDesc {
                label: Some("DeviceLocalBuffer"),
                size: source.len() as u64,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC,
                memory_location: MemoryLocation::GpuOnly,
            },
            &source,
        )
        .expect("upload device local buffer");

    let readback = device
        .create_buffer(&BufferDesc {
            label: Some("DeviceLocalReadback"),
            size: source.len() as u64,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            memory_location: MemoryLocation::GpuToCpu,
        })
        .expect("create readback buffer");

    let mut encoder = device.create_command_encoder().expect("create encoder");
    encoder.copy_buffer_to_buffer(&local, 0, &readback, 0, source.len() as u64);
    queue.submit(encoder.finish()).expect("submit copy");

    let mut result = [0u8; 16];
    readback
        .read(&mut result)
        .expect("read device local buffer");
    assert_eq!(result, source, "device local upload must round trip");
}

#[test]
fn test_timestamp_queries() {
    let (_instance, device, queue) = require_gpu!();

    if device.timestamp_valid_bits() == 0 {
        eprintln!("Skipping test: the selected queue family does not support timestamps");
        return;
    }

    let pool = device
        .create_timestamp_pool(2)
        .expect("create timestamp pool");

    let mut encoder = device.create_command_encoder().expect("create encoder");
    encoder.reset_timestamp_pool(&pool);
    encoder
        .write_timestamp(&pool, 0)
        .expect("write first timestamp");
    encoder
        .write_timestamp(&pool, 1)
        .expect("write second timestamp");
    queue.submit(encoder.finish()).expect("submit timestamps");

    let values = pool.read().expect("read timestamps");
    assert_eq!(values.len(), 2);
    assert!(
        values[0] > 0,
        "the first timestamp should be a valid GPU tick, got {values:?}"
    );
    assert!(
        values[1] >= values[0],
        "timestamps must be monotonic, got {values:?}"
    );

    // Reusing the pool requires a reset, which must be recorded before the new writes.
    let mut encoder = device.create_command_encoder().expect("create encoder");
    encoder.reset_timestamp_pool(&pool);
    encoder
        .write_timestamp(&pool, 0)
        .expect("write reused timestamp");
    encoder
        .write_timestamp(&pool, 1)
        .expect("write reused second timestamp");
    queue
        .submit(encoder.finish())
        .expect("submit reused timestamps");
    let reused = pool.read().expect("read reused timestamps");
    assert!(reused[0] > 0, "reused timestamp must be written");
}

#[test]
fn test_timestamp_index_validation() {
    let (_instance, device, _queue) = require_gpu!();

    if device.timestamp_valid_bits() == 0 {
        eprintln!("Skipping test: the selected queue family does not support timestamps");
        return;
    }

    let pool = device.create_timestamp_pool(1).expect("create pool");
    let mut encoder = device.create_command_encoder().expect("create encoder");
    assert!(
        encoder.write_timestamp(&pool, 1).is_err(),
        "writing past the end of the pool must fail"
    );
}
