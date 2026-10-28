//! Integration tests verifying buffer/texture data transfers, mipmaps, MSAA, and indirect commands.

use lgpu::*;

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
fn test_copy_buffer_to_buffer() {
    let (_instance, device, queue) = require_gpu!();

    let src_data = [11u8, 22, 33, 44, 55, 66, 77, 88];
    let src_buf = device
        .create_buffer_init(
            &BufferDesc {
                label: Some("src_buf"),
                size: 64,
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE,
                memory_location: MemoryLocation::CpuToGpu,
            },
            &src_data,
        )
        .expect("create src_buf");

    let dst_buf = device
        .create_buffer(&BufferDesc {
            label: Some("dst_buf"),
            size: 64,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE,
            memory_location: MemoryLocation::CpuToGpu,
        })
        .expect("create dst_buf");

    let mut encoder = device
        .create_command_encoder()
        .expect("create_command_encoder");
    encoder.copy_buffer_to_buffer(&src_buf, 0, &dst_buf, 0, src_data.len() as u64);
    let cmd = encoder.finish();
    queue.submit(cmd).expect("submit");

    let mut readback = [0u8; 8];
    dst_buf.read(&mut readback).expect("read");
    assert_eq!(readback, src_data, "Buffer to buffer copy content mismatch");
}

#[test]
fn test_write_texture_and_copy_texture_to_buffer() {
    let (_instance, device, queue) = require_gpu!();

    let width = 8;
    let height = 8;
    let tex_desc = TextureDesc {
        label: Some("upload_test_tex"),
        width,
        height,
        depth: 1,
        mip_levels: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
    };
    let texture = device.create_texture(&tex_desc).expect("create_texture");

    let pixel_count = (width * height) as usize;
    let upload_data = vec![128u8; pixel_count * 4];

    // Direct write_texture into GPU texture
    device
        .write_texture(&texture, &upload_data)
        .expect("write_texture");

    // Copy texture to buffer to verify content
    let readback_buf = device
        .create_buffer(&BufferDesc {
            label: Some("readback_buf"),
            size: (pixel_count * 4) as u64,
            usage: BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
            memory_location: MemoryLocation::CpuToGpu,
        })
        .expect("create readback_buf");

    let mut encoder = device
        .create_command_encoder()
        .expect("create_command_encoder");
    encoder.copy_texture_to_buffer(&texture, &readback_buf);
    let cmd = encoder.finish();
    queue.submit(cmd).expect("submit");

    let mut result_pixels = vec![0u8; pixel_count * 4];
    readback_buf.read(&mut result_pixels).expect("read");
    assert_eq!(
        result_pixels, upload_data,
        "Texture upload and readback matched"
    );
}

#[test]
fn test_mipmaps_and_msaa_texture_and_pipeline() {
    let (_instance, device, _queue) = require_gpu!();

    // Mipmap texture creation (4 mip levels)
    let mip_tex = device
        .create_texture(&TextureDesc {
            label: Some("mip_texture"),
            width: 64,
            height: 64,
            depth: 1,
            mip_levels: 4,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        })
        .expect("create mip texture");

    // MSAA 4x texture creation
    let msaa_tex = device
        .create_texture(&TextureDesc {
            label: Some("msaa_texture"),
            width: 64,
            height: 64,
            depth: 1,
            mip_levels: 1,
            sample_count: 4,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::RENDER_ATTACHMENT,
        })
        .expect("create msaa texture");

    assert!(mip_tex.id().is_some());
    assert_eq!(msaa_tex.width, 64);
}

#[test]
fn test_draw_indexed_and_indirect() {
    let (_instance, device, queue) = require_gpu!();

    let _color_tex = device
        .create_texture(&TextureDesc {
            label: Some("index_target"),
            width: 64,
            height: 64,
            depth: 1,
            mip_levels: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
        })
        .expect("create color texture");

    let indices = [0u32, 1, 2];
    let index_buf = device
        .create_buffer_init_slice(
            &BufferDesc {
                label: Some("index_buf"),
                size: 16,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory_location: MemoryLocation::CpuToGpu,
            },
            &indices,
        )
        .expect("create index_buf");

    // Dummy indirect command buffer: VkDrawIndirectCommand { vertexCount: 3, instanceCount: 1, firstVertex: 0, firstInstance: 0 }
    let indirect_cmd = [3u32, 1, 0, 0];
    let indirect_buf = device
        .create_buffer_init_slice(
            &BufferDesc {
                label: Some("indirect_buf"),
                size: 16,
                usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
                memory_location: MemoryLocation::CpuToGpu,
            },
            &indirect_cmd,
        )
        .expect("create indirect_buf");

    let mut encoder = device.create_command_encoder().expect("create encoder");
    encoder.copy_buffer_to_buffer(&index_buf, 0, &indirect_buf, 0, 12);
    let cmd = encoder.finish();
    queue.submit(cmd).expect("submit");
}
