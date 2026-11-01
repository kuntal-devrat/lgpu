//! Integration coverage for 1D/3D/array/cube textures and the extended depth formats.

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

fn readback(
    device: &lgpu::Device,
    queue: &lgpu::Queue,
    texture: &lgpu::Texture,
    bytes: u64,
) -> Vec<u8> {
    let buffer = device
        .create_buffer(&BufferDesc {
            label: Some("TextureReadback"),
            size: bytes,
            usage: BufferUsage::TRANSFER_DST,
            memory_location: MemoryLocation::GpuToCpu,
        })
        .expect("create readback buffer");
    let mut encoder = device.create_command_encoder().expect("create encoder");
    encoder.copy_texture_to_buffer(texture, &buffer);
    queue.submit(encoder.finish()).expect("submit readback");

    let mut data = vec![0u8; bytes as usize];
    buffer.read(&mut data).expect("read back texture");
    data
}

#[test]
fn test_volume_texture_upload_and_readback() {
    let (_instance, device, queue) = require_gpu!();

    const W: u32 = 2;
    const H: u32 = 2;
    const D: u32 = 2;
    let texture = device
        .create_texture(&TextureDesc {
            label: Some("Volume"),
            width: W,
            height: H,
            depth: D,
            array_layers: 1,
            dimension: TextureDimension::D3,
            mip_levels: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        })
        .expect("create 3D texture");

    let mut texels = Vec::new();
    for slice in 0..D {
        for _ in 0..(W * H) {
            texels.extend_from_slice(&[slice as u8 * 10 + 1, 2, 3, 255]);
        }
    }
    device
        .write_texture_mip(&texture, 0, &texels, None)
        .expect("upload every slice of the volume");

    let data = readback(&device, &queue, &texture, (W * H * D * 4) as u64);
    assert_eq!(data.len(), texels.len(), "readback must cover every slice");
    assert_eq!(&data[0..4], &[1, 2, 3, 255], "slice 0 must round-trip");
    let slice_one = ((W * H) * 4) as usize;
    assert_eq!(
        &data[slice_one..slice_one + 4],
        &[11, 2, 3, 255],
        "slice 1 must round-trip"
    );
}

#[test]
fn test_array_texture_layers_and_cube_mips() {
    let (_instance, device, _queue) = require_gpu!();

    const LAYERS: u32 = 3;
    let array = device
        .create_texture(&TextureDesc {
            label: Some("Array"),
            width: 2,
            height: 2,
            depth: 1,
            array_layers: LAYERS,
            dimension: TextureDimension::D2Array,
            mip_levels: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        })
        .expect("create 2D array texture");

    let texels = vec![128u8; (2 * 2 * 4 * LAYERS) as usize];
    device
        .write_texture_mip(&array, 0, &texels, None)
        .expect("upload all array layers");

    let cube = device
        .create_texture(&TextureDesc {
            label: Some("Cube"),
            width: 8,
            height: 8,
            depth: 1,
            array_layers: 6,
            dimension: TextureDimension::Cube,
            mip_levels: 4,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
        })
        .expect("create cube map");

    device
        .write_texture_mip(&cube, 0, &vec![200u8; 8 * 8 * 4 * 6], None)
        .expect("upload every cube face");
    device
        .generate_mipmaps(&cube)
        .expect("generate cube mip chain for all faces");
}

#[test]
fn test_dimension_and_format_validation() {
    let (_instance, device, _queue) = require_gpu!();

    // A cube map must be square.
    assert!(
        device
            .create_texture(&TextureDesc {
                label: Some("BadCube"),
                width: 8,
                height: 4,
                depth: 1,
                array_layers: 6,
                dimension: TextureDimension::Cube,
                mip_levels: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::TEXTURE_BINDING,
            })
            .is_err(),
        "non-square cube maps must be rejected"
    );

    // Cube maps need a multiple of six layers.
    assert!(
        device
            .create_texture(&TextureDesc {
                label: Some("BadCubeLayers"),
                width: 8,
                height: 8,
                depth: 1,
                array_layers: 4,
                dimension: TextureDimension::Cube,
                mip_levels: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::TEXTURE_BINDING,
            })
            .is_err(),
        "cube maps with layer counts that are not a multiple of six must be rejected"
    );

    // 1D textures have a single texel in the other axes.
    assert!(
        device
            .create_texture(&TextureDesc {
                label: Some("Bad1D"),
                width: 16,
                height: 2,
                depth: 1,
                array_layers: 1,
                dimension: TextureDimension::D1,
                mip_levels: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsage::TEXTURE_BINDING,
            })
            .is_err(),
        "1D textures with height > 1 must be rejected"
    );

    // Depth/stencil formats are usable as attachments.
    for format in [
        TextureFormat::Depth16Unorm,
        TextureFormat::Depth24Stencil8,
        TextureFormat::Depth32FloatStencil8,
    ] {
        let depth = device
            .create_texture(&TextureDesc {
                label: Some("DepthStencil"),
                width: 16,
                height: 16,
                depth: 1,
                array_layers: 1,
                dimension: TextureDimension::D2,
                mip_levels: 1,
                sample_count: 1,
                format,
                usage: TextureUsage::RENDER_ATTACHMENT,
            })
            .unwrap_or_else(|e| panic!("{format:?} depth attachment must be creatable: {e}"));
        assert_eq!(depth.format, format);
        assert!(
            format.is_depth() && format.has_stencil() == (format != TextureFormat::Depth16Unorm)
        );
    }
}
