//! Integration tests for cross-platform backends (Metal, D3D12) and native Surface configuration.

#![allow(unused_imports)]
use lgpu_hal::{
    AddressMode, Backend, BufferDesc, BufferUsage, FilterMode, MemoryLocation, SamplerDesc,
    TextureDesc, TextureFormat, TextureUsage,
};

#[test]
fn test_metal_backend_complete_bindless_and_storage() {
    #[cfg(feature = "metal")]
    {
        use lgpu_hal::metal::{MetalAdapter, MetalDevice};

        let adapter = MetalAdapter::new();
        assert_eq!(adapter.info().backend, Backend::Metal);

        let device = MetalDevice::new();

        // 1. Test Buffer Allocation, Bindless Slot, and Read/Write Storage
        let buf_desc = BufferDesc {
            label: Some("MetalBuffer1"),
            size: 256,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            memory_location: MemoryLocation::CpuToGpu,
        };
        let buf1 = device.create_buffer(&buf_desc).expect("Create buffer failed");
        let slot1 = device.register_bindless_buffer(&buf1).expect("Register slot 1");
        assert_eq!(slot1, 0);

        let data = vec![42u8; 128];
        device.write_buffer(&buf1, 10, &data).expect("write_buffer failed");
        let mut readback = vec![0u8; 128];
        device.read_buffer(&buf1, 10, &mut readback).expect("read_buffer failed");
        assert_eq!(readback, data);

        // 2. Test Second Buffer Slot Increment
        let buf_desc2 = BufferDesc {
            label: Some("MetalBuffer2"),
            size: 64,
            usage: BufferUsage::STORAGE,
            memory_location: MemoryLocation::GpuOnly,
        };
        let buf2 = device.create_buffer(&buf_desc2).expect("Create buffer 2 failed");
        let slot2 = device.register_bindless_buffer(&buf2).expect("Register slot 2");
        assert_eq!(slot2, 1);

        // 3. Test Recycling of Slot
        device.unregister_bindless_buffer(slot1);
        let buf3 = device.create_buffer(&buf_desc2).expect("Create buffer 3 failed");
        let slot3 = device.register_bindless_buffer(&buf3).expect("Register slot 3");
        assert_eq!(slot3, slot1, "Recycled LIFO slot should match unallocated slot");

        // 4. Test Texture Allocation and Bindless Slot
        let tex_desc = TextureDesc {
            label: Some("MetalTex"),
            width: 512,
            height: 512,
            depth: 1,
            mip_levels: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
        };
        let tex = device.create_texture(&tex_desc).expect("Create texture failed");
        let tex_slot = device.register_bindless_texture(&tex).expect("Register texture failed");
        assert_eq!(tex_slot, 0);
        device.unregister_bindless_texture(tex_slot);

        // 5. Test Sampler Allocation and Bindless Slot
        let sampler_desc = SamplerDesc {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
        };
        let sampler = device.create_sampler(&sampler_desc).expect("Create sampler failed");
        let sampler_slot = device.register_bindless_sampler(&sampler).expect("Register sampler failed");
        assert_eq!(sampler_slot, 0);
        device.unregister_bindless_sampler(sampler_slot);
    }
}

#[test]
fn test_d3d12_backend_complete_bindless_and_storage() {
    #[cfg(feature = "d3d12")]
    {
        use lgpu_hal::d3d12::{D3D12Adapter, D3D12Device};

        let adapter = D3D12Adapter::new();
        assert_eq!(adapter.info().backend, Backend::DirectX12);

        let device = D3D12Device::new();

        // 1. Test Buffer Allocation, Bindless Slot, and Read/Write Storage
        let buf_desc = BufferDesc {
            label: Some("D3D12Buffer1"),
            size: 512,
            usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
            memory_location: MemoryLocation::CpuToGpu,
        };
        let buf1 = device.create_buffer(&buf_desc).expect("Create buffer failed");
        let slot1 = device.register_bindless_buffer(&buf1).expect("Register slot 1");
        assert_eq!(slot1, 0);

        let data = vec![99u8; 256];
        device.write_buffer(&buf1, 32, &data).expect("write_buffer failed");
        let mut readback = vec![0u8; 256];
        device.read_buffer(&buf1, 32, &mut readback).expect("read_buffer failed");
        assert_eq!(readback, data);

        // 2. Test Second Buffer Slot Increment
        let buf_desc2 = BufferDesc {
            label: Some("D3D12Buffer2"),
            size: 128,
            usage: BufferUsage::STORAGE,
            memory_location: MemoryLocation::GpuOnly,
        };
        let buf2 = device.create_buffer(&buf_desc2).expect("Create buffer 2 failed");
        let slot2 = device.register_bindless_buffer(&buf2).expect("Register slot 2");
        assert_eq!(slot2, 1);

        // 3. Test Recycling of Slot
        device.unregister_bindless_buffer(slot1);
        let buf3 = device.create_buffer(&buf_desc2).expect("Create buffer 3 failed");
        let slot3 = device.register_bindless_buffer(&buf3).expect("Register slot 3");
        assert_eq!(slot3, slot1, "Recycled LIFO slot should match unallocated slot");

        // 4. Test Texture Allocation and Bindless Slot
        let tex_desc = TextureDesc {
            label: Some("D3D12Tex"),
            width: 1024,
            height: 1024,
            depth: 1,
            mip_levels: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
        };
        let tex = device.create_texture(&tex_desc).expect("Create texture failed");
        let tex_slot = device.register_bindless_texture(&tex).expect("Register texture failed");
        assert_eq!(tex_slot, 0);
        device.unregister_bindless_texture(tex_slot);

        // 5. Test Sampler Allocation and Bindless Slot
        let sampler_desc = SamplerDesc {
            label: None,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
        };
        let sampler = device.create_sampler(&sampler_desc).expect("Create sampler failed");
        let sampler_slot = device.register_bindless_sampler(&sampler).expect("Register sampler failed");
        assert_eq!(sampler_slot, 0);
        device.unregister_bindless_sampler(sampler_slot);
    }
}

#[test]
fn test_surface_configuration_and_types() {
    use lgpu::{PresentMode, SurfaceConfiguration, TextureFormat, TextureUsage};

    let config = SurfaceConfiguration {
        usage: TextureUsage::RENDER_ATTACHMENT,
        format: TextureFormat::Bgra8Unorm,
        width: 1920,
        height: 1080,
        present_mode: PresentMode::Mailbox,
    };

    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, TextureFormat::Bgra8Unorm);
    assert_eq!(config.present_mode, PresentMode::Mailbox);
}
