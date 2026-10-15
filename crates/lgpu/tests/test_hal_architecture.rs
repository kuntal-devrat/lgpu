//! Integration tests verifying HAL architecture, modular submodule exports, and backend traits.

use lgpu_hal::{BufferDesc, BufferUsage, HalAdapter, HalBackend, HalDevice, HalInstance, MemoryLocation, TextureFormat, VulkanBackend};

#[test]
fn test_texture_format_vk_conversions() {
    let formats = [
        TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8Srgb,
        TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8Srgb,
        TextureFormat::R32Float,
        TextureFormat::Rgba16Float,
        TextureFormat::Rgba32Float,
        TextureFormat::Depth32Float,
    ];

    for fmt in formats {
        let vk_fmt = fmt.to_vk();
        let back_fmt = TextureFormat::from_vk(vk_fmt);
        assert_eq!(back_fmt, Some(fmt), "Failed roundtrip for format {:?}", fmt);
    }
}

#[test]
fn test_lgpu_modular_exports() {
    // Verify each modular submodule in lgpu is accessible
    use lgpu::buffer::Buffer;
    use lgpu::instance::Instance;

    let instance = Instance::new().expect("Failed to create instance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    if adapters.is_empty() {
        return;
    }

    let (device, _queue) = adapters[0].request_device(&instance).expect("Failed to create device");
    let buf: Buffer = device.create_buffer(&BufferDesc {
        label: Some("modular_test_buf"),
        size: 256,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    }).expect("Failed to create buffer");

    assert!(buf.id().is_some());
}

#[test]
fn test_hal_vulkan_modular_exports() {
    // Verify that lgpu_hal::vulkan submodules are individually accessible and cleanly organized
    use lgpu_hal::vulkan::{
        buffer::VulkanBuffer,
        command::VulkanCommandRecorder,
        device::VulkanDevice,
        instance::{VulkanAdapter, VulkanInstance},
        pipeline::VulkanShader,
        swapchain::VulkanSwapchain,
        sync::VulkanTimelineSemaphore,
        texture::{VulkanSampler, VulkanTexture},
    };

    let instance = VulkanInstance::new().expect("Failed to create VulkanInstance");
    let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    if adapters.is_empty() {
        return;
    }

    let dev: VulkanDevice = instance.create_device(&adapters[0]).expect("Failed to create VulkanDevice");
    assert!(dev.queue_family_index < 32);

    // Ensure types from submodules are usable
    fn _assert_types(
        _b: Option<VulkanBuffer>,
        _t: Option<VulkanTexture>,
        _s: Option<VulkanSampler>,
        _c: Option<VulkanCommandRecorder>,
        _sc: Option<VulkanSwapchain>,
        _sem: Option<VulkanTimelineSemaphore>,
        _sh: Option<VulkanShader>,
        _ad: Option<VulkanAdapter>,
    ) {}
    _assert_types(None, None, None, None, None, None, None, None);
}

#[test]
fn test_hal_traits_abstraction() {
    // Verify HalBackend, HalInstance, HalAdapter, HalDevice traits work uniformly
    let instance = <VulkanBackend as HalBackend>::Instance::new().expect("Failed to create instance via HalBackend");
    let adapters = HalInstance::enumerate_adapters(&instance).expect("HalInstance::enumerate_adapters");
    if adapters.is_empty() {
        return;
    }

    let info = HalAdapter::info(&adapters[0]);
    assert!(!info.name.is_empty());

    let device = HalInstance::create_device(&instance, &adapters[0]).expect("HalInstance::create_device");

    let buf = HalDevice::create_buffer(&device, &BufferDesc {
        label: Some("hal_buf"),
        size: 128,
        usage: BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        memory_location: MemoryLocation::CpuToGpu,
    }).expect("HalDevice::create_buffer");

    let slot = HalDevice::register_bindless_buffer(&device, &buf).expect("HalDevice::register_bindless_buffer");
    assert!(slot < 16384);

    let test_bytes = [42u8; 16];
    HalDevice::write_buffer(&device, &buf, 0, &test_bytes).expect("HalDevice::write_buffer");

    let mut read_bytes = [0u8; 16];
    HalDevice::read_buffer(&device, &buf, 0, &mut read_bytes).expect("HalDevice::read_buffer");
    assert_eq!(read_bytes, test_bytes);

    HalDevice::unregister_bindless_buffer(&device, slot);
    HalDevice::destroy_buffer(&device, buf);
}
