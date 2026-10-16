//! Unified HAL backend abstraction trait.

use crate::{AdapterInfo, BufferDesc, HalError, SamplerDesc, TextureDesc};

pub trait HalInstance: Send + Sync + 'static {
    type Adapter: HalAdapter;
    type Device: HalDevice;

    fn enumerate_adapters(&self) -> Result<Vec<Self::Adapter>, HalError>;
    fn create_device(&self, adapter: &Self::Adapter) -> Result<Self::Device, HalError>;
}

pub trait HalAdapter: Send + Sync + 'static {
    fn info(&self) -> &AdapterInfo;
}

pub trait HalDevice: Send + Sync + 'static {
    type Buffer: Send + Sync + 'static;
    type Texture: Send + Sync + 'static;
    type Sampler: Send + Sync + 'static;
    type CommandRecorder: Send + Sync + 'static;

    fn create_buffer(&self, desc: &BufferDesc) -> Result<Self::Buffer, HalError>;
    fn destroy_buffer(&self, buffer: Self::Buffer);
    fn write_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), HalError>;
    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        dest: &mut [u8],
    ) -> Result<(), HalError>;

    fn create_texture(&self, desc: &TextureDesc) -> Result<Self::Texture, HalError>;
    fn destroy_texture(&self, texture: Self::Texture);

    fn create_sampler(&self, desc: &SamplerDesc) -> Result<Self::Sampler, HalError>;
    fn destroy_sampler(&self, sampler: Self::Sampler);

    fn create_command_recorder(&self) -> Result<Self::CommandRecorder, HalError>;

    fn register_bindless_buffer(&self, buffer: &Self::Buffer) -> Result<u32, HalError>;
    fn unregister_bindless_buffer(&self, slot: u32);

    fn register_bindless_texture(&self, texture: &Self::Texture) -> Result<u32, HalError>;
    fn unregister_bindless_texture(&self, slot: u32);

    fn register_bindless_storage_image(&self, texture: &Self::Texture) -> Result<u32, HalError>;
    fn unregister_bindless_storage_image(&self, slot: u32);

    fn register_bindless_sampler(&self, sampler: &Self::Sampler) -> Result<u32, HalError>;
    fn unregister_bindless_sampler(&self, slot: u32);
}

pub trait HalBackend: Send + Sync + 'static {
    type Instance: HalInstance<Adapter = Self::Adapter, Device = Self::Device>;
    type Adapter: HalAdapter;
    type Device: HalDevice;
}

#[cfg(feature = "vulkan")]
pub struct VulkanBackend;

#[cfg(feature = "vulkan")]
impl HalBackend for VulkanBackend {
    type Instance = crate::vulkan::VulkanInstance;
    type Adapter = crate::vulkan::VulkanAdapter;
    type Device = crate::vulkan::VulkanDevice;
}

#[cfg(feature = "vulkan")]
impl HalInstance for crate::vulkan::VulkanInstance {
    type Adapter = crate::vulkan::VulkanAdapter;
    type Device = crate::vulkan::VulkanDevice;

    fn enumerate_adapters(&self) -> Result<Vec<Self::Adapter>, HalError> {
        self.enumerate_adapters()
    }

    fn create_device(&self, adapter: &Self::Adapter) -> Result<Self::Device, HalError> {
        self.create_device(adapter)
    }
}

#[cfg(feature = "vulkan")]
impl HalAdapter for crate::vulkan::VulkanAdapter {
    fn info(&self) -> &AdapterInfo {
        self.info()
    }
}

#[cfg(feature = "vulkan")]
impl HalDevice for crate::vulkan::VulkanDevice {
    type Buffer = crate::vulkan::VulkanBuffer;
    type Texture = crate::vulkan::VulkanTexture;
    type Sampler = crate::vulkan::VulkanSampler;
    type CommandRecorder = crate::vulkan::VulkanCommandRecorder;

    fn create_buffer(&self, desc: &BufferDesc) -> Result<Self::Buffer, HalError> {
        self.create_buffer(desc)
    }

    fn destroy_buffer(&self, buffer: Self::Buffer) {
        self.destroy_buffer(buffer)
    }

    fn write_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), HalError> {
        self.write_buffer(buffer, offset, bytes)
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        dest: &mut [u8],
    ) -> Result<(), HalError> {
        self.read_buffer(buffer, offset, dest)
    }

    fn create_texture(&self, desc: &TextureDesc) -> Result<Self::Texture, HalError> {
        self.create_texture(desc)
    }

    fn destroy_texture(&self, texture: Self::Texture) {
        self.destroy_texture(texture)
    }

    fn create_sampler(&self, desc: &SamplerDesc) -> Result<Self::Sampler, HalError> {
        self.create_sampler(desc)
    }

    fn destroy_sampler(&self, sampler: Self::Sampler) {
        self.destroy_sampler(sampler)
    }

    fn create_command_recorder(&self) -> Result<Self::CommandRecorder, HalError> {
        self.create_command_recorder()
    }

    fn register_bindless_buffer(&self, buffer: &Self::Buffer) -> Result<u32, HalError> {
        self.register_bindless_buffer(buffer)
    }

    fn unregister_bindless_buffer(&self, slot: u32) {
        self.unregister_bindless_buffer(slot)
    }

    fn register_bindless_texture(&self, texture: &Self::Texture) -> Result<u32, HalError> {
        self.register_bindless_texture(texture)
    }

    fn unregister_bindless_texture(&self, slot: u32) {
        self.unregister_bindless_texture(slot)
    }

    fn register_bindless_storage_image(&self, texture: &Self::Texture) -> Result<u32, HalError> {
        self.register_bindless_storage_image(texture)
    }

    fn unregister_bindless_storage_image(&self, slot: u32) {
        self.unregister_bindless_storage_image(slot)
    }

    fn register_bindless_sampler(&self, sampler: &Self::Sampler) -> Result<u32, HalError> {
        self.register_bindless_sampler(sampler)
    }

    fn unregister_bindless_sampler(&self, slot: u32) {
        self.unregister_bindless_sampler(slot)
    }
}
