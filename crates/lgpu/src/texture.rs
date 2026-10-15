//! GPU texture, image views, and samplers.

use std::sync::Arc;
use crate::TextureFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerId(pub u32);

pub struct Texture {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_texture: Option<lgpu_hal::vulkan::VulkanTexture>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub id: Option<TextureId>,
    pub owned: bool,
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Texture")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("id", &self.id)
            .field("owned", &self.owned)
            .finish()
    }
}

pub struct Sampler {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_sampler: Option<lgpu_hal::vulkan::VulkanSampler>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub id: Option<SamplerId>,
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("id", &self.id)
            .finish()
    }
}

impl Texture {
    pub fn id(&self) -> Option<TextureId> {
        self.id
    }

    pub fn is_owned(&self) -> bool {
        self.owned
    }

    pub fn clone_unowned(&self) -> Self {
        #[cfg(feature = "vulkan")]
        {
            Self {
                hal_texture: self.hal_texture.as_ref().map(|t| lgpu_hal::vulkan::VulkanTexture {
                    image: t.image,
                    view: t.view,
                    memory: t.memory,
                    block_id: t.block_id,
                    offset: t.offset,
                    allocated_size: t.allocated_size,
                    layout: std::sync::Mutex::new(t.current_layout()),
                    width: t.width,
                    height: t.height,
                    mip_levels: t.mip_levels,
                    sample_count: t.sample_count,
                    format: t.format,
                    owned: false,
                }),
                vulkan_device: Arc::clone(&self.vulkan_device),
                width: self.width,
                height: self.height,
                format: self.format,
                id: self.id,
                owned: false,
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Self {
                width: self.width,
                height: self.height,
                format: self.format,
                id: self.id,
                owned: false,
            }
        }
    }
}

impl Sampler {
    pub fn id(&self) -> Option<SamplerId> {
        self.id
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if self.owned {
                if let Some(tex_id) = self.id {
                    self.vulkan_device.unregister_bindless_texture(tex_id.0);
                }
                if let Some(tex) = self.hal_texture.take() {
                    self.vulkan_device.destroy_texture(tex);
                }
            }
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(sampler_id) = self.id {
                self.vulkan_device.unregister_bindless_sampler(sampler_id.0);
            }
            if let Some(sampler) = self.hal_sampler.take() {
                self.vulkan_device.destroy_sampler(sampler);
            }
        }
    }
}
