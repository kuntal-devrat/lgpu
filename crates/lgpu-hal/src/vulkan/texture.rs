//! Vulkan texture and sampler resource management.

use super::VulkanDevice;
use crate::{
    AddressMode, FilterMode, HalError, SamplerDesc, TextureDesc, TextureFormat, TextureUsage,
};
use ash::vk;

pub struct VulkanTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub block_id: Option<usize>,
    pub offset: u64,
    pub allocated_size: u64,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub sample_count: u32,
    pub format: vk::Format,
    pub layout: std::sync::Mutex<vk::ImageLayout>,
    pub owned: bool,
}

unsafe impl Send for VulkanTexture {}
unsafe impl Sync for VulkanTexture {}

pub struct VulkanSampler {
    pub sampler: vk::Sampler,
}

unsafe impl Send for VulkanSampler {}
unsafe impl Sync for VulkanSampler {}

impl VulkanTexture {
    #[inline]
    pub fn current_layout(&self) -> vk::ImageLayout {
        *self.layout.lock().unwrap()
    }

    #[inline]
    pub fn transition_layout(&self, new_layout: vk::ImageLayout) -> vk::ImageLayout {
        let mut lock = self.layout.lock().unwrap();
        let old = *lock;
        *lock = new_layout;
        old
    }
}

impl VulkanDevice {
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<VulkanTexture, HalError> {
        if desc.width == 0 || desc.height == 0 || desc.depth == 0 {
            return Err(HalError::InvalidParameter(
                "Texture dimensions must be greater than 0".to_string(),
            ));
        }

        let vk_format = desc.format.to_vk();

        let mut usage = vk::ImageUsageFlags::empty();
        if desc.usage.contains(TextureUsage::COPY_SRC) {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if desc.usage.contains(TextureUsage::COPY_DST) {
            usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if desc.usage.contains(TextureUsage::TEXTURE_BINDING) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT) {
            if desc.format == TextureFormat::Depth32Float {
                usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            } else {
                usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            }
        }

        let mip_levels = desc.mip_levels.max(1);
        let sample_count = match desc.sample_count {
            2 => vk::SampleCountFlags::TYPE_2,
            4 => vk::SampleCountFlags::TYPE_4,
            8 => vk::SampleCountFlags::TYPE_8,
            16 => vk::SampleCountFlags::TYPE_16,
            32 => vk::SampleCountFlags::TYPE_32,
            64 => vk::SampleCountFlags::TYPE_64,
            _ => vk::SampleCountFlags::TYPE_1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth.max(1),
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(sample_count)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            self.device
                .create_image(&image_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateImage failed: {e}")))?
        };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let (mem_type_index, is_coherent) = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc = self.allocator.lock().unwrap().allocate(
            &self.device,
            mem_reqs.size,
            mem_reqs.alignment,
            mem_type_index,
            false,
            is_coherent,
            super::allocator::AllocationType::Optimal,
        )?;

        unsafe {
            if let Err(e) = self
                .device
                .bind_image_memory(image, alloc.memory, alloc.offset)
            {
                if let Some(block_id) = alloc.block_id {
                    self.allocator.lock().unwrap().free(
                        block_id,
                        alloc.offset,
                        mem_reqs.size,
                        &self.device,
                    );
                } else {
                    self.device.free_memory(alloc.memory, None);
                }
                self.device.destroy_image(image, None);
                return Err(HalError::BackendError(format!(
                    "vkBindImageMemory failed: {e}"
                )));
            }
        }

        let aspect_mask = if desc.format == TextureFormat::Depth32Float {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(subresource_range);

        let view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateImageView failed: {e}")))?
        };

        Ok(VulkanTexture {
            image,
            view,
            memory: alloc.memory,
            block_id: alloc.block_id,
            offset: alloc.offset,
            allocated_size: mem_reqs.size,
            width: desc.width,
            height: desc.height,
            mip_levels,
            sample_count: desc.sample_count.max(1),
            format: vk_format,
            layout: std::sync::Mutex::new(vk::ImageLayout::UNDEFINED),
            owned: true,
        })
    }

    pub fn destroy_texture(&self, texture: VulkanTexture) {
        if !texture.owned {
            return;
        }
        unsafe {
            if texture.view != vk::ImageView::null() {
                self.device.destroy_image_view(texture.view, None);
            }
            if texture.image != vk::Image::null() {
                self.device.destroy_image(texture.image, None);
            }
            if texture.memory != vk::DeviceMemory::null() {
                if let Some(block_id) = texture.block_id {
                    self.allocator.lock().unwrap().free(
                        block_id,
                        texture.offset,
                        texture.allocated_size,
                        &self.device,
                    );
                } else {
                    self.device.free_memory(texture.memory, None);
                }
            }
        }
    }

    pub fn register_bindless_texture(&self, texture: &VulkanTexture) -> Result<u32, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let slot = heap.allocate_texture_slot()?;
        heap.write_texture_descriptor(
            &self.device,
            slot,
            texture.view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        Ok(slot)
    }

    pub fn unregister_bindless_texture(&self, slot: u32) {
        if let Some(heap) = &self.bindless_heap {
            heap.free_texture_slot(slot);
        }
    }

    pub fn register_bindless_storage_image(
        &self,
        texture: &VulkanTexture,
    ) -> Result<u32, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let slot = heap.allocate_storage_image_slot()?;
        heap.write_storage_image_descriptor(
            &self.device,
            slot,
            texture.view,
            vk::ImageLayout::GENERAL,
        );
        Ok(slot)
    }

    pub fn unregister_bindless_storage_image(&self, slot: u32) {
        if let Some(heap) = &self.bindless_heap {
            heap.free_storage_image_slot(slot);
        }
    }

    pub fn write_texture_descriptors_batch(
        &self,
        writes: &[(u32, &VulkanTexture)],
    ) -> Result<(), HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let raw_writes: Vec<(u32, ash::vk::ImageView, ash::vk::ImageLayout)> = writes
            .iter()
            .map(|(slot, tex)| {
                (
                    *slot,
                    tex.view,
                    ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )
            })
            .collect();
        heap.write_texture_descriptors_batch(&self.device, &raw_writes);
        Ok(())
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> Result<VulkanSampler, HalError> {
        let vk_filter = |f: FilterMode| match f {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        };
        let vk_mipmap_mode = |f: FilterMode| match f {
            FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
        };
        let vk_address_mode = |a: AddressMode| match a {
            AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        };

        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk_filter(desc.mag_filter))
            .min_filter(vk_filter(desc.min_filter))
            .mipmap_mode(vk_mipmap_mode(desc.mipmap_filter))
            .address_mode_u(vk_address_mode(desc.address_mode_u))
            .address_mode_v(vk_address_mode(desc.address_mode_v))
            .address_mode_w(vk_address_mode(desc.address_mode_w))
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .compare_enable(false)
            .compare_op(vk::CompareOp::NEVER)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false);

        let sampler = unsafe {
            self.device
                .create_sampler(&create_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateSampler failed: {e}")))?
        };

        Ok(VulkanSampler { sampler })
    }

    pub fn destroy_sampler(&self, sampler: VulkanSampler) {
        unsafe {
            self.device.destroy_sampler(sampler.sampler, None);
        }
    }

    pub fn register_bindless_sampler(&self, sampler: &VulkanSampler) -> Result<u32, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let slot = heap.allocate_sampler_slot()?;
        heap.write_sampler_descriptor(&self.device, slot, sampler.sampler);
        Ok(slot)
    }

    pub fn unregister_bindless_sampler(&self, slot: u32) {
        if let Some(heap) = &self.bindless_heap {
            heap.free_sampler_slot(slot);
        }
    }
}
