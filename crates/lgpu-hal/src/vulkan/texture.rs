//! Vulkan texture and sampler resource management.

use super::VulkanDevice;
use crate::{
    AddressMode, FilterMode, HalError, SamplerDesc, TextureDesc, TextureDimension, TextureUsage,
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
    /// Slices of a 3D image; 1 for every other dimension.
    pub depth: u32,
    /// Image layers (`6` per cube, 1 for plain 2D/3D).
    pub array_layers: u32,
    pub dimension: crate::TextureDimension,
    pub mip_levels: u32,
    pub sample_count: u32,
    /// Whether the image was created with sampled usage. Only then is a transition to
    /// `SHADER_READ_ONLY_OPTIMAL` legal.
    pub supports_sampled: bool,
    pub usage_flags: vk::ImageUsageFlags,
    pub format: vk::Format,
    /// Shared with every clone of this image so layout tracking cannot diverge between handles
    /// that reference the same `VkImage`.
    pub layout: std::sync::Arc<std::sync::Mutex<vk::ImageLayout>>,
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
    pub fn clone_unowned(&self) -> Self {
        Self {
            image: self.image,
            view: self.view,
            memory: self.memory,
            block_id: self.block_id,
            offset: self.offset,
            allocated_size: self.allocated_size,
            width: self.width,
            height: self.height,
            depth: self.depth,
            array_layers: self.array_layers,
            dimension: self.dimension,
            mip_levels: self.mip_levels,
            sample_count: self.sample_count,
            supports_sampled: self.supports_sampled,
            usage_flags: self.usage_flags,
            format: self.format,
            layout: std::sync::Arc::clone(&self.layout),
            owned: false,
        }
    }

    #[inline]
    pub fn current_layout(&self) -> vk::ImageLayout {
        match self.layout.lock() {
            Ok(g) => *g,
            Err(p) => *p.into_inner(),
        }
    }

    #[inline]
    pub fn transition_layout(&self, new_layout: vk::ImageLayout) -> vk::ImageLayout {
        let mut lock = match self.layout.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
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
            if desc.format.is_depth() {
                usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            } else {
                usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            }
        }

        let mip_levels = desc.mip_levels.max(1);

        let (image_type, extent, array_layers, view_type) = match desc.dimension {
            TextureDimension::D1 => {
                if desc.height != 1 || desc.depth != 1 {
                    return Err(HalError::InvalidParameter(
                        "A 1D texture must have height == 1 and depth == 1".to_string(),
                    ));
                }
                (
                    vk::ImageType::TYPE_1D,
                    vk::Extent3D {
                        width: desc.width,
                        height: 1,
                        depth: 1,
                    },
                    1,
                    vk::ImageViewType::TYPE_1D,
                )
            }
            TextureDimension::D2 => (
                vk::ImageType::TYPE_2D,
                vk::Extent3D {
                    width: desc.width,
                    height: desc.height,
                    depth: 1,
                },
                1,
                vk::ImageViewType::TYPE_2D,
            ),
            TextureDimension::D2Array => {
                if desc.array_layers == 0 {
                    return Err(HalError::InvalidParameter(
                        "A 2D array texture needs at least one layer".to_string(),
                    ));
                }
                (
                    vk::ImageType::TYPE_2D,
                    vk::Extent3D {
                        width: desc.width,
                        height: desc.height,
                        depth: 1,
                    },
                    desc.array_layers,
                    vk::ImageViewType::TYPE_2D_ARRAY,
                )
            }
            TextureDimension::D3 => (
                vk::ImageType::TYPE_3D,
                vk::Extent3D {
                    width: desc.width,
                    height: desc.height,
                    depth: desc.depth.max(1),
                },
                1,
                vk::ImageViewType::TYPE_3D,
            ),
            TextureDimension::Cube | TextureDimension::CubeArray => {
                if desc.width != desc.height {
                    return Err(HalError::InvalidParameter(format!(
                        "A cube map must be square, but got {}x{}",
                        desc.width, desc.height
                    )));
                }
                if desc.array_layers < 6 || desc.array_layers % 6 != 0 {
                    return Err(HalError::InvalidParameter(format!(
                        "A cube map needs a multiple of 6 layers, but got {}",
                        desc.array_layers
                    )));
                }
                (
                    vk::ImageType::TYPE_2D,
                    vk::Extent3D {
                        width: desc.width,
                        height: desc.height,
                        depth: 1,
                    },
                    desc.array_layers,
                    if desc.dimension == TextureDimension::Cube {
                        vk::ImageViewType::CUBE
                    } else {
                        vk::ImageViewType::CUBE_ARRAY
                    },
                )
            }
        };

        // A texture may not have more mip levels than log2(largest extent) + 1.
        let largest = desc
            .width
            .max(desc.height)
            .max(if image_type == vk::ImageType::TYPE_3D {
                desc.depth
            } else {
                1
            });
        let max_mip_levels = 32 - largest.leading_zeros();
        if mip_levels > max_mip_levels {
            return Err(HalError::InvalidParameter(format!(
                "mip_levels {mip_levels} exceeds the maximum {max_mip_levels} for a {largest}-texel texture"
            )));
        }
        let sample_count = match desc.sample_count {
            1 => vk::SampleCountFlags::TYPE_1,
            2 => vk::SampleCountFlags::TYPE_2,
            4 => vk::SampleCountFlags::TYPE_4,
            8 => vk::SampleCountFlags::TYPE_8,
            16 => vk::SampleCountFlags::TYPE_16,
            32 => vk::SampleCountFlags::TYPE_32,
            64 => vk::SampleCountFlags::TYPE_64,
            other => {
                return Err(HalError::InvalidParameter(format!(
                    "Unsupported sample_count {other}: expected 1, 2, 4, 8, 16, 32, or 64"
                )))
            }
        };

        let (sharing_mode, sharing_families) = self.sharing();
        let image_info = vk::ImageCreateInfo::default()
            .image_type(image_type)
            .format(vk_format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(sample_count)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(sharing_mode)
            .queue_family_indices(sharing_families)
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

        let mut alloc_lock = match self.allocator.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let alloc = alloc_lock.allocate(
            &self.device,
            mem_reqs.size,
            mem_reqs.alignment,
            mem_type_index,
            false,
            is_coherent,
            super::allocator::AllocationType::Optimal,
        )?;
        drop(alloc_lock);

        unsafe {
            if let Err(e) = self
                .device
                .bind_image_memory(image, alloc.memory, alloc.offset)
            {
                if let Some(block_id) = alloc.block_id {
                    let mut alloc_lock = match self.allocator.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    alloc_lock.free(block_id, alloc.offset, mem_reqs.size, &self.device);
                } else {
                    self.device.free_memory(alloc.memory, None);
                }
                self.device.destroy_image(image, None);
                return Err(HalError::BackendError(format!(
                    "vkBindImageMemory failed: {e}"
                )));
            }
        }

        let aspect_mask = if desc.format.is_depth() {
            vk::ImageAspectFlags::DEPTH
                | if desc.format.has_stencil() {
                    vk::ImageAspectFlags::STENCIL
                } else {
                    vk::ImageAspectFlags::empty()
                }
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(array_layers);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(vk_format)
            .subresource_range(subresource_range);

        let view = match unsafe { self.device.create_image_view(&view_info, None) } {
            Ok(v) => v,
            Err(e) => {
                // Release the image and its memory instead of leaking them on the error path.
                unsafe {
                    self.device.destroy_image(image, None);
                    if let Some(block_id) = alloc.block_id {
                        let mut alloc_lock = match self.allocator.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        alloc_lock.free(block_id, alloc.offset, mem_reqs.size, &self.device);
                    } else {
                        self.device.free_memory(alloc.memory, None);
                    }
                }
                return Err(HalError::BackendError(format!(
                    "vkCreateImageView failed: {e}"
                )));
            }
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
            depth: extent.depth,
            array_layers,
            dimension: desc.dimension,
            mip_levels,
            sample_count: desc.sample_count.max(1),
            supports_sampled: desc.usage.contains(TextureUsage::TEXTURE_BINDING),
            usage_flags: usage,
            format: vk_format,
            layout: std::sync::Arc::new(std::sync::Mutex::new(vk::ImageLayout::UNDEFINED)),
            owned: true,
        })
    }

    pub fn destroy_texture(&self, texture: VulkanTexture) {
        if !texture.owned {
            return;
        }

        let staging_guard = match self.staging_ring.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(ref ring) = *staging_guard {
            if ring.current_timeline_value > 0 {
                let completed = unsafe {
                    self.device
                        .get_semaphore_counter_value(ring.timeline_semaphore)
                        .unwrap_or(0)
                };
                if completed < ring.current_timeline_value {
                    let sems = [ring.timeline_semaphore];
                    let vals = [ring.current_timeline_value];
                    let wait_info = vk::SemaphoreWaitInfo::default()
                        .semaphores(&sems)
                        .values(&vals);
                    unsafe {
                        let _ = self.device.wait_semaphores(&wait_info, u64::MAX);
                    }
                }
            }
        }
        drop(staging_guard);

        unsafe {
            if texture.view != vk::ImageView::null() {
                self.device.destroy_image_view(texture.view, None);
            }
            if texture.image != vk::Image::null() {
                self.device.destroy_image(texture.image, None);
            }
            if texture.memory != vk::DeviceMemory::null() {
                if let Some(block_id) = texture.block_id {
                    let mut alloc_lock = match self.allocator.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    alloc_lock.free(
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
            .compare_enable(desc.compare_enable)
            .compare_op(desc.compare_op.to_vk())
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
