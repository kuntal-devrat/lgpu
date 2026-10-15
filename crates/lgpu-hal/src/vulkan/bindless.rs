//! Bindless Descriptor Heap implementation for Vulkan 1.2+
//!
//! Provides a persistent, global descriptor array where buffers, textures,
//! and samplers can be indexed directly by a 32-bit integer in shaders.

use crate::HalError;
use ash::vk;
use std::sync::Mutex;

pub const DEFAULT_MAX_STORAGE_BUFFERS: u32 = 16384;
pub const DEFAULT_MAX_TEXTURES: u32 = 16384;
pub const DEFAULT_MAX_SAMPLERS: u32 = 256;
pub const PUSH_CONSTANT_SIZE: u32 = 128;

pub struct VulkanBindlessHeap {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub pipeline_layout: vk::PipelineLayout,
    pub max_buffers: u32,
    pub max_textures: u32,
    pub max_samplers: u32,
    buffer_slots: Mutex<SlotPool>,
    texture_slots: Mutex<SlotPool>,
    sampler_slots: Mutex<SlotPool>,
}

struct SlotPool {
    free: Vec<u32>,
    next: u32,
}

impl VulkanBindlessHeap {
    pub fn new(
        device: &ash::Device,
        max_buffers: u32,
        max_textures: u32,
        max_samplers: u32,
    ) -> Result<Self, HalError> {
        // 1. Descriptor Set Layout Bindings
        // Binding 0: Storage Buffers array
        // Binding 1: Sampled Images array
        // Binding 2: Samplers array
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(max_buffers)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(max_textures)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(max_samplers)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let binding_flags = [
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];

        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut flags_info);

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| {
                    HalError::PipelineError(format!("Failed to create bindless set layout: {e}"))
                })?
        };

        // 2. Global Pipeline Layout (with 128 bytes Push Constants)
        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(PUSH_CONSTANT_SIZE)];

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| {
                    HalError::PipelineError(format!(
                        "Failed to create bindless pipeline layout: {e}"
                    ))
                })?
        };

        // 3. Descriptor Pool
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(max_buffers),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(max_textures),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(max_samplers),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    HalError::PipelineError(format!(
                        "Failed to create bindless descriptor pool: {e}"
                    ))
                })?
        };

        // 4. Allocate the single global Descriptor Set
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&alloc_info).map_err(|e| {
                HalError::PipelineError(format!("Failed to allocate bindless descriptor set: {e}"))
            })?
        };

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set: descriptor_sets[0],
            pipeline_layout,
            max_buffers,
            max_textures,
            max_samplers,
            buffer_slots: Mutex::new(SlotPool {
                free: Vec::new(),
                next: 0,
            }),
            texture_slots: Mutex::new(SlotPool {
                free: Vec::new(),
                next: 0,
            }),
            sampler_slots: Mutex::new(SlotPool {
                free: Vec::new(),
                next: 0,
            }),
        })
    }

    pub fn allocate_buffer_slot(&self) -> Result<u32, HalError> {
        let mut pool = self.buffer_slots.lock().unwrap();
        if let Some(slot) = pool.free.pop() {
            return Ok(slot);
        }
        if pool.next >= self.max_buffers {
            return Err(HalError::OutOfMemory(
                "Exceeded max bindless buffer capacity".to_string(),
            ));
        }
        let slot = pool.next;
        pool.next += 1;
        Ok(slot)
    }

    pub fn free_buffer_slot(&self, slot: u32) {
        let mut pool = self.buffer_slots.lock().unwrap();
        pool.free.push(slot);
    }

    pub fn write_buffer_descriptor(
        &self,
        device: &ash::Device,
        slot: u32,
        buffer: vk::Buffer,
        size: u64,
    ) {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(size);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            device.update_descriptor_sets(&[write], &[]);
        }
    }

    pub fn allocate_texture_slot(&self) -> Result<u32, HalError> {
        let mut pool = self.texture_slots.lock().unwrap();
        if let Some(slot) = pool.free.pop() {
            return Ok(slot);
        }
        if pool.next >= self.max_textures {
            return Err(HalError::OutOfMemory(
                "Exceeded max bindless texture capacity".to_string(),
            ));
        }
        let slot = pool.next;
        pool.next += 1;
        Ok(slot)
    }

    pub fn free_texture_slot(&self, slot: u32) {
        let mut pool = self.texture_slots.lock().unwrap();
        pool.free.push(slot);
    }

    pub fn write_texture_descriptor(
        &self,
        device: &ash::Device,
        slot: u32,
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(image_view)
            .image_layout(image_layout);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info));

        unsafe {
            device.update_descriptor_sets(&[write], &[]);
        }
    }

    pub fn allocate_sampler_slot(&self) -> Result<u32, HalError> {
        let mut pool = self.sampler_slots.lock().unwrap();
        if let Some(slot) = pool.free.pop() {
            return Ok(slot);
        }
        if pool.next >= self.max_samplers {
            return Err(HalError::OutOfMemory(
                "Exceeded max bindless sampler capacity".to_string(),
            ));
        }
        let slot = pool.next;
        pool.next += 1;
        Ok(slot)
    }

    pub fn free_sampler_slot(&self, slot: u32) {
        let mut pool = self.sampler_slots.lock().unwrap();
        pool.free.push(slot);
    }

    pub fn write_sampler_descriptor(&self, device: &ash::Device, slot: u32, sampler: vk::Sampler) {
        let sampler_info = vk::DescriptorImageInfo::default().sampler(sampler);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_set)
            .dst_binding(2)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&sampler_info));

        unsafe {
            device.update_descriptor_sets(&[write], &[]);
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
