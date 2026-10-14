//! Vulkan backend implementation for lgpu-hal
//!
//! Uses zero-cost dynamic loading via `ash` to interface with `libvulkan.so.1`,
//! `vulkan-1.dll`, or `MoltenVK`.

use ash::vk;
use std::ffi::CString;

pub mod bindless;

use crate::{
    AdapterInfo, Backend, BufferDesc, BufferUsage, DeviceType, HalError, MemoryLocation,
    TextureDesc, TextureFormat, TextureUsage,
};

pub use ash::vk::Pipeline as RawPipeline;

pub struct VulkanInstance {
    _entry: ash::Entry,
    instance: ash::Instance,
}

pub struct VulkanAdapter {
    physical_device: vk::PhysicalDevice,
    info: AdapterInfo,
    queue_family_index: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

pub struct VulkanDevice {
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub bindless_heap: Option<bindless::VulkanBindlessHeap>,
}

pub struct VulkanBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    mapped_ptr: Option<*mut u8>,
}

pub struct VulkanTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
}

pub struct VulkanShader {
    module: vk::ShaderModule,
}

pub struct VulkanComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

pub struct VulkanCommandRecorder {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

impl VulkanInstance {
    pub fn new() -> Result<Self, HalError> {
        let entry = unsafe {
            ash::Entry::load().map_err(|e| HalError::LibraryLoadError(e.to_string()))?
        };

        let app_name = CString::new("lgpu Application").unwrap();
        let engine_name = CString::new("lgpu").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(|e| HalError::BackendError(format!("Failed to create Vulkan instance: {e}")))?
        };

        Ok(Self {
            _entry: entry,
            instance,
        })
    }

    pub fn enumerate_adapters(&self) -> Result<Vec<VulkanAdapter>, HalError> {
        let physical_devices = unsafe {
            self.instance
                .enumerate_physical_devices()
                .map_err(|e| HalError::BackendError(e.to_string()))?
        };

        if physical_devices.is_empty() {
            return Err(HalError::NoAdapterFound);
        }

        let mut adapters = Vec::new();
        for pd in physical_devices {
            let props = unsafe { self.instance.get_physical_device_properties(pd) };
            let mem_props = unsafe { self.instance.get_physical_device_memory_properties(pd) };
            let queue_props =
                unsafe { self.instance.get_physical_device_queue_family_properties(pd) };

            // Find a queue family that supports both graphics and compute (or at least compute)
            let mut selected_queue = None;
            for (idx, qp) in queue_props.iter().enumerate() {
                if qp.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    selected_queue = Some(idx as u32);
                    if qp.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        break;
                    }
                }
            }

            let queue_family_index = match selected_queue {
                Some(idx) => idx,
                None => continue,
            };

            let name_bytes: Vec<u8> = props
                .device_name
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            let name = String::from_utf8_lossy(&name_bytes).to_string();

            let device_type = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::DiscreteGpu,
                vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::IntegratedGpu,
                vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::VirtualGpu,
                vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
                _ => DeviceType::Other,
            };

            let adapter_info = AdapterInfo {
                name,
                backend: Backend::Vulkan,
                device_type,
                vendor_id: props.vendor_id,
                device_id: props.device_id,
                supports_dynamic_rendering: true,
                supports_timeline_semaphores: true,
                supports_descriptor_indexing: true,
                max_compute_workgroup_invocations: props
                    .limits
                    .max_compute_work_group_invocations,
                max_compute_workgroup_size: props.limits.max_compute_work_group_size,
            };

            adapters.push(VulkanAdapter {
                physical_device: pd,
                info: adapter_info,
                queue_family_index,
                memory_properties: mem_props,
            });
        }

        Ok(adapters)
    }

    pub fn create_device(&self, adapter: &VulkanAdapter) -> Result<VulkanDevice, HalError> {
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(adapter.queue_family_index)
            .queue_priorities(&queue_priorities);

        let queue_infos = [queue_create_info];

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true);

        let mut features = vk::PhysicalDeviceFeatures::default();
        features.shader_int64 = vk::TRUE;

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&features)
            .push_next(&mut features12)
            .push_next(&mut features13);

        let device = unsafe {
            self.instance
                .create_device(adapter.physical_device, &device_create_info, None)
                .or_else(|_| {
                    // Fallback without features12 if unsupported on legacy drivers
                    let fallback_create_info = vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queue_infos)
                        .enabled_features(&features);
                    self.instance
                        .create_device(adapter.physical_device, &fallback_create_info, None)
                })
                .map_err(|e| HalError::DeviceCreationFailed(e.to_string()))?
        };

        let queue = unsafe { device.get_device_queue(adapter.queue_family_index, 0) };

        let bindless_heap = bindless::VulkanBindlessHeap::new(
            &device,
            bindless::DEFAULT_MAX_STORAGE_BUFFERS,
            bindless::DEFAULT_MAX_TEXTURES,
            bindless::DEFAULT_MAX_SAMPLERS,
        ).ok();

        Ok(VulkanDevice {
            device,
            queue,
            queue_family_index: adapter.queue_family_index,
            memory_properties: adapter.memory_properties,
            bindless_heap,
        })
    }
}

impl VulkanAdapter {
    pub fn info(&self) -> &AdapterInfo {
        &self.info
    }
}

impl VulkanDevice {
    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, HalError> {
        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & properties)
                    == properties
            {
                return Ok(i);
            }
        }
        Err(HalError::OutOfMemory("Suitable memory type not found".to_string()))
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<VulkanBuffer, HalError> {
        let mut vk_usage = vk::BufferUsageFlags::empty();
        if desc.usage.contains(BufferUsage::TRANSFER_SRC) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if desc.usage.contains(BufferUsage::TRANSFER_DST) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if desc.usage.contains(BufferUsage::STORAGE) {
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if desc.usage.contains(BufferUsage::UNIFORM) {
            vk_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if desc.usage.contains(BufferUsage::VERTEX) {
            vk_usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if desc.usage.contains(BufferUsage::INDEX) {
            vk_usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if desc.usage.contains(BufferUsage::INDIRECT) {
            vk_usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateBuffer failed: {e}")))?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let mem_props = match desc.memory_location {
            MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::CpuToGpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
            MemoryLocation::GpuToCpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
        };

        let mem_type_index = self.find_memory_type(mem_reqs.memory_type_bits, mem_props)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| HalError::OutOfMemory(format!("vkAllocateMemory failed: {e}")))?
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| HalError::BackendError(format!("vkBindBufferMemory failed: {e}")))?;
        }

        let mut mapped_ptr = None;
        if desc.memory_location != MemoryLocation::GpuOnly {
            let ptr = unsafe {
                self.device
                    .map_memory(memory, 0, desc.size, vk::MemoryMapFlags::empty())
                    .map_err(|e| HalError::MapError(e.to_string()))?
            };
            mapped_ptr = Some(ptr as *mut u8);
        }

        Ok(VulkanBuffer {
            buffer,
            memory,
            size: desc.size,
            mapped_ptr,
        })
    }

    pub fn write_buffer(&self, buffer: &VulkanBuffer, offset: u64, data: &[u8]) -> Result<(), HalError> {
        let ptr = buffer
            .mapped_ptr
            .ok_or_else(|| HalError::MapError("Buffer is not host-mapped".to_string()))?;
        if offset + data.len() as u64 > buffer.size {
            return Err(HalError::CommandError("Buffer write overflow".to_string()));
        }
        unsafe {
            let dst = ptr.add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        Ok(())
    }

    pub fn read_buffer(&self, buffer: &VulkanBuffer, offset: u64, data: &mut [u8]) -> Result<(), HalError> {
        let ptr = buffer
            .mapped_ptr
            .ok_or_else(|| HalError::MapError("Buffer is not host-mapped".to_string()))?;
        if offset + data.len() as u64 > buffer.size {
            return Err(HalError::CommandError("Buffer read overflow".to_string()));
        }
        unsafe {
            let src = ptr.add(offset as usize);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), data.len());
        }
        Ok(())
    }

    pub fn create_texture(&self, desc: &TextureDesc) -> Result<VulkanTexture, HalError> {
        let vk_format = match desc.format {
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            TextureFormat::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
            TextureFormat::R32Float => vk::Format::R32_SFLOAT,
            TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        };

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

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth.max(1),
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
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
        let mem_type_index = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| HalError::OutOfMemory(format!("vkAllocateMemory failed for image: {e}")))?
        };

        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| HalError::BackendError(format!("vkBindImageMemory failed: {e}")))?;
        }

        let aspect_mask = if desc.format == TextureFormat::Depth32Float {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(1)
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
            memory,
            width: desc.width,
            height: desc.height,
            format: vk_format,
        })
    }

    pub fn destroy_texture(&self, texture: VulkanTexture) {
        unsafe {
            self.device.destroy_image_view(texture.view, None);
            self.device.destroy_image(texture.image, None);
            self.device.free_memory(texture.memory, None);
        }
    }

    pub fn cmd_clear_and_render(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        clear_color: [f32; 4],
    ) {
        unsafe {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let barrier_to_attach = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_attach],
            );

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear_color,
                },
            };

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value);

            let color_attachments = [color_attachment];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: texture.width,
                    height: texture.height,
                },
            };

            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(&color_attachments);

            self.device
                .cmd_begin_rendering(recorder.command_buffer, &rendering_info);

            self.device.cmd_end_rendering(recorder.command_buffer);

            let barrier_to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_src],
            );
        }
    }

    pub fn cmd_copy_image_to_buffer(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        buffer: &VulkanBuffer,
    ) {
        unsafe {
            let subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(texture.width)
                .buffer_image_height(texture.height)
                .image_subresource(subresource)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: texture.width,
                    height: texture.height,
                    depth: 1,
                });

            self.device.cmd_copy_image_to_buffer(
                recorder.command_buffer,
                texture.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.buffer,
                &[copy_region],
            );
        }
    }

    pub fn create_shader_from_spirv(&self, spirv_words: &[u32]) -> Result<VulkanShader, HalError> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
        let module = unsafe {
            self.device
                .create_shader_module(&create_info, None)
                .map_err(|e| HalError::ShaderError(e.to_string()))?
        };
        Ok(VulkanShader { module })
    }

    pub fn create_compute_pipeline(
        &self,
        shader: &VulkanShader,
        buffer_count: u32,
    ) -> Result<VulkanComputePipeline, HalError> {
        // Create descriptor set layout for storage buffers
        let mut bindings = Vec::with_capacity(buffer_count as usize);
        for i in 0..buffer_count {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

        let layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let entry_point = CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| HalError::PipelineError(e.to_string()))?
        };

        // Create descriptor pool and allocate descriptor set
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(buffer_count.max(1))];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        Ok(VulkanComputePipeline {
            pipeline: pipelines[0],
            layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set: descriptor_sets[0],
        })
    }

    pub fn bind_buffers_to_compute_pipeline(
        &self,
        pipeline: &VulkanComputePipeline,
        buffers: &[&VulkanBuffer],
    ) {
        let mut buffer_infos = Vec::new();
        let mut writes = Vec::new();

        for (_binding, buffer) in buffers.iter().enumerate() {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.size),
            );
        }

        for (binding, info) in buffer_infos.iter().enumerate() {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(pipeline.descriptor_set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info));
            writes.push(write);
        }

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn create_command_recorder(&self) -> Result<VulkanCommandRecorder, HalError> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            self.device
                .create_command_pool(&pool_info, None)
                .map_err(|e| HalError::CommandError(e.to_string()))?
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| HalError::CommandError(e.to_string()))?
        };

        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| HalError::CommandError(e.to_string()))?;
        }

        Ok(VulkanCommandRecorder {
            command_pool,
            command_buffer,
        })
    }

    pub fn register_bindless_buffer(&self, buffer: &VulkanBuffer) -> Result<u32, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let slot = heap.allocate_buffer_slot()?;
        heap.write_buffer_descriptor(&self.device, slot, buffer.buffer, buffer.size);
        Ok(slot)
    }

    pub fn unregister_bindless_buffer(&self, slot: u32) {
        if let Some(heap) = &self.bindless_heap {
            heap.free_buffer_slot(slot);
        }
    }

    pub fn create_bindless_compute_pipeline(
        &self,
        shader: &VulkanShader,
    ) -> Result<vk::Pipeline, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        let entry_point = CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(heap.pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| HalError::PipelineError(e.to_string()))?
        };

        Ok(pipelines[0])
    }

    pub fn dispatch_bindless(
        &self,
        recorder: &VulkanCommandRecorder,
        pipeline: vk::Pipeline,
        push_constants: &[u8],
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        unsafe {
            self.device.cmd_bind_pipeline(
                recorder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                recorder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                heap.pipeline_layout,
                0,
                &[heap.descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                self.device.cmd_push_constants(
                    recorder.command_buffer,
                    heap.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    push_constants,
                );
            }
            self.device.cmd_dispatch(recorder.command_buffer, x, y, z);
        }
        Ok(())
    }

    pub fn destroy_raw_pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.device.destroy_pipeline(pipeline, None);
        }
    }

    pub fn dispatch(
        &self,
        recorder: &VulkanCommandRecorder,
        pipeline: &VulkanComputePipeline,
        x: u32,
        y: u32,
        z: u32,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                recorder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                recorder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[pipeline.descriptor_set],
                &[],
            );
            self.device.cmd_dispatch(recorder.command_buffer, x, y, z);
        }
    }

    pub fn submit_and_wait(&self, recorder: VulkanCommandRecorder) -> Result<(), HalError> {
        unsafe {
            self.device
                .end_command_buffer(recorder.command_buffer)
                .map_err(|e| HalError::CommandError(e.to_string()))?;

            let command_buffers = [recorder.command_buffer];
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&command_buffers);

            let fence_info = vk::FenceCreateInfo::default();
            let fence = self
                .device
                .create_fence(&fence_info, None)
                .map_err(|e| HalError::CommandError(e.to_string()))?;

            self.device
                .queue_submit(self.queue, &[submit_info], fence)
                .map_err(|e| HalError::CommandError(e.to_string()))?;

            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| HalError::CommandError(e.to_string()))?;

            self.device.destroy_fence(fence, None);
            self.device.destroy_command_pool(recorder.command_pool, None);
        }
        Ok(())
    }

    pub fn destroy_buffer(&self, buffer: VulkanBuffer) {
        unsafe {
            if let Some(_) = buffer.mapped_ptr {
                self.device.unmap_memory(buffer.memory);
            }
            self.device.destroy_buffer(buffer.buffer, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    pub fn destroy_shader(&self, shader: VulkanShader) {
        unsafe {
            self.device.destroy_shader_module(shader.module, None);
        }
    }

    pub fn destroy_compute_pipeline(&self, pipeline: VulkanComputePipeline) {
        unsafe {
            self.device
                .destroy_descriptor_pool(pipeline.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
            self.device.destroy_pipeline(pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(pipeline.layout, None);
        }
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(heap) = self.bindless_heap.take() {
                heap.destroy(&self.device);
            }
            self.device.destroy_device(None);
        }
    }
}
