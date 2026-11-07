//! Vulkan command recording, render passes, copy operations, and queue submission.

use super::{
    bindless, VulkanBuffer, VulkanComputePipeline, VulkanDevice, VulkanTexture,
    VulkanTimelineSemaphore,
};
use crate::HalError;
use ash::vk;

pub struct VulkanCommandRecorder {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    /// Queue family the command buffer was allocated for; submission must target it.
    pub queue_family_index: u32,
}

unsafe impl Send for VulkanCommandRecorder {}

#[derive(Clone, Copy)]
pub enum DrawCommand<'a> {
    Array {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    Indexed {
        index_buffer: &'a VulkanBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        index_type: vk::IndexType,
    },
    Indirect {
        indirect_buffer: &'a VulkanBuffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
    },
    IndexedIndirect {
        index_buffer: &'a VulkanBuffer,
        indirect_buffer: &'a VulkanBuffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
        index_type: vk::IndexType,
    },
}

pub const MAX_CACHED_COMMAND_RECORDERS: usize = 32;

impl VulkanDevice {
    pub fn create_command_recorder(&self) -> Result<VulkanCommandRecorder, HalError> {
        self.create_recorder_on(self.queue_family_index)
    }

    /// Creates a command recorder for the dedicated compute queue family.
    pub fn create_compute_command_recorder(&self) -> Result<VulkanCommandRecorder, HalError> {
        let family = self.compute_queue_family_index.ok_or_else(|| {
            HalError::CommandError("This adapter has no separate compute queue family".to_string())
        })?;
        self.create_recorder_on(family)
    }

    fn create_recorder_on(&self, family: u32) -> Result<VulkanCommandRecorder, HalError> {
        // Harvest any completed async command recorders back into cache
        {
            let current_val = unsafe {
                self.device
                    .get_semaphore_counter_value(self.internal_frame_semaphore)
                    .unwrap_or(0)
            };
            let mut pending = match self.pending_async_recorders.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            pending.retain_mut(|(val, recorders)| {
                if *val <= current_val {
                    for r in recorders.drain(..) {
                        let cache_mutex = self.recorder_cache_for(r.queue_family_index);
                        let mut cache = match cache_mutex.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        if cache.len() < MAX_CACHED_COMMAND_RECORDERS {
                            cache.push((r.command_pool, r.command_buffer));
                        } else {
                            unsafe {
                                self.device.destroy_command_pool(r.command_pool, None);
                            }
                        }
                    }
                    false
                } else {
                    true
                }
            });

            if let Ok(mut ring_guard) = self.staging_ring.try_lock() {
                if let Some(ref mut ring) = *ring_guard {
                    let completed = unsafe {
                        self.device
                            .get_semaphore_counter_value(ring.timeline_semaphore)
                            .unwrap_or(0)
                    };
                    ring.pending_recorders.retain_mut(|(val, r)| {
                        if *val <= completed {
                            unsafe {
                                self.device
                                    .reset_command_buffer(
                                        r.command_buffer,
                                        vk::CommandBufferResetFlags::empty(),
                                    )
                                    .ok();
                            }
                            let cache_mutex = self.recorder_cache_for(r.queue_family_index);
                            let mut cache = match cache_mutex.lock() {
                                Ok(g) => g,
                                Err(p) => p.into_inner(),
                            };
                            if cache.len() < MAX_CACHED_COMMAND_RECORDERS {
                                cache.push((r.command_pool, r.command_buffer));
                            } else {
                                unsafe {
                                    self.device.destroy_command_pool(r.command_pool, None);
                                }
                            }
                            false
                        } else {
                            true
                        }
                    });
                }
            }
        }

        let cached_item = {
            let mut cache = match self.recorder_cache_for(family).lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            cache.pop()
        };

        let (command_pool, command_buffer) = match cached_item {
            Some((pool, cb)) => {
                unsafe {
                    self.device
                        .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                        .ok();
                }
                (pool, cb)
            }
            None => {
                let pool_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

                let command_pool = unsafe {
                    self.device
                        .create_command_pool(&pool_info, None)
                        .map_err(|e| {
                            HalError::CommandError(format!("Failed to create command pool: {e}"))
                        })?
                };

                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                let command_buffer =
                    match unsafe { self.device.allocate_command_buffers(&alloc_info) } {
                        Ok(bufs) if !bufs.is_empty() => bufs[0],
                        Ok(_) => {
                            unsafe {
                                self.device.destroy_command_pool(command_pool, None);
                            }
                            return Err(HalError::CommandError(
                                "Allocated 0 command buffers".to_string(),
                            ));
                        }
                        Err(e) => {
                            unsafe {
                                self.device.destroy_command_pool(command_pool, None);
                            }
                            return Err(HalError::CommandError(format!(
                                "Failed to allocate command buffer: {e}"
                            )));
                        }
                    };

                (command_pool, command_buffer)
            }
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        if let Err(e) = unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
        } {
            let mut cache = match self.recorder_cache_for(family).lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if cache.len() < MAX_CACHED_COMMAND_RECORDERS {
                cache.push((command_pool, command_buffer));
            } else {
                unsafe {
                    self.device.destroy_command_pool(command_pool, None);
                }
            }
            return Err(HalError::CommandError(e.to_string()));
        }

        Ok(VulkanCommandRecorder {
            command_pool,
            command_buffer,
            queue_family_index: family,
        })
    }

    pub fn recycle_command_recorder(&self, recorder: VulkanCommandRecorder) {
        unsafe {
            let _ = self.device.reset_command_buffer(
                recorder.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            );
        }
        let mut cache = match self.recorder_cache_for(recorder.queue_family_index).lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if cache.len() < MAX_CACHED_COMMAND_RECORDERS {
            cache.push((recorder.command_pool, recorder.command_buffer));
        } else {
            unsafe {
                self.device
                    .destroy_command_pool(recorder.command_pool, None);
            }
        }
    }

    pub fn bind_buffers_to_compute_pipeline(
        &self,
        pipeline: &VulkanComputePipeline,
        buffers: &[&VulkanBuffer],
    ) {
        let mut buffer_infos = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            buffer_infos.push([vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .offset(0)
                .range(buffer.size)]);
        }

        let mut writes = Vec::with_capacity(buffers.len());
        for (binding, info_arr) in buffer_infos.iter().enumerate() {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(pipeline.descriptor_set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info_arr);
            writes.push(write);
        }

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
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

        if push_constants.len() > bindless::PUSH_CONSTANT_SIZE as usize {
            return Err(HalError::CommandError(format!(
                "Push constants size {} exceeds maximum allowed size of {}",
                push_constants.len(),
                bindless::PUSH_CONSTANT_SIZE
            )));
        }

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
                let mut aligned_pc = [0u8; bindless::PUSH_CONSTANT_SIZE as usize];
                aligned_pc[..push_constants.len()].copy_from_slice(push_constants);
                let aligned_len = ((push_constants.len() + 3) / 4) * 4;

                self.device.cmd_push_constants(
                    recorder.command_buffer,
                    heap.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &aligned_pc[..aligned_len],
                );
            }
            self.device.cmd_dispatch(recorder.command_buffer, x, y, z);
        }
        Ok(())
    }

    fn cmd_clear_and_render_internal(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth: Option<(&VulkanTexture, f32)>,
        clear_color: [f32; 4],
    ) {
        unsafe {
            let color_subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let old_color_layout =
                color_texture.transition_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let (src_stage2, src_access2) = match old_color_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                _ => (
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    vk::AccessFlags2::empty(),
                ),
            };

            let color_barrier_to_attach = vk::ImageMemoryBarrier2::default()
                .old_layout(old_color_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_stage_mask(src_stage2)
                .src_access_mask(src_access2)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource);

            let mut barriers = vec![color_barrier_to_attach];

            let mut depth_attachment_val = None;
            if let Some((depth_texture, clear_depth)) = depth {
                let depth_subresource = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let old_depth_layout =
                    depth_texture.transition_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
                let (depth_src_stage2, depth_src_access2) = match old_depth_layout {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                    ),
                };

                let depth_barrier_to_attach = vk::ImageMemoryBarrier2::default()
                    .old_layout(old_depth_layout)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .src_stage_mask(depth_src_stage2)
                    .src_access_mask(depth_src_access2)
                    .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                    .dst_access_mask(
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(depth_texture.image)
                    .subresource_range(depth_subresource);

                barriers.push(depth_barrier_to_attach);

                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_texture.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: clear_depth,
                            stencil: 0,
                        },
                    });
                depth_attachment_val = Some(depth_attachment);
            }

            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            self.device
                .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear_color,
                },
            };

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(color_texture.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value);

            let color_attachments = [color_attachment];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: color_texture.width,
                    height: color_texture.height,
                },
            };

            let mut rendering_info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(&color_attachments);

            if let Some(ref da) = depth_attachment_val {
                rendering_info = rendering_info.depth_attachment(da);
            }

            self.device
                .cmd_begin_rendering(recorder.command_buffer, &rendering_info);
            self.device.cmd_end_rendering(recorder.command_buffer);

            if !color_texture.owned {
                let barrier_to_post = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(vk::AccessFlags2::empty())
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(color_texture.image)
                    .subresource_range(color_subresource);

                let post_barriers = [barrier_to_post];
                let post_dep = vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
                self.device
                    .cmd_pipeline_barrier2(recorder.command_buffer, &post_dep);

                color_texture.transition_layout(vk::ImageLayout::PRESENT_SRC_KHR);
            }
            if let Some((depth_texture, _)) = depth {
                depth_texture.transition_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
            }
        }
    }

    pub fn cmd_clear_and_render(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        clear_color: [f32; 4],
    ) -> Result<(), HalError> {
        self.cmd_clear_and_render_internal(recorder, texture, None, clear_color);
        Ok(())
    }

    pub fn cmd_clear_and_render_with_depth(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth_texture: &VulkanTexture,
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), HalError> {
        self.cmd_clear_and_render_internal(
            recorder,
            color_texture,
            Some((depth_texture, clear_depth)),
            clear_color,
        );
        Ok(())
    }

    /// Draws with the default state: no MSAA resolve target and no vertex buffers bound.
    fn cmd_draw_internal(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth: Option<(&VulkanTexture, crate::DepthLoadOp)>,
        pipeline: vk::Pipeline,
        draw_cmd: DrawCommand<'_>,
        push_constants: &[u8],
        color_load_op: crate::ColorLoadOp,
    ) -> Result<(), HalError> {
        self.cmd_draw_internal_ext(
            recorder,
            color_texture,
            depth,
            None,
            pipeline,
            draw_cmd,
            &[],
            push_constants,
            color_load_op,
        )
    }

    /// Full draw path used by the high level API, with optional MSAA resolve and vertex buffers.
    pub fn cmd_draw_with_ops_ex(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        resolve_texture: Option<&VulkanTexture>,
        depth: Option<(&VulkanTexture, crate::DepthLoadOp)>,
        pipeline: vk::Pipeline,
        draw_cmd: DrawCommand<'_>,
        vertex_buffers: &[(vk::Buffer, u64)],
        push_constants: &[u8],
        color_load_op: crate::ColorLoadOp,
    ) -> Result<(), HalError> {
        self.cmd_draw_internal_ext(
            recorder,
            color_texture,
            depth,
            resolve_texture,
            pipeline,
            draw_cmd,
            vertex_buffers,
            push_constants,
            color_load_op,
        )
    }

    fn cmd_draw_internal_ext(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth: Option<(&VulkanTexture, crate::DepthLoadOp)>,
        resolve_texture: Option<&VulkanTexture>,
        pipeline: vk::Pipeline,
        draw_cmd: DrawCommand<'_>,
        vertex_buffers: &[(vk::Buffer, u64)],
        push_constants: &[u8],
        color_load_op: crate::ColorLoadOp,
    ) -> Result<(), HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        if push_constants.len() > bindless::PUSH_CONSTANT_SIZE as usize {
            return Err(HalError::CommandError(format!(
                "Push constants size {} exceeds maximum allowed size of {}",
                push_constants.len(),
                bindless::PUSH_CONSTANT_SIZE
            )));
        }

        if let Some(resolve) = resolve_texture {
            if color_texture.sample_count <= 1 {
                return Err(HalError::InvalidParameter(format!(
                    "A resolve target requires a multisampled color attachment, but the attachment has {} sample(s)",
                    color_texture.sample_count
                )));
            }
            if resolve.sample_count != 1 {
                return Err(HalError::InvalidParameter(format!(
                    "The resolve target must be single sampled, but it has {} samples",
                    resolve.sample_count
                )));
            }
            if resolve.format != color_texture.format {
                return Err(HalError::InvalidParameter(format!(
                    "Resolve target format {} does not match attachment format {}",
                    resolve.format.as_raw(),
                    color_texture.format.as_raw()
                )));
            }
            if resolve.width != color_texture.width || resolve.height != color_texture.height {
                return Err(HalError::InvalidParameter(format!(
                    "Resolve target is {}x{} but the attachment is {}x{}",
                    resolve.width, resolve.height, color_texture.width, color_texture.height
                )));
            }
        }

        unsafe {
            let color_subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(color_texture.mip_levels.max(1))
                .base_array_layer(0)
                .layer_count(1);

            let old_color_layout =
                color_texture.transition_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let (src_stage2, src_access2) = match old_color_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                _ => (
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    vk::AccessFlags2::empty(),
                ),
            };

            let color_barrier_to_attach = vk::ImageMemoryBarrier2::default()
                .old_layout(old_color_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_stage_mask(src_stage2)
                .src_access_mask(src_access2)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource);

            let mut barriers = vec![color_barrier_to_attach];

            let mut depth_attachment_val = None;
            if let Some((depth_texture, depth_load_op)) = depth {
                let depth_subresource = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(depth_texture.mip_levels.max(1))
                    .base_array_layer(0)
                    .layer_count(1);

                let old_depth_layout =
                    depth_texture.transition_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);

                let (depth_src_stage2, depth_src_access2) = match old_depth_layout {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                    ),
                };

                let depth_barrier_to_attach = vk::ImageMemoryBarrier2::default()
                    .old_layout(old_depth_layout)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .src_stage_mask(depth_src_stage2)
                    .src_access_mask(depth_src_access2)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .dst_access_mask(
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(depth_texture.image)
                    .subresource_range(depth_subresource);

                barriers.push(depth_barrier_to_attach);

                let (depth_load_op_vk, clear_depth_val) = match depth_load_op {
                    crate::LoadOp::Clear(d) => (
                        vk::AttachmentLoadOp::CLEAR,
                        vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: d,
                                stencil: 0,
                            },
                        },
                    ),
                    crate::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
                    crate::LoadOp::DontCare => {
                        (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default())
                    }
                };

                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_texture.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(depth_load_op_vk)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear_depth_val);
                depth_attachment_val = Some(depth_attachment);
            }

            if let Some(resolve) = resolve_texture {
                let old_layout =
                    resolve.transition_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                let (src_stage, src_access) = match old_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                    ),
                };
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .old_layout(old_layout)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(resolve.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        ),
                );
            }

            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            self.device
                .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);

            let (color_load_op_vk, clear_color_value) = match color_load_op {
                crate::LoadOp::Clear(c) => (
                    vk::AttachmentLoadOp::CLEAR,
                    vk::ClearValue {
                        color: vk::ClearColorValue { float32: c },
                    },
                ),
                crate::LoadOp::Load => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
                crate::LoadOp::DontCare => {
                    (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default())
                }
            };

            let mut color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(color_texture.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(color_load_op_vk)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_color_value);

            if let Some(resolve) = resolve_texture {
                color_attachment = color_attachment
                    .resolve_image_view(resolve.view)
                    .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            }

            let color_attachments = [color_attachment];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: color_texture.width,
                    height: color_texture.height,
                },
            };

            let mut rendering_info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(&color_attachments);

            if let Some(ref da) = depth_attachment_val {
                rendering_info = rendering_info.depth_attachment(da);
            }

            self.device
                .cmd_begin_rendering(recorder.command_buffer, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: color_texture.width as f32,
                height: color_texture.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device
                .cmd_set_viewport(recorder.command_buffer, 0, &[viewport]);

            let scissor = render_area;
            self.device
                .cmd_set_scissor(recorder.command_buffer, 0, &[scissor]);

            self.device.cmd_bind_pipeline(
                recorder.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                recorder.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                heap.pipeline_layout,
                0,
                &[heap.descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                let mut aligned_pc = [0u8; bindless::PUSH_CONSTANT_SIZE as usize];
                aligned_pc[..push_constants.len()].copy_from_slice(push_constants);
                let aligned_len = ((push_constants.len() + 3) / 4) * 4;

                self.device.cmd_push_constants(
                    recorder.command_buffer,
                    heap.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &aligned_pc[..aligned_len],
                );
            }

            if !vertex_buffers.is_empty() {
                let buffers: Vec<vk::Buffer> = vertex_buffers.iter().map(|(b, _)| *b).collect();
                let offsets: Vec<u64> = vertex_buffers.iter().map(|(_, o)| *o).collect();
                self.device
                    .cmd_bind_vertex_buffers(recorder.command_buffer, 0, &buffers, &offsets);
            }

            match draw_cmd {
                DrawCommand::Array {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    self.device.cmd_draw(
                        recorder.command_buffer,
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    );
                }
                DrawCommand::Indexed {
                    index_buffer,
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                    index_type,
                } => {
                    self.device.cmd_bind_index_buffer(
                        recorder.command_buffer,
                        index_buffer.buffer,
                        0,
                        index_type,
                    );
                    self.device.cmd_draw_indexed(
                        recorder.command_buffer,
                        index_count,
                        instance_count,
                        first_index,
                        vertex_offset,
                        first_instance,
                    );
                }
                DrawCommand::Indirect {
                    indirect_buffer,
                    offset,
                    draw_count,
                    stride,
                } => {
                    self.device.cmd_draw_indirect(
                        recorder.command_buffer,
                        indirect_buffer.buffer,
                        offset,
                        draw_count,
                        stride,
                    );
                }
                DrawCommand::IndexedIndirect {
                    index_buffer,
                    indirect_buffer,
                    offset,
                    draw_count,
                    stride,
                    index_type,
                } => {
                    self.device.cmd_bind_index_buffer(
                        recorder.command_buffer,
                        index_buffer.buffer,
                        0,
                        index_type,
                    );
                    self.device.cmd_draw_indexed_indirect(
                        recorder.command_buffer,
                        indirect_buffer.buffer,
                        offset,
                        draw_count,
                        stride,
                    );
                }
            }

            self.device.cmd_end_rendering(recorder.command_buffer);

            if !color_texture.owned {
                let barrier_to_post = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(vk::AccessFlags2::empty())
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(color_texture.image)
                    .subresource_range(color_subresource);

                let post_barriers = [barrier_to_post];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
                self.device
                    .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);

                color_texture.transition_layout(vk::ImageLayout::PRESENT_SRC_KHR);
            }
            if let Some((depth_texture, _)) = depth {
                depth_texture.transition_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
            }

            if let Some(resolve) = resolve_texture {
                // A resolve target that was not created with sampled usage may not be moved to
                // SHADER_READ_ONLY_OPTIMAL; leaving it as a color attachment is valid.
                let new_layout = if !resolve.owned {
                    Some(vk::ImageLayout::PRESENT_SRC_KHR)
                } else if resolve.supports_sampled {
                    Some(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                } else {
                    None
                };

                let Some(new_layout) = new_layout else {
                    return Ok(());
                };

                let dst_stage = if resolve.owned {
                    vk::PipelineStageFlags2::FRAGMENT_SHADER
                        | vk::PipelineStageFlags2::COMPUTE_SHADER
                } else {
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE
                };
                let dst_access = if resolve.owned {
                    vk::AccessFlags2::SHADER_READ
                } else {
                    vk::AccessFlags2::empty()
                };
                let barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(new_layout)
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(resolve.image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                let dep = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));
                self.device
                    .cmd_pipeline_barrier2(recorder.command_buffer, &dep);
                resolve.transition_layout(new_layout);
            }
        }
        Ok(())
    }

    pub fn cmd_draw(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        vertex_count: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) -> Result<(), HalError> {
        self.cmd_draw_internal(
            recorder,
            texture,
            None,
            pipeline,
            DrawCommand::Array {
                vertex_count,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            push_constants,
            crate::LoadOp::Clear(clear_color),
        )
    }

    pub fn cmd_draw_instanced(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) -> Result<(), HalError> {
        self.cmd_draw_internal(
            recorder,
            texture,
            None,
            pipeline,
            DrawCommand::Array {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            },
            push_constants,
            crate::LoadOp::Clear(clear_color),
        )
    }

    pub fn cmd_draw_triangle(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        clear_color: [f32; 4],
    ) -> Result<(), HalError> {
        self.cmd_draw(recorder, texture, pipeline, 3, &[], clear_color)
    }

    pub fn cmd_draw_with_depth(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth_texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        vertex_count: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), HalError> {
        self.cmd_draw_internal(
            recorder,
            color_texture,
            Some((depth_texture, crate::LoadOp::Clear(clear_depth))),
            pipeline,
            DrawCommand::Array {
                vertex_count,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            push_constants,
            crate::LoadOp::Clear(clear_color),
        )
    }

    pub fn cmd_draw_mrt(
        &self,
        recorder: &VulkanCommandRecorder,
        color_textures: &[&VulkanTexture],
        depth_texture: Option<&VulkanTexture>,
        pipeline: vk::Pipeline,
        vertex_buffers: &[(vk::Buffer, u64)],
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: Option<f32>,
    ) -> Result<(), HalError> {
        if color_textures.is_empty() {
            return Err(HalError::InvalidParameter(
                "color_textures cannot be empty for MRT".to_string(),
            ));
        }

        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        if push_constants.len() > bindless::PUSH_CONSTANT_SIZE as usize {
            return Err(HalError::CommandError(format!(
                "Push constants size {} exceeds maximum allowed size of {}",
                push_constants.len(),
                bindless::PUSH_CONSTANT_SIZE
            )));
        }

        unsafe {
            let mut barriers = Vec::with_capacity(color_textures.len() + 1);
            let mut color_subresources = Vec::with_capacity(color_textures.len());

            for tex in color_textures {
                let subresource = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(tex.mip_levels.max(1))
                    .base_array_layer(0)
                    .layer_count(1);
                color_subresources.push(subresource);

                let old_layout = tex.transition_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
                let (src_stage2, src_access2) = match old_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                        vk::PipelineStageFlags2::TRANSFER,
                        vk::AccessFlags2::TRANSFER_READ,
                    ),
                    _ => (
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                    ),
                };

                let barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_stage_mask(src_stage2)
                    .src_access_mask(src_access2)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(tex.image)
                    .subresource_range(subresource);

                barriers.push(barrier);
            }

            let mut depth_attachment_val = None;
            if let (Some(depth_tex), Some(depth_val)) = (depth_texture, clear_depth) {
                let depth_subresource = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let old_depth_layout =
                    depth_tex.transition_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
                let (depth_src_stage2, depth_src_access2) = match old_depth_layout {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                    ),
                };

                let depth_barrier = vk::ImageMemoryBarrier2::default()
                    .old_layout(old_depth_layout)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .src_stage_mask(depth_src_stage2)
                    .src_access_mask(depth_src_access2)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .dst_access_mask(
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(depth_tex.image)
                    .subresource_range(depth_subresource);

                barriers.push(depth_barrier);

                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_tex.view)
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: depth_val,
                            stencil: 0,
                        },
                    });
                depth_attachment_val = Some(depth_attachment);
            }

            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            self.device
                .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear_color,
                },
            };

            let attachments: Vec<vk::RenderingAttachmentInfo> = color_textures
                .iter()
                .map(|tex| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(tex.view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(clear_value)
                })
                .collect();

            let first_tex = color_textures[0];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: first_tex.width,
                    height: first_tex.height,
                },
            };

            let mut rendering_info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(&attachments);

            if let Some(ref da) = depth_attachment_val {
                rendering_info = rendering_info.depth_attachment(da);
            }

            self.device
                .cmd_begin_rendering(recorder.command_buffer, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: first_tex.width as f32,
                height: first_tex.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device
                .cmd_set_viewport(recorder.command_buffer, 0, &[viewport]);
            self.device
                .cmd_set_scissor(recorder.command_buffer, 0, &[render_area]);

            self.device.cmd_bind_pipeline(
                recorder.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                recorder.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                heap.pipeline_layout,
                0,
                &[heap.descriptor_set],
                &[],
            );

            if !push_constants.is_empty() {
                let mut aligned_pc = [0u8; bindless::PUSH_CONSTANT_SIZE as usize];
                aligned_pc[..push_constants.len()].copy_from_slice(push_constants);
                let aligned_len = ((push_constants.len() + 3) / 4) * 4;

                self.device.cmd_push_constants(
                    recorder.command_buffer,
                    heap.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &aligned_pc[..aligned_len],
                );
            }

            if !vertex_buffers.is_empty() {
                let buffers: Vec<vk::Buffer> = vertex_buffers.iter().map(|(b, _)| *b).collect();
                let offsets: Vec<u64> = vertex_buffers.iter().map(|(_, o)| *o).collect();
                self.device
                    .cmd_bind_vertex_buffers(recorder.command_buffer, 0, &buffers, &offsets);
            }

            self.device.cmd_draw(
                recorder.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );

            self.device.cmd_end_rendering(recorder.command_buffer);

            let mut post_barriers = Vec::new();
            for (tex, &subres) in color_textures.iter().zip(color_subresources.iter()) {
                if !tex.owned {
                    let barrier = vk::ImageMemoryBarrier2::default()
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                        .dst_access_mask(vk::AccessFlags2::empty())
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(tex.image)
                        .subresource_range(subres);
                    post_barriers.push(barrier);
                    tex.transition_layout(vk::ImageLayout::PRESENT_SRC_KHR);
                }
            }

            if !post_barriers.is_empty() {
                let post_dep = vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
                self.device
                    .cmd_pipeline_barrier2(recorder.command_buffer, &post_dep);
            }
        }
        Ok(())
    }

    pub fn cmd_draw_indexed(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        index_buffer: &VulkanBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        index_type: vk::IndexType,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) -> Result<(), HalError> {
        self.cmd_draw_internal(
            recorder,
            texture,
            None,
            pipeline,
            DrawCommand::Indexed {
                index_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
                index_type,
            },
            push_constants,
            crate::LoadOp::Clear(clear_color),
        )
    }

    pub fn cmd_draw_indexed_with_depth(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth_texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        index_buffer: &VulkanBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        index_type: vk::IndexType,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), HalError> {
        self.cmd_draw_internal(
            recorder,
            color_texture,
            Some((depth_texture, crate::LoadOp::Clear(clear_depth))),
            pipeline,
            DrawCommand::Indexed {
                index_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
                index_type,
            },
            push_constants,
            crate::LoadOp::Clear(clear_color),
        )
    }

    pub fn cmd_draw_indirect(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        indirect_buffer: &VulkanBuffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) -> Result<(), HalError> {
        self.cmd_draw_internal(
            recorder,
            texture,
            None,
            pipeline,
            DrawCommand::Indirect {
                indirect_buffer,
                offset,
                draw_count,
                stride,
            },
            push_constants,
            crate::LoadOp::Clear(clear_color),
        )
    }

    pub fn cmd_draw_indirect_with_depth(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth_texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        indirect_buffer: &VulkanBuffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), HalError> {
        self.cmd_draw_internal(
            recorder,
            color_texture,
            Some((depth_texture, crate::LoadOp::Clear(clear_depth))),
            pipeline,
            DrawCommand::Indirect {
                indirect_buffer,
                offset,
                draw_count,
                stride,
            },
            push_constants,
            crate::LoadOp::Clear(clear_color),
        )
    }

    pub fn cmd_draw_with_ops(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth: Option<(&VulkanTexture, crate::DepthLoadOp)>,
        pipeline: vk::Pipeline,
        draw_cmd: DrawCommand<'_>,
        push_constants: &[u8],
        color_load_op: crate::ColorLoadOp,
    ) -> Result<(), HalError> {
        self.cmd_draw_internal(
            recorder,
            color_texture,
            depth,
            pipeline,
            draw_cmd,
            push_constants,
            color_load_op,
        )
    }

    pub fn cmd_dispatch_indirect(
        &self,
        recorder: &VulkanCommandRecorder,
        pipeline: vk::Pipeline,
        indirect_buffer: &VulkanBuffer,
        offset: u64,
        push_constants: &[u8],
    ) -> Result<(), HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        if push_constants.len() > bindless::PUSH_CONSTANT_SIZE as usize {
            return Err(HalError::CommandError(format!(
                "Push constants size {} exceeds maximum allowed size of {}",
                push_constants.len(),
                bindless::PUSH_CONSTANT_SIZE
            )));
        }

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
                let mut aligned_pc = [0u8; bindless::PUSH_CONSTANT_SIZE as usize];
                aligned_pc[..push_constants.len()].copy_from_slice(push_constants);
                let aligned_len = ((push_constants.len() + 3) / 4) * 4;

                self.device.cmd_push_constants(
                    recorder.command_buffer,
                    heap.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &aligned_pc[..aligned_len],
                );
            }
            self.device.cmd_dispatch_indirect(
                recorder.command_buffer,
                indirect_buffer.buffer,
                offset,
            );
        }
        Ok(())
    }

    pub fn cmd_copy_buffer_to_buffer(
        &self,
        recorder: &VulkanCommandRecorder,
        src: &VulkanBuffer,
        dst: &VulkanBuffer,
        size: u64,
        src_offset: u64,
        dst_offset: u64,
    ) {
        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);
        unsafe {
            self.device
                .cmd_copy_buffer(recorder.command_buffer, src.buffer, dst.buffer, &[region]);
        }
    }

    pub fn cmd_copy_buffer_to_image(
        &self,
        recorder: &VulkanCommandRecorder,
        buffer: &VulkanBuffer,
        texture: &VulkanTexture,
    ) {
        unsafe {
            let mut lock = match texture.layout.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let old_layout = *lock;

            if old_layout != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(Self::image_aspect(texture))
                    .base_mip_level(0)
                    .level_count(texture.mip_levels.max(1))
                    .base_array_layer(0)
                    .layer_count(Self::copy_layer_count(texture));

                let (src_stage2, src_access2) = match old_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                        vk::PipelineStageFlags2::FRAGMENT_SHADER
                            | vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_READ,
                    ),
                    _ => (
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                    ),
                };

                let barrier_to_dst = vk::ImageMemoryBarrier2::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_stage_mask(src_stage2)
                    .src_access_mask(src_access2)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(subresource_range);

                let barriers = [barrier_to_dst];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                self.device
                    .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);

                *lock = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            }

            let subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(Self::image_aspect(texture))
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(Self::copy_layer_count(texture));

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(subresource)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: texture.width,
                    height: texture.height,
                    depth: texture.depth.max(1),
                });

            self.device.cmd_copy_buffer_to_image(
                recorder.command_buffer,
                buffer.buffer,
                texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
        }
    }

    pub fn write_texture(&self, texture: &VulkanTexture, data: &[u8]) -> Result<(), HalError> {
        if texture.sample_count > 1 {
            return Err(HalError::InvalidParameter(format!(
                "Cannot upload to a multisampled texture ({} samples)",
                texture.sample_count
            )));
        }
        if let Some(format) = crate::TextureFormat::from_vk(texture.format) {
            if format.is_depth() {
                return Err(HalError::InvalidParameter(
                    "write_texture does not support depth formats".to_string(),
                ));
            }
            let slices = texture.depth.max(1) as u64
                * if texture.depth > 1 {
                    1
                } else {
                    texture.array_layers.max(1) as u64
                };
            let required = (texture.width as u64)
                .saturating_mul(texture.height as u64)
                .saturating_mul(format.bytes_per_pixel() as u64)
                .saturating_mul(slices);
            if (data.len() as u64) < required {
                return Err(HalError::InvalidParameter(format!(
                    "Texture upload of {} bytes is smaller than the {} bytes required for a {}x{} {:?} texture",
                    data.len(),
                    required,
                    texture.width,
                    texture.height,
                    format
                )));
            }
        }

        let mut staging_guard = match self.staging_ring.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if let Some(ref mut ring) = *staging_guard {
            let data_len = data.len() as u64;
            if data_len <= ring.capacity {
                let (offset, aligned_size) = ring.suballocate(&self.device, data_len)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        ring.mapped_ptr.add(offset as usize),
                        data.len(),
                    );
                }

                let recorder = self.create_command_recorder()?;

                unsafe {
                    let subresource_range = vk::ImageSubresourceRange::default()
                        .aspect_mask(Self::image_aspect(texture))
                        .base_mip_level(0)
                        .level_count(texture.mip_levels.max(1))
                        .base_array_layer(0)
                        .layer_count(Self::copy_layer_count(texture));

                    let old_layout = texture.current_layout();
                    let (src_stage2, src_access2) = match old_layout {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                            vk::PipelineStageFlags2::FRAGMENT_SHADER
                                | vk::PipelineStageFlags2::COMPUTE_SHADER,
                            vk::AccessFlags2::SHADER_READ,
                        ),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                            vk::PipelineStageFlags2::TRANSFER,
                            vk::AccessFlags2::TRANSFER_WRITE,
                        ),
                        _ => (
                            vk::PipelineStageFlags2::TOP_OF_PIPE,
                            vk::AccessFlags2::empty(),
                        ),
                    };

                    let barrier_to_dst = vk::ImageMemoryBarrier2::default()
                        .old_layout(old_layout)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_stage_mask(src_stage2)
                        .src_access_mask(src_access2)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(texture.image)
                        .subresource_range(subresource_range);

                    let pre_barriers = [barrier_to_dst];
                    let pre_dep =
                        vk::DependencyInfo::default().image_memory_barriers(&pre_barriers);
                    self.device
                        .cmd_pipeline_barrier2(recorder.command_buffer, &pre_dep);

                    let subresource_layers = vk::ImageSubresourceLayers::default()
                        .aspect_mask(Self::image_aspect(texture))
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(Self::copy_layer_count(texture));

                    let copy_region = vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(subresource_layers)
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width: texture.width,
                            height: texture.height,
                            depth: texture.depth.max(1),
                        });

                    self.device.cmd_copy_buffer_to_image(
                        recorder.command_buffer,
                        ring.buffer,
                        texture.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy_region],
                    );

                    let barrier_to_read = vk::ImageMemoryBarrier2::default()
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::FRAGMENT_SHADER
                                | vk::PipelineStageFlags2::COMPUTE_SHADER,
                        )
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(texture.image)
                        .subresource_range(subresource_range);

                    let post_barriers = [barrier_to_read];
                    let post_dep =
                        vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
                    self.device
                        .cmd_pipeline_barrier2(recorder.command_buffer, &post_dep);

                    texture.transition_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

                    self.device
                        .end_command_buffer(recorder.command_buffer)
                        .map_err(|e| HalError::CommandError(e.to_string()))?;
                }

                // Submit asynchronously, signaling timeline semaphore
                ring.current_timeline_value += 1;
                let timeline_val = ring.current_timeline_value;

                let cmd_buffers = [recorder.command_buffer];
                let signal_sems = [ring.timeline_semaphore];
                let signal_vals = [timeline_val];

                let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                    .signal_semaphore_values(&signal_vals);

                let submit_info = vk::SubmitInfo::default()
                    .command_buffers(&cmd_buffers)
                    .signal_semaphores(&signal_sems)
                    .push_next(&mut timeline_info);

                {
                    let queue = match self.queue.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    unsafe {
                        self.device
                            .queue_submit(*queue, &[submit_info], vk::Fence::null())
                            .map_err(|e| HalError::CommandError(e.to_string()))?;
                    }
                }

                ring.pending_batches
                    .push_back(super::staging::StagingBatch {
                        offset,
                        size: aligned_size,
                        timeline_value: timeline_val,
                    });
                ring.pending_recorders.push((timeline_val, recorder));

                return Ok(());
            }
        }
        drop(staging_guard);

        // Fallback for oversized uploads or if ring is uninitialized
        let staging_buf = self.create_buffer(&crate::BufferDesc {
            label: Some("staging_texture_upload_fallback"),
            size: data.len() as u64,
            usage: crate::BufferUsage::TRANSFER_SRC,
            memory_location: crate::MemoryLocation::CpuToGpu,
        })?;

        self.write_buffer(&staging_buf, 0, data)?;

        let recorder = self.create_command_recorder()?;
        self.cmd_copy_buffer_to_image(&recorder, &staging_buf, texture);

        unsafe {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(Self::image_aspect(texture))
                .base_mip_level(0)
                .level_count(texture.mip_levels.max(1))
                .base_array_layer(0)
                .layer_count(Self::copy_layer_count(texture));

            let barrier_to_read = vk::ImageMemoryBarrier2::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::FRAGMENT_SHADER
                        | vk::PipelineStageFlags2::COMPUTE_SHADER,
                )
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource_range);

            let post_barriers = [barrier_to_read];
            let post_dep = vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
            self.device
                .cmd_pipeline_barrier2(recorder.command_buffer, &post_dep);
            texture.transition_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }

        self.submit_and_wait(recorder)?;
        self.destroy_buffer(staging_buf);

        Ok(())
    }

    fn image_aspect(texture: &VulkanTexture) -> vk::ImageAspectFlags {
        match texture.format {
            vk::Format::D16_UNORM | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
            vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::COLOR,
        }
    }

    /// Layers a whole-image copy touches: 3D images copy every slice of layer 0, arrays copy
    /// every layer of slice 0.
    fn copy_layer_count(texture: &VulkanTexture) -> u32 {
        if texture.depth > 1 {
            1
        } else {
            texture.array_layers.max(1)
        }
    }

    /// Transitions the whole image and returns the previous layout.
    #[allow(clippy::too_many_arguments)]
    fn cmd_transition_full(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        new_layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) -> vk::ImageLayout {
        let old_layout = texture.transition_layout(new_layout);
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(Self::image_aspect(texture))
            .base_mip_level(0)
            .level_count(texture.mip_levels.max(1))
            .base_array_layer(0)
            .layer_count(Self::copy_layer_count(texture));
        let barrier = vk::ImageMemoryBarrier2::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(texture.image)
            .subresource_range(range);
        let barriers = [barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe {
            self.device
                .cmd_pipeline_barrier2(recorder.command_buffer, &dep);
        }
        old_layout
    }

    /// Uploads one mip level. `row_pitch` is in bytes; `None` means tightly packed rows.
    pub fn write_texture_mip(
        &self,
        texture: &VulkanTexture,
        mip_level: u32,
        data: &[u8],
        row_pitch: Option<u32>,
    ) -> Result<(), HalError> {
        if texture.sample_count > 1 {
            return Err(HalError::InvalidParameter(
                "Cannot upload to a multisampled texture".to_string(),
            ));
        }
        if mip_level >= texture.mip_levels {
            return Err(HalError::InvalidParameter(format!(
                "Mip level {mip_level} is out of range for a texture with {} levels",
                texture.mip_levels
            )));
        }
        let format = crate::TextureFormat::from_vk(texture.format).ok_or_else(|| {
            HalError::InvalidParameter("Unsupported texture format for upload".to_string())
        })?;
        if format == crate::TextureFormat::Depth32Float {
            return Err(HalError::InvalidParameter(
                "Texture uploads do not support depth formats".to_string(),
            ));
        }

        let mip_width = (texture.width >> mip_level).max(1);
        let mip_height = (texture.height >> mip_level).max(1);
        let mip_depth = if texture.depth > 1 {
            (texture.depth >> mip_level).max(1)
        } else {
            1
        };
        let layer_count = Self::copy_layer_count(texture);
        let slice_count = mip_depth as u64 * layer_count as u64;
        let bytes_per_pixel = format.bytes_per_pixel();
        let packed_row = mip_width * bytes_per_pixel;
        let row_bytes = row_pitch.unwrap_or(packed_row);
        if row_bytes < packed_row {
            return Err(HalError::InvalidParameter(format!(
                "Row pitch {row_bytes} is smaller than the {packed_row} byte packed row"
            )));
        }
        let slice_bytes = row_bytes as u64 * (mip_height as u64 - 1) + packed_row as u64;
        let required = slice_bytes * slice_count;
        if (data.len() as u64) < required {
            return Err(HalError::InvalidParameter(format!(
                "Mip {mip_level} needs {required} bytes, but {} were given",
                data.len()
            )));
        }

        let staging = self.create_buffer(&crate::BufferDesc {
            label: Some("texture_mip_upload"),
            size: required,
            usage: crate::BufferUsage::TRANSFER_SRC,
            memory_location: crate::MemoryLocation::CpuToGpu,
        })?;
        if let Err(e) = self.write_buffer(&staging, 0, &data[..required as usize]) {
            self.destroy_buffer(staging);
            return Err(e);
        }

        let recorder = self.create_command_recorder()?;
        self.cmd_transition_full(
            &recorder,
            texture,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );

        let tightly_packed = row_bytes == packed_row;
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(if tightly_packed {
                0
            } else {
                row_bytes / bytes_per_pixel
            })
            .buffer_image_height(if tightly_packed { 0 } else { mip_height })
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(Self::image_aspect(texture))
                    .mip_level(mip_level)
                    .base_array_layer(0)
                    .layer_count(layer_count),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: mip_width,
                height: mip_height,
                depth: mip_depth,
            });

        unsafe {
            self.device.cmd_copy_buffer_to_image(
                recorder.command_buffer,
                staging.buffer,
                texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        let final_layout = if texture.supports_sampled {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else if texture
            .usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::GENERAL
        };
        let (dst_stage, dst_access) = if texture.supports_sampled {
            (
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            )
        } else {
            (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            )
        };
        self.cmd_transition_full(
            &recorder,
            texture,
            final_layout,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage,
            dst_access,
        );

        let result = self.submit_and_wait(recorder);
        self.destroy_buffer(staging);
        result
    }

    /// Generates the mip chain by blitting each level from the previous one. The texture must
    /// have been created with `TEXTURE_BINDING | COPY_SRC | COPY_DST`.
    pub fn generate_mipmaps(&self, texture: &VulkanTexture) -> Result<(), HalError> {
        if texture.mip_levels <= 1 {
            return Ok(());
        }
        if texture.sample_count > 1 {
            return Err(HalError::InvalidParameter(
                "Mipmap generation does not support multisampled textures".to_string(),
            ));
        }
        let required = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        if !texture.usage_flags.contains(required) {
            return Err(HalError::InvalidParameter(
                "Mipmap generation requires TEXTURE_BINDING | COPY_SRC | COPY_DST".to_string(),
            ));
        }

        let recorder = self.create_command_recorder()?;
        let aspect = Self::image_aspect(texture);

        // GENERAL is legal for both the blit source and destination, which keeps the tracked
        // image layout consistent across levels.
        self.cmd_transition_full(
            &recorder,
            texture,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
        );

        let layer_count = Self::copy_layer_count(texture);
        for level in 1..texture.mip_levels {
            let mut barriers = Vec::with_capacity(2);

            // If level > 1, the previous blit wrote to (level - 1). Synchronize that write for TRANSFER_READ.
            if level > 1 {
                let prev_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .base_mip_level(level - 1)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(layer_count);
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .image(texture.image)
                        .subresource_range(prev_range),
                );
            }

            let dst_range = vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(level)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(layer_count);
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .image(texture.image)
                    .subresource_range(dst_range),
            );
            let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);

            let src_extent = vk::Extent3D {
                width: (texture.width >> (level - 1)).max(1),
                height: (texture.height >> (level - 1)).max(1),
                depth: if texture.depth > 1 {
                    (texture.depth >> (level - 1)).max(1)
                } else {
                    1
                },
            };
            let dst_extent = vk::Extent3D {
                width: (texture.width >> level).max(1),
                height: (texture.height >> level).max(1),
                depth: if texture.depth > 1 {
                    (texture.depth >> level).max(1)
                } else {
                    1
                },
            };

            let blit = vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(aspect)
                        .mip_level(level - 1)
                        .base_array_layer(0)
                        .layer_count(layer_count),
                )
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_extent.width as i32,
                        y: src_extent.height as i32,
                        z: src_extent.depth as i32,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(aspect)
                        .mip_level(level)
                        .base_array_layer(0)
                        .layer_count(layer_count),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst_extent.width as i32,
                        y: dst_extent.height as i32,
                        z: dst_extent.depth as i32,
                    },
                ]);

            unsafe {
                self.device
                    .cmd_pipeline_barrier2(recorder.command_buffer, &dep);
                self.device.cmd_blit_image(
                    recorder.command_buffer,
                    texture.image,
                    vk::ImageLayout::GENERAL,
                    texture.image,
                    vk::ImageLayout::GENERAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }
        }

        self.cmd_transition_full(
            &recorder,
            texture,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );

        self.submit_and_wait(recorder)
    }

    pub fn cmd_copy_image_to_buffer(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        buffer: &VulkanBuffer,
    ) {
        unsafe {
            let mut lock = match texture.layout.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let old_layout = *lock;

            if old_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(Self::image_aspect(texture))
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(Self::copy_layer_count(texture));

                let (src_stage2, src_access2) = match old_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                    ),
                };

                let barrier_to_src = vk::ImageMemoryBarrier2::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_stage_mask(src_stage2)
                    .src_access_mask(src_access2)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(subresource_range);

                let barriers = [barrier_to_src];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                self.device
                    .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);

                *lock = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            }

            let subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(Self::image_aspect(texture))
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(Self::copy_layer_count(texture));

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(subresource)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: texture.width,
                    height: texture.height,
                    depth: texture.depth.max(1),
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

    fn submit_batches_internal(
        &self,
        recorders: Vec<VulkanCommandRecorder>,
        wait_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        signal_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        sync_fence: Option<vk::Fence>,
        queue_family: u32,
    ) -> Result<(), HalError> {
        if recorders.is_empty() {
            return Ok(());
        }

        unsafe {
            let mut vk_cmd_buffers = Vec::with_capacity(recorders.len());
            for r in &recorders {
                self.device
                    .end_command_buffer(r.command_buffer)
                    .map_err(|e| HalError::CommandError(e.to_string()))?;
                vk_cmd_buffers.push(r.command_buffer);
            }

            let mut vk_wait_sems = Vec::with_capacity(wait_semaphores.len() + 1);
            let mut vk_wait_values = Vec::with_capacity(wait_semaphores.len() + 1);
            let mut vk_wait_stages = Vec::with_capacity(wait_semaphores.len() + 1);
            for (sem, val) in wait_semaphores {
                vk_wait_sems.push(sem.semaphore);
                vk_wait_values.push(*val);
                vk_wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }

            let staging_guard = match self.staging_ring.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if let Some(ref ring) = *staging_guard {
                if ring.current_timeline_value > 0 {
                    vk_wait_sems.push(ring.timeline_semaphore);
                    vk_wait_values.push(ring.current_timeline_value);
                    vk_wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
                }
            }
            drop(staging_guard);

            let mut vk_signal_sems = Vec::with_capacity(signal_semaphores.len() + 1);
            let mut vk_signal_values = Vec::with_capacity(signal_semaphores.len() + 1);
            for (sem, val) in signal_semaphores {
                vk_signal_sems.push(sem.semaphore);
                vk_signal_values.push(*val);
            }

            let fence = sync_fence.unwrap_or(vk::Fence::null());
            let next_val = {
                // The guard must cover the whole submission, including the timeline value
                // allocation, or concurrent submits can signal out of order and races the queue.
                let queue_mutex = if queue_family == self.queue_family_index {
                    &self.queue
                } else {
                    self.compute_queue.as_ref().ok_or_else(|| {
                        HalError::CommandError(
                            "No compute queue is available on this device".to_string(),
                        )
                    })?
                };
                let queue_guard = match queue_mutex.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                let next_val = self
                    .internal_frame_counter
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                    + 1;
                vk_signal_sems.push(self.internal_frame_semaphore);
                vk_signal_values.push(next_val);

                let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                    .wait_semaphore_values(&vk_wait_values)
                    .signal_semaphore_values(&vk_signal_values);

                let submit_info = vk::SubmitInfo::default()
                    .command_buffers(&vk_cmd_buffers)
                    .wait_semaphores(&vk_wait_sems)
                    .wait_dst_stage_mask(&vk_wait_stages)
                    .signal_semaphores(&vk_signal_sems)
                    .push_next(&mut timeline_info);

                if let Err(e) = self
                    .device
                    .queue_submit(*queue_guard, &[submit_info], fence)
                {
                    return Err(HalError::CommandError(e.to_string()));
                }
                next_val
            };

            if let Some(f) = sync_fence {
                let wait_res = self.device.wait_for_fences(&[f], true, u64::MAX);
                self.device.reset_fences(&[f]).ok();
                {
                    let mut fence_cache = match self.fence_cache.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    fence_cache.push(f);
                }
                wait_res.map_err(|e| HalError::CommandError(e.to_string()))?;

                // Acquire locks in standard order: pending_async_recorders THEN recorder cache
                let current_val = self
                    .device
                    .get_semaphore_counter_value(self.internal_frame_semaphore)
                    .unwrap_or(0);
                let mut pending = match self.pending_async_recorders.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                for r in recorders {
                    let cache_mutex = self.recorder_cache_for(r.queue_family_index);
                    let mut cache = match cache_mutex.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    if cache.len() < MAX_CACHED_COMMAND_RECORDERS {
                        cache.push((r.command_pool, r.command_buffer));
                    } else {
                        self.device.destroy_command_pool(r.command_pool, None);
                    }
                }

                pending.retain_mut(|(val, async_recorders)| {
                    if *val <= current_val {
                        for r in async_recorders.drain(..) {
                            let cache_mutex = self.recorder_cache_for(r.queue_family_index);
                            let mut cache = match cache_mutex.lock() {
                                Ok(g) => g,
                                Err(p) => p.into_inner(),
                            };
                            if cache.len() < MAX_CACHED_COMMAND_RECORDERS {
                                cache.push((r.command_pool, r.command_buffer));
                            } else {
                                self.device.destroy_command_pool(r.command_pool, None);
                            }
                        }
                        false
                    } else {
                        true
                    }
                });
            } else {
                let mut pending = match self.pending_async_recorders.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                pending.push((next_val, recorders));
            }
        }
        Ok(())
    }

    pub fn submit_and_wait(&self, recorder: VulkanCommandRecorder) -> Result<(), HalError> {
        let fence = {
            let mut cache = match self.fence_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            cache.pop()
        };
        let fence = match fence {
            Some(f) => f,
            None => {
                let fence_info = vk::FenceCreateInfo::default();
                unsafe {
                    self.device.create_fence(&fence_info, None).map_err(|e| {
                        HalError::CommandError(format!("Failed to create fence: {e}"))
                    })?
                }
            }
        };
        let res = self.submit_batches_internal(
            vec![recorder],
            &[],
            &[],
            Some(fence),
            self.queue_family_index,
        );
        if res.is_err() {
            unsafe {
                self.device.reset_fences(&[fence]).ok();
                let mut cache = match self.fence_cache.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                cache.push(fence);
            }
        }
        res
    }

    pub fn submit_batches_and_wait(
        &self,
        recorders: Vec<VulkanCommandRecorder>,
    ) -> Result<(), HalError> {
        let fence = {
            let mut cache = match self.fence_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            cache.pop()
        };
        let fence = match fence {
            Some(f) => f,
            None => {
                let fence_info = vk::FenceCreateInfo::default();
                unsafe {
                    self.device.create_fence(&fence_info, None).map_err(|e| {
                        HalError::CommandError(format!("Failed to create fence: {e}"))
                    })?
                }
            }
        };
        let res =
            self.submit_batches_internal(recorders, &[], &[], Some(fence), self.queue_family_index);
        if res.is_err() {
            unsafe {
                self.device.reset_fences(&[fence]).ok();
                let mut cache = match self.fence_cache.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                cache.push(fence);
            }
        }
        res
    }

    pub fn submit_batches_async(
        &self,
        recorders: Vec<VulkanCommandRecorder>,
        wait_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        signal_semaphores: &[(&VulkanTimelineSemaphore, u64)],
    ) -> Result<(), HalError> {
        self.submit_batches_async_on(
            self.queue_family_index,
            recorders,
            wait_semaphores,
            signal_semaphores,
        )
    }

    /// Async submission on an explicit queue family, used for the async compute queue.
    pub fn submit_batches_async_on(
        &self,
        queue_family: u32,
        recorders: Vec<VulkanCommandRecorder>,
        wait_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        signal_semaphores: &[(&VulkanTimelineSemaphore, u64)],
    ) -> Result<(), HalError> {
        self.submit_batches_internal(
            recorders,
            wait_semaphores,
            signal_semaphores,
            None,
            queue_family,
        )
    }

    /// Submits a recorder to the dedicated compute queue and waits for completion.
    pub fn submit_compute_and_wait(&self, recorder: VulkanCommandRecorder) -> Result<(), HalError> {
        let family = self.compute_queue_family_index.ok_or_else(|| {
            HalError::CommandError("This adapter has no separate compute queue family".to_string())
        })?;
        let fence = {
            let mut cache = match self.fence_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            cache.pop()
        };
        let fence = match fence {
            Some(f) => f,
            None => {
                let fence_info = vk::FenceCreateInfo::default();
                unsafe {
                    self.device.create_fence(&fence_info, None).map_err(|e| {
                        HalError::CommandError(format!("Failed to create fence: {e}"))
                    })?
                }
            }
        };
        let res = self.submit_batches_internal(vec![recorder], &[], &[], Some(fence), family);
        if res.is_err() {
            unsafe {
                self.device.reset_fences(&[fence]).ok();
                let mut cache = match self.fence_cache.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                cache.push(fence);
            }
        }
        res
    }

    /// Submits several recorders to the dedicated compute queue and waits for completion.
    pub fn submit_compute_batches_and_wait(
        &self,
        recorders: Vec<VulkanCommandRecorder>,
    ) -> Result<(), HalError> {
        let family = self.compute_queue_family_index.ok_or_else(|| {
            HalError::CommandError("This adapter has no separate compute queue family".to_string())
        })?;
        let fence = {
            let mut cache = match self.fence_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            cache.pop()
        };
        let fence = match fence {
            Some(f) => f,
            None => {
                let fence_info = vk::FenceCreateInfo::default();
                unsafe {
                    self.device.create_fence(&fence_info, None).map_err(|e| {
                        HalError::CommandError(format!("Failed to create fence: {e}"))
                    })?
                }
            }
        };
        let res = self.submit_batches_internal(recorders, &[], &[], Some(fence), family);
        if res.is_err() {
            unsafe {
                self.device.reset_fences(&[fence]).ok();
                let mut cache = match self.fence_cache.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                cache.push(fence);
            }
        }
        res
    }

    /// Labels the following commands in GPU debuggers such as RenderDoc.
    pub fn cmd_begin_debug_label(
        &self,
        recorder: &VulkanCommandRecorder,
        name: &str,
        color: [f32; 4],
    ) {
        let Some(loader) = &self.debug_utils_device else {
            return;
        };
        let Ok(name) = std::ffi::CString::new(name) else {
            return;
        };
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&name)
            .color(color);
        unsafe {
            loader.cmd_begin_debug_utils_label(recorder.command_buffer, &label);
        }
    }

    /// Closes a label opened with [`VulkanDevice::cmd_begin_debug_label`].
    pub fn cmd_end_debug_label(&self, recorder: &VulkanCommandRecorder) {
        let Some(loader) = &self.debug_utils_device else {
            return;
        };
        unsafe {
            loader.cmd_end_debug_utils_label(recorder.command_buffer);
        }
    }

    pub fn cmd_memory_barrier(&self, recorder: &VulkanCommandRecorder) {
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::TRANSFER
                    | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            )
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::TRANSFER
                    | vk::PipelineStageFlags2::VERTEX_INPUT
                    | vk::PipelineStageFlags2::DRAW_INDIRECT,
            )
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::SHADER_READ);
        let dep_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier));
        unsafe {
            self.device
                .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);
        }
    }

    pub fn cmd_buffer_barrier(&self, recorder: &VulkanCommandRecorder, buffer: &VulkanBuffer) {
        let barrier = vk::BufferMemoryBarrier2::default()
            .buffer(buffer.buffer)
            .offset(0)
            .size(buffer.size)
            .src_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::TRANSFER
                    | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            )
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::TRANSFER
                    | vk::PipelineStageFlags2::VERTEX_INPUT
                    | vk::PipelineStageFlags2::HOST,
            )
            .dst_access_mask(
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::TRANSFER_READ
                    | vk::AccessFlags2::HOST_READ,
            )
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        let dep_info =
            vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&barrier));
        unsafe {
            self.device
                .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);
        }
    }
}
