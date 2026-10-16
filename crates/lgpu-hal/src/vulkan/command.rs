//! Vulkan command recording, render passes, copy operations, and queue submission.

use super::{
    bindless, VulkanBuffer, VulkanComputePipeline, VulkanDevice, VulkanTexture,
    VulkanTimelineSemaphore,
};
use crate::HalError;
use ash::vk;

pub struct VulkanCommandRecorder {
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) command_buffer: vk::CommandBuffer,
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

impl VulkanDevice {
    pub fn create_command_recorder(&self) -> Result<VulkanCommandRecorder, HalError> {
        // Harvest any completed async command recorders back into cache
        {
            let current_val = unsafe {
                self.device
                    .get_semaphore_counter_value(self.internal_frame_semaphore)
                    .unwrap_or(0)
            };
            let mut pending = self.pending_async_recorders.lock().unwrap();
            let mut cache = self.command_recorder_cache.lock().unwrap();
            pending.retain_mut(|(val, recorders)| {
                if *val <= current_val {
                    for r in recorders.drain(..) {
                        cache.push((r.command_pool, r.command_buffer));
                    }
                    false
                } else {
                    true
                }
            });
        }

        let (command_pool, command_buffer) = match self.command_recorder_cache.lock().unwrap().pop()
        {
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
                    .queue_family_index(self.queue_family_index)
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
            self.command_recorder_cache
                .lock()
                .unwrap()
                .push((command_pool, command_buffer));
            return Err(HalError::CommandError(e.to_string()));
        }

        Ok(VulkanCommandRecorder {
            command_pool,
            command_buffer,
        })
    }

    pub fn recycle_command_recorder(&self, recorder: VulkanCommandRecorder) {
        unsafe {
            let _ = self.device.reset_command_buffer(
                recorder.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            );
        }
        self.command_recorder_cache
            .lock()
            .unwrap()
            .push((recorder.command_pool, recorder.command_buffer));
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

            let (src_stage, src_access) = match old_color_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                ),
                _ => (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                ),
            };

            let color_barrier_to_attach = vk::ImageMemoryBarrier::default()
                .old_layout(old_color_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource)
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

            let mut barriers = vec![color_barrier_to_attach];
            let mut wait_dst_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let mut total_src_stage = src_stage;

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
                let (depth_src_stage, depth_src_access) = match old_depth_layout {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::AccessFlags::empty(),
                    ),
                };

                let depth_barrier_to_attach = vk::ImageMemoryBarrier::default()
                    .old_layout(old_depth_layout)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(depth_texture.image)
                    .subresource_range(depth_subresource)
                    .src_access_mask(depth_src_access)
                    .dst_access_mask(
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    );

                barriers.push(depth_barrier_to_attach);
                total_src_stage |= depth_src_stage;
                wait_dst_stage |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;

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

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                total_src_stage,
                wait_dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

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

            let barrier_to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource)
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

            color_texture.transition_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
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
            let color_subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(color_texture.mip_levels.max(1))
                .base_array_layer(0)
                .layer_count(1);

            let old_color_layout =
                color_texture.transition_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let (src_stage, src_access) = match old_color_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                ),
                _ => (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                ),
            };

            let color_barrier_to_attach = vk::ImageMemoryBarrier::default()
                .old_layout(old_color_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource)
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

            let mut barriers = vec![color_barrier_to_attach];
            let mut total_src_stage = src_stage;
            let mut wait_dst_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;

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

                let (depth_src_stage, depth_src_access) = match old_depth_layout {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::AccessFlags::empty(),
                    ),
                };

                let depth_barrier_to_attach = vk::ImageMemoryBarrier::default()
                    .old_layout(old_depth_layout)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(depth_texture.image)
                    .subresource_range(depth_subresource)
                    .src_access_mask(depth_src_access)
                    .dst_access_mask(
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    );

                barriers.push(depth_barrier_to_attach);
                total_src_stage |= depth_src_stage;
                wait_dst_stage |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;

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

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                total_src_stage,
                wait_dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

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

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(color_texture.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(color_load_op_vk)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_color_value);

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

            let barrier_to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource)
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

            color_texture.transition_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            if let Some((depth_texture, _)) = depth {
                depth_texture.transition_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
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
            let mut total_src_stage = vk::PipelineStageFlags::empty();
            let mut wait_dst_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
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
                let (src_stage, src_access) = match old_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    ),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_READ,
                    ),
                    _ => (
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::AccessFlags::empty(),
                    ),
                };

                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(tex.image)
                    .subresource_range(subresource)
                    .src_access_mask(src_access)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

                barriers.push(barrier);
                total_src_stage |= src_stage;
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
                let (depth_src_stage, depth_src_access) = match old_depth_layout {
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::AccessFlags::empty(),
                    ),
                };

                let depth_barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(old_depth_layout)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(depth_tex.image)
                    .subresource_range(depth_subresource)
                    .src_access_mask(depth_src_access)
                    .dst_access_mask(
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    );

                barriers.push(depth_barrier);
                total_src_stage |= depth_src_stage;
                wait_dst_stage |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;

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

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                total_src_stage,
                wait_dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

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

            self.device.cmd_draw(
                recorder.command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );

            self.device.cmd_end_rendering(recorder.command_buffer);

            let post_barriers: Vec<vk::ImageMemoryBarrier> = color_textures
                .iter()
                .zip(color_subresources.iter())
                .map(|(tex, &subres)| {
                    vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(tex.image)
                        .subresource_range(subres)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                })
                .collect();

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &post_barriers,
            );

            for tex in color_textures {
                tex.transition_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
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
            let mut lock = texture.layout.lock().unwrap();
            let old_layout = *lock;

            if old_layout != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(texture.mip_levels.max(1))
                    .base_array_layer(0)
                    .layer_count(1);

                let (src_stage, src_access) = match old_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    ),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                        vk::PipelineStageFlags::FRAGMENT_SHADER
                            | vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::SHADER_READ,
                    ),
                    _ => (
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::AccessFlags::empty(),
                    ),
                };

                let barrier_to_dst = vk::ImageMemoryBarrier::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(subresource_range)
                    .src_access_mask(src_access)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

                self.device.cmd_pipeline_barrier(
                    recorder.command_buffer,
                    src_stage,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_to_dst],
                );

                *lock = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            }

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
        let staging_buf = self.create_buffer(&crate::BufferDesc {
            label: Some("staging_texture_upload"),
            size: data.len() as u64,
            usage: crate::BufferUsage::TRANSFER_SRC,
            memory_location: crate::MemoryLocation::CpuToGpu,
        })?;

        self.write_buffer(&staging_buf, 0, data)?;

        let recorder = self.create_command_recorder()?;
        self.cmd_copy_buffer_to_image(&recorder, &staging_buf, texture);

        unsafe {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(texture.mip_levels.max(1))
                .base_array_layer(0)
                .layer_count(1);

            let barrier_to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_read],
            );
            *texture.layout.lock().unwrap() = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        }

        self.submit_and_wait(recorder)?;
        self.wait_idle()?;
        self.destroy_buffer(staging_buf);

        Ok(())
    }

    pub fn cmd_copy_image_to_buffer(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        buffer: &VulkanBuffer,
    ) {
        unsafe {
            let mut lock = texture.layout.lock().unwrap();
            let old_layout = *lock;

            if old_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let (src_stage, src_access) = match old_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::AccessFlags::empty(),
                    ),
                };

                let barrier_to_src = vk::ImageMemoryBarrier::default()
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(subresource_range)
                    .src_access_mask(src_access)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

                self.device.cmd_pipeline_barrier(
                    recorder.command_buffer,
                    src_stage,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_to_src],
                );

                *lock = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            }

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

    fn submit_batches_internal(
        &self,
        recorders: Vec<VulkanCommandRecorder>,
        wait_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        signal_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        sync_fence: Option<vk::Fence>,
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

            let mut vk_wait_sems = Vec::with_capacity(wait_semaphores.len());
            let mut vk_wait_values = Vec::with_capacity(wait_semaphores.len());
            let mut vk_wait_stages = Vec::with_capacity(wait_semaphores.len());
            for (sem, val) in wait_semaphores {
                vk_wait_sems.push(sem.semaphore);
                vk_wait_values.push(*val);
                vk_wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }

            let next_val = self
                .internal_frame_counter
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                + 1;
            let mut vk_signal_sems = Vec::with_capacity(signal_semaphores.len() + 1);
            let mut vk_signal_values = Vec::with_capacity(signal_semaphores.len() + 1);
            for (sem, val) in signal_semaphores {
                vk_signal_sems.push(sem.semaphore);
                vk_signal_values.push(*val);
            }
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

            let fence = sync_fence.unwrap_or(vk::Fence::null());
            {
                let queue = self.queue.lock().unwrap();
                if let Err(e) = self.device.queue_submit(*queue, &[submit_info], fence) {
                    return Err(HalError::CommandError(e.to_string()));
                }
            }

            if let Some(f) = sync_fence {
                let wait_res = self.device.wait_for_fences(&[f], true, u64::MAX);
                self.device.reset_fences(&[f]).ok();
                self.fence_cache.lock().unwrap().push(f);
                wait_res.map_err(|e| HalError::CommandError(e.to_string()))?;

                // Acquire locks in standard order: pending_async_recorders THEN command_recorder_cache
                let current_val = self
                    .device
                    .get_semaphore_counter_value(self.internal_frame_semaphore)
                    .unwrap_or(0);
                let mut pending = self.pending_async_recorders.lock().unwrap();
                let mut cache = self.command_recorder_cache.lock().unwrap();
                for r in recorders {
                    cache.push((r.command_pool, r.command_buffer));
                }

                pending.retain_mut(|(val, async_recorders)| {
                    if *val <= current_val {
                        for r in async_recorders.drain(..) {
                            cache.push((r.command_pool, r.command_buffer));
                        }
                        false
                    } else {
                        true
                    }
                });
            } else {
                self.pending_async_recorders
                    .lock()
                    .unwrap()
                    .push((next_val, recorders));
            }
        }
        Ok(())
    }

    pub fn submit_and_wait(&self, recorder: VulkanCommandRecorder) -> Result<(), HalError> {
        let fence = self.fence_cache.lock().unwrap().pop();
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
        let res = self.submit_batches_internal(vec![recorder], &[], &[], Some(fence));
        if res.is_err() {
            unsafe {
                self.device.reset_fences(&[fence]).ok();
                self.fence_cache.lock().unwrap().push(fence);
            }
        }
        res
    }

    pub fn submit_batches_and_wait(
        &self,
        recorders: Vec<VulkanCommandRecorder>,
    ) -> Result<(), HalError> {
        let fence = self.fence_cache.lock().unwrap().pop();
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
        let res = self.submit_batches_internal(recorders, &[], &[], Some(fence));
        if res.is_err() {
            unsafe {
                self.device.reset_fences(&[fence]).ok();
                self.fence_cache.lock().unwrap().push(fence);
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
        self.submit_batches_internal(recorders, wait_semaphores, signal_semaphores, None)
    }

    pub fn cmd_memory_barrier(&self, recorder: &VulkanCommandRecorder) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::SHADER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::VERTEX_INPUT
                    | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    pub fn cmd_buffer_barrier(&self, recorder: &VulkanCommandRecorder, buffer: &VulkanBuffer) {
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buffer.buffer)
            .offset(0)
            .size(buffer.size)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::HOST_READ,
            )
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        unsafe {
            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::VERTEX_INPUT
                    | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }
}
