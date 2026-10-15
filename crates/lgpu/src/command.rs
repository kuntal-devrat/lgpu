//! Command encoders, buffers, and recording passes.

use std::sync::Arc;
use crate::{
    BindlessComputePipeline, Buffer, ComputePipeline, Error, GraphicsPipeline, Texture,
};

pub struct CommandEncoder {
    #[cfg(feature = "vulkan")]
    pub(crate) recorder: Option<lgpu_hal::vulkan::VulkanCommandRecorder>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

pub struct CommandBuffer {
    #[cfg(feature = "vulkan")]
    pub(crate) recorder: lgpu_hal::vulkan::VulkanCommandRecorder,
}

impl CommandEncoder {
    pub fn dispatch_compute(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[&Buffer],
        x: u32,
        y: u32,
        z: u32,
    ) {
        #[cfg(feature = "vulkan")]
        {
            let raw_pipeline = pipeline.hal_pipeline.as_ref().unwrap();
            let raw_buffers: Vec<&lgpu_hal::vulkan::VulkanBuffer> = buffers
                .iter()
                .map(|b| b.hal_buffer.as_ref().unwrap())
                .collect();

            self.vulkan_device
                .bind_buffers_to_compute_pipeline(raw_pipeline, &raw_buffers);

            let recorder = self.recorder.as_ref().unwrap();
            self.vulkan_device
                .dispatch(recorder, raw_pipeline, x, y, z);
        }
    }

    pub fn dispatch_bindless_compute(
        &mut self,
        pipeline: &BindlessComputePipeline,
        push_constants: &[u8],
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("BindlessComputePipeline has been destroyed");
            self.vulkan_device.dispatch_bindless(
                recorder,
                raw_pipeline,
                push_constants,
                x,
                y,
                z,
            )?;
            Ok(())
        }
    }

    /// Clears the color attachment and executes dynamic rendering transitions.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{CommandEncoder, Texture};
    ///
    /// # fn example(encoder: &mut CommandEncoder, target: &Texture) {
    /// encoder.clear_and_render_texture(target, [0.1, 0.2, 0.3, 1.0]).expect("Clear failed");
    /// # }
    /// ```
    pub fn clear_and_render_texture(
        &mut self,
        texture: &Texture,
        clear_color: [f32; 4],
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.cmd_clear_and_render(recorder, raw_tex, clear_color)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (texture, clear_color);
            Ok(())
        }
    }

    pub fn draw(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            self.vulkan_device.cmd_draw(
                recorder,
                raw_tex,
                raw_pipeline,
                vertex_count,
                push_constants,
                clear_color,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (texture, pipeline, vertex_count, push_constants, clear_color);
            Ok(())
        }
    }

    pub fn draw_with_ops(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            self.vulkan_device.cmd_draw_with_ops(
                recorder,
                raw_tex,
                None,
                raw_pipeline,
                lgpu_hal::vulkan::DrawCommand::Array {
                    vertex_count,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
                push_constants,
                color_load_op,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (texture, pipeline, vertex_count, push_constants, color_load_op);
            Ok(())
        }
    }

    pub fn clear_and_render_with_depth(
        &mut self,
        color_texture: &Texture,
        depth_texture: &Texture,
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_color = color_texture.hal_texture.as_ref().unwrap();
            let raw_depth = depth_texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.cmd_clear_and_render_with_depth(
                recorder,
                raw_color,
                raw_depth,
                clear_color,
                clear_depth,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (color_texture, depth_texture, clear_color, clear_depth);
            Ok(())
        }
    }

    pub fn draw_with_depth(
        &mut self,
        color_texture: &Texture,
        depth_texture: &Texture,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_color = color_texture.hal_texture.as_ref().unwrap();
            let raw_depth = depth_texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            self.vulkan_device.cmd_draw_with_depth(
                recorder,
                raw_color,
                raw_depth,
                raw_pipeline,
                vertex_count,
                push_constants,
                clear_color,
                clear_depth,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (color_texture, depth_texture, pipeline, vertex_count, push_constants, clear_color, clear_depth);
            Ok(())
        }
    }

    pub fn draw_with_depth_and_ops(
        &mut self,
        color_texture: &Texture,
        depth_texture: &Texture,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
        depth_load_op: lgpu_hal::DepthLoadOp,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_color = color_texture.hal_texture.as_ref().unwrap();
            let raw_depth = depth_texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            self.vulkan_device.cmd_draw_with_ops(
                recorder,
                raw_color,
                Some((raw_depth, depth_load_op)),
                raw_pipeline,
                lgpu_hal::vulkan::DrawCommand::Array {
                    vertex_count,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
                push_constants,
                color_load_op,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (color_texture, depth_texture, pipeline, vertex_count, push_constants, color_load_op, depth_load_op);
            Ok(())
        }
    }

    pub fn render_triangle(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        clear_color: [f32; 4],
    ) -> Result<(), Error> {
        self.draw(texture, pipeline, 3, &[], clear_color)
    }

    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &Buffer,
        source_offset: u64,
        destination: &Buffer,
        destination_offset: u64,
        size: u64,
    ) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_src = source.hal_buffer.as_ref().unwrap();
            let raw_dst = destination.hal_buffer.as_ref().unwrap();
            self.vulkan_device.cmd_copy_buffer_to_buffer(
                recorder,
                raw_src,
                raw_dst,
                size,
                source_offset,
                destination_offset,
            );
        }
    }

    pub fn copy_buffer_to_texture(
        &mut self,
        buffer: &Buffer,
        texture: &Texture,
    ) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_buf = buffer.hal_buffer.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.cmd_copy_buffer_to_image(recorder, raw_buf, raw_tex);
        }
    }

    pub fn dispatch_indirect(
        &mut self,
        pipeline: &BindlessComputePipeline,
        indirect_buffer: &Buffer,
        indirect_offset: u64,
        push_constants: &[u8],
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_buf = indirect_buffer.hal_buffer.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("BindlessComputePipeline has been destroyed");
            self.vulkan_device.cmd_dispatch_indirect(
                recorder,
                raw_pipeline,
                raw_buf,
                indirect_offset,
                push_constants,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (pipeline, indirect_buffer, indirect_offset, push_constants);
            Ok(())
        }
    }

    pub fn draw_indexed(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        index_buffer: &Buffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            let raw_idx_buf = index_buffer.hal_buffer.as_ref().unwrap();
            self.vulkan_device.cmd_draw_indexed(
                recorder,
                raw_tex,
                raw_pipeline,
                raw_idx_buf,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
                lgpu_hal::vulkan::vk::IndexType::UINT32,
                push_constants,
                clear_color,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (texture, pipeline, index_buffer, index_count, instance_count, first_index, vertex_offset, first_instance, push_constants, clear_color);
            Ok(())
        }
    }

    pub fn draw_indexed_with_depth(
        &mut self,
        color_texture: &Texture,
        depth_texture: &Texture,
        pipeline: &GraphicsPipeline,
        index_buffer: &Buffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_color = color_texture.hal_texture.as_ref().unwrap();
            let raw_depth = depth_texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            let raw_idx_buf = index_buffer.hal_buffer.as_ref().unwrap();
            self.vulkan_device.cmd_draw_indexed_with_depth(
                recorder,
                raw_color,
                raw_depth,
                raw_pipeline,
                raw_idx_buf,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
                lgpu_hal::vulkan::vk::IndexType::UINT32,
                push_constants,
                clear_color,
                clear_depth,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (color_texture, depth_texture, pipeline, index_buffer, index_count, instance_count, first_index, vertex_offset, first_instance, push_constants, clear_color, clear_depth);
            Ok(())
        }
    }

    pub fn draw_indexed_with_ops(
        &mut self,
        color_texture: &Texture,
        depth_texture: Option<&Texture>,
        pipeline: &GraphicsPipeline,
        index_buffer: &Buffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
        depth_load_op: Option<lgpu_hal::DepthLoadOp>,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_color = color_texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            let raw_idx_buf = index_buffer.hal_buffer.as_ref().unwrap();
            let depth_tuple = match (depth_texture, depth_load_op) {
                (Some(dt), Some(op)) => Some((dt.hal_texture.as_ref().unwrap(), op)),
                _ => None,
            };
            self.vulkan_device.cmd_draw_with_ops(
                recorder,
                raw_color,
                depth_tuple,
                raw_pipeline,
                lgpu_hal::vulkan::DrawCommand::Indexed {
                    index_buffer: raw_idx_buf,
                    index_count,
                    instance_count,
                    first_index,
                    vertex_offset,
                    first_instance,
                    index_type: lgpu_hal::vulkan::vk::IndexType::UINT32,
                },
                push_constants,
                color_load_op,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (color_texture, depth_texture, pipeline, index_buffer, index_count, instance_count, first_index, vertex_offset, first_instance, push_constants, color_load_op, depth_load_op);
            Ok(())
        }
    }

    pub fn draw_indirect(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        indirect_buffer: &Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            let raw_ind_buf = indirect_buffer.hal_buffer.as_ref().unwrap();
            self.vulkan_device.cmd_draw_indirect(
                recorder,
                raw_tex,
                raw_pipeline,
                raw_ind_buf,
                offset,
                draw_count,
                stride,
                push_constants,
                clear_color,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (texture, pipeline, indirect_buffer, offset, draw_count, stride, push_constants, clear_color);
            Ok(())
        }
    }

    pub fn draw_indirect_with_depth(
        &mut self,
        color_texture: &Texture,
        depth_texture: &Texture,
        pipeline: &GraphicsPipeline,
        indirect_buffer: &Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_color = color_texture.hal_texture.as_ref().unwrap();
            let raw_depth = depth_texture.hal_texture.as_ref().unwrap();
            let raw_pipeline = pipeline
                .pipeline
                .expect("GraphicsPipeline has been destroyed");
            let raw_ind_buf = indirect_buffer.hal_buffer.as_ref().unwrap();
            self.vulkan_device.cmd_draw_indirect_with_depth(
                recorder,
                raw_color,
                raw_depth,
                raw_pipeline,
                raw_ind_buf,
                offset,
                draw_count,
                stride,
                push_constants,
                clear_color,
                clear_depth,
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (color_texture, depth_texture, pipeline, indirect_buffer, offset, draw_count, stride, push_constants, clear_color, clear_depth);
            Ok(())
        }
    }

    pub fn copy_texture_to_buffer(
        &mut self,
        texture: &Texture,
        buffer: &Buffer,
    ) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            let raw_buf = buffer.hal_buffer.as_ref().unwrap();
            self.vulkan_device.cmd_copy_image_to_buffer(recorder, raw_tex, raw_buf);
        }
    }

    pub fn memory_barrier(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            self.vulkan_device.cmd_memory_barrier(recorder);
        }
    }

    pub fn buffer_barrier(&mut self, buffer: &Buffer) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_buf = buffer.hal_buffer.as_ref().unwrap();
            self.vulkan_device.cmd_buffer_barrier(recorder, raw_buf);
        }
    }

    pub fn transition_to_present(&mut self, texture: &Texture) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.cmd_transition_to_present(recorder, raw_tex);
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = texture;
        }
    }

    /// Finishes recording and returns an executable [`CommandBuffer`] for submission.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{CommandEncoder, Queue};
    ///
    /// # fn example(encoder: CommandEncoder, queue: &Queue) {
    /// let cmd = encoder.finish();
    /// queue.submit(cmd).expect("Submit failed");
    /// # }
    /// ```
    pub fn finish(mut self) -> CommandBuffer {
        #[cfg(feature = "vulkan")]
        {
            CommandBuffer {
                recorder: self.recorder.take().unwrap(),
            }
        }
    }

    /// Begins a scoped [`RenderPass`] for executing one or more draw calls with customizable load/store ops.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{ColorLoadOp, CommandEncoder, RenderPassDesc, Texture};
    ///
    /// # fn example(encoder: &mut CommandEncoder, target: &Texture) {
    /// let pass_desc = RenderPassDesc {
    ///     color_texture: target,
    ///     color_load_op: ColorLoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
    ///     depth_attachment: None,
    /// };
    /// let mut pass = encoder.begin_render_pass(pass_desc);
    /// // pass.draw(...);
    /// # }
    /// ```
    pub fn begin_render_pass<'a>(&'a mut self, desc: RenderPassDesc<'a>) -> RenderPass<'a> {
        RenderPass {
            encoder: self,
            desc,
        }
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(recorder) = self.recorder.take() {
                self.vulkan_device.recycle_command_recorder(recorder);
            }
        }
    }
}

pub struct RenderPassDesc<'a> {
    pub color_texture: &'a Texture,
    pub color_load_op: lgpu_hal::ColorLoadOp,
    pub depth_attachment: Option<(&'a Texture, lgpu_hal::DepthLoadOp)>,
}

pub struct RenderPass<'a> {
    encoder: &'a mut CommandEncoder,
    desc: RenderPassDesc<'a>,
}

impl<'a> RenderPass<'a> {
    pub fn draw(
        &mut self,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
    ) -> Result<(), Error> {
        if let Some((depth_tex, depth_op)) = self.desc.depth_attachment {
            self.encoder.draw_with_depth_and_ops(
                self.desc.color_texture,
                depth_tex,
                pipeline,
                vertex_count,
                push_constants,
                self.desc.color_load_op,
                depth_op,
            )?;
        } else {
            self.encoder.draw_with_ops(
                self.desc.color_texture,
                pipeline,
                vertex_count,
                push_constants,
                self.desc.color_load_op,
            )?;
        }
        // Subsequent draws in the same pass preserve (LOAD) previously drawn contents
        self.desc.color_load_op = lgpu_hal::ColorLoadOp::Load;
        if let Some((d, _)) = self.desc.depth_attachment {
            self.desc.depth_attachment = Some((d, lgpu_hal::DepthLoadOp::Load));
        }
        Ok(())
    }

    pub fn draw_indexed(
        &mut self,
        pipeline: &GraphicsPipeline,
        index_buffer: &Buffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        push_constants: &[u8],
    ) -> Result<(), Error> {
        let (depth_tex, depth_op) = match self.desc.depth_attachment {
            Some((dt, dop)) => (Some(dt), Some(dop)),
            None => (None, None),
        };
        self.encoder.draw_indexed_with_ops(
            self.desc.color_texture,
            depth_tex,
            pipeline,
            index_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            push_constants,
            self.desc.color_load_op,
            depth_op,
        )?;
        self.desc.color_load_op = lgpu_hal::ColorLoadOp::Load;
        if let Some((d, _)) = self.desc.depth_attachment {
            self.desc.depth_attachment = Some((d, lgpu_hal::DepthLoadOp::Load));
        }
        Ok(())
    }
}
