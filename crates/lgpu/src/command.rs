//! Command encoders, buffers, and recording passes.

use crate::{BindlessComputePipeline, Buffer, ComputePipeline, Error, GraphicsPipeline, Texture};
#[allow(unused_imports)]
use std::sync::Arc;

pub struct CommandEncoder {
    pub(crate) recorder: Option<lgpu_hal::vulkan::VulkanCommandRecorder>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    /// Vertex buffers bound for the next draws, indexed by binding number.
    pub(crate) vertex_buffers: Vec<Option<(lgpu_hal::vulkan::vk::Buffer, u64)>>,
}

pub struct CommandBuffer {
    pub(crate) recorder: Option<lgpu_hal::vulkan::VulkanCommandRecorder>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        if let Some(recorder) = self.recorder.take() {
            self.vulkan_device.recycle_command_recorder(recorder);
        }
    }
}

impl CommandEncoder {
    #[inline]
    fn rec(&self) -> Result<&lgpu_hal::vulkan::VulkanCommandRecorder, Error> {
        self.recorder.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "CommandEncoder is closed or invalid".to_string(),
            ))
        })
    }

    #[inline]
    fn get_tex(texture: &Texture) -> Result<&lgpu_hal::vulkan::VulkanTexture, Error> {
        texture.hal_texture.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Texture hal resource is missing or destroyed".to_string(),
            ))
        })
    }

    #[inline]
    fn get_buf(buffer: &Buffer) -> Result<&lgpu_hal::vulkan::VulkanBuffer, Error> {
        buffer.hal_buffer.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Buffer hal resource is missing or destroyed".to_string(),
            ))
        })
    }

    #[inline]
    fn get_gfx_pipe(pipeline: &GraphicsPipeline) -> Result<lgpu_hal::vulkan::RawPipeline, Error> {
        pipeline.pipeline.ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "GraphicsPipeline has been destroyed".to_string(),
            ))
        })
    }

    #[inline]
    fn get_compute_pipe(
        pipeline: &BindlessComputePipeline,
    ) -> Result<lgpu_hal::vulkan::RawPipeline, Error> {
        pipeline.pipeline.ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "BindlessComputePipeline has been destroyed".to_string(),
            ))
        })
    }

    /// Binds a vertex buffer to a binding slot for every draw recorded afterwards.
    /// The layout passed to the graphics pipeline describes the bindings this draws from.
    pub fn bind_vertex_buffer(&mut self, binding: u32, buffer: &Buffer, offset: u64) {
        let Some(hal_buffer) = buffer.hal_buffer.as_ref() else {
            return;
        };
        let index = binding as usize;
        if self.vertex_buffers.len() <= index {
            self.vertex_buffers.resize(index + 1, None);
        }
        self.vertex_buffers[index] = Some((hal_buffer.buffer, offset));
    }

    /// Clears all vertex buffer bindings.
    pub fn unbind_vertex_buffers(&mut self) {
        self.vertex_buffers.clear();
    }

    fn vertex_bindings(&self) -> Result<Vec<(lgpu_hal::vulkan::vk::Buffer, u64)>, Error> {
        let mut bindings = Vec::with_capacity(self.vertex_buffers.len());
        for (binding, slot) in self.vertex_buffers.iter().enumerate() {
            match slot {
                Some(binding_value) => bindings.push(*binding_value),
                None => {
                    return Err(Error::Hal(lgpu_hal::HalError::InvalidParameter(format!(
                        "Vertex buffer binding {binding} is not set, but binding {}",
                        bindings.len()
                    ))))
                }
            }
        }
        Ok(bindings)
    }

    pub fn dispatch_compute(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[&Buffer],
        x: u32,
        y: u32,
        z: u32,
    ) {
        let (Some(raw_pipeline), Some(recorder)) =
            (pipeline.hal_pipeline.as_ref(), self.recorder.as_ref())
        else {
            return;
        };
        let mut raw_buffers = Vec::with_capacity(buffers.len());
        for b in buffers {
            if let Some(buf) = b.hal_buffer.as_ref() {
                raw_buffers.push(buf);
            } else {
                return;
            }
        }

        self.vulkan_device
            .bind_buffers_to_compute_pipeline(raw_pipeline, &raw_buffers);

        self.vulkan_device.dispatch(recorder, raw_pipeline, x, y, z);
    }

    pub fn dispatch_bindless_compute(
        &mut self,
        pipeline: &BindlessComputePipeline,
        push_constants: &[u8],
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), Error> {
        let recorder = self.rec()?;
        let raw_pipeline = Self::get_compute_pipe(pipeline)?;
        self.vulkan_device
            .dispatch_bindless(recorder, raw_pipeline, push_constants, x, y, z)?;
        Ok(())
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
        let recorder = self.rec()?;
        let raw_tex = Self::get_tex(texture)?;
        self.vulkan_device
            .cmd_clear_and_render(recorder, raw_tex, clear_color)?;
        Ok(())
    }

    pub fn draw(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) -> Result<(), Error> {
        self.draw_with_ops_resolve(
            texture,
            None,
            pipeline,
            vertex_count,
            push_constants,
            lgpu_hal::ColorLoadOp::Clear(clear_color),
        )
    }

    pub fn draw_with_ops(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
    ) -> Result<(), Error> {
        self.draw_with_ops_resolve(
            texture,
            None,
            pipeline,
            vertex_count,
            push_constants,
            color_load_op,
        )
    }

    /// Draws with an optional MSAA resolve target. When `resolve` is set, the color texture
    /// must be multisampled and the resolve texture single sampled, with matching size and
    /// format. The resolved image is left in `SHADER_READ_ONLY_OPTIMAL`.
    pub fn draw_with_ops_resolve(
        &mut self,
        color_texture: &Texture,
        resolve: Option<&Texture>,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
    ) -> Result<(), Error> {
        self.draw_full(
            color_texture,
            resolve,
            None,
            pipeline,
            lgpu_hal::vulkan::DrawCommand::Array {
                vertex_count,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            push_constants,
            color_load_op,
        )
    }

    /// Draws with an optional depth attachment and MSAA resolve target.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_depth_and_ops_resolve(
        &mut self,
        color_texture: &Texture,
        resolve: Option<&Texture>,
        depth_texture: &Texture,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
        depth_load_op: lgpu_hal::DepthLoadOp,
    ) -> Result<(), Error> {
        self.draw_full(
            color_texture,
            resolve,
            Some((depth_texture, depth_load_op)),
            pipeline,
            lgpu_hal::vulkan::DrawCommand::Array {
                vertex_count,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            push_constants,
            color_load_op,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_full(
        &mut self,
        color_texture: &Texture,
        resolve: Option<&Texture>,
        depth: Option<(&Texture, lgpu_hal::DepthLoadOp)>,
        pipeline: &GraphicsPipeline,
        draw_cmd: lgpu_hal::vulkan::DrawCommand<'_>,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
    ) -> Result<(), Error> {
        let recorder = self.rec()?;
        let raw_color = Self::get_tex(color_texture)?;
        let raw_resolve = match resolve {
            Some(texture) => Some(Self::get_tex(texture)?),
            None => None,
        };
        let raw_depth = match depth {
            Some((texture, op)) => Some((Self::get_tex(texture)?, op)),
            None => None,
        };
        let raw_pipeline = Self::get_gfx_pipe(pipeline)?;
        let vertex_buffers = self.vertex_bindings()?;
        self.vulkan_device.cmd_draw_with_ops_ex(
            recorder,
            raw_color,
            raw_resolve,
            raw_depth,
            raw_pipeline,
            draw_cmd,
            &vertex_buffers,
            push_constants,
            color_load_op,
        )?;
        Ok(())
    }

    pub fn clear_and_render_with_depth(
        &mut self,
        color_texture: &Texture,
        depth_texture: &Texture,
        clear_color: [f32; 4],
        clear_depth: f32,
    ) -> Result<(), Error> {
        let recorder = self.rec()?;
        let raw_color = Self::get_tex(color_texture)?;
        let raw_depth = Self::get_tex(depth_texture)?;
        self.vulkan_device.cmd_clear_and_render_with_depth(
            recorder,
            raw_color,
            raw_depth,
            clear_color,
            clear_depth,
        )?;
        Ok(())
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
        self.draw_with_depth_and_ops_resolve(
            color_texture,
            None,
            depth_texture,
            pipeline,
            vertex_count,
            push_constants,
            lgpu_hal::ColorLoadOp::Clear(clear_color),
            lgpu_hal::DepthLoadOp::Clear(clear_depth),
        )
    }

    pub fn draw_mrt(
        &mut self,
        color_textures: &[&Texture],
        depth_texture: Option<&Texture>,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: Option<f32>,
    ) -> Result<(), Error> {
        let recorder = self.rec()?;
        let mut raw_colors = Vec::with_capacity(color_textures.len());
        for t in color_textures {
            raw_colors.push(Self::get_tex(t)?);
        }
        let raw_depth = match depth_texture {
            Some(dt) => Some(Self::get_tex(dt)?),
            None => None,
        };
        let raw_pipeline = Self::get_gfx_pipe(pipeline)?;
        let vertex_buffers = self.vertex_bindings()?;
        self.vulkan_device.cmd_draw_mrt(
            recorder,
            &raw_colors,
            raw_depth,
            raw_pipeline,
            &vertex_buffers,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            push_constants,
            clear_color,
            clear_depth,
        )?;
        Ok(())
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
        self.draw_with_depth_and_ops_resolve(
            color_texture,
            None,
            depth_texture,
            pipeline,
            vertex_count,
            push_constants,
            color_load_op,
            depth_load_op,
        )
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
        let (Some(recorder), Some(raw_src), Some(raw_dst)) = (
            self.recorder.as_ref(),
            source.hal_buffer.as_ref(),
            destination.hal_buffer.as_ref(),
        ) else {
            return;
        };
        self.vulkan_device.cmd_copy_buffer_to_buffer(
            recorder,
            raw_src,
            raw_dst,
            size,
            source_offset,
            destination_offset,
        );
    }

    pub fn copy_buffer_to_texture(&mut self, buffer: &Buffer, texture: &Texture) {
        let (Some(recorder), Some(raw_buf), Some(raw_tex)) = (
            self.recorder.as_ref(),
            buffer.hal_buffer.as_ref(),
            texture.hal_texture.as_ref(),
        ) else {
            return;
        };
        self.vulkan_device
            .cmd_copy_buffer_to_image(recorder, raw_buf, raw_tex);
    }

    pub fn dispatch_indirect(
        &mut self,
        pipeline: &BindlessComputePipeline,
        indirect_buffer: &Buffer,
        indirect_offset: u64,
        push_constants: &[u8],
    ) -> Result<(), Error> {
        let recorder = self.rec()?;
        let raw_buf = Self::get_buf(indirect_buffer)?;
        let raw_pipeline = Self::get_compute_pipe(pipeline)?;
        self.vulkan_device.cmd_dispatch_indirect(
            recorder,
            raw_pipeline,
            raw_buf,
            indirect_offset,
            push_constants,
        )?;
        Ok(())
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
        self.draw_indexed_with_ops_resolve(
            texture,
            None,
            None,
            pipeline,
            index_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            push_constants,
            lgpu_hal::ColorLoadOp::Clear(clear_color),
        )
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
        self.draw_indexed_with_ops_resolve(
            color_texture,
            None,
            Some((depth_texture, lgpu_hal::DepthLoadOp::Clear(clear_depth))),
            pipeline,
            index_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            push_constants,
            lgpu_hal::ColorLoadOp::Clear(clear_color),
        )
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
        let depth_tuple = match (depth_texture, depth_load_op) {
            (Some(dt), Some(op)) => Some((dt, op)),
            _ => None,
        };
        self.draw_indexed_with_ops_resolve(
            color_texture,
            None,
            depth_tuple,
            pipeline,
            index_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            push_constants,
            color_load_op,
        )
    }

    /// Indexed draw with optional depth attachment and MSAA resolve target.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indexed_with_ops_resolve(
        &mut self,
        color_texture: &Texture,
        resolve: Option<&Texture>,
        depth: Option<(&Texture, lgpu_hal::DepthLoadOp)>,
        pipeline: &GraphicsPipeline,
        index_buffer: &Buffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
    ) -> Result<(), Error> {
        let raw_index = Self::get_buf(index_buffer)?;
        self.draw_full(
            color_texture,
            resolve,
            depth,
            pipeline,
            lgpu_hal::vulkan::DrawCommand::Indexed {
                index_buffer: raw_index,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
                index_type: lgpu_hal::vulkan::vk::IndexType::UINT32,
            },
            push_constants,
            color_load_op,
        )
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
        self.draw_indirect_with_ops_resolve(
            texture,
            None,
            None,
            pipeline,
            indirect_buffer,
            offset,
            draw_count,
            stride,
            push_constants,
            lgpu_hal::ColorLoadOp::Clear(clear_color),
        )
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
        self.draw_indirect_with_ops_resolve(
            color_texture,
            None,
            Some((depth_texture, lgpu_hal::DepthLoadOp::Clear(clear_depth))),
            pipeline,
            indirect_buffer,
            offset,
            draw_count,
            stride,
            push_constants,
            lgpu_hal::ColorLoadOp::Clear(clear_color),
        )
    }

    /// Indirect draw with optional depth attachment and MSAA resolve target.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_indirect_with_ops_resolve(
        &mut self,
        color_texture: &Texture,
        resolve: Option<&Texture>,
        depth: Option<(&Texture, lgpu_hal::DepthLoadOp)>,
        pipeline: &GraphicsPipeline,
        indirect_buffer: &Buffer,
        offset: u64,
        draw_count: u32,
        stride: u32,
        push_constants: &[u8],
        color_load_op: lgpu_hal::ColorLoadOp,
    ) -> Result<(), Error> {
        let raw_indirect = Self::get_buf(indirect_buffer)?;
        self.draw_full(
            color_texture,
            resolve,
            depth,
            pipeline,
            lgpu_hal::vulkan::DrawCommand::Indirect {
                indirect_buffer: raw_indirect,
                offset,
                draw_count,
                stride,
            },
            push_constants,
            color_load_op,
        )
    }

    pub fn copy_texture_to_buffer(&mut self, texture: &Texture, buffer: &Buffer) {
        let (Some(recorder), Some(raw_tex), Some(raw_buf)) = (
            self.recorder.as_ref(),
            texture.hal_texture.as_ref(),
            buffer.hal_buffer.as_ref(),
        ) else {
            return;
        };
        self.vulkan_device
            .cmd_copy_image_to_buffer(recorder, raw_tex, raw_buf);
    }

    /// Resets every query in a timestamp pool. Call this before writing timestamps in a
    /// command buffer that reuses the pool.
    pub fn reset_timestamp_pool(&mut self, pool: &crate::TimestampPool) {
        if let (Some(recorder), Ok(raw)) = (self.recorder.as_ref(), pool.raw()) {
            self.vulkan_device
                .cmd_reset_query_pool(recorder, raw, 0, pool.count);
        }
    }

    /// Writes a GPU timestamp into `index` of `pool`.
    pub fn write_timestamp(
        &mut self,
        pool: &crate::TimestampPool,
        index: u32,
    ) -> Result<(), Error> {
        let recorder = self.rec()?;
        let raw = pool.raw()?;
        if index >= pool.count {
            return Err(Error::Hal(lgpu_hal::HalError::InvalidParameter(format!(
                "Timestamp index {index} is out of range for a pool of {}",
                pool.count
            ))));
        }
        self.vulkan_device.cmd_write_timestamp(recorder, raw, index);
        Ok(())
    }

    /// Opens a named region in GPU debuggers such as RenderDoc. Has no effect when the
    /// debug utils extension is unavailable.
    pub fn begin_debug_label(&mut self, name: &str, color: [f32; 4]) {
        if let Some(recorder) = self.recorder.as_ref() {
            self.vulkan_device
                .cmd_begin_debug_label(recorder, name, color);
        }
    }

    /// Closes the region opened by [`CommandEncoder::begin_debug_label`].
    pub fn end_debug_label(&mut self) {
        if let Some(recorder) = self.recorder.as_ref() {
            self.vulkan_device.cmd_end_debug_label(recorder);
        }
    }

    pub fn memory_barrier(&mut self) {
        if let Some(recorder) = self.recorder.as_ref() {
            self.vulkan_device.cmd_memory_barrier(recorder);
        }
    }

    pub fn buffer_barrier(&mut self, buffer: &Buffer) {
        if let (Some(recorder), Some(raw_buf)) =
            (self.recorder.as_ref(), buffer.hal_buffer.as_ref())
        {
            self.vulkan_device.cmd_buffer_barrier(recorder, raw_buf);
        }
    }

    pub fn transition_to_present(&mut self, texture: &Texture) {
        if let (Some(recorder), Some(raw_tex)) =
            (self.recorder.as_ref(), texture.hal_texture.as_ref())
        {
            self.vulkan_device
                .cmd_transition_to_present(recorder, raw_tex);
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
    #[allow(unused_mut)]
    pub fn finish(mut self) -> CommandBuffer {
        let recorder = self
            .recorder
            .take()
            .expect("CommandEncoder has already been finished or recorder missing");
        let vulkan_device = Arc::clone(&self.vulkan_device);
        CommandBuffer {
            recorder: Some(recorder),
            vulkan_device,
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
        self.begin_render_pass_with_resolve(desc, None)
    }

    /// Begins a scoped render pass with an optional MSAA resolve target.
    pub fn begin_render_pass_with_resolve<'a>(
        &'a mut self,
        desc: RenderPassDesc<'a>,
        resolve: Option<&'a Texture>,
    ) -> RenderPass<'a> {
        RenderPass {
            encoder: self,
            desc,
            resolve,
        }
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        if let Some(recorder) = self.recorder.take() {
            self.vulkan_device.recycle_command_recorder(recorder);
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
    resolve: Option<&'a Texture>,
}

impl<'a> RenderPass<'a> {
    pub fn draw(
        &mut self,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
    ) -> Result<(), Error> {
        if let Some((depth_tex, depth_op)) = self.desc.depth_attachment {
            self.encoder.draw_with_depth_and_ops_resolve(
                self.desc.color_texture,
                self.resolve,
                depth_tex,
                pipeline,
                vertex_count,
                push_constants,
                self.desc.color_load_op,
                depth_op,
            )?;
        } else {
            self.encoder.draw_with_ops_resolve(
                self.desc.color_texture,
                self.resolve,
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
        self.encoder.draw_indexed_with_ops_resolve(
            self.desc.color_texture,
            self.resolve,
            match (depth_tex, depth_op) {
                (Some(tex), Some(op)) => Some((tex, op)),
                _ => None,
            },
            pipeline,
            index_buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            push_constants,
            self.desc.color_load_op,
        )?;
        self.desc.color_load_op = lgpu_hal::ColorLoadOp::Load;
        if let Some((d, _)) = self.desc.depth_attachment {
            self.desc.depth_attachment = Some((d, lgpu_hal::DepthLoadOp::Load));
        }
        Ok(())
    }
}
