//! Logical GPU device and queue submission.

use std::sync::Arc;
use crate::{
    BindlessComputePipeline, Buffer, BufferDesc, BufferId, CommandBuffer, CommandEncoder,
    ComputePipeline, Error, GraphicsPipeline, Sampler, SamplerDesc, SamplerId,
    Shader, SurfaceTexture, Texture, TextureDesc, TextureFormat, TextureId,
    TextureUsage, TimelineSemaphore,
};

pub struct Device {
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

pub struct Queue {
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

impl Device {
    /// Creates an uninitialized GPU [`Buffer`] according to [`BufferDesc`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{BufferDesc, BufferUsage, Device, MemoryLocation};
    ///
    /// # fn example(device: &Device) {
    /// let desc = BufferDesc {
    ///     label: Some("StorageBuffer"),
    ///     size: 1024,
    ///     usage: BufferUsage::STORAGE,
    ///     memory_location: MemoryLocation::GpuOnly,
    /// };
    /// let buffer = device.create_buffer(&desc).expect("Failed to create buffer");
    /// # }
    /// ```
    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<Buffer, Error> {
        #[cfg(feature = "vulkan")]
        {
            let hal_buffer = self.vulkan_device.create_buffer(desc)?;
            let id = if self.vulkan_device.bindless_heap.is_some() {
                Some(BufferId(self.vulkan_device.register_bindless_buffer(&hal_buffer)?))
            } else {
                None
            };

            Ok(Buffer {
                hal_buffer: Some(hal_buffer),
                vulkan_device: Arc::clone(&self.vulkan_device),
                size: desc.size,
                id,
            })
        }
    }

    /// Creates a host-mapped GPU [`Buffer`] initialized with the provided byte slice.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{BufferDesc, BufferUsage, Device, MemoryLocation};
    ///
    /// # fn example(device: &Device) {
    /// let desc = BufferDesc {
    ///     label: Some("InitBuffer"),
    ///     size: 64,
    ///     usage: BufferUsage::STORAGE,
    ///     memory_location: MemoryLocation::CpuToGpu,
    /// };
    /// let initial_bytes = [1u8, 2, 3, 4];
    /// let buffer = device.create_buffer_init(&desc, &initial_bytes).unwrap();
    /// # }
    /// ```
    pub fn create_buffer_init(&self, desc: &BufferDesc, data: &[u8]) -> Result<Buffer, Error> {
        let buffer = self.create_buffer(desc)?;
        buffer.write(data)?;
        Ok(buffer)
    }

    /// Creates a host-mapped GPU [`Buffer`] initialized with a typed slice implementing [`bytemuck::NoUninit`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{BufferDesc, BufferUsage, Device, MemoryLocation};
    ///
    /// # fn example(device: &Device) {
    /// let desc = BufferDesc {
    ///     label: Some("FloatBuffer"),
    ///     size: 16,
    ///     usage: BufferUsage::STORAGE,
    ///     memory_location: MemoryLocation::CpuToGpu,
    /// };
    /// let floats = [1.0f32, 2.0, 3.0, 4.0];
    /// let buffer = device.create_buffer_init_slice(&desc, &floats).unwrap();
    /// # }
    /// ```
    pub fn create_buffer_init_slice<T: bytemuck::NoUninit>(
        &self,
        desc: &BufferDesc,
        data: &[T],
    ) -> Result<Buffer, Error> {
        self.create_buffer_init(desc, bytemuck::cast_slice(data))
    }

    pub fn create_shader_spirv(&self, spirv_words: &[u32]) -> Result<Shader, Error> {
        #[cfg(feature = "vulkan")]
        {
            let hal_shader = self.vulkan_device.create_shader_from_spirv(spirv_words)?;
            Ok(Shader {
                hal_shader: Some(hal_shader),
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
    }

    pub fn create_compute_pipeline(
        &self,
        shader: &Shader,
        buffer_count: u32,
    ) -> Result<ComputePipeline, Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_shader = shader.hal_shader.as_ref().unwrap();
            let hal_pipeline = self
                .vulkan_device
                .create_compute_pipeline(raw_shader, buffer_count)?;
            Ok(ComputePipeline {
                hal_pipeline: Some(hal_pipeline),
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
    }

    pub fn create_bindless_compute_pipeline(
        &self,
        shader: &Shader,
    ) -> Result<BindlessComputePipeline, Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_shader = shader.hal_shader.as_ref().unwrap();
            let pipeline = self
                .vulkan_device
                .create_bindless_compute_pipeline(raw_shader)?;
            Ok(BindlessComputePipeline {
                pipeline: Some(pipeline),
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
    }

    /// Creates a GPU [`Texture`] according to [`TextureDesc`].
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{Device, TextureDesc, TextureFormat, TextureUsage};
    ///
    /// # fn example(device: &Device) {
    /// let desc = TextureDesc::new_2d(
    ///     512,
    ///     512,
    ///     TextureFormat::Rgba8Unorm,
    ///     TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    /// );
    /// let texture = device.create_texture(&desc).expect("Failed to create texture");
    /// # }
    /// ```
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture, Error> {
        #[cfg(feature = "vulkan")]
        {
            let hal_texture = self.vulkan_device.create_texture(desc)?;
            let id = if desc.usage.contains(TextureUsage::TEXTURE_BINDING)
                && self.vulkan_device.bindless_heap.is_some()
            {
                Some(TextureId(
                    self.vulkan_device.register_bindless_texture(&hal_texture)?,
                ))
            } else {
                None
            };
            Ok(Texture {
                hal_texture: Some(hal_texture),
                vulkan_device: Arc::clone(&self.vulkan_device),
                width: desc.width,
                height: desc.height,
                format: desc.format,
                id,
                owned: true,
            })
        }
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> Result<Sampler, Error> {
        #[cfg(feature = "vulkan")]
        {
            let hal_sampler = self.vulkan_device.create_sampler(desc)?;
            let id = if self.vulkan_device.bindless_heap.is_some() {
                Some(SamplerId(
                    self.vulkan_device.register_bindless_sampler(&hal_sampler)?,
                ))
            } else {
                None
            };
            Ok(Sampler {
                hal_sampler: Some(hal_sampler),
                vulkan_device: Arc::clone(&self.vulkan_device),
                id,
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = desc;
            Err(HalError::NoAdapterFound.into())
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        shader: &Shader,
        vs_entry: &str,
        fs_entry: &str,
        color_format: TextureFormat,
    ) -> Result<GraphicsPipeline, Error> {
        self.create_graphics_pipeline_with_depth(
            shader,
            vs_entry,
            fs_entry,
            color_format,
            None,
            false,
            false,
        )
    }

    pub fn create_graphics_pipeline_with_depth(
        &self,
        shader: &Shader,
        vs_entry: &str,
        fs_entry: &str,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        depth_test: bool,
        depth_write: bool,
    ) -> Result<GraphicsPipeline, Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_shader = shader.hal_shader.as_ref().unwrap();
            let pipeline = self.vulkan_device.create_graphics_pipeline_with_depth(
                raw_shader,
                vs_entry,
                fs_entry,
                color_format,
                depth_format,
                depth_test,
                depth_write,
            )?;
            Ok(GraphicsPipeline {
                pipeline: Some(pipeline),
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (shader, vs_entry, fs_entry, color_format, depth_format, depth_test, depth_write);
            Err(HalError::NoAdapterFound.into())
        }
    }

    pub fn create_graphics_pipeline_with_desc(
        &self,
        shader: &Shader,
        desc: &lgpu_hal::GraphicsPipelineDesc,
    ) -> Result<GraphicsPipeline, Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_shader = shader.hal_shader.as_ref().unwrap();
            let pipeline = self.vulkan_device.create_graphics_pipeline_with_desc(raw_shader, desc)?;
            Ok(GraphicsPipeline {
                pipeline: Some(pipeline),
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (shader, desc);
            Err(HalError::NoAdapterFound.into())
        }
    }

    /// Creates a new [`CommandEncoder`] for recording graphics and compute commands.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::Device;
    ///
    /// # fn example(device: &Device) {
    /// let mut encoder = device.create_command_encoder().expect("Failed to create encoder");
    /// // Record barriers, clear passes, or dispatches...
    /// let command_buffer = encoder.finish();
    /// # }
    /// ```
    pub fn create_command_encoder(&self) -> Result<CommandEncoder, Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.vulkan_device.create_command_recorder()?;
            Ok(CommandEncoder {
                recorder: Some(recorder),
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> Result<TimelineSemaphore, Error> {
        #[cfg(feature = "vulkan")]
        {
            let hal_semaphore = self.vulkan_device.create_timeline_semaphore(initial_value)?;
            Ok(TimelineSemaphore {
                hal_semaphore: Some(hal_semaphore),
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = initial_value;
            Err(HalError::NoAdapterFound.into())
        }
    }

    pub fn wait_idle(&self) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            self.vulkan_device.wait_idle()?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Ok(())
        }
    }

    pub fn write_texture(&self, texture: &Texture, data: &[u8]) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.write_texture(raw_tex, data)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (texture, data);
            Ok(())
        }
    }
}

impl Queue {
    /// Submits a [`CommandBuffer`] to the GPU queue and waits synchronously for execution to finish.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{CommandBuffer, Queue};
    ///
    /// # fn example(queue: &Queue, cmd: CommandBuffer) {
    /// queue.submit(cmd).expect("GPU submission failed");
    /// # }
    /// ```
    pub fn submit(&self, command_buffer: CommandBuffer) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            self.vulkan_device
                .submit_and_wait(command_buffer.recorder)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = command_buffer;
            Ok(())
        }
    }

    pub fn submit_surface(
        &self,
        command_buffer: CommandBuffer,
        surface_texture: SurfaceTexture,
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            self.vulkan_device.submit_for_surface(
                command_buffer.recorder,
                surface_texture.acquire_semaphore,
                surface_texture.render_semaphore,
                surface_texture.frame_fence,
            )?;
            self.vulkan_device.queue_present_raw(
                surface_texture.swapchain,
                surface_texture.image_index,
                &[surface_texture.render_semaphore],
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (command_buffer, surface_texture);
            Ok(())
        }
    }

    pub fn submit_batches(&self, command_buffers: Vec<CommandBuffer>) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorders = command_buffers.into_iter().map(|cb| cb.recorder).collect();
            self.vulkan_device.submit_batches_and_wait(recorders)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = command_buffers;
            Ok(())
        }
    }

    pub fn submit_async(
        &self,
        command_buffer: CommandBuffer,
        wait_semaphores: &[(&TimelineSemaphore, u64)],
        signal_semaphores: &[(&TimelineSemaphore, u64)],
    ) -> Result<(), Error> {
        self.submit_batches_async(vec![command_buffer], wait_semaphores, signal_semaphores)
    }

    pub fn submit_batches_async(
        &self,
        command_buffers: Vec<CommandBuffer>,
        wait_semaphores: &[(&TimelineSemaphore, u64)],
        signal_semaphores: &[(&TimelineSemaphore, u64)],
    ) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let recorders = command_buffers.into_iter().map(|cb| cb.recorder).collect();
            let raw_wait: Vec<(&lgpu_hal::vulkan::VulkanTimelineSemaphore, u64)> = wait_semaphores
                .iter()
                .map(|(s, v)| (s.hal_semaphore.as_ref().unwrap(), *v))
                .collect();
            let raw_signal: Vec<(&lgpu_hal::vulkan::VulkanTimelineSemaphore, u64)> = signal_semaphores
                .iter()
                .map(|(s, v)| (s.hal_semaphore.as_ref().unwrap(), *v))
                .collect();
            self.vulkan_device
                .submit_batches_async(recorders, &raw_wait, &raw_signal)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (command_buffers, wait_semaphores, signal_semaphores);
            Ok(())
        }
    }

    pub fn write_texture(&self, texture: &Texture, data: &[u8]) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.write_texture(raw_tex, data)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (texture, data);
            Ok(())
        }
    }

    /// Writes byte data into a host-mapped [`Buffer`] at the specified offset.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::{Buffer, Queue};
    ///
    /// # fn example(queue: &Queue, buffer: &Buffer) {
    /// queue.write_buffer(buffer, 0, &[1u8, 2, 3, 4]).expect("Failed to write buffer");
    /// # }
    /// ```
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Result<(), Error> {
        buffer.write_at(offset, data)
    }
}
