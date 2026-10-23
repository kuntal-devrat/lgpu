//! Logical GPU device and queue submission.

use crate::{
    BindlessComputePipeline, Buffer, BufferDesc, BufferId, CommandBuffer, CommandEncoder,
    ComputePipeline, Error, GraphicsPipeline, Sampler, SamplerDesc, SamplerId, Shader,
    SurfaceTexture, Texture, TextureDesc, TextureFormat, TextureId, TextureUsage,
    TimelineSemaphore,
};
use std::sync::Arc;

pub struct Device {
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub(crate) _instance: Arc<lgpu_hal::vulkan::VulkanInstance>,
}

pub struct Queue {
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub(crate) _instance: Arc<lgpu_hal::vulkan::VulkanInstance>,
}

impl Device {
    /// Convenience helper that initializes a default Vulkan 1.3 [`crate::Instance`], selects
    /// the first available physical GPU adapter, and returns the initialized [`Device`] and [`Queue`].
    ///
    /// Ideal for quick prototyping, headless compute jobs, and test suites.
    ///
    /// # Example
    ///
    /// ```no_run
    /// let (device, queue) = lgpu::Device::default_device().expect("Failed to initialize GPU");
    /// ```
    pub fn default_device() -> Result<(Device, Queue), Error> {
        let instance = crate::Instance::new()?;
        let adapters = instance.enumerate_adapters()?;
        if adapters.is_empty() {
            return Err(Error::Hal(lgpu_hal::HalError::NoAdapterFound));
        }
        adapters[0].request_device(&instance)
    }

    /// Convenience helper that finishes recording on the given [`CommandEncoder`],
    /// submits it to the queue, and blocks the CPU until execution completes.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lgpu::Device;
    /// # fn example(device: &Device, queue: &lgpu::Queue) -> Result<(), lgpu::Error> {
    /// let encoder = device.create_command_encoder()?;
    /// device.run(queue, encoder)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run(&self, queue: &Queue, encoder: CommandEncoder) -> Result<(), Error> {
        queue.run(encoder)
    }

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
        let hal_buffer = self.vulkan_device.create_buffer(desc)?;
        let id = if self.vulkan_device.bindless_heap.is_some() {
            Some(BufferId(
                self.vulkan_device.register_bindless_buffer(&hal_buffer)?,
            ))
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
        let hal_shader = self.vulkan_device.create_shader_from_spirv(spirv_words)?;
        Ok(Shader {
            hal_shader: Some(hal_shader),
            vulkan_device: Arc::clone(&self.vulkan_device),
        })
    }

    pub fn create_compute_pipeline(
        &self,
        shader: &Shader,
        buffer_count: u32,
    ) -> Result<ComputePipeline, Error> {
        let raw_shader = shader.hal_shader.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Shader hal resource is missing or destroyed".to_string(),
            ))
        })?;
        let hal_pipeline = self
            .vulkan_device
            .create_compute_pipeline(raw_shader, buffer_count)?;
        Ok(ComputePipeline {
            hal_pipeline: Some(hal_pipeline),
            vulkan_device: Arc::clone(&self.vulkan_device),
        })
    }

    pub fn create_bindless_compute_pipeline(
        &self,
        shader: &Shader,
    ) -> Result<BindlessComputePipeline, Error> {
        let raw_shader = shader.hal_shader.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Shader hal resource is missing or destroyed".to_string(),
            ))
        })?;
        let pipeline = self
            .vulkan_device
            .create_bindless_compute_pipeline(raw_shader)?;
        Ok(BindlessComputePipeline {
            pipeline: Some(pipeline),
            vulkan_device: Arc::clone(&self.vulkan_device),
        })
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
        let storage_id = if desc.usage.contains(TextureUsage::STORAGE_BINDING)
            && self.vulkan_device.bindless_heap.is_some()
        {
            Some(TextureId(
                self.vulkan_device
                    .register_bindless_storage_image(&hal_texture)?,
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
            storage_id,
            owned: true,
        })
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> Result<Sampler, Error> {
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

    /// Performs a batched update of bindless buffer descriptors in a single driver call.
    pub fn write_buffer_descriptors_batch(&self, writes: &[(u32, &Buffer)]) -> Result<(), Error> {
        let mut raw = Vec::with_capacity(writes.len());
        for (slot, buf) in writes {
            let hal_buf = buf.hal_buffer.as_ref().ok_or_else(|| {
                Error::Hal(lgpu_hal::HalError::InvalidParameter(
                    "Buffer hal resource is missing or destroyed".to_string(),
                ))
            })?;
            raw.push((*slot, hal_buf));
        }
        self.vulkan_device.write_buffer_descriptors_batch(&raw)?;
        Ok(())
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
        let raw_shader = shader.hal_shader.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Shader hal resource is missing or destroyed".to_string(),
            ))
        })?;
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

    pub fn create_graphics_pipeline_with_desc(
        &self,
        shader: &Shader,
        desc: &lgpu_hal::GraphicsPipelineDesc,
    ) -> Result<GraphicsPipeline, Error> {
        let raw_shader = shader.hal_shader.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Shader hal resource is missing or destroyed".to_string(),
            ))
        })?;
        let pipeline = self
            .vulkan_device
            .create_graphics_pipeline_with_desc(raw_shader, desc)?;
        Ok(GraphicsPipeline {
            pipeline: Some(pipeline),
            vulkan_device: Arc::clone(&self.vulkan_device),
        })
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
        let recorder = self.vulkan_device.create_command_recorder()?;
        Ok(CommandEncoder {
            recorder: Some(recorder),
            vulkan_device: Arc::clone(&self.vulkan_device),
        })
    }

    pub fn create_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<TimelineSemaphore, Error> {
        let hal_semaphore = self
            .vulkan_device
            .create_timeline_semaphore(initial_value)?;
        Ok(TimelineSemaphore {
            hal_semaphore: Some(hal_semaphore),
            vulkan_device: Arc::clone(&self.vulkan_device),
        })
    }

    pub fn wait_idle(&self) -> Result<(), Error> {
        self.vulkan_device.wait_idle()?;
        Ok(())
    }

    pub fn write_texture(&self, texture: &Texture, data: &[u8]) -> Result<(), Error> {
        let raw_tex = texture.hal_texture.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Texture hal resource is missing or destroyed".to_string(),
            ))
        })?;
        self.vulkan_device.write_texture(raw_tex, data)?;
        Ok(())
    }

    /// Reclaims completely empty memory blocks held by the suballocator back to Vulkan.
    ///
    /// By default, `lgpu` automatically frees redundant empty blocks while keeping up to 1
    /// warm block per memory configuration for low-overhead reallocations. Calling `trim_memory`
    /// releases even cached empty blocks, returning maximum GPU and host memory to the OS.
    ///
    /// Returns the number of blocks freed.
    pub fn trim_memory(&self) -> usize {
        self.vulkan_device.trim_memory()
    }

    /// Returns the total number of memory blocks currently held by the internal suballocator.
    pub fn suballocator_block_count(&self) -> usize {
        self.vulkan_device.suballocator_block_count()
    }

    /// Returns the number of empty memory blocks currently held in the suballocator.
    pub fn suballocator_empty_block_count(&self) -> usize {
        self.vulkan_device.suballocator_empty_block_count()
    }
}

impl Queue {
    /// Finishes recording the given [`CommandEncoder`] into a command buffer, submits it
    /// to the GPU queue, and waits synchronously for execution to finish.
    ///
    /// Ideal for one-shot data transfers, headless compute passes, and test fixtures.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lgpu::{Device, Queue};
    /// # fn example(device: &Device, queue: &Queue) -> Result<(), lgpu::Error> {
    /// let encoder = device.create_command_encoder()?;
    /// queue.run(encoder)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run(&self, encoder: CommandEncoder) -> Result<(), Error> {
        let cmd = encoder.finish();
        self.submit(cmd)
    }

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
        self.vulkan_device
            .submit_and_wait(command_buffer.recorder)?;
        Ok(())
    }

    /// Submits command buffer work targeted to a swapchain surface without presenting.
    /// The acquired surface can then be presented explicitly via [`SurfaceTexture::present`].
    pub fn submit_surface_work(
        &self,
        command_buffer: CommandBuffer,
        surface_texture: &SurfaceTexture,
    ) -> Result<(), Error> {
        self.vulkan_device.submit_for_surface(
            command_buffer.recorder,
            surface_texture.acquire_semaphore,
            surface_texture.render_semaphore,
            surface_texture.frame_fence,
        )?;
        Ok(())
    }

    /// Submits command buffer work targeted to a swapchain surface and immediately queues presentation.
    pub fn submit_surface(
        &self,
        command_buffer: CommandBuffer,
        surface_texture: SurfaceTexture,
    ) -> Result<(), Error> {
        self.submit_surface_work(command_buffer, &surface_texture)?;
        surface_texture.present(self)
    }

    pub fn submit_batches(&self, command_buffers: Vec<CommandBuffer>) -> Result<(), Error> {
        let recorders = command_buffers.into_iter().map(|cb| cb.recorder).collect();
        self.vulkan_device.submit_batches_and_wait(recorders)?;
        Ok(())
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
        let recorders = command_buffers.into_iter().map(|cb| cb.recorder).collect();
        let mut raw_wait = Vec::with_capacity(wait_semaphores.len());
        for (s, v) in wait_semaphores {
            let sem = s.hal_semaphore.as_ref().ok_or_else(|| {
                Error::Hal(lgpu_hal::HalError::InvalidParameter(
                    "Wait semaphore is missing or destroyed".to_string(),
                ))
            })?;
            raw_wait.push((sem, *v));
        }
        let mut raw_signal = Vec::with_capacity(signal_semaphores.len());
        for (s, v) in signal_semaphores {
            let sem = s.hal_semaphore.as_ref().ok_or_else(|| {
                Error::Hal(lgpu_hal::HalError::InvalidParameter(
                    "Signal semaphore is missing or destroyed".to_string(),
                ))
            })?;
            raw_signal.push((sem, *v));
        }
        self.vulkan_device
            .submit_batches_async(recorders, &raw_wait, &raw_signal)?;
        Ok(())
    }

    pub fn write_texture(&self, texture: &Texture, data: &[u8]) -> Result<(), Error> {
        let raw_tex = texture.hal_texture.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Texture hal resource is missing or destroyed".to_string(),
            ))
        })?;
        self.vulkan_device.write_texture(raw_tex, data)?;
        Ok(())
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
