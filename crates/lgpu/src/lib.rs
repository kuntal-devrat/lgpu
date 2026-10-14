//! # lgpu (Lean GPU)
//!
//! An ultra-lightweight, bindless-first cross-platform GPU abstraction for Rust and C.
//! Designed for sub-second compile times, minimal binary footprint, and maximum performance.

use std::sync::Arc;

pub use lgpu_hal::{
    AdapterInfo, Backend, BufferDesc, BufferUsage, DeviceType, HalError, MemoryLocation,
    TextureDesc, TextureFormat, TextureUsage,
};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Hardware Abstraction Layer error: {0}")]
    Hal(#[from] HalError),
    #[error("Buffer error: {0}")]
    Buffer(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerId(pub u32);

pub struct Instance {
    #[cfg(feature = "vulkan")]
    vulkan_instance: lgpu_hal::vulkan::VulkanInstance,
}

pub struct Adapter {
    #[cfg(feature = "vulkan")]
    vulkan_adapter: lgpu_hal::vulkan::VulkanAdapter,
}

pub struct Device {
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

pub struct Queue {
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

pub struct Buffer {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_buffer: Option<lgpu_hal::vulkan::VulkanBuffer>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub size: u64,
    pub id: Option<BufferId>,
}

pub struct Texture {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_texture: Option<lgpu_hal::vulkan::VulkanTexture>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

pub struct Shader {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_shader: Option<lgpu_hal::vulkan::VulkanShader>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

pub struct ComputePipeline {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_pipeline: Option<lgpu_hal::vulkan::VulkanComputePipeline>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

pub struct BindlessComputePipeline {
    #[cfg(feature = "vulkan")]
    pub(crate) pipeline: lgpu_hal::vulkan::RawPipeline,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

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

impl Instance {
    pub fn new() -> Result<Self, Error> {
        #[cfg(feature = "vulkan")]
        {
            let vulkan_instance = lgpu_hal::vulkan::VulkanInstance::new()?;
            Ok(Self { vulkan_instance })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Err(HalError::NoAdapterFound.into())
        }
    }

    pub fn enumerate_adapters(&self) -> Result<Vec<Adapter>, Error> {
        #[cfg(feature = "vulkan")]
        {
            let adapters = self.vulkan_instance.enumerate_adapters()?;
            Ok(adapters
                .into_iter()
                .map(|vulkan_adapter| Adapter { vulkan_adapter })
                .collect())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Ok(vec![])
        }
    }
}

impl Adapter {
    pub fn info(&self) -> &AdapterInfo {
        #[cfg(feature = "vulkan")]
        {
            self.vulkan_adapter.info()
        }
    }

    pub fn request_device(&self, instance: &Instance) -> Result<(Device, Queue), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_device = instance
                .vulkan_instance
                .create_device(&self.vulkan_adapter)?;
            let device_arc = Arc::new(raw_device);

            let device = Device {
                vulkan_device: Arc::clone(&device_arc),
            };
            let queue = Queue {
                vulkan_device: device_arc,
            };

            Ok((device, queue))
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Err(HalError::NoAdapterFound.into())
        }
    }
}

impl Device {
    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<Buffer, Error> {
        #[cfg(feature = "vulkan")]
        {
            let hal_buffer = self.vulkan_device.create_buffer(desc)?;
            let id = self
                .vulkan_device
                .register_bindless_buffer(&hal_buffer)
                .ok()
                .map(BufferId);

            Ok(Buffer {
                hal_buffer: Some(hal_buffer),
                vulkan_device: Arc::clone(&self.vulkan_device),
                size: desc.size,
                id,
            })
        }
    }

    pub fn create_buffer_init(&self, desc: &BufferDesc, data: &[u8]) -> Result<Buffer, Error> {
        let buffer = self.create_buffer(desc)?;
        buffer.write(data)?;
        Ok(buffer)
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
                pipeline,
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
    }

    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture, Error> {
        #[cfg(feature = "vulkan")]
        {
            let hal_texture = self.vulkan_device.create_texture(desc)?;
            Ok(Texture {
                hal_texture: Some(hal_texture),
                vulkan_device: Arc::clone(&self.vulkan_device),
                width: desc.width,
                height: desc.height,
                format: desc.format,
            })
        }
    }

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
}

impl Buffer {
    pub fn id(&self) -> Option<BufferId> {
        self.id
    }

    pub fn write(&self, data: &[u8]) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_buf = self.hal_buffer.as_ref().unwrap();
            self.vulkan_device.write_buffer(raw_buf, 0, data)?;
            Ok(())
        }
    }

    pub fn read(&self, data: &mut [u8]) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_buf = self.hal_buffer.as_ref().unwrap();
            self.vulkan_device.read_buffer(raw_buf, 0, data)?;
            Ok(())
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(buf_id) = self.id {
                self.vulkan_device.unregister_bindless_buffer(buf_id.0);
            }
            if let Some(buf) = self.hal_buffer.take() {
                self.vulkan_device.destroy_buffer(buf);
            }
        }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(tex) = self.hal_texture.take() {
                self.vulkan_device.destroy_texture(tex);
            }
        }
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(shader) = self.hal_shader.take() {
                self.vulkan_device.destroy_shader(shader);
            }
        }
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(pipeline) = self.hal_pipeline.take() {
                self.vulkan_device.destroy_compute_pipeline(pipeline);
            }
        }
    }
}

impl Drop for BindlessComputePipeline {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            self.vulkan_device.destroy_raw_pipeline(self.pipeline);
        }
    }
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
            self.vulkan_device.dispatch_bindless(
                recorder,
                pipeline.pipeline,
                push_constants,
                x,
                y,
                z,
            )?;
            Ok(())
        }
    }

    pub fn clear_and_render_texture(
        &mut self,
        texture: &Texture,
        clear_color: [f32; 4],
    ) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.cmd_clear_and_render(recorder, raw_tex, clear_color);
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

    pub fn finish(mut self) -> CommandBuffer {
        #[cfg(feature = "vulkan")]
        {
            CommandBuffer {
                recorder: self.recorder.take().unwrap(),
            }
        }
    }
}

impl Queue {
    pub fn submit(&self, command_buffer: CommandBuffer) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            self.vulkan_device
                .submit_and_wait(command_buffer.recorder)?;
            Ok(())
        }
    }
}
