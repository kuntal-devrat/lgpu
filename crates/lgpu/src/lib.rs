//! # lgpu (Lean GPU)
//!
//! An ultra-lightweight, bindless-first cross-platform GPU abstraction for Rust and C.
//! Designed for sub-second compile times, minimal binary footprint, and maximum performance.

use std::sync::Arc;

pub use lgpu_hal::{
    AdapterInfo, AddressMode, Backend, BufferDesc, BufferUsage, DeviceType, FilterMode, HalError,
    MemoryLocation, SamplerDesc, TextureDesc, TextureFormat, TextureUsage,
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

pub use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    Immediate,
    Mailbox,
    Fifo,
}

#[derive(Debug, Clone)]
pub struct SurfaceConfiguration {
    pub usage: TextureUsage,
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
    pub present_mode: PresentMode,
}

pub struct Surface {
    #[cfg(feature = "vulkan")]
    pub(crate) instance: Arc<lgpu_hal::vulkan::VulkanInstance>,
    #[cfg(feature = "vulkan")]
    pub(crate) raw_surface: lgpu_hal::vulkan::vk::SurfaceKHR,
    #[cfg(feature = "vulkan")]
    pub(crate) swapchain: Option<lgpu_hal::vulkan::VulkanSwapchain>,
    #[cfg(feature = "vulkan")]
    pub(crate) device: Option<Arc<lgpu_hal::vulkan::VulkanDevice>>,
    #[cfg(feature = "vulkan")]
    pub(crate) acquire_semaphore: Option<lgpu_hal::vulkan::vk::Semaphore>,
    #[cfg(feature = "vulkan")]
    pub(crate) render_semaphore: Option<lgpu_hal::vulkan::vk::Semaphore>,
    pub config: Option<SurfaceConfiguration>,
}

pub struct SurfaceTexture {
    pub texture: Texture,
    #[cfg(feature = "vulkan")]
    pub(crate) image_index: u32,
    #[cfg(feature = "vulkan")]
    pub(crate) swapchain: lgpu_hal::vulkan::vk::SwapchainKHR,
    #[cfg(feature = "vulkan")]
    pub(crate) acquire_semaphore: lgpu_hal::vulkan::vk::Semaphore,
    #[cfg(feature = "vulkan")]
    pub(crate) render_semaphore: lgpu_hal::vulkan::vk::Semaphore,
    pub suboptimal: bool,
}

pub struct Instance {
    #[cfg(feature = "vulkan")]
    vulkan_instance: Arc<lgpu_hal::vulkan::VulkanInstance>,
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
    pub id: Option<TextureId>,
}

pub struct Sampler {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_sampler: Option<lgpu_hal::vulkan::VulkanSampler>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub id: Option<SamplerId>,
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

pub struct GraphicsPipeline {
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

pub struct TimelineSemaphore {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_semaphore: Option<lgpu_hal::vulkan::VulkanTimelineSemaphore>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}


impl Instance {
    pub fn new() -> Result<Self, Error> {
        #[cfg(feature = "vulkan")]
        {
            let vulkan_instance = Arc::new(lgpu_hal::vulkan::VulkanInstance::new()?);
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

    pub fn create_surface<W: HasDisplayHandle + HasWindowHandle>(
        &self,
        window: &W,
    ) -> Result<Surface, Error> {
        let display = window
            .display_handle()
            .map_err(|e| HalError::BackendError(format!("Failed to get display handle: {e}")))?
            .as_raw();
        let window_h = window
            .window_handle()
            .map_err(|e| HalError::BackendError(format!("Failed to get window handle: {e}")))?
            .as_raw();
        self.create_surface_raw(display, window_h)
    }

    pub fn create_surface_raw(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Surface, Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_surface = self.vulkan_instance.create_surface(display_handle, window_handle)?;
            Ok(Surface {
                instance: Arc::clone(&self.vulkan_instance),
                raw_surface,
                swapchain: None,
                device: None,
                acquire_semaphore: None,
                render_semaphore: None,
                config: None,
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (display_handle, window_handle);
            Err(HalError::BackendError("Vulkan not enabled for surface creation".to_string()).into())
        }
    }
}

impl Surface {
    pub fn configure(&mut self, device: &Device, config: &SurfaceConfiguration) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let vulkan_device = &device.vulkan_device;
            let surface_loader = self.instance.surface_loader.as_ref().ok_or_else(|| {
                HalError::BackendError("Surface loader not available".to_string())
            })?;

            let vk_format = match config.format {
                TextureFormat::Bgra8Unorm => Some(lgpu_hal::vulkan::vk::Format::B8G8R8A8_UNORM),
                TextureFormat::Bgra8Srgb => Some(lgpu_hal::vulkan::vk::Format::B8G8R8A8_SRGB),
                TextureFormat::Rgba8Unorm => Some(lgpu_hal::vulkan::vk::Format::R8G8B8A8_UNORM),
                TextureFormat::Rgba8Srgb => Some(lgpu_hal::vulkan::vk::Format::R8G8B8A8_SRGB),
                _ => None,
            };

            let vk_present_mode = match config.present_mode {
                PresentMode::Immediate => Some(lgpu_hal::vulkan::vk::PresentModeKHR::IMMEDIATE),
                PresentMode::Mailbox => Some(lgpu_hal::vulkan::vk::PresentModeKHR::MAILBOX),
                PresentMode::Fifo => Some(lgpu_hal::vulkan::vk::PresentModeKHR::FIFO),
            };

            let old_swapchain = self.swapchain.take().map(|sc| {
                let raw_sc = sc.swapchain;
                vulkan_device.destroy_swapchain(sc);
                raw_sc
            }).unwrap_or(lgpu_hal::vulkan::vk::SwapchainKHR::null());

            let swapchain = vulkan_device.create_swapchain(
                self.raw_surface,
                surface_loader,
                config.width,
                config.height,
                old_swapchain,
                vk_format,
                vk_present_mode,
            )?;

            if self.acquire_semaphore.is_none() {
                let sem_info = lgpu_hal::vulkan::vk::SemaphoreCreateInfo::default();
                let acq = unsafe {
                    vulkan_device.device.create_semaphore(&sem_info, None)
                        .map_err(|e| HalError::BackendError(format!("Failed to create acquire semaphore: {e}")))?
                };
                let rend = unsafe {
                    vulkan_device.device.create_semaphore(&sem_info, None)
                        .map_err(|e| HalError::BackendError(format!("Failed to create render semaphore: {e}")))?
                };
                self.acquire_semaphore = Some(acq);
                self.render_semaphore = Some(rend);
            }

            self.swapchain = Some(swapchain);
            self.device = Some(Arc::clone(vulkan_device));
            self.config = Some(config.clone());
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (device, config);
            Ok(())
        }
    }

    pub fn get_current_texture(&mut self) -> Result<SurfaceTexture, Error> {
        #[cfg(feature = "vulkan")]
        {
            let swapchain = self.swapchain.as_ref().ok_or_else(|| {
                HalError::BackendError("Surface is not configured".to_string())
            })?;
            let device = self.device.as_ref().ok_or_else(|| {
                HalError::BackendError("Surface has no associated device".to_string())
            })?;
            let acq_sem = self.acquire_semaphore.unwrap();
            let rend_sem = self.render_semaphore.unwrap();

            let (image_index, suboptimal) = device.acquire_next_image(swapchain, acq_sem, u64::MAX)?;

            let image = swapchain.images[image_index as usize];
            let view = swapchain.views[image_index as usize];
            let config = self.config.as_ref().unwrap();

            let hal_texture = lgpu_hal::vulkan::VulkanTexture {
                image,
                view,
                memory: lgpu_hal::vulkan::vk::DeviceMemory::null(),
                current_layout: std::sync::atomic::AtomicI32::new(lgpu_hal::vulkan::vk::ImageLayout::UNDEFINED.as_raw()),
                width: swapchain.width,
                height: swapchain.height,
                format: swapchain.format,
            };

            let texture = Texture {
                hal_texture: Some(hal_texture),
                vulkan_device: Arc::clone(device),
                width: swapchain.width,
                height: swapchain.height,
                format: config.format,
                id: None,
            };

            Ok(SurfaceTexture {
                texture,
                image_index,
                swapchain: swapchain.swapchain,
                acquire_semaphore: acq_sem,
                render_semaphore: rend_sem,
                suboptimal,
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Err(HalError::BackendError("Vulkan not enabled".to_string()).into())
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(device) = &self.device {
                let _ = device.wait_idle();
                if let Some(acq) = self.acquire_semaphore.take() {
                    unsafe { device.device.destroy_semaphore(acq, None); }
                }
                if let Some(rend) = self.render_semaphore.take() {
                    unsafe { device.device.destroy_semaphore(rend, None); }
                }
                if let Some(swapchain) = self.swapchain.take() {
                    device.destroy_swapchain(swapchain);
                }
            }
            self.instance.destroy_surface(self.raw_surface);
        }
    }
}

impl SurfaceTexture {
    pub fn present(self, queue: &Queue) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            queue.vulkan_device.queue_present_raw(
                self.swapchain,
                self.image_index,
                &[self.render_semaphore],
            )?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = queue;
            Ok(())
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
                pipeline,
                vulkan_device: Arc::clone(&self.vulkan_device),
            })
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (shader, vs_entry, fs_entry, color_format, depth_format, depth_test, depth_write);
            Err(HalError::NoAdapterFound.into())
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
}

impl Buffer {
    pub fn id(&self) -> Option<BufferId> {
        self.id
    }

    pub fn write(&self, data: &[u8]) -> Result<(), Error> {
        self.write_at(0, data)
    }

    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_buf = self.hal_buffer.as_ref().unwrap();
            self.vulkan_device.write_buffer(raw_buf, offset, data)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (offset, data);
            Ok(())
        }
    }

    pub fn read(&self, data: &mut [u8]) -> Result<(), Error> {
        self.read_at(0, data)
    }

    pub fn read_at(&self, offset: u64, data: &mut [u8]) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_buf = self.hal_buffer.as_ref().unwrap();
            self.vulkan_device.read_buffer(raw_buf, offset, data)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (offset, data);
            Ok(())
        }
    }
}

impl Texture {
    pub fn id(&self) -> Option<TextureId> {
        self.id
    }
}

impl Sampler {
    pub fn id(&self) -> Option<SamplerId> {
        self.id
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
            if let Some(tex_id) = self.id {
                self.vulkan_device.unregister_bindless_texture(tex_id.0);
            }
            if let Some(tex) = self.hal_texture.take() {
                self.vulkan_device.destroy_texture(tex);
            }
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(sampler_id) = self.id {
                self.vulkan_device.unregister_bindless_sampler(sampler_id.0);
            }
            if let Some(sampler) = self.hal_sampler.take() {
                self.vulkan_device.destroy_sampler(sampler);
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

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            self.vulkan_device.destroy_raw_pipeline(self.pipeline);
        }
    }
}

impl TimelineSemaphore {
    pub fn value(&self) -> Result<u64, Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_sem = self.hal_semaphore.as_ref().unwrap();
            let val = self.vulkan_device.get_timeline_semaphore_value(raw_sem)?;
            Ok(val)
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Ok(0)
        }
    }

    pub fn wait(&self, value: u64, timeout_nanos: u64) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_sem = self.hal_semaphore.as_ref().unwrap();
            self.vulkan_device.wait_timeline_semaphore(raw_sem, value, timeout_nanos)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (value, timeout_nanos);
            Ok(())
        }
    }

    pub fn signal(&self, value: u64) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_sem = self.hal_semaphore.as_ref().unwrap();
            self.vulkan_device.signal_timeline_semaphore(raw_sem, value)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = value;
            Ok(())
        }
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(sem) = self.hal_semaphore.take() {
                self.vulkan_device.destroy_timeline_semaphore(sem);
            }
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

    pub fn draw(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        vertex_count: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_tex = texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.cmd_draw(
                recorder,
                raw_tex,
                pipeline.pipeline,
                vertex_count,
                push_constants,
                clear_color,
            );
        }
    }

    pub fn clear_and_render_with_depth(
        &mut self,
        color_texture: &Texture,
        depth_texture: &Texture,
        clear_color: [f32; 4],
        clear_depth: f32,
    ) {
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
            );
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
    ) {
        #[cfg(feature = "vulkan")]
        {
            let recorder = self.recorder.as_ref().unwrap();
            let raw_color = color_texture.hal_texture.as_ref().unwrap();
            let raw_depth = depth_texture.hal_texture.as_ref().unwrap();
            self.vulkan_device.cmd_draw_with_depth(
                recorder,
                raw_color,
                raw_depth,
                pipeline.pipeline,
                vertex_count,
                push_constants,
                clear_color,
                clear_depth,
            );
        }
    }

    pub fn render_triangle(
        &mut self,
        texture: &Texture,
        pipeline: &GraphicsPipeline,
        clear_color: [f32; 4],
    ) {
        self.draw(texture, pipeline, 3, &[], clear_color);
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
}

