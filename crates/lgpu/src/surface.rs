//! Native window surface and presentation.

use std::sync::Arc;
use crate::{
    Device, Error, HalError, Queue, Texture, TextureFormat, TextureUsage,
};

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
    pub(crate) acquire_semaphores: Vec<lgpu_hal::vulkan::vk::Semaphore>,
    #[cfg(feature = "vulkan")]
    pub(crate) render_semaphores: Vec<lgpu_hal::vulkan::vk::Semaphore>,
    #[cfg(feature = "vulkan")]
    pub(crate) in_flight_fences: Vec<lgpu_hal::vulkan::vk::Fence>,
    #[cfg(feature = "vulkan")]
    pub(crate) current_frame: usize,
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
    #[cfg(feature = "vulkan")]
    pub(crate) frame_fence: Option<lgpu_hal::vulkan::vk::Fence>,
    pub suboptimal: bool,
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
                TextureFormat::Bgra8Unorm
                | TextureFormat::Bgra8Srgb
                | TextureFormat::Rgba8Unorm
                | TextureFormat::Rgba8Srgb => Some(config.format.to_vk()),
                _ => None,
            };

            let vk_present_mode = match config.present_mode {
                PresentMode::Immediate => Some(lgpu_hal::vulkan::vk::PresentModeKHR::IMMEDIATE),
                PresentMode::Mailbox => Some(lgpu_hal::vulkan::vk::PresentModeKHR::MAILBOX),
                PresentMode::Fifo => Some(lgpu_hal::vulkan::vk::PresentModeKHR::FIFO),
            };

            // Wait for GPU to finish work before re-creating swapchain
            let _ = vulkan_device.wait_idle();

            // Keep old swapchain handle valid until new swapchain is created
            let old_swapchain = self.swapchain.take();
            let old_raw = old_swapchain
                .as_ref()
                .map(|sc| sc.swapchain)
                .unwrap_or(lgpu_hal::vulkan::vk::SwapchainKHR::null());

            let swapchain_res = vulkan_device.create_swapchain(
                self.raw_surface,
                surface_loader,
                config.width,
                config.height,
                old_raw,
                vk_format,
                vk_present_mode,
            );

            // Destroy the old swapchain now that create_swapchain has executed
            if let Some(old) = old_swapchain {
                vulkan_device.destroy_swapchain(old);
            }

            let swapchain = swapchain_res?;

            // Create per-frame-in-flight sync objects if not already allocated
            const MAX_FRAMES_IN_FLIGHT: usize = 2;
            if self.acquire_semaphores.is_empty() {
                let sem_info = lgpu_hal::vulkan::vk::SemaphoreCreateInfo::default();
                let fence_info = lgpu_hal::vulkan::vk::FenceCreateInfo::default()
                    .flags(lgpu_hal::vulkan::vk::FenceCreateFlags::SIGNALED);

                let mut created_acqs = Vec::new();
                let mut created_rends = Vec::new();
                let mut created_fences = Vec::new();

                let mut create_sync_objects = || -> Result<(), HalError> {
                    for _ in 0..MAX_FRAMES_IN_FLIGHT {
                        let acq = unsafe {
                            vulkan_device.device.create_semaphore(&sem_info, None)
                                .map_err(|e| HalError::BackendError(format!("Failed to create acquire semaphore: {e}")))?
                        };
                        created_acqs.push(acq);
                        let rend = unsafe {
                            vulkan_device.device.create_semaphore(&sem_info, None)
                                .map_err(|e| HalError::BackendError(format!("Failed to create render semaphore: {e}")))?
                        };
                        created_rends.push(rend);
                        let fence = unsafe {
                            vulkan_device.device.create_fence(&fence_info, None)
                                .map_err(|e| HalError::BackendError(format!("Failed to create in-flight fence: {e}")))?
                        };
                        created_fences.push(fence);
                    }
                    Ok(())
                };

                if let Err(e) = create_sync_objects() {
                    unsafe {
                        for s in created_acqs {
                            vulkan_device.device.destroy_semaphore(s, None);
                        }
                        for s in created_rends {
                            vulkan_device.device.destroy_semaphore(s, None);
                        }
                        for f in created_fences {
                            vulkan_device.device.destroy_fence(f, None);
                        }
                    }
                    return Err(e.into());
                }

                self.acquire_semaphores.extend(created_acqs);
                self.render_semaphores.extend(created_rends);
                self.in_flight_fences.extend(created_fences);
            }

            self.swapchain = Some(swapchain);
            self.device = Some(Arc::clone(vulkan_device));
            self.config = Some(config.clone());
            self.current_frame = 0;
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

            if self.acquire_semaphores.is_empty() || self.render_semaphores.is_empty() || self.in_flight_fences.is_empty() {
                return Err(HalError::BackendError("Surface sync primitives not initialized".to_string()).into());
            }

            let frame_idx = self.current_frame;
            let fence = self.in_flight_fences[frame_idx];
            let acq_sem = self.acquire_semaphores[frame_idx];
            let rend_sem = self.render_semaphores[frame_idx];

            // Wait for previous frame using this slot to complete on GPU
            unsafe {
                device.device.wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(|e| HalError::BackendError(format!("Failed to wait for frame fence: {e}")))?;
                device.device.reset_fences(&[fence])
                    .map_err(|e| HalError::BackendError(format!("Failed to reset frame fence: {e}")))?;
            }

            let (image_index, suboptimal) = device.acquire_next_image(swapchain, acq_sem, u64::MAX)?;

            let image = swapchain.images[image_index as usize];
            let view = swapchain.views[image_index as usize];
            let config = self.config.as_ref().ok_or_else(|| {
                HalError::BackendError("Surface configuration missing".to_string())
            })?;

            let hal_texture = lgpu_hal::vulkan::VulkanTexture {
                image,
                view,
                memory: lgpu_hal::vulkan::vk::DeviceMemory::null(),
                block_id: None,
                offset: 0,
                allocated_size: 0,
                layout: std::sync::Mutex::new(lgpu_hal::vulkan::vk::ImageLayout::UNDEFINED),
                width: swapchain.width,
                height: swapchain.height,
                mip_levels: 1,
                sample_count: 1,
                format: swapchain.format,
                owned: false,
            };

            let texture = Texture {
                hal_texture: Some(hal_texture),
                vulkan_device: Arc::clone(device),
                width: swapchain.width,
                height: swapchain.height,
                format: config.format,
                id: None,
                owned: false,
            };

            self.current_frame = (self.current_frame + 1) % self.acquire_semaphores.len();

            Ok(SurfaceTexture {
                texture,
                image_index,
                swapchain: swapchain.swapchain,
                acquire_semaphore: acq_sem,
                render_semaphore: rend_sem,
                frame_fence: Some(fence),
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
                for acq in self.acquire_semaphores.drain(..) {
                    unsafe { device.device.destroy_semaphore(acq, None); }
                }
                for rend in self.render_semaphores.drain(..) {
                    unsafe { device.device.destroy_semaphore(rend, None); }
                }
                for fence in self.in_flight_fences.drain(..) {
                    unsafe { device.device.destroy_fence(fence, None); }
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
