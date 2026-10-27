//! Native window surface and presentation.

use crate::{Device, Error, HalError, Queue, Texture, TextureFormat, TextureUsage};
use std::sync::Arc;

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
    pub(crate) instance: Arc<lgpu_hal::vulkan::VulkanInstance>,
    pub(crate) raw_surface: lgpu_hal::vulkan::vk::SurfaceKHR,
    pub(crate) swapchain: Option<lgpu_hal::vulkan::VulkanSwapchain>,
    pub(crate) device: Option<Arc<lgpu_hal::vulkan::VulkanDevice>>,
    pub(crate) acquire_semaphores: Vec<lgpu_hal::vulkan::vk::Semaphore>,
    pub(crate) render_semaphores: Vec<lgpu_hal::vulkan::vk::Semaphore>,
    pub(crate) in_flight_fences: Vec<lgpu_hal::vulkan::vk::Fence>,
    pub(crate) current_frame: usize,
    pub config: Option<SurfaceConfiguration>,
}

pub struct SurfaceTexture {
    pub texture: Texture,
    pub(crate) image_index: u32,
    pub(crate) swapchain: lgpu_hal::vulkan::vk::SwapchainKHR,
    pub(crate) acquire_semaphore: lgpu_hal::vulkan::vk::Semaphore,
    pub(crate) render_semaphore: lgpu_hal::vulkan::vk::Semaphore,
    pub(crate) frame_fence: Option<lgpu_hal::vulkan::vk::Fence>,
    pub suboptimal: bool,
    pub(crate) submitted: std::sync::atomic::AtomicBool,
    pub(crate) presented: std::sync::atomic::AtomicBool,
}

impl Surface {
    pub fn configure(
        &mut self,
        device: &Device,
        config: &SurfaceConfiguration,
    ) -> Result<(), Error> {
        let vulkan_device = &device.vulkan_device;
        self.device = Some(Arc::clone(vulkan_device));
        self.config = Some(config.clone());

        // Handle zero-size windows (minimized) gracefully by pausing rendering
        if config.width == 0 || config.height == 0 {
            return Ok(());
        }

        let surface_loader =
            self.instance.surface_loader.as_ref().ok_or_else(|| {
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

        // Wait for current in-flight fences and GPU idle before re-creating swapchain
        if !self.in_flight_fences.is_empty() {
            unsafe {
                let _ =
                    vulkan_device
                        .device
                        .wait_for_fences(&self.in_flight_fences, true, u64::MAX);
            }
        }
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

        // Drain previous sync objects on reconfiguration to ensure all fences are reset to SIGNALED
        for acq in self.acquire_semaphores.drain(..) {
            unsafe {
                vulkan_device.device.destroy_semaphore(acq, None);
            }
        }
        for rend in self.render_semaphores.drain(..) {
            unsafe {
                vulkan_device.device.destroy_semaphore(rend, None);
            }
        }
        for fence in self.in_flight_fences.drain(..) {
            unsafe {
                vulkan_device.device.destroy_fence(fence, None);
            }
        }

        // Create per-frame-in-flight sync objects (triple buffering)
        const MAX_FRAMES_IN_FLIGHT: usize = 3;
        let sem_info = lgpu_hal::vulkan::vk::SemaphoreCreateInfo::default();
        let fence_info = lgpu_hal::vulkan::vk::FenceCreateInfo::default()
            .flags(lgpu_hal::vulkan::vk::FenceCreateFlags::SIGNALED);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let acq = unsafe {
                vulkan_device
                    .device
                    .create_semaphore(&sem_info, None)
                    .map_err(|e| HalError::BackendError(e.to_string()))?
            };
            let fence = unsafe {
                vulkan_device
                    .device
                    .create_fence(&fence_info, None)
                    .map_err(|e| HalError::BackendError(e.to_string()))?
            };
            self.acquire_semaphores.push(acq);
            self.in_flight_fences.push(fence);
        }

        // Create one render semaphore per swapchain image to ensure safe semaphore reuse
        for _ in 0..swapchain.images.len() {
            let rend = unsafe {
                vulkan_device
                    .device
                    .create_semaphore(&sem_info, None)
                    .map_err(|e| HalError::BackendError(e.to_string()))?
            };
            self.render_semaphores.push(rend);
        }

        self.swapchain = Some(swapchain);
        self.device = Some(Arc::clone(vulkan_device));
        self.config = Some(config.clone());
        self.current_frame = 0;
        Ok(())
    }

    pub fn get_current_texture(&mut self) -> Result<SurfaceTexture, Error> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| HalError::BackendError("Surface configuration missing".to_string()))?;

        if config.width == 0 || config.height == 0 {
            return Err(Error::Hal(HalError::NotReady(
                "Window extent is zero (minimized)".to_string(),
            )));
        }

        let swapchain = self
            .swapchain
            .as_ref()
            .ok_or_else(|| HalError::NotReady("Surface is not configured".to_string()))?;
        let device = self.device.as_ref().ok_or_else(|| {
            HalError::BackendError("Surface has no associated device".to_string())
        })?;

        if self.acquire_semaphores.is_empty()
            || self.render_semaphores.is_empty()
            || self.in_flight_fences.is_empty()
        {
            return Err(HalError::BackendError(
                "Surface sync primitives not initialized".to_string(),
            )
            .into());
        }

        let frame_idx = self.current_frame;
        let fence = self.in_flight_fences[frame_idx];
        let acq_sem = self.acquire_semaphores[frame_idx];

        // 1. Wait for previous frame using this slot to complete on GPU
        unsafe {
            device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| {
                    HalError::BackendError(format!("Failed to wait for frame fence: {e}"))
                })?;
        }

        // 2. Acquire next image. If it returns OutOfDate / NotReady, the fence is STILL signaled
        // so re-calling get_current_texture after reconfiguration won't deadlock!
        let (image_index, suboptimal) = device.acquire_next_image(swapchain, acq_sem, u64::MAX)?;

        let rend_sem = self.render_semaphores[image_index as usize];

        // 3. Reset fence only after acquire succeeds
        unsafe {
            device
                .device
                .reset_fences(&[fence])
                .map_err(|e| HalError::BackendError(format!("Failed to reset frame fence: {e}")))?;
        }

        let image = swapchain.images[image_index as usize];
        let view = swapchain.views[image_index as usize];
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| HalError::BackendError("Surface configuration missing".to_string()))?;

        let texture = Texture {
            hal_texture: Some(lgpu_hal::vulkan::VulkanTexture {
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
                format: config.format.to_vk(),
                owned: false,
            }),
            vulkan_device: Arc::clone(device),
            width: swapchain.width,
            height: swapchain.height,
            format: config.format,
            id: None,
            storage_id: None,
            slot_guard: None,
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
            submitted: std::sync::atomic::AtomicBool::new(false),
            presented: std::sync::atomic::AtomicBool::new(false),
        })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        if let Some(device) = &self.device {
            let _ = device.wait_idle();
            for acq in self.acquire_semaphores.drain(..) {
                unsafe {
                    device.device.destroy_semaphore(acq, None);
                }
            }
            for rend in self.render_semaphores.drain(..) {
                unsafe {
                    device.device.destroy_semaphore(rend, None);
                }
            }
            for fence in self.in_flight_fences.drain(..) {
                unsafe {
                    device.device.destroy_fence(fence, None);
                }
            }
            if let Some(swapchain) = self.swapchain.take() {
                device.destroy_swapchain(swapchain);
            }
        }
        self.instance.destroy_surface(self.raw_surface);
    }
}

impl SurfaceTexture {
    pub fn present(self, queue: &Queue) -> Result<(), Error> {
        self.presented
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let res = queue.vulkan_device.queue_present_raw(
            self.swapchain,
            self.image_index,
            &[self.render_semaphore],
        );
        match res {
            Ok(_) => Ok(()),
            Err(HalError::Suboptimal(_)) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

impl Drop for SurfaceTexture {
    fn drop(&mut self) {
        if self.presented.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }

        // Mark as presented to prevent any recursive or duplicate presents
        self.presented
            .store(true, std::sync::atomic::Ordering::SeqCst);

        let device = &self.texture.vulkan_device;

        if !self.submitted.load(std::sync::atomic::Ordering::SeqCst) {
            // Unsubmitted frame: fence was reset in get_current_texture and acquire_semaphore was signaled.
            // Submit a quick barrier transition to PRESENT_SRC_KHR to signal frame_fence and consume acquire_semaphore.
            if let Ok(recorder) = device.create_command_recorder() {
                let subresource = lgpu_hal::vulkan::vk::ImageSubresourceRange::default()
                    .aspect_mask(lgpu_hal::vulkan::vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let img = self
                    .texture
                    .hal_texture
                    .as_ref()
                    .map(|t| t.image)
                    .unwrap_or(lgpu_hal::vulkan::vk::Image::null());

                let barrier = lgpu_hal::vulkan::vk::ImageMemoryBarrier2::default()
                    .old_layout(lgpu_hal::vulkan::vk::ImageLayout::UNDEFINED)
                    .new_layout(lgpu_hal::vulkan::vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_stage_mask(lgpu_hal::vulkan::vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(lgpu_hal::vulkan::vk::AccessFlags2::empty())
                    .dst_stage_mask(lgpu_hal::vulkan::vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(lgpu_hal::vulkan::vk::AccessFlags2::empty())
                    .src_queue_family_index(lgpu_hal::vulkan::vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(lgpu_hal::vulkan::vk::QUEUE_FAMILY_IGNORED)
                    .image(img)
                    .subresource_range(subresource);

                let dep = lgpu_hal::vulkan::vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));

                unsafe {
                    device
                        .device
                        .cmd_pipeline_barrier2(recorder.command_buffer, &dep);
                }

                let _ = device.submit_for_surface(
                    recorder,
                    self.acquire_semaphore,
                    self.render_semaphore,
                    self.frame_fence,
                );
            }

            // Return the acquired image back to the presentation engine waiting on render_semaphore.
            let _ = device.queue_present_raw(
                self.swapchain,
                self.image_index,
                &[self.render_semaphore],
            );
        } else {
            // Frame work was submitted to GPU, but present() was never called.
            // Return the image back to the presentation engine waiting on render_semaphore.
            let _ = device.queue_present_raw(
                self.swapchain,
                self.image_index,
                &[self.render_semaphore],
            );
        }
    }
}
