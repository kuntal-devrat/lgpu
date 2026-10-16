//! Vulkan swapchain presentation and surface integration.

use super::{VulkanCommandRecorder, VulkanDevice, VulkanTexture};
use crate::HalError;
use ash::khr::surface;
use ash::vk;

pub struct VulkanSwapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
}

impl VulkanDevice {
    pub fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        surface_loader: &surface::Instance,
        width: u32,
        height: u32,
        old_swapchain: vk::SwapchainKHR,
        preferred_format: Option<vk::Format>,
        preferred_present_mode: Option<vk::PresentModeKHR>,
    ) -> Result<VulkanSwapchain, HalError> {
        let loader = self.swapchain_loader.as_ref().ok_or_else(|| {
            HalError::BackendError("VK_KHR_swapchain is not supported or not enabled".to_string())
        })?;

        let caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, surface)
                .map_err(|e| {
                    HalError::BackendError(format!("Failed to query surface capabilities: {e}"))
                })?
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(self.physical_device, surface)
                .map_err(|e| {
                    HalError::BackendError(format!("Failed to query surface formats: {e}"))
                })?
        };

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, surface)
                .map_err(|e| {
                    HalError::BackendError(format!("Failed to query surface present modes: {e}"))
                })?
        };

        if formats.is_empty() {
            return Err(HalError::BackendError(
                "No surface formats supported".to_string(),
            ));
        }

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let selected_format = if let Some(pref) = preferred_format {
            formats
                .iter()
                .find(|f| f.format == pref)
                .cloned()
                .unwrap_or(formats[0])
        } else {
            formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_UNORM
                        || f.format == vk::Format::R8G8B8A8_UNORM
                        || f.format == vk::Format::B8G8R8A8_SRGB
                        || f.format == vk::Format::R8G8B8A8_SRGB
                })
                .cloned()
                .unwrap_or(formats[0])
        };

        let present_mode = match preferred_present_mode {
            Some(mode) if present_modes.contains(&mode) => mode,
            _ => {
                if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                    vk::PresentModeKHR::MAILBOX
                } else {
                    vk::PresentModeKHR::FIFO
                }
            }
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(selected_format.format)
            .image_color_space(selected_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe {
            loader
                .create_swapchain(&create_info, None)
                .map_err(|e| HalError::BackendError(format!("Failed to create swapchain: {e}")))?
        };

        let images = match unsafe { loader.get_swapchain_images(swapchain) } {
            Ok(imgs) => imgs,
            Err(e) => {
                unsafe {
                    loader.destroy_swapchain(swapchain, None);
                }
                return Err(HalError::BackendError(format!(
                    "Failed to get swapchain images: {e}"
                )));
            }
        };

        let mut views = Vec::with_capacity(images.len());
        for &img in &images {
            let view_create_info = vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(selected_format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let view = match unsafe { self.device.create_image_view(&view_create_info, None) } {
                Ok(v) => v,
                Err(e) => {
                    unsafe {
                        for v in views {
                            self.device.destroy_image_view(v, None);
                        }
                        loader.destroy_swapchain(swapchain, None);
                    }
                    return Err(HalError::BackendError(format!(
                        "Failed to create swapchain image view: {e}"
                    )));
                }
            };
            views.push(view);
        }

        Ok(VulkanSwapchain {
            swapchain,
            images,
            views,
            width: extent.width,
            height: extent.height,
            format: selected_format.format,
        })
    }

    pub fn destroy_swapchain(&self, swapchain: VulkanSwapchain) {
        if let Some(loader) = &self.swapchain_loader {
            unsafe {
                for view in swapchain.views {
                    self.device.destroy_image_view(view, None);
                }
                loader.destroy_swapchain(swapchain.swapchain, None);
            }
        }
    }

    pub fn acquire_next_image(
        &self,
        swapchain: &VulkanSwapchain,
        semaphore: vk::Semaphore,
        timeout_nanos: u64,
    ) -> Result<(u32, bool), HalError> {
        let loader = self
            .swapchain_loader
            .as_ref()
            .ok_or_else(|| HalError::BackendError("VK_KHR_swapchain is not enabled".to_string()))?;
        unsafe {
            loader
                .acquire_next_image(
                    swapchain.swapchain,
                    timeout_nanos,
                    semaphore,
                    vk::Fence::null(),
                )
                .map_err(|e| HalError::BackendError(format!("acquire_next_image failed: {e}")))
        }
    }

    pub fn queue_present(
        &self,
        swapchain: &VulkanSwapchain,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool, HalError> {
        self.queue_present_raw(swapchain.swapchain, image_index, wait_semaphores)
    }

    pub fn queue_present_raw(
        &self,
        swapchain: vk::SwapchainKHR,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool, HalError> {
        let loader = self
            .swapchain_loader
            .as_ref()
            .ok_or_else(|| HalError::BackendError("VK_KHR_swapchain is not enabled".to_string()))?;

        let swapchains = [swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let queue = self.queue.lock().unwrap();
        unsafe {
            loader
                .queue_present(*queue, &present_info)
                .map_err(|e| HalError::BackendError(format!("queue_present failed: {e}")))
        }
    }

    pub fn cmd_transition_to_present(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
    ) {
        let old_layout = texture.transition_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let new_layout = vk::ImageLayout::PRESENT_SRC_KHR;

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let (src_stage, src_access) = match old_layout {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            _ => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::TRANSFER_READ,
            ),
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::empty())
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(texture.image)
            .subresource_range(subresource_range);

        let barriers = [barrier];
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        unsafe {
            self.device
                .cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);
        }
    }

    pub fn submit_for_surface(
        &self,
        recorder: VulkanCommandRecorder,
        wait_semaphore: vk::Semaphore,
        signal_semaphore: vk::Semaphore,
        fence: Option<vk::Fence>,
    ) -> Result<(), HalError> {
        unsafe {
            self.device
                .end_command_buffer(recorder.command_buffer)
                .map_err(|e| HalError::CommandError(e.to_string()))?;

            let cmd_buffers = [recorder.command_buffer];
            let wait_sems = [wait_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let wait_values = [0u64];

            let next_val = self
                .internal_frame_counter
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                + 1;
            let signal_sems = [signal_semaphore, self.internal_frame_semaphore];
            let signal_values = [0u64, next_val];

            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cmd_buffers)
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&signal_sems)
                .push_next(&mut timeline_info);

            let raw_fence = fence.unwrap_or(vk::Fence::null());
            {
                let queue = self.queue.lock().unwrap();
                self.device
                    .queue_submit(*queue, &[submit_info], raw_fence)
                    .map_err(|e| HalError::CommandError(e.to_string()))?;
            }

            self.pending_async_recorders
                .lock()
                .unwrap()
                .push((next_val, vec![recorder]));

            Ok(())
        }
    }
}
