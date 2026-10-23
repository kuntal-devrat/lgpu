//! Vulkan logical device and execution context.

use super::{allocator, bindless, VulkanCommandRecorder};
use crate::HalError;
use ash::vk;

pub struct VulkanDevice {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: std::sync::Mutex<vk::Queue>,
    pub queue_family_index: u32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub non_coherent_atom_size: u64,
    pub bindless_heap: Option<bindless::VulkanBindlessHeap>,
    pub swapchain_loader: Option<ash::khr::swapchain::Device>,
    pub command_recorder_cache: std::sync::Mutex<Vec<(vk::CommandPool, vk::CommandBuffer)>>,
    pub fence_cache: std::sync::Mutex<Vec<vk::Fence>>,
    pub internal_frame_semaphore: vk::Semaphore,
    pub internal_frame_counter: std::sync::atomic::AtomicU64,
    pub pending_async_recorders: std::sync::Mutex<Vec<(u64, Vec<VulkanCommandRecorder>)>>,
    pub allocator: std::sync::Mutex<allocator::VulkanSuballocator>,
    pub pipeline_cache: vk::PipelineCache,
}

impl VulkanDevice {
    pub(crate) fn find_memory_type(
        &self,
        type_filter: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Result<(u32, bool), HalError> {
        let wants_host = required.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        if wants_host {
            let preferred = required | vk::MemoryPropertyFlags::HOST_COHERENT;
            for i in 0..self.memory_properties.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && (self.memory_properties.memory_types[i as usize].property_flags & preferred)
                        == preferred
                {
                    return Ok((i, true));
                }
            }
        }

        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & required)
                    == required
            {
                let is_coherent = (self.memory_properties.memory_types[i as usize].property_flags
                    & vk::MemoryPropertyFlags::HOST_COHERENT)
                    == vk::MemoryPropertyFlags::HOST_COHERENT;
                return Ok((i, is_coherent));
            }
        }
        Err(HalError::OutOfMemory(
            "Suitable memory type not found".to_string(),
        ))
    }

    pub fn wait_idle(&self) -> Result<(), HalError> {
        unsafe {
            self.device
                .device_wait_idle()
                .map_err(|e| HalError::BackendError(format!("device_wait_idle failed: {e}")))?;
        }
        let mut pending = self.pending_async_recorders.lock().unwrap();
        let mut cache = self.command_recorder_cache.lock().unwrap();
        for (_, recorders) in pending.drain(..) {
            for r in recorders {
                unsafe {
                    self.device
                        .reset_command_buffer(
                            r.command_buffer,
                            vk::CommandBufferResetFlags::empty(),
                        )
                        .ok();
                }
                cache.push((r.command_pool, r.command_buffer));
            }
        }
        Ok(())
    }

    /// Reclaims completely empty memory blocks in the suballocator back to Vulkan.
    pub fn trim_memory(&self) -> usize {
        self.allocator.lock().unwrap().trim(&self.device)
    }

    /// Returns the number of memory blocks currently held by the suballocator.
    pub fn suballocator_block_count(&self) -> usize {
        self.allocator.lock().unwrap().block_count()
    }

    /// Returns the number of empty memory blocks currently held in the suballocator.
    pub fn suballocator_empty_block_count(&self) -> usize {
        self.allocator.lock().unwrap().empty_block_count()
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(heap) = self.bindless_heap.take() {
                heap.destroy(&self.device);
            }
            self.device
                .destroy_semaphore(self.internal_frame_semaphore, None);
            for (pool, _) in self.command_recorder_cache.lock().unwrap().drain(..) {
                self.device.destroy_command_pool(pool, None);
            }
            for (_, recorders) in self.pending_async_recorders.lock().unwrap().drain(..) {
                for r in recorders {
                    self.device.destroy_command_pool(r.command_pool, None);
                }
            }
            for fence in self.fence_cache.lock().unwrap().drain(..) {
                self.device.destroy_fence(fence, None);
            }
            self.allocator.lock().unwrap().destroy(&self.device);
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_device(None);
        }
    }
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}
