//! Vulkan logical device and execution context.

use super::{allocator, bindless, staging, VulkanCommandRecorder};
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
    pub staging_ring: std::sync::Mutex<Option<staging::VulkanStagingRing>>,
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
        let mut pending = match self.pending_async_recorders.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut cache = match self.command_recorder_cache.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
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
                if cache.len() < 32 {
                    cache.push((r.command_pool, r.command_buffer));
                } else {
                    unsafe {
                        self.device.destroy_command_pool(r.command_pool, None);
                    }
                }
            }
        }
        Ok(())
    }

    /// Reclaims completely empty memory blocks in the suballocator and idle command pools back to Vulkan.
    pub fn trim_memory(&self) -> usize {
        let mut freed = match self.allocator.lock() {
            Ok(mut alloc) => alloc.trim(&self.device),
            Err(p) => p.into_inner().trim(&self.device),
        };
        let mut cache = match self.command_recorder_cache.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        for (pool, _) in cache.drain(..) {
            unsafe {
                self.device.destroy_command_pool(pool, None);
            }
            freed += 1;
        }
        freed
    }

    /// Returns the number of memory blocks currently held by the suballocator.
    pub fn suballocator_block_count(&self) -> usize {
        match self.allocator.lock() {
            Ok(a) => a.block_count(),
            Err(p) => p.into_inner().block_count(),
        }
    }

    /// Returns the number of empty memory blocks currently held in the suballocator.
    pub fn suballocator_empty_block_count(&self) -> usize {
        match self.allocator.lock() {
            Ok(a) => a.empty_block_count(),
            Err(p) => p.into_inner().empty_block_count(),
        }
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

            let mut command_cache = match self.command_recorder_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            for (pool, _) in command_cache.drain(..) {
                self.device.destroy_command_pool(pool, None);
            }

            let mut pending = match self.pending_async_recorders.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            for (_, recorders) in pending.drain(..) {
                for r in recorders {
                    self.device.destroy_command_pool(r.command_pool, None);
                }
            }

            let mut fence_cache = match self.fence_cache.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            for fence in fence_cache.drain(..) {
                self.device.destroy_fence(fence, None);
            }

            let staging_ring = match self.staging_ring.lock() {
                Ok(mut g) => g.take(),
                Err(p) => p.into_inner().take(),
            };
            if let Some(mut ring) = staging_ring {
                ring.destroy(&self.device);
            }

            let mut alloc = match self.allocator.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            alloc.destroy(&self.device);

            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
            self.device.destroy_device(None);
        }
    }
}

unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}
