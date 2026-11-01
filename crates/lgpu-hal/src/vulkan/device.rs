//! Vulkan logical device and execution context.

use super::{allocator, bindless, staging, VulkanCommandRecorder};
use crate::HalError;
use ash::vk;

pub struct VulkanDevice {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: std::sync::Mutex<vk::Queue>,
    pub queue_family_index: u32,
    /// Dedicated compute queue, when the adapter exposes a second compute-capable family.
    pub compute_queue: Option<std::sync::Mutex<vk::Queue>>,
    pub compute_queue_family_index: Option<u32>,
    /// Every queue family resources must be shareable with; one entry means exclusive access.
    pub queue_family_indices: Vec<u32>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub non_coherent_atom_size: u64,
    pub timestamp_valid_bits: u32,
    pub bindless_heap: Option<bindless::VulkanBindlessHeap>,
    pub swapchain_loader: Option<ash::khr::swapchain::Device>,
    pub debug_utils_device: Option<ash::ext::debug_utils::Device>,
    pub command_recorder_cache: std::sync::Mutex<Vec<(vk::CommandPool, vk::CommandBuffer)>>,
    pub compute_recorder_cache: std::sync::Mutex<Vec<(vk::CommandPool, vk::CommandBuffer)>>,
    pub fence_cache: std::sync::Mutex<Vec<vk::Fence>>,
    pub internal_frame_semaphore: vk::Semaphore,
    pub internal_frame_counter: std::sync::atomic::AtomicU64,
    pub pending_async_recorders: std::sync::Mutex<Vec<(u64, Vec<VulkanCommandRecorder>)>>,
    pub allocator: std::sync::Mutex<allocator::VulkanSuballocator>,
    pub staging_ring: std::sync::Mutex<Option<staging::VulkanStagingRing>>,
    pub pipeline_cache: vk::PipelineCache,
}

impl VulkanDevice {
    /// Sharing mode and family list for resources that may be used from every queue this device
    /// owns. With a single family, exclusive ownership is both cheaper and sufficient.
    pub fn sharing(&self) -> (vk::SharingMode, &[u32]) {
        if self.queue_family_indices.len() > 1 {
            (vk::SharingMode::CONCURRENT, &self.queue_family_indices)
        } else {
            (vk::SharingMode::EXCLUSIVE, &[])
        }
    }

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

    /// Returns the serialized pipeline cache contents, suitable for persisting and passing to
    /// `VulkanInstance::create_device_with_cache` on the next run.
    pub fn get_pipeline_cache_data(&self) -> Result<Vec<u8>, HalError> {
        unsafe {
            self.device
                .get_pipeline_cache_data(self.pipeline_cache)
                .map_err(|e| HalError::BackendError(format!("vkGetPipelineCacheData failed: {e}")))
        }
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

    pub fn create_timestamp_query_pool(&self, count: u32) -> Result<vk::QueryPool, HalError> {
        if self.timestamp_valid_bits == 0 {
            return Err(HalError::InvalidParameter(
                "The selected queue family does not support timestamps".to_string(),
            ));
        }
        if count == 0 {
            return Err(HalError::InvalidParameter(
                "A timestamp query pool needs at least one query".to_string(),
            ));
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        unsafe {
            self.device
                .create_query_pool(&info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateQueryPool failed: {e}")))
        }
    }

    pub fn destroy_query_pool(&self, pool: vk::QueryPool) {
        unsafe {
            self.device.destroy_query_pool(pool, None);
        }
    }

    pub fn cmd_write_timestamp(
        &self,
        recorder: &VulkanCommandRecorder,
        pool: vk::QueryPool,
        index: u32,
    ) {
        unsafe {
            self.device.cmd_write_timestamp2(
                recorder.command_buffer,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                pool,
                index,
            );
        }
    }

    pub fn cmd_reset_query_pool(
        &self,
        recorder: &VulkanCommandRecorder,
        pool: vk::QueryPool,
        first: u32,
        count: u32,
    ) {
        unsafe {
            self.device
                .cmd_reset_query_pool(recorder.command_buffer, pool, first, count);
        }
    }

    /// Blocks until the queries are available and returns their raw values.
    pub fn read_timestamp_query_pool(
        &self,
        pool: vk::QueryPool,
        count: u32,
    ) -> Result<Vec<u64>, HalError> {
        let mut values = vec![0u64; count as usize];
        // Deliberately no WAIT flag: waiting on a query that was reset but never written would
        // block until the driver declares the device lost. Waiting for submission completion is
        // the caller's job.
        match unsafe {
            self.device
                .get_query_pool_results(pool, 0, &mut values, vk::QueryResultFlags::TYPE_64)
        } {
            Ok(()) => Ok(values),
            Err(vk::Result::NOT_READY) => Err(HalError::NotReady(
                "GPU timestamps are not available yet; wait for the submission to finish"
                    .to_string(),
            )),
            Err(e) => Err(HalError::BackendError(format!(
                "vkGetQueryPoolResults failed: {e}"
            ))),
        }
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
