//! Persistent CPU-mapped asynchronous ring buffer for texture and buffer staging.

use super::VulkanCommandRecorder;
use crate::HalError;
use ash::vk;
use std::collections::VecDeque;

pub const DEFAULT_STAGING_RING_SIZE: u64 = 64 * 1024 * 1024; // 64 MiB persistent ring
pub const STAGING_ALIGNMENT: u64 = 256; // 256-byte alignment for buffer-to-image copy

pub struct StagingBatch {
    pub offset: u64,
    pub size: u64,
    pub timeline_value: u64,
}

pub struct VulkanStagingRing {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped_ptr: *mut u8,
    pub capacity: u64,
    pub head: u64,
    pub tail: u64,
    pub timeline_semaphore: vk::Semaphore,
    pub current_timeline_value: u64,
    pub pending_batches: VecDeque<StagingBatch>,
    pub pending_recorders: Vec<(u64, VulkanCommandRecorder)>,
}

unsafe impl Send for VulkanStagingRing {}
unsafe impl Sync for VulkanStagingRing {}

impl VulkanStagingRing {
    pub fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        capacity: u64,
    ) -> Result<Self, HalError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device.create_buffer(&buffer_info, None).map_err(|e| {
                HalError::BackendError(format!("Failed to create staging ring buffer: {e}"))
            })?
        };

        let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Find HOST_VISIBLE | HOST_COHERENT memory
        let mut selected_type = None;
        let preferred =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        for i in 0..mem_props.memory_type_count {
            if (mem_reqs.memory_type_bits & (1 << i)) != 0
                && (mem_props.memory_types[i as usize].property_flags & preferred) == preferred
            {
                selected_type = Some(i);
                break;
            }
        }

        let memory_type_index = selected_type
            .or_else(|| {
                for i in 0..mem_props.memory_type_count {
                    if (mem_reqs.memory_type_bits & (1 << i)) != 0
                        && (mem_props.memory_types[i as usize].property_flags
                            & vk::MemoryPropertyFlags::HOST_VISIBLE)
                            == vk::MemoryPropertyFlags::HOST_VISIBLE
                    {
                        return Some(i);
                    }
                }
                None
            })
            .ok_or_else(|| {
                HalError::OutOfMemory("No host-visible memory for staging ring".to_string())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe {
            device.allocate_memory(&alloc_info, None).map_err(|e| {
                HalError::OutOfMemory(format!("Failed to allocate staging ring memory: {e}"))
            })?
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| HalError::BackendError(format!("Failed to bind staging ring: {e}")))?;
        }

        let mapped = unsafe {
            device
                .map_memory(memory, 0, capacity, vk::MemoryMapFlags::empty())
                .map_err(|e| HalError::MapError(e.to_string()))?
        };

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);

        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let timeline_semaphore = unsafe {
            device.create_semaphore(&sem_info, None).map_err(|e| {
                HalError::BackendError(format!("Failed to create staging timeline semaphore: {e}"))
            })?
        };

        Ok(Self {
            buffer,
            memory,
            mapped_ptr: mapped as *mut u8,
            capacity,
            head: 0,
            tail: 0,
            timeline_semaphore,
            current_timeline_value: 0,
            pending_batches: VecDeque::new(),
            pending_recorders: Vec::new(),
        })
    }

    /// Suballocates a slice of `size` bytes from the ring buffer.
    /// If capacity wraps around or in-flight batches overlap, waits only on the retired batches needed.
    pub fn suballocate(&mut self, device: &ash::Device, size: u64) -> Result<(u64, u64), HalError> {
        let aligned_size = (size + STAGING_ALIGNMENT - 1) & !(STAGING_ALIGNMENT - 1);
        if aligned_size > self.capacity {
            return Err(HalError::InvalidParameter(format!(
                "Requested size {size} exceeds staging ring capacity {}",
                self.capacity
            )));
        }

        // 1. Harvest finished batches
        self.poll_completed(device);

        // 2. Try allocating at self.head
        let (offset, next_head) = if self.head + aligned_size <= self.capacity {
            (self.head, self.head + aligned_size)
        } else {
            // Wrap around to offset 0
            (0, aligned_size)
        };

        // 3. If any pending batches overlap the requested range [offset .. offset + aligned_size],
        // wait for the newest overlapping batch to finish.
        if let Some(wait_val) = self.find_overlapping_timeline_val(offset, aligned_size) {
            let sems = [self.timeline_semaphore];
            let values = [wait_val];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&sems)
                .values(&values);

            unsafe {
                device.wait_semaphores(&wait_info, u64::MAX).map_err(|e| {
                    HalError::BackendError(format!("Failed waiting for staging space: {e}"))
                })?;
            }
            self.poll_completed(device);
        }

        self.head = next_head;
        Ok((offset, aligned_size))
    }

    fn find_overlapping_timeline_val(&self, offset: u64, size: u64) -> Option<u64> {
        let mut max_val = None;
        let end = offset + size;
        for batch in &self.pending_batches {
            let batch_end = batch.offset + batch.size;
            if offset < batch_end && batch.offset < end {
                max_val = Some(batch.timeline_value);
            }
        }
        max_val
    }

    pub fn poll_completed(&mut self, device: &ash::Device) {
        let completed = unsafe {
            device
                .get_semaphore_counter_value(self.timeline_semaphore)
                .unwrap_or(0)
        };

        while let Some(batch) = self.pending_batches.front() {
            if batch.timeline_value <= completed {
                self.tail = batch.offset + batch.size;
                self.pending_batches.pop_front();
            } else {
                break;
            }
        }

        if self.pending_batches.is_empty() {
            self.head = 0;
            self.tail = 0;
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if self.current_timeline_value > 0 {
                let sems = [self.timeline_semaphore];
                let values = [self.current_timeline_value];
                let wait_info = vk::SemaphoreWaitInfo::default()
                    .semaphores(&sems)
                    .values(&values);
                let _ = device.wait_semaphores(&wait_info, u64::MAX);
            }
            for (_, r) in self.pending_recorders.drain(..) {
                device.destroy_command_pool(r.command_pool, None);
            }
            device.destroy_semaphore(self.timeline_semaphore, None);
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}
