//! Vulkan buffer resource management and operations.

use super::VulkanDevice;
use crate::{BufferDesc, BufferUsage, HalError, MemoryLocation};
use ash::vk;

pub struct VulkanBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) block_id: Option<usize>,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) allocated_size: u64,
    pub(crate) mapped_ptr: Option<*mut u8>,
    pub(crate) is_coherent: bool,
    pub(crate) sync_lock: std::sync::Mutex<()>,
}

unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

impl VulkanDevice {
    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<VulkanBuffer, HalError> {
        if desc.size == 0 {
            return Err(HalError::InvalidParameter(
                "Buffer size must be greater than 0".to_string(),
            ));
        }

        let mut vk_usage = vk::BufferUsageFlags::empty();
        if desc.usage.contains(BufferUsage::TRANSFER_SRC) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if desc.usage.contains(BufferUsage::TRANSFER_DST) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if desc.usage.contains(BufferUsage::STORAGE) {
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if desc.usage.contains(BufferUsage::UNIFORM) {
            vk_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if desc.usage.contains(BufferUsage::VERTEX) {
            vk_usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if desc.usage.contains(BufferUsage::INDEX) {
            vk_usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if desc.usage.contains(BufferUsage::INDIRECT) {
            vk_usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateBuffer failed: {e}")))?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let mem_props = match desc.memory_location {
            MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::CpuToGpu => vk::MemoryPropertyFlags::HOST_VISIBLE,
            MemoryLocation::GpuToCpu => vk::MemoryPropertyFlags::HOST_VISIBLE,
        };

        let (mem_type_index, is_coherent) =
            self.find_memory_type(mem_reqs.memory_type_bits, mem_props)?;
        let is_host_visible = desc.memory_location != MemoryLocation::GpuOnly;

        let alloc = self.allocator.lock().unwrap().allocate(
            &self.device,
            mem_reqs.size,
            mem_reqs.alignment,
            mem_type_index,
            is_host_visible,
            is_coherent,
            super::allocator::AllocationType::Linear,
        )?;

        unsafe {
            if let Err(e) = self
                .device
                .bind_buffer_memory(buffer, alloc.memory, alloc.offset)
            {
                if let Some(block_id) = alloc.block_id {
                    self.allocator.lock().unwrap().free(
                        block_id,
                        alloc.offset,
                        mem_reqs.size,
                        &self.device,
                    );
                } else {
                    self.device.free_memory(alloc.memory, None);
                }
                self.device.destroy_buffer(buffer, None);
                return Err(HalError::BackendError(format!(
                    "vkBindBufferMemory failed: {e}"
                )));
            }
        }

        Ok(VulkanBuffer {
            buffer,
            memory: alloc.memory,
            block_id: alloc.block_id,
            offset: alloc.offset,
            size: desc.size,
            allocated_size: mem_reqs.size,
            mapped_ptr: alloc.mapped_ptr,
            is_coherent: alloc.is_coherent,
            sync_lock: std::sync::Mutex::new(()),
        })
    }

    pub fn write_buffer(
        &self,
        buffer: &VulkanBuffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), HalError> {
        let _guard = buffer.sync_lock.lock().unwrap();
        let ptr = buffer
            .mapped_ptr
            .ok_or_else(|| HalError::MapError("Buffer is not host-mapped".to_string()))?;
        if offset
            .checked_add(data.len() as u64)
            .is_none_or(|end| end > buffer.size)
        {
            return Err(HalError::CommandError("Buffer write overflow".to_string()));
        }
        unsafe {
            let dst = ptr.add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());

            if !buffer.is_coherent {
                let atom = self.non_coherent_atom_size.max(1);
                let mem_offset = buffer.offset + offset;
                let aligned_offset = (mem_offset / atom) * atom;
                let end = mem_offset + data.len() as u64;
                let aligned_end = ((end + atom - 1) / atom) * atom;
                let aligned_size = aligned_end - aligned_offset;

                let range = vk::MappedMemoryRange::default()
                    .memory(buffer.memory)
                    .offset(aligned_offset)
                    .size(aligned_size);
                self.device
                    .flush_mapped_memory_ranges(&[range])
                    .map_err(|e| {
                        HalError::MapError(format!("vkFlushMappedMemoryRanges failed: {e}"))
                    })?;
            }
        }
        Ok(())
    }

    pub fn read_buffer(
        &self,
        buffer: &VulkanBuffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), HalError> {
        let _guard = buffer.sync_lock.lock().unwrap();
        let ptr = buffer
            .mapped_ptr
            .ok_or_else(|| HalError::MapError("Buffer is not host-mapped".to_string()))?;
        if offset
            .checked_add(data.len() as u64)
            .is_none_or(|end| end > buffer.size)
        {
            return Err(HalError::CommandError("Buffer read overflow".to_string()));
        }
        unsafe {
            if !buffer.is_coherent {
                let atom = self.non_coherent_atom_size.max(1);
                let mem_offset = buffer.offset + offset;
                let aligned_offset = (mem_offset / atom) * atom;
                let end = mem_offset + data.len() as u64;
                let aligned_end = ((end + atom - 1) / atom) * atom;
                let aligned_size = aligned_end - aligned_offset;

                let range = vk::MappedMemoryRange::default()
                    .memory(buffer.memory)
                    .offset(aligned_offset)
                    .size(aligned_size);
                self.device
                    .invalidate_mapped_memory_ranges(&[range])
                    .map_err(|e| {
                        HalError::MapError(format!("vkInvalidateMappedMemoryRanges failed: {e}"))
                    })?;
            }

            let src = ptr.add(offset as usize);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), data.len());
        }
        Ok(())
    }

    pub fn destroy_buffer(&self, mut buffer: VulkanBuffer) {
        let _guard = buffer.sync_lock.lock().unwrap();
        unsafe {
            if buffer.block_id.is_none() && buffer.mapped_ptr.take().is_some() {
                self.device.unmap_memory(buffer.memory);
            } else {
                buffer.mapped_ptr.take();
            }
            self.device.destroy_buffer(buffer.buffer, None);
            if let Some(block_id) = buffer.block_id {
                self.allocator.lock().unwrap().free(
                    block_id,
                    buffer.offset,
                    buffer.allocated_size,
                    &self.device,
                );
            } else {
                self.device.free_memory(buffer.memory, None);
            }
        }
    }

    pub fn register_bindless_buffer(&self, buffer: &VulkanBuffer) -> Result<u32, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let slot = heap.allocate_buffer_slot()?;
        heap.write_buffer_descriptor(&self.device, slot, buffer.buffer, buffer.size);
        Ok(slot)
    }

    pub fn unregister_bindless_buffer(&self, slot: u32) {
        if let Some(heap) = &self.bindless_heap {
            heap.free_buffer_slot(slot);
        }
    }

    pub fn write_buffer_descriptors_batch(
        &self,
        writes: &[(u32, &VulkanBuffer)],
    ) -> Result<(), HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let raw_writes: Vec<(u32, ash::vk::Buffer, u64)> = writes
            .iter()
            .map(|(slot, buf)| (*slot, buf.buffer, buf.size))
            .collect();
        heap.write_buffer_descriptors_batch(&self.device, &raw_writes);
        Ok(())
    }
}
