//! High-performance Vulkan memory suballocator.
//!
//! Suballocates small buffer and texture memory allocations from large contiguous
//! blocks (32 MiB by default), avoiding driver allocation limits (e.g., 4096 allocations)
//! and eliminating redundant vkAllocateMemory, vkMapMemory, and vkUnmapMemory calls.

use crate::HalError;
use ash::vk;

pub const DEFAULT_BLOCK_SIZE: u64 = 32 * 1024 * 1024; // 32 MiB
pub const SUBALLOCATION_THRESHOLD: u64 = 16 * 1024 * 1024; // 16 MiB

#[derive(Debug)]
pub struct AllocationResult {
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub block_id: Option<usize>,
    pub mapped_ptr: Option<*mut u8>,
    pub is_coherent: bool,
}

pub struct MemoryBlock {
    pub id: usize,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub memory_type_index: u32,
    pub is_host_visible: bool,
    pub mapped_base: Option<*mut u8>,
    pub is_coherent: bool,
    pub free_chunks: Vec<(u64, u64)>, // (offset, size)
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl MemoryBlock {
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let align = alignment.max(1);
        for i in 0..self.free_chunks.len() {
            let (chunk_offset, chunk_size) = self.free_chunks[i];
            let aligned_offset = (chunk_offset + align - 1) & !(align - 1);
            let padding = aligned_offset - chunk_offset;
            if chunk_size >= padding + size {
                // Remove chunk and insert pre/post gaps
                self.free_chunks.remove(i);
                if padding > 0 {
                    self.free_chunks.push((chunk_offset, padding));
                }
                let remaining = chunk_size - (padding + size);
                if remaining > 0 {
                    self.free_chunks.push((aligned_offset + size, remaining));
                }
                self.free_chunks.sort_unstable_by_key(|&(off, _)| off);
                return Some(aligned_offset);
            }
        }
        None
    }

    pub fn free(&mut self, offset: u64, size: u64) {
        self.free_chunks.push((offset, size));
        self.free_chunks.sort_unstable_by_key(|&(off, _)| off);

        // Merge contiguous free chunks
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.free_chunks.len());
        for chunk in &self.free_chunks {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == chunk.0 {
                    last.1 += chunk.1;
                    continue;
                }
            }
            merged.push(*chunk);
        }
        self.free_chunks = merged;
    }
}

pub struct VulkanSuballocator {
    blocks: Vec<MemoryBlock>,
    next_block_id: usize,
}

impl VulkanSuballocator {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            next_block_id: 0,
        }
    }

    pub fn allocate(
        &mut self,
        device: &ash::Device,
        size: u64,
        alignment: u64,
        memory_type_index: u32,
        is_host_visible: bool,
        is_coherent: bool,
    ) -> Result<AllocationResult, HalError> {
        // Large allocations exceed threshold: use dedicated allocation
        if size >= SUBALLOCATION_THRESHOLD {
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(memory_type_index);

            let memory = unsafe {
                device.allocate_memory(&alloc_info, None).map_err(|e| {
                    HalError::OutOfMemory(format!("vkAllocateMemory dedicated failed: {e}"))
                })?
            };

            let mapped_ptr = if is_host_visible {
                let ptr = unsafe {
                    device
                        .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                        .map_err(|e| HalError::MapError(e.to_string()))?
                };
                Some(ptr as *mut u8)
            } else {
                None
            };

            return Ok(AllocationResult {
                memory,
                offset: 0,
                block_id: None,
                mapped_ptr,
                is_coherent,
            });
        }

        // Search existing blocks for matching memory type, host-visibility, and sufficient space
        for block in &mut self.blocks {
            if block.memory_type_index == memory_type_index
                && block.is_host_visible == is_host_visible
            {
                if let Some(offset) = block.allocate(size, alignment) {
                    let mapped_ptr = block
                        .mapped_base
                        .map(|base| unsafe { base.add(offset as usize) });
                    return Ok(AllocationResult {
                        memory: block.memory,
                        offset,
                        block_id: Some(block.id),
                        mapped_ptr,
                        is_coherent: block.is_coherent,
                    });
                }
            }
        }

        // Allocate a new block of DEFAULT_BLOCK_SIZE
        let block_size = DEFAULT_BLOCK_SIZE.max(size);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(block_size)
            .memory_type_index(memory_type_index);

        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| HalError::OutOfMemory(format!("vkAllocateMemory block failed: {e}")))?
        };

        let mapped_base = if is_host_visible {
            let ptr = unsafe {
                device
                    .map_memory(memory, 0, block_size, vk::MemoryMapFlags::empty())
                    .map_err(|e| HalError::MapError(e.to_string()))?
            };
            Some(ptr as *mut u8)
        } else {
            None
        };

        let block_id = self.next_block_id;
        self.next_block_id += 1;

        let mut block = MemoryBlock {
            id: block_id,
            memory,
            size: block_size,
            memory_type_index,
            is_host_visible,
            mapped_base,
            is_coherent,
            free_chunks: vec![(0, block_size)],
        };

        let offset = block
            .allocate(size, alignment)
            .expect("Fresh block must have sufficient space for allocation");

        let mapped_ptr = block
            .mapped_base
            .map(|base| unsafe { base.add(offset as usize) });

        self.blocks.push(block);

        Ok(AllocationResult {
            memory,
            offset,
            block_id: Some(block_id),
            mapped_ptr,
            is_coherent,
        })
    }

    pub fn free(&mut self, block_id: usize, offset: u64, size: u64) {
        if let Some(block) = self.blocks.iter_mut().find(|b| b.id == block_id) {
            block.free(offset, size);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for block in self.blocks.drain(..) {
                if block.mapped_base.is_some() {
                    device.unmap_memory(block.memory);
                }
                device.free_memory(block.memory, None);
            }
        }
    }
}
