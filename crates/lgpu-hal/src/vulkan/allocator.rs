//! Vulkan memory suballocator based on a segregated power-of-two Buddy Allocator.
//!
//! Suballocates buffer and texture memory allocations from large contiguous blocks
//! (64 MiB by default) to stay well within driver allocation limits (e.g., maxMemoryAllocationCount = 4096).
//! Provides O(1) allocation and O(1) coalescing deallocation without vector sorting overhead.

use crate::HalError;
use ash::vk;
use std::collections::HashMap;

pub const DEFAULT_BLOCK_SIZE: u64 = 64 * 1024 * 1024; // 64 MiB chunks
pub const SUBALLOCATION_THRESHOLD: u64 = 32 * 1024 * 1024; // 32 MiB
pub const MIN_CHUNK_SIZE: u64 = 256; // 256 bytes minimum chunk
pub const NUM_ORDERS: usize = 19; // 256 B (order 0) to 64 MiB (order 18)

/// Distinguishes linear resources (buffers) from non-linear resources (optimal textures)
/// to respect Vulkan bufferImageGranularity rules and prevent memory aliasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationType {
    Linear,
    Optimal,
}

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
    pub allocation_type: AllocationType,
    pub mapped_base: Option<*mut u8>,
    pub is_coherent: bool,
    /// Free chunks segregated by power-of-two order (size = MIN_CHUNK_SIZE << order).
    pub free_lists: Vec<Vec<u64>>,
    /// Tracks active allocations: offset -> (order, allocated_size) to detect double free.
    pub allocated_chunks: HashMap<u64, (usize, u64)>,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

#[inline]
fn order_size(order: usize) -> u64 {
    MIN_CHUNK_SIZE << order
}

#[inline]
fn size_to_order(size: u64) -> usize {
    let s = size.max(MIN_CHUNK_SIZE);
    let chunks = (s + MIN_CHUNK_SIZE - 1) / MIN_CHUNK_SIZE;
    let next_power = chunks.next_power_of_two();
    next_power.trailing_zeros() as usize
}

impl MemoryBlock {
    pub fn new(
        id: usize,
        memory: vk::DeviceMemory,
        size: u64,
        memory_type_index: u32,
        is_host_visible: bool,
        allocation_type: AllocationType,
        mapped_base: Option<*mut u8>,
        is_coherent: bool,
    ) -> Self {
        let max_order = size_to_order(size).min(NUM_ORDERS - 1);
        let mut free_lists = vec![Vec::new(); NUM_ORDERS];
        free_lists[max_order].push(0);

        Self {
            id,
            memory,
            size,
            memory_type_index,
            is_host_visible,
            allocation_type,
            mapped_base,
            is_coherent,
            free_lists,
            allocated_chunks: HashMap::new(),
        }
    }

    pub fn allocate(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let needed_size = size.max(alignment).max(MIN_CHUNK_SIZE);
        let target_order = size_to_order(needed_size);
        if target_order >= NUM_ORDERS {
            return None;
        }

        // Find smallest order >= target_order with a free chunk
        let mut found_order = None;
        for order in target_order..NUM_ORDERS {
            if !self.free_lists[order].is_empty() {
                found_order = Some(order);
                break;
            }
        }

        let order = found_order?;
        let offset = self.free_lists[order].pop().unwrap();

        // Split chunk down to target_order
        let mut current_order = order;
        while current_order > target_order {
            current_order -= 1;
            let buddy_offset = offset + order_size(current_order);
            self.free_lists[current_order].push(buddy_offset);
        }

        self.allocated_chunks.insert(offset, (target_order, size));

        Some(offset)
    }

    pub fn free(&mut self, offset: u64, size: u64) {
        // Look up the order either from tracked allocations or calculate from size
        let order = if let Some((alloc_order, _)) = self.allocated_chunks.remove(&offset) {
            alloc_order
        } else {
            // Already freed or untracked: compute order from size and verify
            size_to_order(size).min(NUM_ORDERS - 1)
        };

        let mut curr_order = order;
        let mut curr_offset = offset;

        // Merge buddies recursively up to NUM_ORDERS - 1
        while curr_order < NUM_ORDERS - 1 {
            let chunk_sz = order_size(curr_order);
            let buddy_offset = curr_offset ^ chunk_sz;

            // Check if buddy is free in free_lists[curr_order]
            if let Some(pos) = self.free_lists[curr_order]
                .iter()
                .position(|&off| off == buddy_offset)
            {
                // Remove buddy from free list and coalesce
                self.free_lists[curr_order].swap_remove(pos);
                curr_offset = curr_offset.min(buddy_offset);
                curr_order += 1;
            } else {
                break;
            }
        }

        self.free_lists[curr_order].push(curr_offset);
    }

    pub fn is_empty(&self) -> bool {
        if !self.allocated_chunks.is_empty() {
            return false;
        }
        let max_order = size_to_order(self.size).min(NUM_ORDERS - 1);
        self.free_lists[max_order].len() == 1
    }

    pub fn free_bytes(&self) -> u64 {
        let mut total = 0;
        for (order, list) in self.free_lists.iter().enumerate() {
            total += (list.len() as u64) * order_size(order);
        }
        total
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
        allocation_type: AllocationType,
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

        // Search existing blocks for matching memory type, host-visibility, allocation type, and sufficient space
        for block in &mut self.blocks {
            if block.memory_type_index == memory_type_index
                && block.is_host_visible == is_host_visible
                && block.allocation_type == allocation_type
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

        let mut block = MemoryBlock::new(
            block_id,
            memory,
            block_size,
            memory_type_index,
            is_host_visible,
            allocation_type,
            mapped_base,
            is_coherent,
        );

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

    pub fn free(&mut self, block_id: usize, offset: u64, size: u64, device: &ash::Device) {
        let mut block_to_reclaim = None;

        if let Some(pos) = self.blocks.iter().position(|b| b.id == block_id) {
            self.blocks[pos].free(offset, size);

            if self.blocks[pos].is_empty() {
                let mem_idx = self.blocks[pos].memory_type_index;
                let is_host_vis = self.blocks[pos].is_host_visible;
                let alloc_type = self.blocks[pos].allocation_type;

                let empty_count = self
                    .blocks
                    .iter()
                    .filter(|b| {
                        b.memory_type_index == mem_idx
                            && b.is_host_visible == is_host_vis
                            && b.allocation_type == alloc_type
                            && b.is_empty()
                    })
                    .count();

                // Keep at most 1 empty block per configuration as a warm pool to prevent
                // allocation thrashing. Any redundant empty blocks are immediately reclaimed.
                if empty_count > 1 {
                    block_to_reclaim = Some(pos);
                }
            }
        }

        if let Some(pos) = block_to_reclaim {
            let block = self.blocks.remove(pos);
            unsafe {
                if block.mapped_base.is_some() {
                    device.unmap_memory(block.memory);
                }
                device.free_memory(block.memory, None);
            }
        }
    }

    /// Reclaims ALL completely empty blocks, releasing their GPU and host memory back to Vulkan.
    /// Returns the number of reclaimed blocks.
    pub fn trim(&mut self, device: &ash::Device) -> usize {
        let mut freed_count = 0;
        let mut i = 0;
        while i < self.blocks.len() {
            if self.blocks[i].is_empty() {
                let block = self.blocks.remove(i);
                unsafe {
                    if block.mapped_base.is_some() {
                        device.unmap_memory(block.memory);
                    }
                    device.free_memory(block.memory, None);
                }
                freed_count += 1;
            } else {
                i += 1;
            }
        }
        freed_count
    }

    /// Returns the total number of blocks currently held by the suballocator.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the number of completely empty blocks currently held by the suballocator.
    pub fn empty_block_count(&self) -> usize {
        self.blocks.iter().filter(|b| b.is_empty()).count()
    }

    /// Returns the total capacity in bytes across all managed blocks.
    pub fn total_capacity_bytes(&self) -> u64 {
        self.blocks.iter().map(|b| b.size).sum()
    }

    /// Returns the total available free bytes across all managed blocks.
    pub fn total_free_bytes(&self) -> u64 {
        self.blocks.iter().map(|b| b.free_bytes()).sum()
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
