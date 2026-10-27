//! GPU buffer allocation and data streaming.

use crate::Error;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

pub(crate) struct BufferSlotGuard {
    pub(crate) device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub(crate) id: Option<BufferId>,
}

impl Drop for BufferSlotGuard {
    fn drop(&mut self) {
        if let Some(buf_id) = self.id {
            self.device.unregister_bindless_buffer(buf_id.0);
        }
    }
}

pub struct Buffer {
    pub(crate) hal_buffer: Option<lgpu_hal::vulkan::VulkanBuffer>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub size: u64,
    pub id: Option<BufferId>,
    pub(crate) slot_guard: Option<Arc<BufferSlotGuard>>,
    pub owned: bool,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("size", &self.size)
            .field("id", &self.id)
            .field("owned", &self.owned)
            .finish()
    }
}

impl Buffer {
    /// Returns the optional bindless buffer identifier.
    #[inline]
    pub fn id(&self) -> Option<BufferId> {
        self.id
    }

    /// Returns whether this buffer instance owns the underlying GPU resource.
    #[inline]
    pub fn is_owned(&self) -> bool {
        self.owned
    }

    /// Clones an unowned reference to this buffer. Dropping the unowned clone will
    /// neither prematurely unregister its bindless slot nor destroy the underlying Vulkan buffer.
    pub fn clone_unowned(&self) -> Self {
        Self {
            hal_buffer: self
                .hal_buffer
                .as_ref()
                .map(|b| lgpu_hal::vulkan::VulkanBuffer {
                    buffer: b.buffer,
                    memory: b.memory,
                    block_id: b.block_id,
                    offset: b.offset,
                    size: b.size,
                    allocated_size: b.allocated_size,
                    mapped_ptr: b.mapped_ptr,
                    is_coherent: b.is_coherent,
                    sync_lock: std::sync::Mutex::new(()),
                    owned: false,
                }),
            vulkan_device: Arc::clone(&self.vulkan_device),
            size: self.size,
            id: self.id,
            slot_guard: self.slot_guard.clone(),
            owned: false,
        }
    }

    /// Writes data to the beginning of the buffer.
    pub fn write(&self, data: &[u8]) -> Result<(), Error> {
        self.write_at(0, data)
    }

    /// Writes data into the buffer at a specific byte offset.
    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), Error> {
        let raw_buf = self.hal_buffer.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Buffer hal resource is missing or destroyed".to_string(),
            ))
        })?;
        self.vulkan_device.write_buffer(raw_buf, offset, data)?;
        Ok(())
    }

    /// Reads data from the beginning of the buffer into a destination slice.
    pub fn read(&self, data: &mut [u8]) -> Result<(), Error> {
        self.read_at(0, data)
    }

    /// Reads data from the buffer at a specific byte offset into a destination slice.
    pub fn read_at(&self, offset: u64, data: &mut [u8]) -> Result<(), Error> {
        let raw_buf = self.hal_buffer.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "Buffer hal resource is missing or destroyed".to_string(),
            ))
        })?;
        self.vulkan_device.read_buffer(raw_buf, offset, data)?;
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.owned {
            if let Some(buf) = self.hal_buffer.take() {
                self.vulkan_device.destroy_buffer(buf);
            }
        }
    }
}
