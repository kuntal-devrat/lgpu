//! GPU buffer allocation and data streaming.

use crate::Error;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

pub struct Buffer {
    pub(crate) hal_buffer: Option<lgpu_hal::vulkan::VulkanBuffer>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub size: u64,
    pub id: Option<BufferId>,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("size", &self.size)
            .field("id", &self.id)
            .finish()
    }
}

impl Buffer {
    /// Returns the optional bindless buffer identifier.
    #[inline]
    pub fn id(&self) -> Option<BufferId> {
        self.id
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
        if let Some(buf_id) = self.id {
            self.vulkan_device.unregister_bindless_buffer(buf_id.0);
        }
        if let Some(buf) = self.hal_buffer.take() {
            self.vulkan_device.destroy_buffer(buf);
        }
    }
}
