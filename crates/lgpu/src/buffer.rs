//! GPU buffer allocation and data streaming.

use std::sync::Arc;
use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

pub struct Buffer {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_buffer: Option<lgpu_hal::vulkan::VulkanBuffer>,
    #[cfg(feature = "vulkan")]
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
    pub fn id(&self) -> Option<BufferId> {
        self.id
    }

    /// Writes data to the beginning of the buffer.
    ///
    /// # Example
    /// ```no_run
    /// use lgpu::Buffer;
    ///
    /// # fn example(buffer: &Buffer) -> Result<(), lgpu::Error> {
    /// buffer.write(&[1, 2, 3, 4])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn write(&self, data: &[u8]) -> Result<(), Error> {
        self.write_at(0, data)
    }

    /// Writes data into the buffer at a specific byte offset.
    ///
    /// # Example
    /// ```no_run
    /// use lgpu::Buffer;
    ///
    /// # fn example(buffer: &Buffer) -> Result<(), lgpu::Error> {
    /// buffer.write_at(16, &[10, 20, 30, 40])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_buf = self.hal_buffer.as_ref().unwrap();
            self.vulkan_device.write_buffer(raw_buf, offset, data)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (offset, data);
            Ok(())
        }
    }

    /// Reads data from the beginning of the buffer into a destination slice.
    ///
    /// # Example
    /// ```no_run
    /// use lgpu::Buffer;
    ///
    /// # fn example(buffer: &Buffer) -> Result<(), lgpu::Error> {
    /// let mut output = [0u8; 16];
    /// buffer.read(&mut output)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read(&self, data: &mut [u8]) -> Result<(), Error> {
        self.read_at(0, data)
    }

    /// Reads data from the buffer at a specific byte offset into a destination slice.
    ///
    /// # Example
    /// ```no_run
    /// use lgpu::Buffer;
    ///
    /// # fn example(buffer: &Buffer) -> Result<(), lgpu::Error> {
    /// let mut output = [0u8; 8];
    /// buffer.read_at(8, &mut output)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_at(&self, offset: u64, data: &mut [u8]) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_buf = self.hal_buffer.as_ref().unwrap();
            self.vulkan_device.read_buffer(raw_buf, offset, data)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (offset, data);
            Ok(())
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(buf_id) = self.id {
                self.vulkan_device.unregister_bindless_buffer(buf_id.0);
            }
            if let Some(buf) = self.hal_buffer.take() {
                self.vulkan_device.destroy_buffer(buf);
            }
        }
    }
}
