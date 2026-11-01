//! GPU timeline semaphores and fine-grained CPU-GPU synchronization.

use crate::Error;
use std::sync::Arc;

/// A pool of GPU timestamp queries.
///
/// Record timings with [`crate::CommandEncoder::reset_timestamp_pool`] and
/// [`crate::CommandEncoder::write_timestamp`], then read them after submission with
/// [`TimestampPool::read`]. Values are raw tick counts; divide by the device timestamp period
/// to obtain nanoseconds.
pub struct TimestampPool {
    pub(crate) pool: Option<lgpu_hal::vulkan::vk::QueryPool>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
    pub count: u32,
}

impl TimestampPool {
    pub fn count(&self) -> u32 {
        self.count
    }

    pub(crate) fn raw(&self) -> Result<lgpu_hal::vulkan::vk::QueryPool, Error> {
        self.pool.ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "TimestampPool is destroyed".to_string(),
            ))
        })
    }

    /// Blocks until the GPU has written every query and returns the raw tick counts.
    pub fn read(&self) -> Result<Vec<u64>, Error> {
        let pool = self.raw()?;
        Ok(self
            .vulkan_device
            .read_timestamp_query_pool(pool, self.count)?)
    }
}

impl Drop for TimestampPool {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            self.vulkan_device.destroy_query_pool(pool);
        }
    }
}

pub struct TimelineSemaphore {
    pub(crate) hal_semaphore: Option<lgpu_hal::vulkan::VulkanTimelineSemaphore>,
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

impl TimelineSemaphore {
    pub fn value(&self) -> Result<u64, Error> {
        let raw_sem = self.hal_semaphore.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "TimelineSemaphore is uninitialized or destroyed".to_string(),
            ))
        })?;
        let val = self.vulkan_device.get_timeline_semaphore_value(raw_sem)?;
        Ok(val)
    }

    pub fn wait(&self, value: u64, timeout_nanos: u64) -> Result<(), Error> {
        let raw_sem = self.hal_semaphore.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "TimelineSemaphore is uninitialized or destroyed".to_string(),
            ))
        })?;
        self.vulkan_device
            .wait_timeline_semaphore(raw_sem, value, timeout_nanos)?;
        Ok(())
    }

    pub fn signal(&self, value: u64) -> Result<(), Error> {
        let raw_sem = self.hal_semaphore.as_ref().ok_or_else(|| {
            Error::Hal(lgpu_hal::HalError::InvalidParameter(
                "TimelineSemaphore is uninitialized or destroyed".to_string(),
            ))
        })?;
        self.vulkan_device
            .signal_timeline_semaphore(raw_sem, value)?;
        Ok(())
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        if let Some(sem) = self.hal_semaphore.take() {
            self.vulkan_device.destroy_timeline_semaphore(sem);
        }
    }
}
