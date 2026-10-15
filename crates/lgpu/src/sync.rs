//! GPU timeline semaphores and fine-grained CPU-GPU synchronization.

use std::sync::Arc;
use crate::Error;

pub struct TimelineSemaphore {
    #[cfg(feature = "vulkan")]
    pub(crate) hal_semaphore: Option<lgpu_hal::vulkan::VulkanTimelineSemaphore>,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_device: Arc<lgpu_hal::vulkan::VulkanDevice>,
}

impl TimelineSemaphore {
    pub fn value(&self) -> Result<u64, Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_sem = self.hal_semaphore.as_ref().unwrap();
            let val = self.vulkan_device.get_timeline_semaphore_value(raw_sem)?;
            Ok(val)
        }
        #[cfg(not(feature = "vulkan"))]
        {
            Ok(0)
        }
    }

    pub fn wait(&self, value: u64, timeout_nanos: u64) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_sem = self.hal_semaphore.as_ref().unwrap();
            self.vulkan_device.wait_timeline_semaphore(raw_sem, value, timeout_nanos)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = (value, timeout_nanos);
            Ok(())
        }
    }

    pub fn signal(&self, value: u64) -> Result<(), Error> {
        #[cfg(feature = "vulkan")]
        {
            let raw_sem = self.hal_semaphore.as_ref().unwrap();
            self.vulkan_device.signal_timeline_semaphore(raw_sem, value)?;
            Ok(())
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = value;
            Ok(())
        }
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        {
            if let Some(sem) = self.hal_semaphore.take() {
                self.vulkan_device.destroy_timeline_semaphore(sem);
            }
        }
    }
}
