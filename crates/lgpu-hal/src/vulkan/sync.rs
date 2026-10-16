//! Vulkan timeline semaphore synchronization.

use super::VulkanDevice;
use crate::HalError;
use ash::vk;

pub struct VulkanTimelineSemaphore {
    pub semaphore: vk::Semaphore,
}

unsafe impl Send for VulkanTimelineSemaphore {}
unsafe impl Sync for VulkanTimelineSemaphore {}

impl VulkanDevice {
    pub fn create_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<VulkanTimelineSemaphore, HalError> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        let semaphore = unsafe {
            self.device
                .create_semaphore(&create_info, None)
                .map_err(|e| {
                    HalError::BackendError(format!("Failed to create timeline semaphore: {e}"))
                })?
        };

        Ok(VulkanTimelineSemaphore { semaphore })
    }

    pub fn wait_timeline_semaphore(
        &self,
        semaphore: &VulkanTimelineSemaphore,
        value: u64,
        timeout_nanos: u64,
    ) -> Result<(), HalError> {
        let semaphores = [semaphore.semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);

        unsafe {
            self.device
                .wait_semaphores(&wait_info, timeout_nanos)
                .map_err(|e| HalError::CommandError(format!("vkWaitSemaphores failed: {e}")))?;
        }
        Ok(())
    }

    pub fn get_timeline_semaphore_value(
        &self,
        semaphore: &VulkanTimelineSemaphore,
    ) -> Result<u64, HalError> {
        unsafe {
            self.device
                .get_semaphore_counter_value(semaphore.semaphore)
                .map_err(|e| {
                    HalError::CommandError(format!("vkGetSemaphoreCounterValue failed: {e}"))
                })
        }
    }

    pub fn signal_timeline_semaphore(
        &self,
        semaphore: &VulkanTimelineSemaphore,
        value: u64,
    ) -> Result<(), HalError> {
        let signal_info = vk::SemaphoreSignalInfo::default()
            .semaphore(semaphore.semaphore)
            .value(value);
        unsafe {
            self.device
                .signal_semaphore(&signal_info)
                .map_err(|e| HalError::CommandError(format!("vkSignalSemaphore failed: {e}")))?;
        }
        Ok(())
    }

    pub fn destroy_timeline_semaphore(&self, semaphore: VulkanTimelineSemaphore) {
        unsafe {
            self.device.destroy_semaphore(semaphore.semaphore, None);
        }
    }
}
