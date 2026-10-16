//! GPU instance and adapter management.

use crate::{
    AdapterInfo, Device, Error, HalError, HasDisplayHandle, HasWindowHandle, Queue,
    RawDisplayHandle, RawWindowHandle, Surface,
};
use std::sync::Arc;

pub struct Instance {
    pub(crate) vulkan_instance: Arc<lgpu_hal::vulkan::VulkanInstance>,
}

pub struct Adapter {
    pub(crate) vulkan_adapter: lgpu_hal::vulkan::VulkanAdapter,
}

impl Instance {
    /// Creates a new `Instance` by dynamically loading the native GPU driver.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::Instance;
    ///
    /// let instance = Instance::new().expect("Failed to initialize GPU instance");
    /// let adapters = instance.enumerate_adapters().expect("Failed to enumerate adapters");
    /// println!("Found {} GPU adapter(s)", adapters.len());
    /// ```
    pub fn new() -> Result<Self, Error> {
        let vulkan_instance = Arc::new(lgpu_hal::vulkan::VulkanInstance::new()?);
        Ok(Self { vulkan_instance })
    }

    /// Enumerates all compatible physical GPU adapters discovered on the system.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::Instance;
    ///
    /// let instance = Instance::new().unwrap();
    /// for adapter in instance.enumerate_adapters().unwrap() {
    ///     println!("Adapter: {}", adapter.info().name);
    /// }
    /// ```
    pub fn enumerate_adapters(&self) -> Result<Vec<Adapter>, Error> {
        let adapters = self.vulkan_instance.enumerate_adapters()?;
        Ok(adapters
            .into_iter()
            .map(|vulkan_adapter| Adapter { vulkan_adapter })
            .collect())
    }

    pub fn create_surface<W: HasDisplayHandle + HasWindowHandle>(
        &self,
        window: &W,
    ) -> Result<Surface, Error> {
        let display = window
            .display_handle()
            .map_err(|e| HalError::BackendError(format!("Failed to get display handle: {e}")))?
            .as_raw();
        let window_h = window
            .window_handle()
            .map_err(|e| HalError::BackendError(format!("Failed to get window handle: {e}")))?
            .as_raw();
        self.create_surface_raw(display, window_h)
    }

    pub fn create_surface_raw(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Surface, Error> {
        let raw_surface = self
            .vulkan_instance
            .create_surface(display_handle, window_handle)?;
        Ok(Surface {
            instance: Arc::clone(&self.vulkan_instance),
            raw_surface,
            swapchain: None,
            device: None,
            acquire_semaphores: Vec::new(),
            render_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            current_frame: 0,
            config: None,
        })
    }

    pub fn create_headless_surface(&self) -> Result<Surface, Error> {
        let raw_surface = self.vulkan_instance.create_headless_surface()?;
        Ok(Surface {
            instance: Arc::clone(&self.vulkan_instance),
            raw_surface,
            swapchain: None,
            device: None,
            acquire_semaphores: Vec::new(),
            render_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            current_frame: 0,
            config: None,
        })
    }
}

impl Adapter {
    pub fn info(&self) -> &AdapterInfo {
        self.vulkan_adapter.info()
    }

    /// Requests a logical [`Device`] and execution [`Queue`] from this adapter.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use lgpu::Instance;
    ///
    /// let instance = Instance::new().unwrap();
    /// let adapter = &instance.enumerate_adapters().unwrap()[0];
    /// let (device, queue) = adapter.request_device(&instance).expect("Failed to create device");
    /// ```
    pub fn request_device(&self, instance: &Instance) -> Result<(Device, Queue), Error> {
        let raw_device = instance
            .vulkan_instance
            .create_device(&self.vulkan_adapter)?;
        let device_arc = Arc::new(raw_device);

        let device = Device {
            vulkan_device: Arc::clone(&device_arc),
        };
        let queue = Queue {
            vulkan_device: device_arc,
        };

        Ok((device, queue))
    }
}
