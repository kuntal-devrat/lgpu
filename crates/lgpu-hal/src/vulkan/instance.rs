//! Vulkan instance, physical device adapters, and surface creation.

use ash::ext::{headless_surface, metal_surface};
use ash::khr::{surface, swapchain, wayland_surface, win32_surface, xcb_surface, xlib_surface};
use ash::vk;
use std::ffi::CString;

use super::{allocator, bindless, VulkanDevice};
use crate::{AdapterInfo, Backend, DeviceType, HalError};

pub struct VulkanInstance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_loader: Option<surface::Instance>,
}

pub struct VulkanAdapter {
    pub physical_device: vk::PhysicalDevice,
    pub info: AdapterInfo,
    pub queue_family_index: u32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub non_coherent_atom_size: u64,
}

impl VulkanInstance {
    pub fn new() -> Result<Self, HalError> {
        let entry =
            unsafe { ash::Entry::load().map_err(|e| HalError::LibraryLoadError(e.to_string()))? };

        let app_name = CString::new("lgpu Application").unwrap();
        let engine_name = CString::new("lgpu").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let supported_extensions = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .unwrap_or_default()
        };
        let is_ext_supported = |name: &std::ffi::CStr| {
            supported_extensions
                .iter()
                .any(|ext| unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) == name })
        };

        let mut enabled_extensions = Vec::new();
        if is_ext_supported(surface::NAME) {
            enabled_extensions.push(surface::NAME.as_ptr());
        }
        if is_ext_supported(xlib_surface::NAME) {
            enabled_extensions.push(xlib_surface::NAME.as_ptr());
        }
        if is_ext_supported(xcb_surface::NAME) {
            enabled_extensions.push(xcb_surface::NAME.as_ptr());
        }
        if is_ext_supported(wayland_surface::NAME) {
            enabled_extensions.push(wayland_surface::NAME.as_ptr());
        }
        if is_ext_supported(win32_surface::NAME) {
            enabled_extensions.push(win32_surface::NAME.as_ptr());
        }
        if is_ext_supported(metal_surface::NAME) {
            enabled_extensions.push(metal_surface::NAME.as_ptr());
        }
        if is_ext_supported(headless_surface::NAME) {
            enabled_extensions.push(headless_surface::NAME.as_ptr());
        }

        #[allow(unused_mut)]
        let mut flags = vk::InstanceCreateFlags::empty();
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let portability_ext = vk::KHR_PORTABILITY_ENUMERATION_NAME;
            if is_ext_supported(portability_ext) {
                enabled_extensions.push(portability_ext.as_ptr());
                flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .flags(flags)
            .enabled_extension_names(&enabled_extensions);

        let instance = unsafe {
            entry.create_instance(&create_info, None).map_err(|e| {
                HalError::BackendError(format!("Failed to create Vulkan instance: {e}"))
            })?
        };

        let surface_loader = if is_ext_supported(surface::NAME) {
            Some(surface::Instance::new(&entry, &instance))
        } else {
            None
        };

        Ok(Self {
            entry,
            instance,
            surface_loader,
        })
    }

    pub fn create_headless_surface(&self) -> Result<vk::SurfaceKHR, HalError> {
        let headless_loader = headless_surface::Instance::new(&self.entry, &self.instance);
        let create_info = vk::HeadlessSurfaceCreateInfoEXT::default();
        unsafe { headless_loader.create_headless_surface(&create_info, None) }
            .map_err(|e| HalError::BackendError(format!("Failed to create headless surface: {e}")))
    }

    pub fn create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<vk::SurfaceKHR, HalError> {
        use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

        match (display_handle, window_handle) {
            (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
                let xlib_loader = xlib_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::XlibSurfaceCreateInfoKHR::default()
                    .dpy(
                        display
                            .display
                            .map(|p| p.as_ptr() as *mut _)
                            .unwrap_or(std::ptr::null_mut()),
                    )
                    .window(window.window as vk::Window);
                unsafe { xlib_loader.create_xlib_surface(&create_info, None) }.map_err(|e| {
                    HalError::BackendError(format!("Failed to create Xlib surface: {e}"))
                })
            }
            (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
                let xcb_loader = xcb_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(
                        display
                            .connection
                            .map(|p| p.as_ptr() as *mut _)
                            .unwrap_or(std::ptr::null_mut()),
                    )
                    .window(window.window.get());
                unsafe { xcb_loader.create_xcb_surface(&create_info, None) }.map_err(|e| {
                    HalError::BackendError(format!("Failed to create XCB surface: {e}"))
                })
            }
            (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
                let wayland_loader = wayland_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(display.display.as_ptr() as *mut _)
                    .surface(window.surface.as_ptr() as *mut _);
                unsafe { wayland_loader.create_wayland_surface(&create_info, None) }.map_err(|e| {
                    HalError::BackendError(format!("Failed to create Wayland surface: {e}"))
                })
            }
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
                let win32_loader = win32_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(window.hinstance.map(|p| p.get() as isize).unwrap_or(0))
                    .hwnd(window.hwnd.get() as isize);
                unsafe { win32_loader.create_win32_surface(&create_info, None) }.map_err(|e| {
                    HalError::BackendError(format!("Failed to create Win32 surface: {e}"))
                })
            }
            (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) => {
                let metal_loader = metal_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::MetalSurfaceCreateInfoEXT::default()
                    .layer(window.ns_view.as_ptr() as *const _);
                unsafe { metal_loader.create_metal_surface(&create_info, None) }.map_err(|e| {
                    HalError::BackendError(format!("Failed to create Metal surface: {e}"))
                })
            }
            (RawDisplayHandle::UiKit(_), RawWindowHandle::UiKit(window)) => {
                let metal_loader = metal_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::MetalSurfaceCreateInfoEXT::default()
                    .layer(window.ui_view.as_ptr() as *const _);
                unsafe { metal_loader.create_metal_surface(&create_info, None) }.map_err(|e| {
                    HalError::BackendError(format!("Failed to create iOS Metal surface: {e}"))
                })
            }
            _ => Err(HalError::BackendError(
                "Unsupported display/window handle pair for Vulkan surface creation".to_string(),
            )),
        }
    }

    pub fn destroy_surface(&self, surface: vk::SurfaceKHR) {
        if let Some(loader) = &self.surface_loader {
            unsafe {
                loader.destroy_surface(surface, None);
            }
        }
    }

    pub fn enumerate_adapters(&self) -> Result<Vec<VulkanAdapter>, HalError> {
        let physical_devices = unsafe {
            self.instance
                .enumerate_physical_devices()
                .map_err(|e| HalError::BackendError(e.to_string()))?
        };

        if physical_devices.is_empty() {
            return Err(HalError::NoAdapterFound);
        }

        let mut adapters = Vec::new();
        for pd in physical_devices {
            let props = unsafe { self.instance.get_physical_device_properties(pd) };
            let mem_props = unsafe { self.instance.get_physical_device_memory_properties(pd) };
            let queue_props = unsafe {
                self.instance
                    .get_physical_device_queue_family_properties(pd)
            };

            // Find a queue family that supports both graphics and compute (or at least compute)
            let mut selected_queue = None;
            for (idx, qp) in queue_props.iter().enumerate() {
                if qp.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    selected_queue = Some(idx as u32);
                    if qp.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        break;
                    }
                }
            }

            let queue_family_index = match selected_queue {
                Some(idx) => idx,
                None => continue,
            };

            let name_bytes: Vec<u8> = props
                .device_name
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            let name = String::from_utf8_lossy(&name_bytes).to_string();

            let device_type = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::DiscreteGpu,
                vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::IntegratedGpu,
                vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::VirtualGpu,
                vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
                _ => DeviceType::Other,
            };

            let mut supports_dynamic_rendering = false;
            let mut supports_timeline_semaphores = false;
            let mut supports_descriptor_indexing = false;

            if props.api_version >= vk::make_api_version(0, 1, 1, 0) {
                let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
                let mut features12 = vk::PhysicalDeviceVulkan12Features::default();

                if props.api_version >= vk::make_api_version(0, 1, 3, 0) {
                    let mut features2 = vk::PhysicalDeviceFeatures2::default()
                        .push_next(&mut features12)
                        .push_next(&mut features13);
                    unsafe {
                        self.instance
                            .get_physical_device_features2(pd, &mut features2);
                    }
                    supports_dynamic_rendering = features13.dynamic_rendering == vk::TRUE;
                    supports_timeline_semaphores = features12.timeline_semaphore == vk::TRUE;
                    supports_descriptor_indexing = features12.descriptor_indexing == vk::TRUE;
                } else if props.api_version >= vk::make_api_version(0, 1, 2, 0) {
                    let mut features2 =
                        vk::PhysicalDeviceFeatures2::default().push_next(&mut features12);
                    unsafe {
                        self.instance
                            .get_physical_device_features2(pd, &mut features2);
                    }
                    supports_timeline_semaphores = features12.timeline_semaphore == vk::TRUE;
                    supports_descriptor_indexing = features12.descriptor_indexing == vk::TRUE;
                }
            }

            let supported_dev_extensions = unsafe {
                self.instance
                    .enumerate_device_extension_properties(pd)
                    .unwrap_or_default()
            };
            let is_dev_ext_supported_bytes = |name: &[u8]| {
                supported_dev_extensions.iter().any(|ext| {
                    let ext_name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
                    ext_name.to_bytes_with_nul() == name
                })
            };

            if !supports_dynamic_rendering
                && is_dev_ext_supported_bytes(b"VK_KHR_dynamic_rendering\0")
            {
                supports_dynamic_rendering = true;
            }
            if !supports_timeline_semaphores
                && is_dev_ext_supported_bytes(b"VK_KHR_timeline_semaphore\0")
            {
                supports_timeline_semaphores = true;
            }
            if !supports_descriptor_indexing
                && is_dev_ext_supported_bytes(b"VK_EXT_descriptor_indexing\0")
            {
                supports_descriptor_indexing = true;
            }

            let adapter_info = AdapterInfo {
                name,
                backend: Backend::Vulkan,
                device_type,
                vendor_id: props.vendor_id,
                device_id: props.device_id,
                supports_dynamic_rendering,
                supports_timeline_semaphores,
                supports_descriptor_indexing,
                max_compute_workgroup_invocations: props.limits.max_compute_work_group_invocations,
                max_compute_workgroup_size: props.limits.max_compute_work_group_size,
                limits: crate::DeviceLimits {
                    max_buffer_size: props.limits.max_storage_buffer_range as u64,
                    max_texture_dimension_1d: props.limits.max_image_dimension1_d,
                    max_texture_dimension_2d: props.limits.max_image_dimension2_d,
                    max_texture_dimension_3d: props.limits.max_image_dimension3_d,
                    max_texture_array_layers: props.limits.max_image_array_layers,
                    min_uniform_buffer_offset_alignment: props
                        .limits
                        .min_uniform_buffer_offset_alignment,
                    min_storage_buffer_offset_alignment: props
                        .limits
                        .min_storage_buffer_offset_alignment,
                    max_compute_workgroup_invocations: props
                        .limits
                        .max_compute_work_group_invocations,
                    max_compute_workgroup_size: props.limits.max_compute_work_group_size,
                },
            };

            adapters.push(VulkanAdapter {
                physical_device: pd,
                info: adapter_info,
                queue_family_index,
                memory_properties: mem_props,
                non_coherent_atom_size: props.limits.non_coherent_atom_size,
            });
        }

        Ok(adapters)
    }

    pub fn create_device(&self, adapter: &VulkanAdapter) -> Result<VulkanDevice, HalError> {
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(adapter.queue_family_index)
            .queue_priorities(&queue_priorities);

        let queue_infos = [queue_create_info];

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true);

        let mut features = vk::PhysicalDeviceFeatures::default();
        features.shader_int64 = vk::TRUE;

        let supported_dev_extensions = unsafe {
            self.instance
                .enumerate_device_extension_properties(adapter.physical_device)
                .unwrap_or_default()
        };
        let is_dev_ext_supported = |name: &std::ffi::CStr| {
            supported_dev_extensions
                .iter()
                .any(|ext| unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) == name })
        };

        let mut enabled_extensions = Vec::new();
        let swapchain_supported = is_dev_ext_supported(swapchain::NAME);
        if swapchain_supported {
            enabled_extensions.push(swapchain::NAME.as_ptr());
        }
        let portability_subset_ext = vk::KHR_PORTABILITY_SUBSET_NAME;
        if is_dev_ext_supported(portability_subset_ext) {
            enabled_extensions.push(portability_subset_ext.as_ptr());
        }

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&features)
            .enabled_extension_names(&enabled_extensions)
            .push_next(&mut features12)
            .push_next(&mut features13);

        if !adapter.info.supports_dynamic_rendering
            || !adapter.info.supports_timeline_semaphores
            || !adapter.info.supports_descriptor_indexing
        {
            return Err(HalError::DeviceCreationFailed(
                "Adapter does not support required features: Vulkan 1.3 dynamic rendering, timeline semaphores, and descriptor indexing are required".to_string(),
            ));
        }

        let device = unsafe {
            self.instance
                .create_device(adapter.physical_device, &device_create_info, None)
                .map_err(|e| HalError::DeviceCreationFailed(format!(
                    "Failed to create Vulkan logical device with required features (Vulkan 1.3+, Dynamic Rendering, Timeline Semaphores, Descriptor Indexing): {e}"
                )))?
        };

        let queue = unsafe { device.get_device_queue(adapter.queue_family_index, 0) };

        let swapchain_loader = if swapchain_supported {
            Some(swapchain::Device::new(&self.instance, &device))
        } else {
            None
        };

        let bindless_heap = bindless::VulkanBindlessHeap::new(
            &device,
            bindless::DEFAULT_MAX_STORAGE_BUFFERS,
            bindless::DEFAULT_MAX_TEXTURES,
            bindless::DEFAULT_MAX_SAMPLERS,
            bindless::DEFAULT_MAX_STORAGE_IMAGES,
        )
        .ok();

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let internal_frame_semaphore = unsafe {
            device
                .create_semaphore(&sem_create_info, None)
                .map_err(|e| {
                    HalError::DeviceCreationFailed(format!(
                        "Failed to create internal timeline semaphore: {e}"
                    ))
                })?
        };

        let pipeline_cache = unsafe {
            let cache_info = vk::PipelineCacheCreateInfo::default();
            device
                .create_pipeline_cache(&cache_info, None)
                .map_err(|e| {
                    HalError::DeviceCreationFailed(format!("Failed to create pipeline cache: {e}"))
                })?
        };

        Ok(VulkanDevice {
            device,
            physical_device: adapter.physical_device,
            queue: std::sync::Mutex::new(queue),
            queue_family_index: adapter.queue_family_index,
            memory_properties: adapter.memory_properties,
            non_coherent_atom_size: adapter.non_coherent_atom_size,
            bindless_heap,
            swapchain_loader,
            command_recorder_cache: std::sync::Mutex::new(Vec::new()),
            fence_cache: std::sync::Mutex::new(Vec::new()),
            internal_frame_semaphore,
            internal_frame_counter: std::sync::atomic::AtomicU64::new(0),
            pending_async_recorders: std::sync::Mutex::new(Vec::new()),
            allocator: std::sync::Mutex::new(allocator::VulkanSuballocator::new()),
            pipeline_cache,
        })
    }
}

impl VulkanAdapter {
    pub fn info(&self) -> &AdapterInfo {
        &self.info
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
