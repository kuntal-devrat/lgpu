//! Vulkan backend implementation for lgpu-hal
//!
//! Uses zero-cost dynamic loading via `ash` to interface with `libvulkan.so.1`,
//! `vulkan-1.dll`, or `MoltenVK`.

pub use ash::vk;
use std::ffi::CString;

pub mod bindless;
use ash::khr::{surface, swapchain, xlib_surface, xcb_surface, wayland_surface, win32_surface};
use ash::ext::metal_surface;

use crate::{
    AdapterInfo, AddressMode, Backend, BufferDesc, BufferUsage, DeviceType, FilterMode, HalError,
    MemoryLocation, SamplerDesc, TextureDesc, TextureFormat, TextureUsage,
};

pub use ash::vk::Pipeline as RawPipeline;

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

pub struct VulkanDevice {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue: std::sync::Mutex<vk::Queue>,
    pub queue_family_index: u32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub non_coherent_atom_size: u64,
    pub bindless_heap: Option<bindless::VulkanBindlessHeap>,
    pub swapchain_loader: Option<swapchain::Device>,
    pub command_recorder_cache: std::sync::Mutex<Vec<(vk::CommandPool, vk::CommandBuffer)>>,
    pub fence_cache: std::sync::Mutex<Vec<vk::Fence>>,
    pub internal_frame_semaphore: vk::Semaphore,
    pub internal_frame_counter: std::sync::atomic::AtomicU64,
    pub pending_async_recorders: std::sync::Mutex<Vec<(u64, Vec<VulkanCommandRecorder>)>>,
}

pub struct VulkanSwapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
}

pub struct VulkanBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    mapped_ptr: Option<*mut u8>,
    is_coherent: bool,
}

pub struct VulkanTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub current_layout: std::sync::atomic::AtomicI32,
}

pub struct VulkanShader {
    module: vk::ShaderModule,
}

pub struct VulkanComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

pub struct VulkanCommandRecorder {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

pub struct VulkanTimelineSemaphore {
    pub semaphore: vk::Semaphore,
}

pub struct VulkanSampler {
    pub sampler: vk::Sampler,
}

unsafe impl Send for VulkanBuffer {}
unsafe impl Sync for VulkanBuffer {}

unsafe impl Send for VulkanTexture {}
unsafe impl Sync for VulkanTexture {}

unsafe impl Send for VulkanSampler {}
unsafe impl Sync for VulkanSampler {}

unsafe impl Send for VulkanShader {}
unsafe impl Sync for VulkanShader {}

unsafe impl Send for VulkanComputePipeline {}
unsafe impl Sync for VulkanComputePipeline {}

unsafe impl Send for VulkanCommandRecorder {}
unsafe impl Send for VulkanTimelineSemaphore {}
unsafe impl Sync for VulkanTimelineSemaphore {}
unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}


impl VulkanInstance {
    pub fn new() -> Result<Self, HalError> {
        let entry = unsafe {
            ash::Entry::load().map_err(|e| HalError::LibraryLoadError(e.to_string()))?
        };

        let app_name = CString::new("lgpu Application").unwrap();
        let engine_name = CString::new("lgpu").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let supported_extensions = unsafe {
            entry.enumerate_instance_extension_properties(None).unwrap_or_default()
        };
        let is_ext_supported = |name: &std::ffi::CStr| {
            supported_extensions.iter().any(|ext| {
                unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) == name }
            })
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
            entry
                .create_instance(&create_info, None)
                .map_err(|e| HalError::BackendError(format!("Failed to create Vulkan instance: {e}")))?
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
                    .dpy(display.display.map(|p| p.as_ptr() as *mut _).unwrap_or(std::ptr::null_mut()))
                    .window(window.window as vk::Window);
                unsafe { xlib_loader.create_xlib_surface(&create_info, None) }
                    .map_err(|e| HalError::BackendError(format!("Failed to create Xlib surface: {e}")))
            }
            (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
                let xcb_loader = xcb_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(display.connection.map(|p| p.as_ptr() as *mut _).unwrap_or(std::ptr::null_mut()))
                    .window(window.window.get());
                unsafe { xcb_loader.create_xcb_surface(&create_info, None) }
                    .map_err(|e| HalError::BackendError(format!("Failed to create XCB surface: {e}")))
            }
            (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
                let wayland_loader = wayland_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(display.display.as_ptr() as *mut _)
                    .surface(window.surface.as_ptr() as *mut _);
                unsafe { wayland_loader.create_wayland_surface(&create_info, None) }
                    .map_err(|e| HalError::BackendError(format!("Failed to create Wayland surface: {e}")))
            }
            (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
                let win32_loader = win32_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(window.hinstance.map(|p| p.get() as isize).unwrap_or(0))
                    .hwnd(window.hwnd.get() as isize);
                unsafe { win32_loader.create_win32_surface(&create_info, None) }
                    .map_err(|e| HalError::BackendError(format!("Failed to create Win32 surface: {e}")))
            }
            (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) => {
                let metal_loader = metal_surface::Instance::new(&self.entry, &self.instance);
                let create_info = vk::MetalSurfaceCreateInfoEXT::default()
                    .layer(window.ns_view.as_ptr() as *const _);
                unsafe { metal_loader.create_metal_surface(&create_info, None) }
                    .map_err(|e| HalError::BackendError(format!("Failed to create Metal surface: {e}")))
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
            let queue_props =
                unsafe { self.instance.get_physical_device_queue_family_properties(pd) };

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

            let adapter_info = AdapterInfo {
                name,
                backend: Backend::Vulkan,
                device_type,
                vendor_id: props.vendor_id,
                device_id: props.device_id,
                supports_dynamic_rendering: true,
                supports_timeline_semaphores: true,
                supports_descriptor_indexing: true,
                max_compute_workgroup_invocations: props
                    .limits
                    .max_compute_work_group_invocations,
                max_compute_workgroup_size: props.limits.max_compute_work_group_size,
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
            supported_dev_extensions.iter().any(|ext| {
                unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) == name }
            })
        };

        let mut enabled_extensions = Vec::new();
        let swapchain_supported = is_dev_ext_supported(swapchain::NAME);
        if swapchain_supported {
            enabled_extensions.push(swapchain::NAME.as_ptr());
        }

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_features(&features)
            .enabled_extension_names(&enabled_extensions)
            .push_next(&mut features12)
            .push_next(&mut features13);

        let device = unsafe {
            self.instance
                .create_device(adapter.physical_device, &device_create_info, None)
                .or_else(|_| {
                    // Fallback without features12 if unsupported on legacy drivers
                    let fallback_create_info = vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queue_infos)
                        .enabled_features(&features)
                        .enabled_extension_names(&enabled_extensions);
                    self.instance
                        .create_device(adapter.physical_device, &fallback_create_info, None)
                })
                .map_err(|e| HalError::DeviceCreationFailed(e.to_string()))?
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
        ).ok();

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let internal_frame_semaphore = unsafe {
            device
                .create_semaphore(&sem_create_info, None)
                .map_err(|e| HalError::DeviceCreationFailed(format!("Failed to create internal timeline semaphore: {e}")))?
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
        })
    }
}

impl VulkanAdapter {
    pub fn info(&self) -> &AdapterInfo {
        &self.info
    }
}

impl VulkanDevice {
    fn find_memory_type(
        &self,
        type_filter: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Result<(u32, bool), HalError> {
        let wants_host = required.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        if wants_host {
            let preferred = required | vk::MemoryPropertyFlags::HOST_COHERENT;
            for i in 0..self.memory_properties.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && (self.memory_properties.memory_types[i as usize].property_flags & preferred) == preferred
                {
                    return Ok((i, true));
                }
            }
        }

        for i in 0..self.memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & required) == required
            {
                let is_coherent = (self.memory_properties.memory_types[i as usize].property_flags
                    & vk::MemoryPropertyFlags::HOST_COHERENT) == vk::MemoryPropertyFlags::HOST_COHERENT;
                return Ok((i, is_coherent));
            }
        }
        Err(HalError::OutOfMemory("Suitable memory type not found".to_string()))
    }

    pub fn create_buffer(&self, desc: &BufferDesc) -> Result<VulkanBuffer, HalError> {
        let mut vk_usage = vk::BufferUsageFlags::empty();
        if desc.usage.contains(BufferUsage::TRANSFER_SRC) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if desc.usage.contains(BufferUsage::TRANSFER_DST) {
            vk_usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if desc.usage.contains(BufferUsage::STORAGE) {
            vk_usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if desc.usage.contains(BufferUsage::UNIFORM) {
            vk_usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if desc.usage.contains(BufferUsage::VERTEX) {
            vk_usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if desc.usage.contains(BufferUsage::INDEX) {
            vk_usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if desc.usage.contains(BufferUsage::INDIRECT) {
            vk_usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateBuffer failed: {e}")))?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let mem_props = match desc.memory_location {
            MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryLocation::CpuToGpu => vk::MemoryPropertyFlags::HOST_VISIBLE,
            MemoryLocation::GpuToCpu => vk::MemoryPropertyFlags::HOST_VISIBLE,
        };

        let (mem_type_index, is_coherent) = self.find_memory_type(mem_reqs.memory_type_bits, mem_props)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| HalError::OutOfMemory(format!("vkAllocateMemory failed: {e}")))?
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| HalError::BackendError(format!("vkBindBufferMemory failed: {e}")))?;
        }

        let mut mapped_ptr = None;
        if desc.memory_location != MemoryLocation::GpuOnly {
            let ptr = unsafe {
                self.device
                    .map_memory(memory, 0, desc.size, vk::MemoryMapFlags::empty())
                    .map_err(|e| HalError::MapError(e.to_string()))?
            };
            mapped_ptr = Some(ptr as *mut u8);
        }

        Ok(VulkanBuffer {
            buffer,
            memory,
            size: desc.size,
            mapped_ptr,
            is_coherent,
        })
    }

    pub fn write_buffer(&self, buffer: &VulkanBuffer, offset: u64, data: &[u8]) -> Result<(), HalError> {
        let ptr = buffer
            .mapped_ptr
            .ok_or_else(|| HalError::MapError("Buffer is not host-mapped".to_string()))?;
        if offset.checked_add(data.len() as u64).map_or(true, |end| end > buffer.size) {
            return Err(HalError::CommandError("Buffer write overflow".to_string()));
        }
        unsafe {
            let dst = ptr.add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());

            if !buffer.is_coherent {
                let atom = self.non_coherent_atom_size.max(1);
                let aligned_offset = (offset / atom) * atom;
                let end = offset + data.len() as u64;
                let range = if end >= buffer.size {
                    vk::MappedMemoryRange::default()
                        .memory(buffer.memory)
                        .offset(aligned_offset)
                        .size(vk::WHOLE_SIZE)
                } else {
                    let aligned_end = ((end + atom - 1) / atom) * atom;
                    let aligned_size = (aligned_end - aligned_offset).min(buffer.size - aligned_offset);
                    vk::MappedMemoryRange::default()
                        .memory(buffer.memory)
                        .offset(aligned_offset)
                        .size(aligned_size)
                };
                self.device
                    .flush_mapped_memory_ranges(&[range])
                    .map_err(|e| HalError::MapError(format!("vkFlushMappedMemoryRanges failed: {e}")))?;
            }
        }
        Ok(())
    }

    pub fn read_buffer(&self, buffer: &VulkanBuffer, offset: u64, data: &mut [u8]) -> Result<(), HalError> {
        let ptr = buffer
            .mapped_ptr
            .ok_or_else(|| HalError::MapError("Buffer is not host-mapped".to_string()))?;
        if offset.checked_add(data.len() as u64).map_or(true, |end| end > buffer.size) {
            return Err(HalError::CommandError("Buffer read overflow".to_string()));
        }
        unsafe {
            if !buffer.is_coherent {
                let atom = self.non_coherent_atom_size.max(1);
                let aligned_offset = (offset / atom) * atom;
                let end = offset + data.len() as u64;
                let range = if end >= buffer.size {
                    vk::MappedMemoryRange::default()
                        .memory(buffer.memory)
                        .offset(aligned_offset)
                        .size(vk::WHOLE_SIZE)
                } else {
                    let aligned_end = ((end + atom - 1) / atom) * atom;
                    let aligned_size = (aligned_end - aligned_offset).min(buffer.size - aligned_offset);
                    vk::MappedMemoryRange::default()
                        .memory(buffer.memory)
                        .offset(aligned_offset)
                        .size(aligned_size)
                };
                self.device
                    .invalidate_mapped_memory_ranges(&[range])
                    .map_err(|e| HalError::MapError(format!("vkInvalidateMappedMemoryRanges failed: {e}")))?;
            }

            let src = ptr.add(offset as usize);
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr(), data.len());
        }
        Ok(())
    }

    pub fn create_texture(&self, desc: &TextureDesc) -> Result<VulkanTexture, HalError> {
        let vk_format = match desc.format {
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            TextureFormat::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
            TextureFormat::R32Float => vk::Format::R32_SFLOAT,
            TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        };

        let mut usage = vk::ImageUsageFlags::empty();
        if desc.usage.contains(TextureUsage::COPY_SRC) {
            usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if desc.usage.contains(TextureUsage::COPY_DST) {
            usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if desc.usage.contains(TextureUsage::TEXTURE_BINDING) {
            usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if desc.usage.contains(TextureUsage::STORAGE_BINDING) {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT) {
            if desc.format == TextureFormat::Depth32Float {
                usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            } else {
                usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            }
        }

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth.max(1),
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            self.device
                .create_image(&image_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateImage failed: {e}")))?
        };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let (mem_type_index, _) = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| HalError::OutOfMemory(format!("vkAllocateMemory failed for image: {e}")))?
        };

        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(|e| HalError::BackendError(format!("vkBindImageMemory failed: {e}")))?;
        }

        let aspect_mask = if desc.format == TextureFormat::Depth32Float {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(subresource_range);

        let view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateImageView failed: {e}")))?
        };

        Ok(VulkanTexture {
            image,
            view,
            memory,
            width: desc.width,
            height: desc.height,
            format: vk_format,
            current_layout: std::sync::atomic::AtomicI32::new(vk::ImageLayout::UNDEFINED.as_raw()),
        })
    }

    pub fn destroy_texture(&self, texture: VulkanTexture) {
        unsafe {
            if texture.memory != vk::DeviceMemory::null() {
                if texture.view != vk::ImageView::null() {
                    self.device.destroy_image_view(texture.view, None);
                }
                self.device.destroy_image(texture.image, None);
                self.device.free_memory(texture.memory, None);
            }
        }
    }

    pub fn cmd_clear_and_render(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        clear_color: [f32; 4],
    ) {
        unsafe {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let old_layout_raw = texture.current_layout.swap(
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
            let old_layout = vk::ImageLayout::from_raw(old_layout_raw);

            let (src_stage, src_access) = match old_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                ),
                _ => (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                ),
            };

            let barrier_to_attach = vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource_range)
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                src_stage,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_attach],
            );

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear_color,
                },
            };

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value);

            let color_attachments = [color_attachment];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: texture.width,
                    height: texture.height,
                },
            };

            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(&color_attachments);

            self.device
                .cmd_begin_rendering(recorder.command_buffer, &rendering_info);

            self.device.cmd_end_rendering(recorder.command_buffer);

            let barrier_to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_src],
            );

            texture.current_layout.store(
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
        }
    }

    pub fn cmd_clear_and_render_with_depth(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth_texture: &VulkanTexture,
        clear_color: [f32; 4],
        clear_depth: f32,
    ) {
        unsafe {
            let color_subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let old_color_layout = vk::ImageLayout::from_raw(
                color_texture.current_layout.swap(
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw(),
                    std::sync::atomic::Ordering::SeqCst,
                ),
            );

            let color_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(old_color_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

            let depth_subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let old_depth_layout = vk::ImageLayout::from_raw(
                depth_texture.current_layout.swap(
                    vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL.as_raw(),
                    std::sync::atomic::Ordering::SeqCst,
                ),
            );

            let depth_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(old_depth_layout)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(depth_texture.image)
                .subresource_range(depth_subresource)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[color_barrier, depth_barrier],
            );

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(color_texture.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: clear_color },
                });

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_texture.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: clear_depth, stencil: 0 },
                });

            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: color_texture.width,
                    height: color_texture.height,
                },
            };

            let color_attachments = [color_attachment];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);


            self.device.cmd_begin_rendering(recorder.command_buffer, &rendering_info);
            self.device.cmd_end_rendering(recorder.command_buffer);

            let barrier_to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_src],
            );

            color_texture.current_layout.store(
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
            depth_texture.current_layout.store(
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
        }
    }


    pub fn cmd_copy_image_to_buffer(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        buffer: &VulkanBuffer,
    ) {
        unsafe {
            let current_raw = texture.current_layout.load(std::sync::atomic::Ordering::SeqCst);
            let current_layout = vk::ImageLayout::from_raw(current_raw);

            if current_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let (src_stage, src_access) = match current_layout {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    ),
                    _ => (
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::AccessFlags::empty(),
                    ),
                };

                let barrier_to_src = vk::ImageMemoryBarrier::default()
                    .old_layout(current_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(subresource_range)
                    .src_access_mask(src_access)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

                self.device.cmd_pipeline_barrier(
                    recorder.command_buffer,
                    src_stage,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_to_src],
                );

                texture.current_layout.store(
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw(),
                    std::sync::atomic::Ordering::SeqCst,
                );
            }

            let subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);

            let copy_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(texture.width)
                .buffer_image_height(texture.height)
                .image_subresource(subresource)
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: texture.width,
                    height: texture.height,
                    depth: 1,
                });

            self.device.cmd_copy_image_to_buffer(
                recorder.command_buffer,
                texture.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.buffer,
                &[copy_region],
            );
        }
    }

    pub fn create_shader_from_spirv(&self, spirv_words: &[u32]) -> Result<VulkanShader, HalError> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
        let module = unsafe {
            self.device
                .create_shader_module(&create_info, None)
                .map_err(|e| HalError::ShaderError(e.to_string()))?
        };
        Ok(VulkanShader { module })
    }

    pub fn create_compute_pipeline(
        &self,
        shader: &VulkanShader,
        buffer_count: u32,
    ) -> Result<VulkanComputePipeline, HalError> {
        // Create descriptor set layout for storage buffers
        let mut bindings = Vec::with_capacity(buffer_count as usize);
        for i in 0..buffer_count {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

        let layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let entry_point = CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| HalError::PipelineError(e.to_string()))?
        };

        // Create descriptor pool and allocate descriptor set
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(buffer_count.max(1))];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);

        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| HalError::PipelineError(e.to_string()))?
        };

        Ok(VulkanComputePipeline {
            pipeline: pipelines[0],
            layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set: descriptor_sets[0],
        })
    }

    pub fn bind_buffers_to_compute_pipeline(
        &self,
        pipeline: &VulkanComputePipeline,
        buffers: &[&VulkanBuffer],
    ) {
        let mut buffer_infos = Vec::new();
        let mut writes = Vec::new();

        for (_binding, buffer) in buffers.iter().enumerate() {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.size),
            );
        }

        for (binding, info) in buffer_infos.iter().enumerate() {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(pipeline.descriptor_set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info));
            writes.push(write);
        }

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn create_command_recorder(&self) -> Result<VulkanCommandRecorder, HalError> {
        // Harvest any completed async command recorders back into cache
        {
            let current_val = unsafe {
                self.device
                    .get_semaphore_counter_value(self.internal_frame_semaphore)
                    .unwrap_or(0)
            };
            let mut pending = self.pending_async_recorders.lock().unwrap();
            let mut cache = self.command_recorder_cache.lock().unwrap();
            pending.retain_mut(|(val, recorders)| {
                if *val <= current_val {
                    for r in recorders.drain(..) {
                        cache.push((r.command_pool, r.command_buffer));
                    }
                    false
                } else {
                    true
                }
            });
        }

        let (command_pool, command_buffer) = self
            .command_recorder_cache
            .lock()
            .unwrap()
            .pop()
            .map(|(pool, cb)| {
                unsafe {
                    self.device
                        .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                        .ok();
                }
                (pool, cb)
            })
            .unwrap_or_else(|| {
                let pool_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

                let command_pool = unsafe {
                    self.device
                        .create_command_pool(&pool_info, None)
                        .expect("Failed to create command pool")
                };

                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);

                let command_buffer = unsafe {
                    self.device
                        .allocate_command_buffers(&alloc_info)
                        .expect("Failed to allocate command buffer")[0]
                };

                (command_pool, command_buffer)
            });

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| HalError::CommandError(e.to_string()))?;
        }

        Ok(VulkanCommandRecorder {
            command_pool,
            command_buffer,
        })
    }

    pub fn register_bindless_buffer(&self, buffer: &VulkanBuffer) -> Result<u32, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let slot = heap.allocate_buffer_slot()?;
        heap.write_buffer_descriptor(&self.device, slot, buffer.buffer, buffer.size);
        Ok(slot)
    }

    pub fn unregister_bindless_buffer(&self, slot: u32) {
        if let Some(heap) = &self.bindless_heap {
            heap.free_buffer_slot(slot);
        }
    }

    pub fn register_bindless_texture(&self, texture: &VulkanTexture) -> Result<u32, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let slot = heap.allocate_texture_slot()?;
        heap.write_texture_descriptor(&self.device, slot, texture.view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        Ok(slot)
    }

    pub fn unregister_bindless_texture(&self, slot: u32) {
        if let Some(heap) = &self.bindless_heap {
            heap.free_texture_slot(slot);
        }
    }

    pub fn create_sampler(&self, desc: &SamplerDesc) -> Result<VulkanSampler, HalError> {
        let vk_filter = |f: FilterMode| match f {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        };
        let vk_mipmap_mode = |f: FilterMode| match f {
            FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
        };
        let vk_address_mode = |a: AddressMode| match a {
            AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        };

        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk_filter(desc.mag_filter))
            .min_filter(vk_filter(desc.min_filter))
            .mipmap_mode(vk_mipmap_mode(desc.mipmap_filter))
            .address_mode_u(vk_address_mode(desc.address_mode_u))
            .address_mode_v(vk_address_mode(desc.address_mode_v))
            .address_mode_w(vk_address_mode(desc.address_mode_w))
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .compare_enable(false)
            .compare_op(vk::CompareOp::NEVER)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false);

        let sampler = unsafe {
            self.device
                .create_sampler(&create_info, None)
                .map_err(|e| HalError::BackendError(format!("vkCreateSampler failed: {e}")))?
        };

        Ok(VulkanSampler { sampler })
    }

    pub fn destroy_sampler(&self, sampler: VulkanSampler) {
        unsafe {
            self.device.destroy_sampler(sampler.sampler, None);
        }
    }

    pub fn register_bindless_sampler(&self, sampler: &VulkanSampler) -> Result<u32, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;
        let slot = heap.allocate_sampler_slot()?;
        heap.write_sampler_descriptor(&self.device, slot, sampler.sampler);
        Ok(slot)
    }

    pub fn unregister_bindless_sampler(&self, slot: u32) {
        if let Some(heap) = &self.bindless_heap {
            heap.free_sampler_slot(slot);
        }
    }


    pub fn create_bindless_compute_pipeline(
        &self,
        shader: &VulkanShader,
    ) -> Result<vk::Pipeline, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        let entry_point = CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(&entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(heap.pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| HalError::PipelineError(e.to_string()))?
        };

        Ok(pipelines[0])
    }

    pub fn dispatch_bindless(
        &self,
        recorder: &VulkanCommandRecorder,
        pipeline: vk::Pipeline,
        push_constants: &[u8],
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<(), HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        if push_constants.len() > bindless::PUSH_CONSTANT_SIZE as usize {
            return Err(HalError::CommandError(format!(
                "Push constants size {} exceeds maximum allowed size of {}",
                push_constants.len(),
                bindless::PUSH_CONSTANT_SIZE
            )));
        }

        unsafe {
            self.device.cmd_bind_pipeline(
                recorder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                recorder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                heap.pipeline_layout,
                0,
                &[heap.descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                let mut aligned_pc = [0u8; bindless::PUSH_CONSTANT_SIZE as usize];
                aligned_pc[..push_constants.len()].copy_from_slice(push_constants);
                let aligned_len = ((push_constants.len() + 3) / 4) * 4;

                self.device.cmd_push_constants(
                    recorder.command_buffer,
                    heap.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &aligned_pc[..aligned_len],
                );
            }
            self.device.cmd_dispatch(recorder.command_buffer, x, y, z);
        }
        Ok(())
    }

    pub fn destroy_raw_pipeline(&self, pipeline: vk::Pipeline) {
        unsafe {
            self.device.destroy_pipeline(pipeline, None);
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        shader: &VulkanShader,
        vs_entry: &str,
        fs_entry: &str,
        color_format: TextureFormat,
    ) -> Result<vk::Pipeline, HalError> {
        self.create_graphics_pipeline_with_depth(
            shader,
            vs_entry,
            fs_entry,
            color_format,
            None,
            false,
            false,
        )
    }

    pub fn create_graphics_pipeline_with_depth(
        &self,
        shader: &VulkanShader,
        vs_entry: &str,
        fs_entry: &str,
        color_format: TextureFormat,
        depth_format: Option<TextureFormat>,
        depth_test: bool,
        depth_write: bool,
    ) -> Result<vk::Pipeline, HalError> {
        let heap = self.bindless_heap.as_ref().ok_or_else(|| {
            HalError::BackendError("Bindless descriptor heap is not enabled".to_string())
        })?;

        let vk_format = match color_format {
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            TextureFormat::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,
            TextureFormat::R32Float => vk::Format::R32_SFLOAT,
            TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        };

        let vs_name = CString::new(vs_entry).unwrap();
        let fs_name = CString::new(fs_entry).unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader.module)
                .name(&vs_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader.module)
                .name(&fs_name),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment);

        let color_formats = [vk_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats);

        let vk_depth_format = match depth_format {
            Some(TextureFormat::Depth32Float) => Some(vk::Format::D32_SFLOAT),
            Some(f) => return Err(HalError::BackendError(format!("Unsupported depth format: {f:?}"))),
            None => None,
        };

        if let Some(df) = vk_depth_format {
            rendering_info = rendering_info.depth_attachment_format(df);
        }

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_test)
            .depth_write_enable(depth_write)
            .depth_compare_op(if depth_test {
                vk::CompareOp::LESS_OR_EQUAL
            } else {
                vk::CompareOp::ALWAYS
            })
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(heap.pipeline_layout)
            .push_next(&mut rendering_info);

        if depth_format.is_some() || depth_test {
            pipeline_info = pipeline_info.depth_stencil_state(&depth_stencil);
        }

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| HalError::PipelineError(format!("create_graphics_pipelines failed: {e}")))?
        };

        Ok(pipelines[0])
    }


    pub fn cmd_draw(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        vertex_count: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
    ) {
        let heap = match &self.bindless_heap {
            Some(h) => h,
            None => return,
        };

        if push_constants.len() > bindless::PUSH_CONSTANT_SIZE as usize {
            eprintln!(
                "Push constants size {} exceeds maximum allowed size of {}",
                push_constants.len(),
                bindless::PUSH_CONSTANT_SIZE
            );
            return;
        }

        unsafe {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let old_layout_raw = texture.current_layout.swap(
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
            let old_layout = vk::ImageLayout::from_raw(old_layout_raw);

            let (src_stage, src_access) = match old_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                ),
                _ => (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                ),
            };

            let barrier_to_attach = vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource_range)
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                src_stage,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_attach],
            );

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear_color,
                },
            };

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(texture.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value);

            let color_attachments = [color_attachment];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: texture.width,
                    height: texture.height,
                },
            };

            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(&color_attachments);

            self.device
                .cmd_begin_rendering(recorder.command_buffer, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: texture.width as f32,
                height: texture.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device
                .cmd_set_viewport(recorder.command_buffer, 0, &[viewport]);

            let scissor = render_area;
            self.device
                .cmd_set_scissor(recorder.command_buffer, 0, &[scissor]);

            self.device.cmd_bind_pipeline(
                recorder.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                recorder.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                heap.pipeline_layout,
                0,
                &[heap.descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                let mut aligned_pc = [0u8; bindless::PUSH_CONSTANT_SIZE as usize];
                aligned_pc[..push_constants.len()].copy_from_slice(push_constants);
                let aligned_len = ((push_constants.len() + 3) / 4) * 4;

                self.device.cmd_push_constants(
                    recorder.command_buffer,
                    heap.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &aligned_pc[..aligned_len],
                );
            }
            self.device.cmd_draw(recorder.command_buffer, vertex_count, 1, 0, 0);

            self.device.cmd_end_rendering(recorder.command_buffer);

            let barrier_to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image)
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_src],
            );

            texture.current_layout.store(
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
        }
    }

    pub fn cmd_draw_triangle(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        clear_color: [f32; 4],
    ) {
        self.cmd_draw(recorder, texture, pipeline, 3, &[], clear_color);
    }

    pub fn cmd_draw_with_depth(
        &self,
        recorder: &VulkanCommandRecorder,
        color_texture: &VulkanTexture,
        depth_texture: &VulkanTexture,
        pipeline: vk::Pipeline,
        vertex_count: u32,
        push_constants: &[u8],
        clear_color: [f32; 4],
        clear_depth: f32,
    ) {
        let heap = match &self.bindless_heap {
            Some(h) => h,
            None => return,
        };

        if push_constants.len() > bindless::PUSH_CONSTANT_SIZE as usize {
            eprintln!(
                "Push constants size {} exceeds maximum allowed size of {}",
                push_constants.len(),
                bindless::PUSH_CONSTANT_SIZE
            );
            return;
        }

        unsafe {
            let color_subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let old_color_layout_raw = color_texture.current_layout.swap(
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
            let old_color_layout = vk::ImageLayout::from_raw(old_color_layout_raw);

            let (src_stage, src_access) = match old_color_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                ),
                _ => (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                ),
            };

            let color_barrier_to_attach = vk::ImageMemoryBarrier::default()
                .old_layout(old_color_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource)
                .src_access_mask(src_access)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

            let depth_subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let old_depth_layout_raw = depth_texture.current_layout.swap(
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
            let old_depth_layout = vk::ImageLayout::from_raw(old_depth_layout_raw);

            let (depth_src_stage, depth_src_access) = match old_depth_layout {
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
                _ => (
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::AccessFlags::empty(),
                ),
            };

            let depth_barrier_to_attach = vk::ImageMemoryBarrier::default()
                .old_layout(old_depth_layout)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(depth_texture.image)
                .subresource_range(depth_subresource)
                .src_access_mask(depth_src_access)
                .dst_access_mask(
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                src_stage | depth_src_stage,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[color_barrier_to_attach, depth_barrier_to_attach],
            );

            let clear_color_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear_color,
                },
            };

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(color_texture.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_color_value);

            let clear_depth_value = vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: clear_depth,
                    stencil: 0,
                },
            };

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_texture.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_depth_value);

            let color_attachments = [color_attachment];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: color_texture.width,
                    height: color_texture.height,
                },
            };

            let rendering_info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);

            self.device
                .cmd_begin_rendering(recorder.command_buffer, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: color_texture.width as f32,
                height: color_texture.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device
                .cmd_set_viewport(recorder.command_buffer, 0, &[viewport]);

            let scissor = render_area;
            self.device
                .cmd_set_scissor(recorder.command_buffer, 0, &[scissor]);

            self.device.cmd_bind_pipeline(
                recorder.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                recorder.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                heap.pipeline_layout,
                0,
                &[heap.descriptor_set],
                &[],
            );
            if !push_constants.is_empty() {
                let mut aligned_pc = [0u8; bindless::PUSH_CONSTANT_SIZE as usize];
                aligned_pc[..push_constants.len()].copy_from_slice(push_constants);
                let aligned_len = ((push_constants.len() + 3) / 4) * 4;

                self.device.cmd_push_constants(
                    recorder.command_buffer,
                    heap.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    &aligned_pc[..aligned_len],
                );
            }
            self.device.cmd_draw(recorder.command_buffer, vertex_count, 1, 0, 0);

            self.device.cmd_end_rendering(recorder.command_buffer);

            let barrier_to_src = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_texture.image)
                .subresource_range(color_subresource)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_to_src],
            );

            color_texture.current_layout.store(
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
            depth_texture.current_layout.store(
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL.as_raw(),
                std::sync::atomic::Ordering::SeqCst,
            );
        }
    }


    pub fn dispatch(
        &self,
        recorder: &VulkanCommandRecorder,
        pipeline: &VulkanComputePipeline,
        x: u32,
        y: u32,
        z: u32,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                recorder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                recorder.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[pipeline.descriptor_set],
                &[],
            );
            self.device.cmd_dispatch(recorder.command_buffer, x, y, z);
        }
    }

    fn submit_batches_internal(
        &self,
        recorders: Vec<VulkanCommandRecorder>,
        wait_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        signal_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        sync_fence: Option<vk::Fence>,
    ) -> Result<(), HalError> {
        if recorders.is_empty() {
            return Ok(());
        }

        unsafe {
            let mut vk_cmd_buffers = Vec::with_capacity(recorders.len());
            for r in &recorders {
                self.device
                    .end_command_buffer(r.command_buffer)
                    .map_err(|e| HalError::CommandError(e.to_string()))?;
                vk_cmd_buffers.push(r.command_buffer);
            }

            let mut vk_wait_sems = Vec::with_capacity(wait_semaphores.len());
            let mut vk_wait_values = Vec::with_capacity(wait_semaphores.len());
            let mut vk_wait_stages = Vec::with_capacity(wait_semaphores.len());
            for (sem, val) in wait_semaphores {
                vk_wait_sems.push(sem.semaphore);
                vk_wait_values.push(*val);
                vk_wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }

            let next_val = self.internal_frame_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            let mut vk_signal_sems = Vec::with_capacity(signal_semaphores.len() + 1);
            let mut vk_signal_values = Vec::with_capacity(signal_semaphores.len() + 1);
            for (sem, val) in signal_semaphores {
                vk_signal_sems.push(sem.semaphore);
                vk_signal_values.push(*val);
            }
            vk_signal_sems.push(self.internal_frame_semaphore);
            vk_signal_values.push(next_val);

            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&vk_wait_values)
                .signal_semaphore_values(&vk_signal_values);

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&vk_cmd_buffers)
                .wait_semaphores(&vk_wait_sems)
                .wait_dst_stage_mask(&vk_wait_stages)
                .signal_semaphores(&vk_signal_sems)
                .push_next(&mut timeline_info);

            let fence = sync_fence.unwrap_or(vk::Fence::null());
            {
                let queue = self.queue.lock().unwrap();
                self.device
                    .queue_submit(*queue, &[submit_info], fence)
                    .map_err(|e| HalError::CommandError(e.to_string()))?;
            }

            if let Some(f) = sync_fence {
                self.device
                    .wait_for_fences(&[f], true, u64::MAX)
                    .map_err(|e| HalError::CommandError(e.to_string()))?;

                self.device.reset_fences(&[f]).ok();
                self.fence_cache.lock().unwrap().push(f);

                let mut cache = self.command_recorder_cache.lock().unwrap();
                for r in recorders {
                    cache.push((r.command_pool, r.command_buffer));
                }
            } else {
                self.pending_async_recorders
                    .lock()
                    .unwrap()
                    .push((next_val, recorders));
            }
        }
        Ok(())
    }

    pub fn submit_and_wait(&self, recorder: VulkanCommandRecorder) -> Result<(), HalError> {
        let fence = self.fence_cache.lock().unwrap().pop().unwrap_or_else(|| {
            let fence_info = vk::FenceCreateInfo::default();
            unsafe {
                self.device
                    .create_fence(&fence_info, None)
                    .expect("Failed to create fence")
            }
        });
        self.submit_batches_internal(vec![recorder], &[], &[], Some(fence))
    }

    pub fn submit_batches_and_wait(&self, recorders: Vec<VulkanCommandRecorder>) -> Result<(), HalError> {
        let fence = self.fence_cache.lock().unwrap().pop().unwrap_or_else(|| {
            let fence_info = vk::FenceCreateInfo::default();
            unsafe {
                self.device
                    .create_fence(&fence_info, None)
                    .expect("Failed to create fence")
            }
        });
        self.submit_batches_internal(recorders, &[], &[], Some(fence))
    }

    pub fn submit_batches_async(
        &self,
        recorders: Vec<VulkanCommandRecorder>,
        wait_semaphores: &[(&VulkanTimelineSemaphore, u64)],
        signal_semaphores: &[(&VulkanTimelineSemaphore, u64)],
    ) -> Result<(), HalError> {
        self.submit_batches_internal(recorders, wait_semaphores, signal_semaphores, None)
    }

    pub fn cmd_memory_barrier(&self, recorder: &VulkanCommandRecorder) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::SHADER_READ);
        unsafe {
            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    pub fn cmd_buffer_barrier(&self, recorder: &VulkanCommandRecorder, buffer: &VulkanBuffer) {
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(buffer.buffer)
            .offset(0)
            .size(buffer.size)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::HOST_READ,
            )
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        unsafe {
            self.device.cmd_pipeline_barrier(
                recorder.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }

    pub fn create_timeline_semaphore(&self, initial_value: u64) -> Result<VulkanTimelineSemaphore, HalError> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        let semaphore = unsafe {
            self.device
                .create_semaphore(&create_info, None)
                .map_err(|e| HalError::BackendError(format!("Failed to create timeline semaphore: {e}")))?
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
                .map_err(|e| HalError::CommandError(format!("vkGetSemaphoreCounterValue failed: {e}")))
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

    pub fn destroy_buffer(&self, buffer: VulkanBuffer) {
        unsafe {
            if let Some(_) = buffer.mapped_ptr {
                self.device.unmap_memory(buffer.memory);
            }
            self.device.destroy_buffer(buffer.buffer, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    pub fn destroy_shader(&self, shader: VulkanShader) {
        unsafe {
            self.device.destroy_shader_module(shader.module, None);
        }
    }

    pub fn destroy_compute_pipeline(&self, pipeline: VulkanComputePipeline) {
        unsafe {
            self.device
                .destroy_descriptor_pool(pipeline.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
            self.device.destroy_pipeline(pipeline.pipeline, None);
            self.device
                .destroy_pipeline_layout(pipeline.layout, None);
        }
    }

    pub fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        surface_loader: &surface::Instance,
        width: u32,
        height: u32,
        old_swapchain: vk::SwapchainKHR,
        preferred_format: Option<vk::Format>,
        preferred_present_mode: Option<vk::PresentModeKHR>,
    ) -> Result<VulkanSwapchain, HalError> {
        let loader = self.swapchain_loader.as_ref().ok_or_else(|| {
            HalError::BackendError("VK_KHR_swapchain is not supported or not enabled".to_string())
        })?;

        let caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, surface)
                .map_err(|e| HalError::BackendError(format!("Failed to query surface capabilities: {e}")))?
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(self.physical_device, surface)
                .map_err(|e| HalError::BackendError(format!("Failed to query surface formats: {e}")))?
        };

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, surface)
                .map_err(|e| HalError::BackendError(format!("Failed to query surface present modes: {e}")))?
        };

        if formats.is_empty() {
            return Err(HalError::BackendError("No surface formats supported".to_string()));
        }

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 && image_count > caps.max_image_count {
            image_count = caps.max_image_count;
        }

        let selected_format = if let Some(pref) = preferred_format {
            formats.iter().find(|f| f.format == pref).cloned().unwrap_or(formats[0])
        } else {
            formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_UNORM
                        || f.format == vk::Format::R8G8B8A8_UNORM
                        || f.format == vk::Format::B8G8R8A8_SRGB
                        || f.format == vk::Format::R8G8B8A8_SRGB
                })
                .cloned()
                .unwrap_or(formats[0])
        };

        let present_mode = match preferred_present_mode {
            Some(mode) if present_modes.contains(&mode) => mode,
            _ => {
                if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                    vk::PresentModeKHR::MAILBOX
                } else {
                    vk::PresentModeKHR::FIFO
                }
            }
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(selected_format.format)
            .image_color_space(selected_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe {
            loader
                .create_swapchain(&create_info, None)
                .map_err(|e| HalError::BackendError(format!("Failed to create swapchain: {e}")))?
        };

        let images = unsafe {
            loader
                .get_swapchain_images(swapchain)
                .map_err(|e| HalError::BackendError(format!("Failed to get swapchain images: {e}")))?
        };

        let mut views = Vec::with_capacity(images.len());
        for &img in &images {
            let view_create_info = vk::ImageViewCreateInfo::default()
                .image(img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(selected_format.format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let view = unsafe {
                self.device
                    .create_image_view(&view_create_info, None)
                    .map_err(|e| HalError::BackendError(format!("Failed to create swapchain image view: {e}")))?
            };
            views.push(view);
        }

        Ok(VulkanSwapchain {
            swapchain,
            images,
            views,
            width: extent.width,
            height: extent.height,
            format: selected_format.format,
        })
    }

    pub fn destroy_swapchain(&self, swapchain: VulkanSwapchain) {
        if let Some(loader) = &self.swapchain_loader {
            unsafe {
                for view in swapchain.views {
                    self.device.destroy_image_view(view, None);
                }
                loader.destroy_swapchain(swapchain.swapchain, None);
            }
        }
    }

    pub fn acquire_next_image(
        &self,
        swapchain: &VulkanSwapchain,
        semaphore: vk::Semaphore,
        timeout_nanos: u64,
    ) -> Result<(u32, bool), HalError> {
        let loader = self.swapchain_loader.as_ref().ok_or_else(|| {
            HalError::BackendError("VK_KHR_swapchain is not enabled".to_string())
        })?;
        unsafe {
            loader
                .acquire_next_image(swapchain.swapchain, timeout_nanos, semaphore, vk::Fence::null())
                .map_err(|e| HalError::BackendError(format!("acquire_next_image failed: {e}")))
        }
    }

    pub fn queue_present(
        &self,
        swapchain: &VulkanSwapchain,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool, HalError> {
        self.queue_present_raw(swapchain.swapchain, image_index, wait_semaphores)
    }

    pub fn queue_present_raw(
        &self,
        swapchain: vk::SwapchainKHR,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool, HalError> {
        let loader = self.swapchain_loader.as_ref().ok_or_else(|| {
            HalError::BackendError("VK_KHR_swapchain is not enabled".to_string())
        })?;

        let swapchains = [swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let queue = self.queue.lock().unwrap();
        unsafe {
            loader
                .queue_present(*queue, &present_info)
                .map_err(|e| HalError::BackendError(format!("queue_present failed: {e}")))
        }
    }

    pub fn wait_idle(&self) -> Result<(), HalError> {
        unsafe {
            self.device
                .device_wait_idle()
                .map_err(|e| HalError::BackendError(format!("device_wait_idle failed: {e}")))
        }
    }

    pub fn cmd_transition_to_present(
        &self,
        recorder: &VulkanCommandRecorder,
        texture: &VulkanTexture,
    ) {
        let current_raw = texture.current_layout.swap(
            vk::ImageLayout::PRESENT_SRC_KHR.as_raw(),
            std::sync::atomic::Ordering::SeqCst,
        );
        let old_layout = vk::ImageLayout::from_raw(current_raw);
        let new_layout = vk::ImageLayout::PRESENT_SRC_KHR;

        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::empty())
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(texture.image)
            .subresource_range(subresource_range);

        let barriers = [barrier];
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        unsafe {
            self.device.cmd_pipeline_barrier2(recorder.command_buffer, &dep_info);
        }
    }

    pub fn submit_for_surface(
        &self,
        recorder: VulkanCommandRecorder,
        wait_semaphore: vk::Semaphore,
        signal_semaphore: vk::Semaphore,
    ) -> Result<(), HalError> {
        unsafe {
            self.device
                .end_command_buffer(recorder.command_buffer)
                .map_err(|e| HalError::CommandError(e.to_string()))?;

            let cmd_buffers = [recorder.command_buffer];
            let wait_sems = [wait_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_sems = [signal_semaphore];

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(&cmd_buffers)
                .wait_semaphores(&wait_sems)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&signal_sems);

            let fence = self.fence_cache.lock().unwrap().pop().unwrap_or_else(|| {
                let fence_info = vk::FenceCreateInfo::default();
                self.device
                    .create_fence(&fence_info, None)
                    .expect("Failed to create fence")
            });
            {
                let queue = self.queue.lock().unwrap();
                self.device
                    .queue_submit(*queue, &[submit_info], fence)
                    .map_err(|e| HalError::CommandError(e.to_string()))?;
            }

            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| HalError::CommandError(e.to_string()))?;

            self.device.reset_fences(&[fence]).ok();
            self.fence_cache.lock().unwrap().push(fence);

            self.command_recorder_cache
                .lock()
                .unwrap()
                .push((recorder.command_pool, recorder.command_buffer));

            Ok(())
        }
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(heap) = self.bindless_heap.take() {
                heap.destroy(&self.device);
            }
            self.device.destroy_semaphore(self.internal_frame_semaphore, None);
            for (pool, _) in self.command_recorder_cache.lock().unwrap().drain(..) {
                self.device.destroy_command_pool(pool, None);
            }
            for (_, recorders) in self.pending_async_recorders.lock().unwrap().drain(..) {
                for r in recorders {
                    self.device.destroy_command_pool(r.command_pool, None);
                }
            }
            for fence in self.fence_cache.lock().unwrap().drain(..) {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_device(None);
        }
    }
}
