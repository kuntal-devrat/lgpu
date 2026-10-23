//! C99 / Zig FFI exports for lgpu

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString};
use std::ptr::{self, NonNull};

use lgpu::{
    AddressMode, Backend, BindlessComputePipeline, BufferDesc, BufferUsage, DeviceType, FilterMode,
    GraphicsPipeline, Instance, MemoryLocation, Sampler, SamplerDesc, Texture, TextureDesc,
    TextureFormat, TextureUsage,
};

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub(crate) fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(msg.into());
    });
}

pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// Struct wrappers for raw pointers
pub struct LgpuInstanceWrapper {
    instance: Instance,
    adapter_names: Vec<CString>,
}

pub struct LgpuDeviceWrapper {
    device: lgpu::Device,
}

pub struct LgpuQueueWrapper {
    queue: lgpu::Queue,
}

pub struct LgpuBufferWrapper {
    pub(crate) buffer: lgpu::Buffer,
}

pub struct LgpuTextureWrapper {
    pub(crate) texture: Texture,
}

pub struct LgpuSamplerWrapper {
    pub(crate) sampler: Sampler,
}

pub struct LgpuShaderWrapper {
    shader: lgpu::Shader,
}

pub struct LgpuComputePipelineWrapper {
    pipeline: lgpu::ComputePipeline,
}

pub struct LgpuBindlessComputePipelineWrapper {
    pub(crate) pipeline: BindlessComputePipeline,
}

pub struct LgpuGraphicsPipelineWrapper {
    pub(crate) pipeline: GraphicsPipeline,
}

pub struct LgpuCommandEncoderWrapper {
    encoder: Option<lgpu::CommandEncoder>,
}

pub struct LgpuCommandBufferWrapper {
    command_buffer: Option<lgpu::CommandBuffer>,
}

pub struct LgpuTimelineSemaphoreWrapper {
    semaphore: lgpu::TimelineSemaphore,
}

pub struct LgpuSurfaceWrapper {
    pub(crate) surface: lgpu::Surface,
}

pub struct LgpuSurfaceTextureWrapper {
    pub(crate) surface_texture: Option<lgpu::SurfaceTexture>,
    pub(crate) texture_wrapper: Option<Box<LgpuTextureWrapper>>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LgpuPresentMode {
    Immediate = 0,
    Mailbox = 1,
    Fifo = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LgpuSurfaceConfig {
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub usage: u32,
    pub present_mode: LgpuPresentMode,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LgpuAdapterInfo {
    pub name: [c_char; 256],
    pub backend: u32,
    pub device_type: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub supports_dynamic_rendering: u8,
    pub supports_timeline_semaphores: u8,
    pub supports_descriptor_indexing: u8,
    pub _padding: u8,
    pub max_compute_workgroup_invocations: u32,
    pub max_compute_workgroup_size: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LgpuDeviceLimits {
    pub max_buffer_size: u64,
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_compute_workgroup_invocations: u32,
    pub max_compute_workgroup_size: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LgpuResult {
    Success = 0,
    ErrorNoAdapter = -1,
    ErrorDeviceCreation = -2,
    ErrorBuffer = -3,
    ErrorShader = -4,
    ErrorPipeline = -5,
    ErrorCommand = -6,
    ErrorGeneric = -99,
}

#[no_mangle]
pub extern "C" fn lgpu_version() -> u32 {
    1
}

/// Creates a new `lgpu` Instance.
///
/// # Safety
///
/// `out_instance` must be a valid, aligned, non-null pointer to a writable `*mut LgpuInstanceWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_instance(
    out_instance: *mut *mut LgpuInstanceWrapper,
) -> LgpuResult {
    if out_instance.is_null() {
        return LgpuResult::ErrorGeneric;
    }

    match Instance::new() {
        Ok(instance) => {
            let mut adapter_names = Vec::new();
            if let Ok(adapters) = instance.enumerate_adapters() {
                for a in adapters {
                    if let Ok(c_str) = CString::new(a.info().name.clone()) {
                        adapter_names.push(c_str);
                    }
                }
            }
            let wrapper = Box::new(LgpuInstanceWrapper {
                instance,
                adapter_names,
            });
            *out_instance = Box::into_raw(wrapper);
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorNoAdapter,
    }
}

/// Returns the number of physical adapters found by the instance.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`, or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_get_adapter_count(instance: *mut LgpuInstanceWrapper) -> u32 {
    if instance.is_null() {
        return 0;
    }
    let wrapper = &*instance;
    wrapper.adapter_names.len() as u32
}

/// Returns the name of the adapter at the specified index as a C string.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`, or null.
/// The returned pointer is valid until `instance` is destroyed.
#[no_mangle]
pub unsafe extern "C" fn lgpu_get_adapter_name(
    instance: *mut LgpuInstanceWrapper,
    index: u32,
) -> *const c_char {
    if instance.is_null() {
        return ptr::null();
    }
    let wrapper = &*instance;
    if (index as usize) < wrapper.adapter_names.len() {
        wrapper.adapter_names[index as usize].as_ptr()
    } else {
        ptr::null()
    }
}

/// Retrieves hardware capabilities and limits for the adapter at `index`.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `out_info` must be a valid, aligned, writable pointer to `LgpuAdapterInfo`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_get_adapter_info(
    instance: *mut LgpuInstanceWrapper,
    index: u32,
    out_info: *mut LgpuAdapterInfo,
) -> LgpuResult {
    if instance.is_null() || out_info.is_null() {
        return LgpuResult::ErrorGeneric;
    }

    let wrapper = &*instance;
    let adapters = match wrapper.instance.enumerate_adapters() {
        Ok(a) => a,
        Err(_) => return LgpuResult::ErrorNoAdapter,
    };

    if (index as usize) >= adapters.len() {
        return LgpuResult::ErrorNoAdapter;
    }

    let info = adapters[index as usize].info();
    let mut c_info = LgpuAdapterInfo {
        name: [0; 256],
        backend: match info.backend {
            Backend::Vulkan => 0,
            Backend::Metal => 1,
            Backend::DirectX12 => 2,
        },
        device_type: match info.device_type {
            DeviceType::DiscreteGpu => 0,
            DeviceType::IntegratedGpu => 1,
            DeviceType::VirtualGpu => 2,
            DeviceType::Cpu => 3,
            DeviceType::Other => 4,
        },
        vendor_id: info.vendor_id,
        device_id: info.device_id,
        supports_dynamic_rendering: info.supports_dynamic_rendering as u8,
        supports_timeline_semaphores: info.supports_timeline_semaphores as u8,
        supports_descriptor_indexing: info.supports_descriptor_indexing as u8,
        _padding: 0,
        max_compute_workgroup_invocations: info.max_compute_workgroup_invocations,
        max_compute_workgroup_size: info.max_compute_workgroup_size,
    };

    let name_bytes = info.name.as_bytes();
    let len = name_bytes.len().min(255);
    for (i, &b) in name_bytes.iter().take(len).enumerate() {
        c_info.name[i] = b as c_char;
    }
    c_info.name[len] = 0;

    *out_info = c_info;
    clear_last_error();
    LgpuResult::Success
}

/// Queries hardware device limits for a specific adapter.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `out_limits` must be a valid, aligned, writable pointer to `LgpuDeviceLimits`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_adapter_get_limits(
    instance: *mut LgpuInstanceWrapper,
    index: u32,
    out_limits: *mut LgpuDeviceLimits,
) -> LgpuResult {
    if instance.is_null() || out_limits.is_null() {
        set_last_error("Null pointer passed to lgpu_adapter_get_limits");
        return LgpuResult::ErrorGeneric;
    }

    let wrapper = &*instance;
    let adapters = match wrapper.instance.enumerate_adapters() {
        Ok(a) => a,
        Err(e) => {
            set_last_error(format!("Failed to enumerate adapters: {e}"));
            return LgpuResult::ErrorNoAdapter;
        }
    };

    if (index as usize) >= adapters.len() {
        set_last_error(format!(
            "Adapter index {index} out of bounds (found {})",
            adapters.len()
        ));
        return LgpuResult::ErrorNoAdapter;
    }

    let info = adapters[index as usize].info();
    let l = info.limits;
    *out_limits = LgpuDeviceLimits {
        max_buffer_size: l.max_buffer_size,
        max_texture_dimension_1d: l.max_texture_dimension_1d,
        max_texture_dimension_2d: l.max_texture_dimension_2d,
        max_texture_dimension_3d: l.max_texture_dimension_3d,
        max_texture_array_layers: l.max_texture_array_layers,
        min_uniform_buffer_offset_alignment: l.min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment: l.min_storage_buffer_offset_alignment,
        max_compute_workgroup_invocations: l.max_compute_workgroup_invocations,
        max_compute_workgroup_size: l.max_compute_workgroup_size,
    };

    clear_last_error();
    LgpuResult::Success
}

/// Retrieves the last error message recorded on the calling thread.
///
/// # Safety
///
/// `out_buf` must point to a writable buffer of at least `max_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn lgpu_get_last_error_message(
    out_buf: *mut c_char,
    max_len: usize,
) -> usize {
    if out_buf.is_null() || max_len == 0 {
        return 0;
    }
    LAST_ERROR.with(|cell| {
        if let Some(ref msg) = *cell.borrow() {
            let bytes = msg.as_bytes();
            let copy_len = bytes.len().min(max_len - 1);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, copy_len);
            *out_buf.add(copy_len) = 0;
            copy_len
        } else {
            *out_buf = 0;
            0
        }
    })
}

/// Requests a logical device and queue from the adapter at `adapter_index`.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `out_device` and `out_queue` must be valid, aligned, writable pointers to receive the created handles.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_device(
    instance: *mut LgpuInstanceWrapper,
    adapter_index: u32,
    out_device: *mut *mut LgpuDeviceWrapper,
    out_queue: *mut *mut LgpuQueueWrapper,
) -> LgpuResult {
    if instance.is_null() || out_device.is_null() || out_queue.is_null() {
        set_last_error("Null pointer passed to lgpu_create_device");
        return LgpuResult::ErrorGeneric;
    }

    let wrapper = &*instance;
    let adapters = match wrapper.instance.enumerate_adapters() {
        Ok(a) => a,
        Err(e) => {
            set_last_error(format!("Failed to enumerate adapters: {e}"));
            return LgpuResult::ErrorNoAdapter;
        }
    };

    if (adapter_index as usize) >= adapters.len() {
        set_last_error(format!("Adapter index {adapter_index} out of bounds"));
        return LgpuResult::ErrorNoAdapter;
    }

    match adapters[adapter_index as usize].request_device(&wrapper.instance) {
        Ok((device, queue)) => {
            *out_device = Box::into_raw(Box::new(LgpuDeviceWrapper { device }));
            *out_queue = Box::into_raw(Box::new(LgpuQueueWrapper { queue }));
            clear_last_error();
            LgpuResult::Success
        }
        Err(e) => {
            set_last_error(format!("Failed to create device: {e}"));
            LgpuResult::ErrorDeviceCreation
        }
    }
}

/// Creates a new GPU buffer with the given size, usage flags, and memory location.
///
/// # Safety
///
/// `device` must be a valid pointer to `LgpuDeviceWrapper`.
/// `out_buffer` must be a valid, aligned, writable pointer to `*mut LgpuBufferWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_buffer(
    device: *mut LgpuDeviceWrapper,
    size: u64,
    usage: u32,
    location: u32,
    out_buffer: *mut *mut LgpuBufferWrapper,
) -> LgpuResult {
    if device.is_null() || out_buffer.is_null() {
        set_last_error("Null pointer passed to lgpu_create_buffer");
        return LgpuResult::ErrorGeneric;
    }

    let dev = &*device;
    let memory_location = match location {
        1 => MemoryLocation::CpuToGpu,
        2 => MemoryLocation::GpuToCpu,
        _ => MemoryLocation::GpuOnly,
    };

    let desc = BufferDesc {
        label: Some("C_Buffer"),
        size,
        usage: BufferUsage::from_bits_truncate(usage),
        memory_location,
    };

    match dev.device.create_buffer(&desc) {
        Ok(buffer) => {
            *out_buffer = Box::into_raw(Box::new(LgpuBufferWrapper { buffer }));
            clear_last_error();
            LgpuResult::Success
        }
        Err(e) => {
            set_last_error(format!("create_buffer failed: {e}"));
            LgpuResult::ErrorBuffer
        }
    }
}

/// Creates a GPU buffer initialized with initial data.
///
/// # Safety
///
/// `device` must be a valid pointer to `LgpuDeviceWrapper`.
/// `data` must point to at least `size` readable bytes.
/// `out_buffer` must be a valid, aligned, writable pointer to `*mut LgpuBufferWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_buffer_init(
    device: *mut LgpuDeviceWrapper,
    data: *const c_void,
    size: usize,
    usage: u32,
    location: u32,
    out_buffer: *mut *mut LgpuBufferWrapper,
) -> LgpuResult {
    if device.is_null() || data.is_null() || out_buffer.is_null() {
        return LgpuResult::ErrorGeneric;
    }

    let dev = &*device;
    let memory_location = match location {
        1 => MemoryLocation::CpuToGpu,
        2 => MemoryLocation::GpuToCpu,
        _ => MemoryLocation::GpuOnly,
    };

    let desc = BufferDesc {
        label: Some("C_BufferInit"),
        size: size as u64,
        usage: BufferUsage::from_bits_truncate(usage),
        memory_location,
    };

    let slice = std::slice::from_raw_parts(data as *const u8, size);
    match dev.device.create_buffer_init(&desc, slice) {
        Ok(buffer) => {
            *out_buffer = Box::into_raw(Box::new(LgpuBufferWrapper { buffer }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorBuffer,
    }
}

/// Returns the bindless resource index for the buffer, or -1 if not registered.
///
/// # Safety
///
/// `buffer` must be a valid pointer to `LgpuBufferWrapper`, or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_buffer_get_bindless_id(buffer: *mut LgpuBufferWrapper) -> i32 {
    if buffer.is_null() {
        return -1;
    }
    let buf = &*buffer;
    match buf.buffer.id() {
        Some(id) => id.0 as i32,
        None => -1,
    }
}

/// Writes data into a CPU-accessible GPU buffer at the given offset.
///
/// # Safety
///
/// `buffer` must be a valid pointer to `LgpuBufferWrapper`.
/// `data` must point to at least `size` readable bytes.
/// Caller must ensure no concurrent host writes or in-flight GPU accesses violate memory safety.
#[no_mangle]
pub unsafe extern "C" fn lgpu_write_buffer(
    buffer: *mut LgpuBufferWrapper,
    offset: u64,
    data: *const c_void,
    size: usize,
) -> LgpuResult {
    if buffer.is_null() || data.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let buf = &*buffer;
    let slice = std::slice::from_raw_parts(data as *const u8, size);
    match buf.buffer.write_at(offset, slice) {
        Ok(_) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorBuffer,
    }
}

/// Reads data from a CPU-accessible GPU buffer at the given offset.
///
/// # Safety
///
/// `buffer` must be a valid pointer to `LgpuBufferWrapper`.
/// `data` must point to at least `size` writable bytes.
/// Caller must ensure GPU operations writing to this buffer have completed.
#[no_mangle]
pub unsafe extern "C" fn lgpu_read_buffer(
    buffer: *mut LgpuBufferWrapper,
    offset: u64,
    data: *mut c_void,
    size: usize,
) -> LgpuResult {
    if buffer.is_null() || data.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let buf = &*buffer;
    let slice = std::slice::from_raw_parts_mut(data as *mut u8, size);
    match buf.buffer.read_at(offset, slice) {
        Ok(_) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorBuffer,
    }
}

/// Creates a shader module from SPIR-V 32-bit words.
///
/// # Safety
///
/// `device` must be a valid pointer to `LgpuDeviceWrapper`.
/// `spirv_words` must point to at least `word_count` valid, readable `u32` words.
/// `out_shader` must be a valid, aligned, writable pointer to `*mut LgpuShaderWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_shader(
    device: *mut LgpuDeviceWrapper,
    spirv_words: *const u32,
    word_count: usize,
    out_shader: *mut *mut LgpuShaderWrapper,
) -> LgpuResult {
    if device.is_null() || spirv_words.is_null() || out_shader.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;
    let words = std::slice::from_raw_parts(spirv_words, word_count);
    match dev.device.create_shader_spirv(words) {
        Ok(shader) => {
            *out_shader = Box::into_raw(Box::new(LgpuShaderWrapper { shader }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorShader,
    }
}

/// Creates a compute pipeline with bound buffer count.
///
/// # Safety
///
/// `device` and `shader` must be valid non-null pointers.
/// `out_pipeline` must be a valid, aligned, writable pointer to `*mut LgpuComputePipelineWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_compute_pipeline(
    device: *mut LgpuDeviceWrapper,
    shader: *mut LgpuShaderWrapper,
    buffer_count: u32,
    out_pipeline: *mut *mut LgpuComputePipelineWrapper,
) -> LgpuResult {
    if device.is_null() || shader.is_null() || out_pipeline.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;
    let sh = &*shader;
    match dev.device.create_compute_pipeline(&sh.shader, buffer_count) {
        Ok(pipeline) => {
            *out_pipeline = Box::into_raw(Box::new(LgpuComputePipelineWrapper { pipeline }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorPipeline,
    }
}

/// Creates a command encoder for recording GPU commands.
///
/// # Safety
///
/// `device` must be a valid pointer to `LgpuDeviceWrapper`.
/// `out_encoder` must be a valid, aligned, writable pointer to `*mut LgpuCommandEncoderWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_command_encoder(
    device: *mut LgpuDeviceWrapper,
    out_encoder: *mut *mut LgpuCommandEncoderWrapper,
) -> LgpuResult {
    if device.is_null() || out_encoder.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;
    match dev.device.create_command_encoder() {
        Ok(encoder) => {
            *out_encoder = Box::into_raw(Box::new(LgpuCommandEncoderWrapper {
                encoder: Some(encoder),
            }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorCommand,
    }
}

/// Dispatches compute workgroups on a command encoder.
///
/// # Safety
///
/// `encoder` and `pipeline` must be valid non-null pointers.
/// If `buffer_count > 0`, `buffers` must point to at least `buffer_count` valid `LgpuBufferWrapper` pointers.
/// `encoder` must not be accessed concurrently on other threads.
#[no_mangle]
pub unsafe extern "C" fn lgpu_dispatch_compute(
    encoder: *mut LgpuCommandEncoderWrapper,
    pipeline: *mut LgpuComputePipelineWrapper,
    buffers: *const *mut LgpuBufferWrapper,
    buffer_count: u32,
    x: u32,
    y: u32,
    z: u32,
) {
    if encoder.is_null() || pipeline.is_null() || buffers.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let pip = &*pipeline;
    let raw_buf_ptrs = std::slice::from_raw_parts(buffers, buffer_count as usize);
    let mut buf_refs = Vec::with_capacity(buffer_count as usize);
    for &ptr in raw_buf_ptrs {
        if !ptr.is_null() {
            buf_refs.push(&(&*ptr).buffer);
        }
    }

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.dispatch_compute(&pip.pipeline, &buf_refs, x, y, z);
    }
}

/// Finishes recording commands and returns an executable command buffer.
///
/// # Safety
///
/// `encoder` must be a valid pointer to `LgpuCommandEncoderWrapper`.
/// The encoder should be freed with `lgpu_destroy_command_encoder` when no longer needed.
#[no_mangle]
pub unsafe extern "C" fn lgpu_finish_encoder(
    encoder: *mut LgpuCommandEncoderWrapper,
) -> *mut LgpuCommandBufferWrapper {
    if encoder.is_null() {
        return ptr::null_mut();
    }
    let enc_wrapper = &mut *encoder;
    if let Some(enc) = enc_wrapper.encoder.take() {
        let cmd = enc.finish();
        Box::into_raw(Box::new(LgpuCommandBufferWrapper {
            command_buffer: Some(cmd),
        }))
    } else {
        ptr::null_mut()
    }
}

/// Submits a recorded command buffer to the GPU queue.
///
/// # Safety
///
/// `queue` and `command_buffer` must be valid non-null pointers.
/// The `command_buffer` wrapper must be cleaned up with `lgpu_destroy_command_buffer` when done.
#[no_mangle]
pub unsafe extern "C" fn lgpu_queue_submit(
    queue: *mut LgpuQueueWrapper,
    command_buffer: *mut LgpuCommandBufferWrapper,
) -> LgpuResult {
    if queue.is_null() || command_buffer.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let q = &*queue;
    let cmd_wrapper = &mut *command_buffer;
    if let Some(cmd) = cmd_wrapper.command_buffer.take() {
        match q.queue.submit(cmd) {
            Ok(_) => LgpuResult::Success,
            Err(_) => LgpuResult::ErrorCommand,
        }
    } else {
        LgpuResult::ErrorGeneric
    }
}

/// Submits multiple command buffers to the GPU queue in batch.
///
/// # Safety
///
/// `queue` must be a valid non-null pointer.
/// `command_buffers` must point to at least `count` valid pointers to `LgpuCommandBufferWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_queue_submit_batches(
    queue: *mut LgpuQueueWrapper,
    command_buffers: *const *mut LgpuCommandBufferWrapper,
    count: u32,
) -> LgpuResult {
    if queue.is_null() || (count > 0 && command_buffers.is_null()) {
        return LgpuResult::ErrorGeneric;
    }
    let q = &*queue;
    let cmd_ptrs = std::slice::from_raw_parts(command_buffers, count as usize);
    let mut cmds = Vec::with_capacity(count as usize);
    for &ptr in cmd_ptrs {
        if ptr.is_null() {
            return LgpuResult::ErrorGeneric;
        }
        let wrapper = &mut *ptr;
        if let Some(cmd) = wrapper.command_buffer.take() {
            cmds.push(cmd);
        } else {
            return LgpuResult::ErrorGeneric;
        }
    }

    match q.queue.submit_batches(cmds) {
        Ok(_) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorCommand,
    }
}

/// Submits command buffers asynchronously with timeline semaphore waits and signals.
///
/// # Safety
///
/// `queue` must be a valid pointer.
/// If `count > 0`, `command_buffers` must point to `count` valid pointers.
/// If `wait_count > 0`, `wait_semaphores` and `wait_values` must point to `wait_count` valid elements.
/// If `signal_count > 0`, `signal_semaphores` and `signal_values` must point to `signal_count` valid elements.
#[no_mangle]
pub unsafe extern "C" fn lgpu_queue_submit_async(
    queue: *mut LgpuQueueWrapper,
    command_buffers: *const *mut LgpuCommandBufferWrapper,
    count: u32,
    wait_semaphores: *const *mut LgpuTimelineSemaphoreWrapper,
    wait_values: *const u64,
    wait_count: u32,
    signal_semaphores: *const *mut LgpuTimelineSemaphoreWrapper,
    signal_values: *const u64,
    signal_count: u32,
) -> LgpuResult {
    if queue.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let q = &*queue;

    let mut cmds = Vec::with_capacity(count as usize);
    if count > 0 {
        if command_buffers.is_null() {
            return LgpuResult::ErrorGeneric;
        }
        let cmd_ptrs = std::slice::from_raw_parts(command_buffers, count as usize);
        for &ptr in cmd_ptrs {
            if ptr.is_null() {
                return LgpuResult::ErrorGeneric;
            }
            let wrapper = &mut *ptr;
            if let Some(cmd) = wrapper.command_buffer.take() {
                cmds.push(cmd);
            } else {
                return LgpuResult::ErrorGeneric;
            }
        }
    }

    let mut wait_tuples = Vec::with_capacity(wait_count as usize);
    if wait_count > 0 {
        if wait_semaphores.is_null() || wait_values.is_null() {
            return LgpuResult::ErrorGeneric;
        }
        let sem_ptrs = std::slice::from_raw_parts(wait_semaphores, wait_count as usize);
        let vals = std::slice::from_raw_parts(wait_values, wait_count as usize);
        for i in 0..wait_count as usize {
            if sem_ptrs[i].is_null() {
                return LgpuResult::ErrorGeneric;
            }
            let sem_wrapper = &*sem_ptrs[i];
            wait_tuples.push((&sem_wrapper.semaphore, vals[i]));
        }
    }

    let mut signal_tuples = Vec::with_capacity(signal_count as usize);
    if signal_count > 0 {
        if signal_semaphores.is_null() || signal_values.is_null() {
            return LgpuResult::ErrorGeneric;
        }
        let sem_ptrs = std::slice::from_raw_parts(signal_semaphores, signal_count as usize);
        let vals = std::slice::from_raw_parts(signal_values, signal_count as usize);
        for i in 0..signal_count as usize {
            if sem_ptrs[i].is_null() {
                return LgpuResult::ErrorGeneric;
            }
            let sem_wrapper = &*sem_ptrs[i];
            signal_tuples.push((&sem_wrapper.semaphore, vals[i]));
        }
    }

    match q
        .queue
        .submit_batches_async(cmds, &wait_tuples, &signal_tuples)
    {
        Ok(_) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorCommand,
    }
}

/// Creates a timeline semaphore with an initial counter value.
///
/// # Safety
///
/// `device` must be a valid non-null pointer.
/// `out_semaphore` must be a valid, aligned, writable pointer to `*mut LgpuTimelineSemaphoreWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_timeline_semaphore(
    device: *mut LgpuDeviceWrapper,
    initial_value: u64,
    out_semaphore: *mut *mut LgpuTimelineSemaphoreWrapper,
) -> LgpuResult {
    if device.is_null() || out_semaphore.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;
    match dev.device.create_timeline_semaphore(initial_value) {
        Ok(semaphore) => {
            *out_semaphore = Box::into_raw(Box::new(LgpuTimelineSemaphoreWrapper { semaphore }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Queries the current counter value of a timeline semaphore.
///
/// # Safety
///
/// `semaphore` must be a valid non-null pointer.
/// `out_value` must be a valid, aligned, writable pointer to `u64`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_get_timeline_semaphore_value(
    semaphore: *mut LgpuTimelineSemaphoreWrapper,
    out_value: *mut u64,
) -> LgpuResult {
    if semaphore.is_null() || out_value.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let sem = &*semaphore;
    match sem.semaphore.value() {
        Ok(val) => {
            *out_value = val;
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorCommand,
    }
}

/// Blocks the CPU host until the timeline semaphore reaches or exceeds `value` or times out.
///
/// # Safety
///
/// `semaphore` must be a valid non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn lgpu_wait_timeline_semaphore(
    semaphore: *mut LgpuTimelineSemaphoreWrapper,
    value: u64,
    timeout_nanos: u64,
) -> LgpuResult {
    if semaphore.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let sem = &*semaphore;
    match sem.semaphore.wait(value, timeout_nanos) {
        Ok(_) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorCommand,
    }
}

/// Signals the timeline semaphore to advance to `value` from the host CPU.
///
/// # Safety
///
/// `semaphore` must be a valid non-null pointer. `value` must be strictly greater than any value previously signaled.
#[no_mangle]
pub unsafe extern "C" fn lgpu_signal_timeline_semaphore(
    semaphore: *mut LgpuTimelineSemaphoreWrapper,
    value: u64,
) -> LgpuResult {
    if semaphore.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let sem = &*semaphore;
    match sem.semaphore.signal(value) {
        Ok(_) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorCommand,
    }
}

/// Inserts a global execution and memory barrier into the command stream.
///
/// # Safety
///
/// `encoder` must be a valid non-null pointer and not accessed concurrently.
#[no_mangle]
pub unsafe extern "C" fn lgpu_encoder_memory_barrier(encoder: *mut LgpuCommandEncoderWrapper) {
    if encoder.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.memory_barrier();
    }
}

/// Inserts an execution and memory barrier for a specific buffer.
///
/// # Safety
///
/// `encoder` and `buffer` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_encoder_buffer_barrier(
    encoder: *mut LgpuCommandEncoderWrapper,
    buffer: *mut LgpuBufferWrapper,
) {
    if encoder.is_null() || buffer.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let buf_wrapper = &*buffer;
    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.buffer_barrier(&buf_wrapper.buffer);
    }
}

/// Destroys a timeline semaphore.
///
/// # Safety
///
/// `semaphore` must be a valid pointer or null. It must not be referenced after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_timeline_semaphore(
    semaphore: *mut LgpuTimelineSemaphoreWrapper,
) {
    if !semaphore.is_null() {
        drop(Box::from_raw(semaphore));
    }
}

/// Creates a 2D or 3D GPU texture.
///
/// # Safety
///
/// `device` must be a valid non-null pointer.
/// `out_texture` must be a valid, aligned, writable pointer to `*mut LgpuTextureWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_texture(
    device: *mut LgpuDeviceWrapper,
    width: u32,
    height: u32,
    depth: u32,
    format: i32,
    usage: u32,
    out_texture: *mut *mut LgpuTextureWrapper,
) -> LgpuResult {
    if device.is_null() || out_texture.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;
    let tex_format = match format {
        0 => TextureFormat::Rgba8Unorm,
        1 => TextureFormat::Rgba8Srgb,
        2 => TextureFormat::Bgra8Unorm,
        3 => TextureFormat::Bgra8Srgb,
        4 => TextureFormat::R32Float,
        5 => TextureFormat::Rgba16Float,
        6 => TextureFormat::Rgba32Float,
        7 => TextureFormat::Depth32Float,
        _ => return LgpuResult::ErrorGeneric,
    };

    let desc = TextureDesc {
        label: Some("C_Texture"),
        width,
        height,
        depth: depth.max(1),
        mip_levels: 1,
        sample_count: 1,
        format: tex_format,
        usage: TextureUsage::from_bits_truncate(usage),
    };

    match dev.device.create_texture(&desc) {
        Ok(texture) => {
            *out_texture = Box::into_raw(Box::new(LgpuTextureWrapper { texture }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Returns the bindless resource index for the texture, or -1 if not registered.
///
/// # Safety
///
/// `texture` must be a valid pointer to `LgpuTextureWrapper`, or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_texture_get_bindless_id(texture: *mut LgpuTextureWrapper) -> i32 {
    if texture.is_null() {
        return -1;
    }
    let tex = &*texture;
    match tex.texture.id() {
        Some(id) => id.0 as i32,
        None => -1,
    }
}

/// Destroys a GPU texture.
///
/// # Safety
///
/// `texture` must be a valid pointer or null. It must not be in-flight or used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_texture(texture: *mut LgpuTextureWrapper) {
    if !texture.is_null() {
        drop(Box::from_raw(texture));
    }
}

/// Creates a texture sampler.
///
/// # Safety
///
/// `device` must be a valid non-null pointer.
/// `out_sampler` must be a valid, aligned, writable pointer to `*mut LgpuSamplerWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_sampler(
    device: *mut LgpuDeviceWrapper,
    min_filter: i32,
    mag_filter: i32,
    mipmap_filter: i32,
    address_u: i32,
    address_v: i32,
    address_w: i32,
    out_sampler: *mut *mut LgpuSamplerWrapper,
) -> LgpuResult {
    if device.is_null() || out_sampler.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;

    let to_filter = |v| match v {
        0 => FilterMode::Nearest,
        _ => FilterMode::Linear,
    };
    let to_addr = |v| match v {
        1 => AddressMode::Repeat,
        2 => AddressMode::MirrorRepeat,
        _ => AddressMode::ClampToEdge,
    };

    let desc = SamplerDesc {
        label: Some("C_Sampler"),
        min_filter: to_filter(min_filter),
        mag_filter: to_filter(mag_filter),
        mipmap_filter: to_filter(mipmap_filter),
        address_mode_u: to_addr(address_u),
        address_mode_v: to_addr(address_v),
        address_mode_w: to_addr(address_w),
    };

    match dev.device.create_sampler(&desc) {
        Ok(sampler) => {
            *out_sampler = Box::into_raw(Box::new(LgpuSamplerWrapper { sampler }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Returns the bindless resource index for the sampler, or -1 if not registered.
///
/// # Safety
///
/// `sampler` must be a valid pointer to `LgpuSamplerWrapper`, or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_sampler_get_bindless_id(sampler: *mut LgpuSamplerWrapper) -> i32 {
    if sampler.is_null() {
        return -1;
    }
    let s = &*sampler;
    match s.sampler.id() {
        Some(id) => id.0 as i32,
        None => -1,
    }
}

/// Destroys a texture sampler.
///
/// # Safety
///
/// `sampler` must be a valid pointer or null. It must not be used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_sampler(sampler: *mut LgpuSamplerWrapper) {
    if !sampler.is_null() {
        drop(Box::from_raw(sampler));
    }
}

/// Creates a graphics pipeline with optional depth testing and writing.
///
/// # Safety
///
/// `device` and `shader` must be valid non-null pointers.
/// `vs_entry` and `fs_entry` must be valid null-terminated C string pointers.
/// `out_pipeline` must be a valid, aligned, writable pointer to `*mut LgpuGraphicsPipelineWrapper`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_graphics_pipeline(
    device: *mut LgpuDeviceWrapper,
    shader: *mut LgpuShaderWrapper,
    vs_entry: *const std::ffi::c_char,
    fs_entry: *const std::ffi::c_char,
    color_format: i32,
    depth_format: i32,
    depth_test: bool,
    depth_write: bool,
    out_pipeline: *mut *mut LgpuGraphicsPipelineWrapper,
) -> LgpuResult {
    if device.is_null()
        || shader.is_null()
        || vs_entry.is_null()
        || fs_entry.is_null()
        || out_pipeline.is_null()
    {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;
    let sh = &*shader;

    let vs_str = match std::ffi::CStr::from_ptr(vs_entry).to_str() {
        Ok(s) => s,
        Err(_) => return LgpuResult::ErrorGeneric,
    };
    let fs_str = match std::ffi::CStr::from_ptr(fs_entry).to_str() {
        Ok(s) => s,
        Err(_) => return LgpuResult::ErrorGeneric,
    };

    let col_fmt = match color_format {
        0 => TextureFormat::Rgba8Unorm,
        1 => TextureFormat::Rgba8Srgb,
        2 => TextureFormat::Bgra8Unorm,
        3 => TextureFormat::Bgra8Srgb,
        4 => TextureFormat::R32Float,
        5 => TextureFormat::Rgba16Float,
        6 => TextureFormat::Rgba32Float,
        7 => TextureFormat::Depth32Float,
        _ => return LgpuResult::ErrorGeneric,
    };

    let dep_fmt = match depth_format {
        7 => Some(TextureFormat::Depth32Float),
        -1 => None,
        _ => return LgpuResult::ErrorGeneric,
    };

    match dev.device.create_graphics_pipeline_with_depth(
        &sh.shader,
        vs_str,
        fs_str,
        col_fmt,
        dep_fmt,
        depth_test,
        depth_write,
    ) {
        Ok(pipeline) => {
            *out_pipeline = Box::into_raw(Box::new(LgpuGraphicsPipelineWrapper { pipeline }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorPipeline,
    }
}

/// Destroys a graphics pipeline.
///
/// # Safety
///
/// `pipeline` must be a valid pointer or null. It must not be used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_graphics_pipeline(
    pipeline: *mut LgpuGraphicsPipelineWrapper,
) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

/// Clears color and depth attachments and begins a rendering pass.
///
/// # Safety
///
/// `encoder`, `color_texture`, `depth_texture`, and `clear_color` must be valid non-null pointers.
/// `color_texture` and `depth_texture` must point to distinct textures to prevent aliasing.
/// `clear_color` must point to at least 4 valid `f32` components [r, g, b, a].
#[no_mangle]
pub unsafe extern "C" fn lgpu_clear_and_render_with_depth(
    encoder: *mut LgpuCommandEncoderWrapper,
    color_texture: *mut LgpuTextureWrapper,
    depth_texture: *mut LgpuTextureWrapper,
    clear_color: *const f32,
    clear_depth: f32,
) {
    if encoder.is_null()
        || color_texture.is_null()
        || depth_texture.is_null()
        || clear_color.is_null()
    {
        return;
    }
    // Prevent attachment aliasing
    if color_texture == depth_texture {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let col = &*color_texture;
    let dep = &*depth_texture;
    let color_slice = std::slice::from_raw_parts(clear_color, 4);
    let color_arr = [
        color_slice[0],
        color_slice[1],
        color_slice[2],
        color_slice[3],
    ];

    if let Some(enc) = &mut enc_wrapper.encoder {
        let _ = enc.clear_and_render_with_depth(&col.texture, &dep.texture, color_arr, clear_depth);
    }
}

/// Clears attachments, binds pipeline, and issues a draw call with depth.
///
/// # Safety
///
/// `encoder`, `color_texture`, `depth_texture`, `pipeline`, and `clear_color` must be valid non-null pointers.
/// `color_texture` and `depth_texture` must point to distinct textures to prevent aliasing.
/// `clear_color` must point to at least 4 valid `f32` components.
/// If `push_constants` is non-null and `push_constants_size > 0`, it must point to readable bytes.
#[no_mangle]
pub unsafe extern "C" fn lgpu_draw_with_depth(
    encoder: *mut LgpuCommandEncoderWrapper,
    color_texture: *mut LgpuTextureWrapper,
    depth_texture: *mut LgpuTextureWrapper,
    pipeline: *mut LgpuGraphicsPipelineWrapper,
    vertex_count: u32,
    push_constants: *const std::ffi::c_void,
    push_constants_size: usize,
    clear_color: *const f32,
    clear_depth: f32,
) {
    if encoder.is_null()
        || color_texture.is_null()
        || depth_texture.is_null()
        || pipeline.is_null()
        || clear_color.is_null()
    {
        return;
    }
    // Prevent attachment aliasing
    if color_texture == depth_texture {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let col = &*color_texture;
    let dep = &*depth_texture;
    let pip = &*pipeline;
    let color_slice = std::slice::from_raw_parts(clear_color, 4);
    let color_arr = [
        color_slice[0],
        color_slice[1],
        color_slice[2],
        color_slice[3],
    ];

    let pc_slice = if !push_constants.is_null() && push_constants_size > 0 {
        std::slice::from_raw_parts(push_constants as *const u8, push_constants_size)
    } else {
        &[]
    };

    if let Some(enc) = &mut enc_wrapper.encoder {
        let _ = enc.draw_with_depth(
            &col.texture,
            &dep.texture,
            &pip.pipeline,
            vertex_count,
            pc_slice,
            color_arr,
            clear_depth,
        );
    }
}

/// Clears a color texture.
///
/// # Safety
///
/// `encoder`, `texture`, and `clear_color` must be valid non-null pointers.
/// `clear_color` must point to at least 4 valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn lgpu_clear_and_render_texture(
    encoder: *mut LgpuCommandEncoderWrapper,
    texture: *mut LgpuTextureWrapper,
    clear_color: *const f32,
) {
    if encoder.is_null() || texture.is_null() || clear_color.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let tex = &*texture;
    let color_slice = std::slice::from_raw_parts(clear_color, 4);
    let color_arr = [
        color_slice[0],
        color_slice[1],
        color_slice[2],
        color_slice[3],
    ];

    if let Some(enc) = &mut enc_wrapper.encoder {
        let _ = enc.clear_and_render_texture(&tex.texture, color_arr);
    }
}

/// Clears color attachment, binds pipeline, and issues a draw call.
///
/// # Safety
///
/// `encoder`, `texture`, `pipeline`, and `clear_color` must be valid non-null pointers.
/// `clear_color` must point to at least 4 valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn lgpu_draw(
    encoder: *mut LgpuCommandEncoderWrapper,
    texture: *mut LgpuTextureWrapper,
    pipeline: *mut LgpuGraphicsPipelineWrapper,
    vertex_count: u32,
    push_constants: *const c_void,
    push_constants_size: usize,
    clear_color: *const f32,
) {
    if encoder.is_null() || texture.is_null() || pipeline.is_null() || clear_color.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let tex = &*texture;
    let pip = &*pipeline;
    let color_slice = std::slice::from_raw_parts(clear_color, 4);
    let color_arr = [
        color_slice[0],
        color_slice[1],
        color_slice[2],
        color_slice[3],
    ];

    let pc_slice = if !push_constants.is_null() && push_constants_size > 0 {
        std::slice::from_raw_parts(push_constants as *const u8, push_constants_size)
    } else {
        &[]
    };

    if let Some(enc) = &mut enc_wrapper.encoder {
        let _ = enc.draw(
            &tex.texture,
            &pip.pipeline,
            vertex_count,
            pc_slice,
            color_arr,
        );
    }
}

/// Renders a test triangle to the given texture.
///
/// # Safety
///
/// `encoder`, `texture`, `pipeline`, and `clear_color` must be valid non-null pointers.
/// `clear_color` must point to at least 4 valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn lgpu_render_triangle(
    encoder: *mut LgpuCommandEncoderWrapper,
    texture: *mut LgpuTextureWrapper,
    pipeline: *mut LgpuGraphicsPipelineWrapper,
    clear_color: *const f32,
) {
    if encoder.is_null() || texture.is_null() || pipeline.is_null() || clear_color.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let tex = &*texture;
    let pip = &*pipeline;
    let color_slice = std::slice::from_raw_parts(clear_color, 4);
    let color_arr = [
        color_slice[0],
        color_slice[1],
        color_slice[2],
        color_slice[3],
    ];

    if let Some(enc) = &mut enc_wrapper.encoder {
        let _ = enc.render_triangle(&tex.texture, &pip.pipeline, color_arr);
    }
}

/// Copies texture image data into a destination buffer.
///
/// # Safety
///
/// `encoder`, `texture`, and `buffer` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_copy_texture_to_buffer(
    encoder: *mut LgpuCommandEncoderWrapper,
    texture: *mut LgpuTextureWrapper,
    buffer: *mut LgpuBufferWrapper,
) {
    if encoder.is_null() || texture.is_null() || buffer.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let tex = &*texture;
    let buf = &*buffer;

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.copy_texture_to_buffer(&tex.texture, &buf.buffer);
    }
}

/// Copies data from one buffer into another buffer on the GPU.
///
/// # Safety
///
/// `encoder`, `source`, and `destination` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_copy_buffer_to_buffer(
    encoder: *mut LgpuCommandEncoderWrapper,
    source: *mut LgpuBufferWrapper,
    source_offset: u64,
    destination: *mut LgpuBufferWrapper,
    destination_offset: u64,
    size: u64,
) {
    if encoder.is_null() || source.is_null() || destination.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let src = &*source;
    let dst = &*destination;

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.copy_buffer_to_buffer(
            &src.buffer,
            source_offset,
            &dst.buffer,
            destination_offset,
            size,
        );
    }
}

/// Copies pixel data from a buffer to a texture.
///
/// # Safety
///
/// `encoder`, `buffer`, and `texture` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_copy_buffer_to_texture(
    encoder: *mut LgpuCommandEncoderWrapper,
    buffer: *mut LgpuBufferWrapper,
    texture: *mut LgpuTextureWrapper,
) {
    if encoder.is_null() || buffer.is_null() || texture.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let buf = &*buffer;
    let tex = &*texture;

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.copy_buffer_to_texture(&buf.buffer, &tex.texture);
    }
}

/// Dispatches a bindless compute pipeline using indirect dispatch parameters from a GPU buffer.
///
/// # Safety
///
/// `encoder`, `pipeline`, and `indirect_buffer` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_dispatch_indirect(
    encoder: *mut LgpuCommandEncoderWrapper,
    pipeline: *mut LgpuBindlessComputePipelineWrapper,
    indirect_buffer: *mut LgpuBufferWrapper,
    indirect_offset: u64,
    push_constants: *const c_void,
    push_constants_size: usize,
) -> LgpuResult {
    if encoder.is_null() || pipeline.is_null() || indirect_buffer.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let enc_wrapper = &mut *encoder;
    let pip = &*pipeline;
    let buf = &*indirect_buffer;

    let pc_slice = if !push_constants.is_null() && push_constants_size > 0 {
        std::slice::from_raw_parts(push_constants as *const u8, push_constants_size)
    } else {
        &[]
    };

    if let Some(enc) = &mut enc_wrapper.encoder {
        match enc.dispatch_indirect(&pip.pipeline, &buf.buffer, indirect_offset, pc_slice) {
            Ok(()) => LgpuResult::Success,
            Err(_) => LgpuResult::ErrorGeneric,
        }
    } else {
        LgpuResult::ErrorGeneric
    }
}

/// Draws indexed geometry to the given render target texture.
///
/// # Safety
///
/// `encoder`, `texture`, `pipeline`, `index_buffer`, and `clear_color` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_draw_indexed(
    encoder: *mut LgpuCommandEncoderWrapper,
    texture: *mut LgpuTextureWrapper,
    pipeline: *mut LgpuGraphicsPipelineWrapper,
    index_buffer: *mut LgpuBufferWrapper,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
    push_constants: *const c_void,
    push_constants_size: usize,
    clear_color: *const f32,
) {
    if encoder.is_null()
        || texture.is_null()
        || pipeline.is_null()
        || index_buffer.is_null()
        || clear_color.is_null()
    {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let tex = &*texture;
    let pip = &*pipeline;
    let idx_buf = &*index_buffer;
    let color_slice = std::slice::from_raw_parts(clear_color, 4);
    let color_arr = [
        color_slice[0],
        color_slice[1],
        color_slice[2],
        color_slice[3],
    ];

    let pc_slice = if !push_constants.is_null() && push_constants_size > 0 {
        std::slice::from_raw_parts(push_constants as *const u8, push_constants_size)
    } else {
        &[]
    };

    if let Some(enc) = &mut enc_wrapper.encoder {
        let _ = enc.draw_indexed(
            &tex.texture,
            &pip.pipeline,
            &idx_buf.buffer,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
            pc_slice,
            color_arr,
        );
    }
}

/// Directly uploads CPU memory to a GPU texture.
///
/// # Safety
///
/// `device`, `texture`, and `data` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_device_write_texture(
    device: *mut LgpuDeviceWrapper,
    texture: *mut LgpuTextureWrapper,
    data: *const u8,
    data_len: usize,
) -> LgpuResult {
    if device.is_null() || texture.is_null() || data.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;
    let tex = &*texture;
    let byte_slice = std::slice::from_raw_parts(data, data_len);

    match dev.device.write_texture(&tex.texture, byte_slice) {
        Ok(()) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Returns the width of the texture in pixels.
///
/// # Safety
///
/// `texture` must be a valid pointer to `LgpuTextureWrapper`, or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_texture_get_width(texture: *mut LgpuTextureWrapper) -> u32 {
    if texture.is_null() {
        return 0;
    }
    let tex = &*texture;
    tex.texture.width
}

/// Returns the height of the texture in pixels.
///
/// # Safety
///
/// `texture` must be a valid pointer to `LgpuTextureWrapper`, or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_texture_get_height(texture: *mut LgpuTextureWrapper) -> u32 {
    if texture.is_null() {
        return 0;
    }
    let tex = &*texture;
    tex.texture.height
}

/// Returns the integer code representing the texture format.
///
/// # Safety
///
/// `texture` must be a valid pointer to `LgpuTextureWrapper`, or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_texture_get_format(texture: *mut LgpuTextureWrapper) -> i32 {
    if texture.is_null() {
        return -1;
    }
    let tex = &*texture;
    match tex.texture.format {
        TextureFormat::Rgba8Unorm => 0,
        TextureFormat::Rgba8Srgb => 1,
        TextureFormat::Bgra8Unorm => 2,
        TextureFormat::Bgra8Srgb => 3,
        TextureFormat::R32Float => 4,
        TextureFormat::Rgba16Float => 5,
        TextureFormat::Rgba32Float => 6,
        TextureFormat::Depth32Float => 7,
    }
}

/// Creates a bindless compute pipeline.
///
/// # Safety
///
/// `device`, `shader`, and `out_pipeline` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_bindless_compute_pipeline(
    device: *mut LgpuDeviceWrapper,
    shader: *mut LgpuShaderWrapper,
    out_pipeline: *mut *mut LgpuBindlessComputePipelineWrapper,
) -> LgpuResult {
    if device.is_null() || shader.is_null() || out_pipeline.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let dev = &*device;
    let sh = &*shader;

    match dev.device.create_bindless_compute_pipeline(&sh.shader) {
        Ok(pipeline) => {
            *out_pipeline =
                Box::into_raw(Box::new(LgpuBindlessComputePipelineWrapper { pipeline }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorPipeline,
    }
}

/// Dispatches compute work on a bindless pipeline with push constants.
///
/// # Safety
///
/// `encoder` and `pipeline` must be valid non-null pointers.
/// If `push_constants` is non-null and `push_constants_size > 0`, it must point to readable bytes.
#[no_mangle]
pub unsafe extern "C" fn lgpu_dispatch_bindless_compute(
    encoder: *mut LgpuCommandEncoderWrapper,
    pipeline: *mut LgpuBindlessComputePipelineWrapper,
    push_constants: *const c_void,
    push_constants_size: usize,
    x: u32,
    y: u32,
    z: u32,
) -> LgpuResult {
    if encoder.is_null() || pipeline.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let enc_wrapper = &mut *encoder;
    let pip = &*pipeline;

    let pc_slice = if !push_constants.is_null() && push_constants_size > 0 {
        std::slice::from_raw_parts(push_constants as *const u8, push_constants_size)
    } else {
        &[]
    };

    if let Some(enc) = &mut enc_wrapper.encoder {
        match enc.dispatch_bindless_compute(&pip.pipeline, pc_slice, x, y, z) {
            Ok(_) => LgpuResult::Success,
            Err(_) => LgpuResult::ErrorCommand,
        }
    } else {
        LgpuResult::ErrorCommand
    }
}

/// Destroys a bindless compute pipeline.
///
/// # Safety
///
/// `pipeline` must be a valid pointer or null. It must not be used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_bindless_compute_pipeline(
    pipeline: *mut LgpuBindlessComputePipelineWrapper,
) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

/// Destroys a GPU buffer.
///
/// # Safety
///
/// `buffer` must be a valid pointer or null. It must not be accessed or in-flight after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_buffer(buffer: *mut LgpuBufferWrapper) {
    if !buffer.is_null() {
        drop(Box::from_raw(buffer));
    }
}

/// Destroys a shader module.
///
/// # Safety
///
/// `shader` must be a valid pointer or null. It must not be used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_shader(shader: *mut LgpuShaderWrapper) {
    if !shader.is_null() {
        drop(Box::from_raw(shader));
    }
}

/// Destroys a compute pipeline.
///
/// # Safety
///
/// `pipeline` must be a valid pointer or null. It must not be used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_compute_pipeline(pipeline: *mut LgpuComputePipelineWrapper) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

/// Destroys a logical GPU device.
///
/// # Safety
///
/// `device` must be a valid pointer or null. It must not be used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_device(device: *mut LgpuDeviceWrapper) {
    if !device.is_null() {
        drop(Box::from_raw(device));
    }
}

/// Destroys a GPU queue handle wrapper.
///
/// # Safety
///
/// `queue` must be a valid pointer or null. It must not be used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_queue(queue: *mut LgpuQueueWrapper) {
    if !queue.is_null() {
        drop(Box::from_raw(queue));
    }
}

/// Destroys an lgpu Instance.
///
/// # Safety
///
/// `instance` must be a valid pointer or null. It must not be used after destruction.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_instance(instance: *mut LgpuInstanceWrapper) {
    if !instance.is_null() {
        drop(Box::from_raw(instance));
    }
}

/// Destroys a command encoder.
///
/// # Safety
///
/// `encoder` must be a valid pointer obtained from `lgpu_create_command_encoder`, or null.
/// Once destroyed, the pointer must not be dereferenced or reused.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_command_encoder(encoder: *mut LgpuCommandEncoderWrapper) {
    if !encoder.is_null() {
        drop(Box::from_raw(encoder));
    }
}

/// Destroys a command buffer.
///
/// # Safety
///
/// `command_buffer` must be a valid pointer obtained from `lgpu_finish_encoder`, or null.
/// Once destroyed, the pointer must not be dereferenced or reused.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_command_buffer(
    command_buffer: *mut LgpuCommandBufferWrapper,
) {
    if !command_buffer.is_null() {
        drop(Box::from_raw(command_buffer));
    }
}

/// Waits for all pending GPU work on `device` to complete.
///
/// # Safety
///
/// `device` must be a valid pointer obtained from `lgpu_create_device`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_device_wait_idle(device: *mut LgpuDeviceWrapper) -> LgpuResult {
    if device.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    match (*device).device.wait_idle() {
        Ok(()) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Reclaims completely empty memory blocks in the suballocator back to Vulkan.
///
/// # Safety
///
/// `device` must be a valid pointer obtained from `lgpu_create_device`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_device_trim_memory(device: *mut LgpuDeviceWrapper) -> u32 {
    if device.is_null() {
        return 0;
    }
    (*device).device.trim_memory() as u32
}

/// Returns the number of memory blocks currently held by the suballocator.
///
/// # Safety
///
/// `device` must be a valid pointer obtained from `lgpu_create_device`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_device_get_suballocator_block_count(
    device: *mut LgpuDeviceWrapper,
) -> u32 {
    if device.is_null() {
        return 0;
    }
    (*device).device.suballocator_block_count() as u32
}

/// Creates a headless surface for offscreen/test presentation.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `out_surface` must be a valid, aligned, non-null pointer to a writable pointer.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_surface_headless(
    instance: *mut LgpuInstanceWrapper,
    out_surface: *mut *mut LgpuSurfaceWrapper,
) -> LgpuResult {
    if instance.is_null() || out_surface.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    match (*instance).instance.create_headless_surface() {
        Ok(surface) => {
            let wrapper = Box::new(LgpuSurfaceWrapper { surface });
            *out_surface = Box::into_raw(wrapper);
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Creates a window surface from Xlib Display and Window.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `dpy` must be a valid Xlib `Display*` and `window` a valid Xlib `Window`.
/// `out_surface` must be a valid, aligned, non-null pointer to a writable pointer.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_surface_from_xlib(
    instance: *mut LgpuInstanceWrapper,
    dpy: *mut c_void,
    window: std::ffi::c_ulong,
    out_surface: *mut *mut LgpuSurfaceWrapper,
) -> LgpuResult {
    if instance.is_null() || out_surface.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let display_handle = raw_window_handle::RawDisplayHandle::Xlib(
        raw_window_handle::XlibDisplayHandle::new(NonNull::new(dpy), 0),
    );
    let window_handle =
        raw_window_handle::RawWindowHandle::Xlib(raw_window_handle::XlibWindowHandle::new(window));
    match (*instance)
        .instance
        .create_surface_raw(display_handle, window_handle)
    {
        Ok(surface) => {
            let wrapper = Box::new(LgpuSurfaceWrapper { surface });
            *out_surface = Box::into_raw(wrapper);
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Creates a window surface from XCB connection and window ID.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `connection` must be a valid XCB connection pointer.
/// `out_surface` must be a valid, aligned, non-null pointer to a writable pointer.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_surface_from_xcb(
    instance: *mut LgpuInstanceWrapper,
    connection: *mut c_void,
    window: u32,
    out_surface: *mut *mut LgpuSurfaceWrapper,
) -> LgpuResult {
    if instance.is_null() || out_surface.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let display_handle = raw_window_handle::RawDisplayHandle::Xcb(
        raw_window_handle::XcbDisplayHandle::new(NonNull::new(connection), 0),
    );
    let non_zero_win = match std::num::NonZeroU32::new(window) {
        Some(w) => w,
        None => return LgpuResult::ErrorGeneric,
    };
    let window_handle = raw_window_handle::RawWindowHandle::Xcb(
        raw_window_handle::XcbWindowHandle::new(non_zero_win),
    );
    match (*instance)
        .instance
        .create_surface_raw(display_handle, window_handle)
    {
        Ok(surface) => {
            let wrapper = Box::new(LgpuSurfaceWrapper { surface });
            *out_surface = Box::into_raw(wrapper);
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Creates a window surface from Wayland display and surface pointers.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `display` and `surface` must be non-null valid Wayland display and surface pointers.
/// `out_surface` must be a valid, aligned, non-null pointer to a writable pointer.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_surface_from_wayland(
    instance: *mut LgpuInstanceWrapper,
    display: *mut c_void,
    surface: *mut c_void,
    out_surface: *mut *mut LgpuSurfaceWrapper,
) -> LgpuResult {
    if instance.is_null() || display.is_null() || surface.is_null() || out_surface.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let display_handle = raw_window_handle::RawDisplayHandle::Wayland(
        raw_window_handle::WaylandDisplayHandle::new(NonNull::new_unchecked(display)),
    );
    let window_handle = raw_window_handle::RawWindowHandle::Wayland(
        raw_window_handle::WaylandWindowHandle::new(NonNull::new_unchecked(surface)),
    );
    match (*instance)
        .instance
        .create_surface_raw(display_handle, window_handle)
    {
        Ok(surface) => {
            let wrapper = Box::new(LgpuSurfaceWrapper { surface });
            *out_surface = Box::into_raw(wrapper);
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Creates a window surface from Win32 HINSTANCE and HWND.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `hwnd` must be a valid non-zero HWND handle.
/// `out_surface` must be a valid, aligned, non-null pointer to a writable pointer.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_surface_from_win32(
    instance: *mut LgpuInstanceWrapper,
    _hinstance: *mut c_void,
    hwnd: *mut c_void,
    out_surface: *mut *mut LgpuSurfaceWrapper,
) -> LgpuResult {
    if instance.is_null() || hwnd.is_null() || out_surface.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let display_handle = raw_window_handle::RawDisplayHandle::Windows(
        raw_window_handle::WindowsDisplayHandle::new(),
    );
    let non_zero_hwnd = match std::num::NonZeroIsize::new(hwnd as isize) {
        Some(h) => h,
        None => return LgpuResult::ErrorGeneric,
    };
    let window_handle = raw_window_handle::RawWindowHandle::Win32(
        raw_window_handle::Win32WindowHandle::new(non_zero_hwnd),
    );
    match (*instance)
        .instance
        .create_surface_raw(display_handle, window_handle)
    {
        Ok(surface) => {
            let wrapper = Box::new(LgpuSurfaceWrapper { surface });
            *out_surface = Box::into_raw(wrapper);
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Creates a window surface from a macOS NSView or CAMetalLayer.
///
/// # Safety
///
/// `instance` must be a valid pointer obtained from `lgpu_create_instance`.
/// `layer_or_view` must be a valid pointer to an NSView or CAMetalLayer.
/// `out_surface` must be a valid, aligned, non-null pointer to a writable pointer.
#[no_mangle]
pub unsafe extern "C" fn lgpu_create_surface_from_metal(
    instance: *mut LgpuInstanceWrapper,
    layer_or_view: *mut c_void,
    out_surface: *mut *mut LgpuSurfaceWrapper,
) -> LgpuResult {
    if instance.is_null() || layer_or_view.is_null() || out_surface.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let display_handle =
        raw_window_handle::RawDisplayHandle::AppKit(raw_window_handle::AppKitDisplayHandle::new());
    let window_handle = raw_window_handle::RawWindowHandle::AppKit(
        raw_window_handle::AppKitWindowHandle::new(NonNull::new_unchecked(layer_or_view)),
    );
    match (*instance)
        .instance
        .create_surface_raw(display_handle, window_handle)
    {
        Ok(surface) => {
            let wrapper = Box::new(LgpuSurfaceWrapper { surface });
            *out_surface = Box::into_raw(wrapper);
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Configures swapchain dimensions, format, and present mode on `surface`.
///
/// # Safety
///
/// `surface`, `device`, and `config` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_surface_configure(
    surface: *mut LgpuSurfaceWrapper,
    device: *mut LgpuDeviceWrapper,
    config: *const LgpuSurfaceConfig,
) -> LgpuResult {
    if surface.is_null() || device.is_null() || config.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let rust_format = match (*config).format {
        0 => TextureFormat::Rgba8Unorm,
        1 => TextureFormat::Rgba8Srgb,
        2 => TextureFormat::Bgra8Unorm,
        3 => TextureFormat::Bgra8Srgb,
        4 => TextureFormat::R32Float,
        5 => TextureFormat::Rgba16Float,
        6 => TextureFormat::Rgba32Float,
        7 => TextureFormat::Depth32Float,
        _ => return LgpuResult::ErrorGeneric,
    };
    let rust_present_mode = match (*config).present_mode {
        LgpuPresentMode::Immediate => lgpu::PresentMode::Immediate,
        LgpuPresentMode::Mailbox => lgpu::PresentMode::Mailbox,
        LgpuPresentMode::Fifo => lgpu::PresentMode::Fifo,
    };
    let rust_usage = TextureUsage::from_bits_truncate((*config).usage);
    let rust_config = lgpu::SurfaceConfiguration {
        usage: rust_usage,
        format: rust_format,
        width: (*config).width,
        height: (*config).height,
        present_mode: rust_present_mode,
    };
    match (*surface)
        .surface
        .configure(&(*device).device, &rust_config)
    {
        Ok(()) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Acquires the next swapchain image as a surface texture.
///
/// # Safety
///
/// `surface` and `out_surface_texture` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_surface_get_current_texture(
    surface: *mut LgpuSurfaceWrapper,
    out_surface_texture: *mut *mut LgpuSurfaceTextureWrapper,
) -> LgpuResult {
    if surface.is_null() || out_surface_texture.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    match (*surface).surface.get_current_texture() {
        Ok(st) => {
            let tex_wrapper = Box::new(LgpuTextureWrapper {
                texture: st.texture.clone_unowned(),
            });
            let wrapper = Box::new(LgpuSurfaceTextureWrapper {
                surface_texture: Some(st),
                texture_wrapper: Some(tex_wrapper),
            });
            *out_surface_texture = Box::into_raw(wrapper);
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorGeneric,
    }
}

/// Retrieves the `lgpu_texture_t` handle from a surface texture for encoding draw commands.
///
/// # Safety
///
/// `surface_texture` must be a valid pointer obtained from `lgpu_surface_get_current_texture`.
#[no_mangle]
pub unsafe extern "C" fn lgpu_surface_texture_get_texture(
    surface_texture: *mut LgpuSurfaceTextureWrapper,
) -> *mut LgpuTextureWrapper {
    if surface_texture.is_null() {
        return ptr::null_mut();
    }
    match &mut (*surface_texture).texture_wrapper {
        Some(tw) => &mut **tw as *mut LgpuTextureWrapper,
        None => ptr::null_mut(),
    }
}

/// Submits a command buffer targeted to a swapchain texture and queues presentation.
///
/// # Safety
///
/// `queue`, `command_buffer`, and `surface_texture` must be valid, live pointers.
#[no_mangle]
pub unsafe extern "C" fn lgpu_queue_submit_surface(
    queue: *mut LgpuQueueWrapper,
    command_buffer: *mut LgpuCommandBufferWrapper,
    surface_texture: *mut LgpuSurfaceTextureWrapper,
) -> LgpuResult {
    if queue.is_null() || command_buffer.is_null() || surface_texture.is_null() {
        return LgpuResult::ErrorGeneric;
    }
    let cmd = match (*command_buffer).command_buffer.take() {
        Some(cmd) => cmd,
        None => return LgpuResult::ErrorCommand,
    };
    let st = match (*surface_texture).surface_texture.take() {
        Some(st) => st,
        None => return LgpuResult::ErrorGeneric,
    };
    match (*queue).queue.submit_surface(cmd, st) {
        Ok(()) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorCommand,
    }
}

/// Frees an acquired surface texture wrapper.
///
/// # Safety
///
/// `surface_texture` must be a valid pointer or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_surface_texture_destroy(
    surface_texture: *mut LgpuSurfaceTextureWrapper,
) {
    if !surface_texture.is_null() {
        drop(Box::from_raw(surface_texture));
    }
}

/// Destroys an lgpu window surface.
///
/// # Safety
///
/// `surface` must be a valid pointer obtained from `lgpu_create_surface_*`, or null.
#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_surface(surface: *mut LgpuSurfaceWrapper) {
    if !surface.is_null() {
        drop(Box::from_raw(surface));
    }
}
