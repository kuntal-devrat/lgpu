//! C99 / Zig FFI exports for lgpu

use std::ffi::{c_char, c_void, CString};
use std::ptr;

use lgpu::{
    AddressMode, Backend, BindlessComputePipeline, BufferDesc, BufferUsage, DeviceType,
    FilterMode, GraphicsPipeline, Instance, MemoryLocation, Sampler, SamplerDesc, Texture,
    TextureDesc, TextureFormat, TextureUsage,
};

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

#[repr(C)]
#[derive(Debug, Clone)]
pub struct LgpuAdapterInfo {
    pub name: [c_char; 256],
    pub backend: u32,
    pub device_type: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub supports_dynamic_rendering: bool,
    pub supports_timeline_semaphores: bool,
    pub supports_descriptor_indexing: bool,
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

#[no_mangle]
pub unsafe extern "C" fn lgpu_get_adapter_count(instance: *mut LgpuInstanceWrapper) -> u32 {
    if instance.is_null() {
        return 0;
    }
    let wrapper = &*instance;
    wrapper.adapter_names.len() as u32
}

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
        supports_dynamic_rendering: info.supports_dynamic_rendering,
        supports_timeline_semaphores: info.supports_timeline_semaphores,
        supports_descriptor_indexing: info.supports_descriptor_indexing,
        max_compute_workgroup_invocations: info.max_compute_workgroup_invocations,
        max_compute_workgroup_size: info.max_compute_workgroup_size,
    };

    let name_bytes = info.name.as_bytes();
    let len = name_bytes.len().min(255);
    for i in 0..len {
        c_info.name[i] = name_bytes[i] as c_char;
    }
    c_info.name[len] = 0;

    *out_info = c_info;
    LgpuResult::Success
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_create_device(
    instance: *mut LgpuInstanceWrapper,
    adapter_index: u32,
    out_device: *mut *mut LgpuDeviceWrapper,
    out_queue: *mut *mut LgpuQueueWrapper,
) -> LgpuResult {
    if instance.is_null() || out_device.is_null() || out_queue.is_null() {
        return LgpuResult::ErrorGeneric;
    }

    let wrapper = &*instance;
    let adapters = match wrapper.instance.enumerate_adapters() {
        Ok(a) => a,
        Err(_) => return LgpuResult::ErrorNoAdapter,
    };

    if (adapter_index as usize) >= adapters.len() {
        return LgpuResult::ErrorNoAdapter;
    }

    match adapters[adapter_index as usize].request_device(&wrapper.instance) {
        Ok((device, queue)) => {
            *out_device = Box::into_raw(Box::new(LgpuDeviceWrapper { device }));
            *out_queue = Box::into_raw(Box::new(LgpuQueueWrapper { queue }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorDeviceCreation,
    }
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_create_buffer(
    device: *mut LgpuDeviceWrapper,
    size: u64,
    usage: u32,
    location: u32,
    out_buffer: *mut *mut LgpuBufferWrapper,
) -> LgpuResult {
    if device.is_null() || out_buffer.is_null() {
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
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorBuffer,
    }
}

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

    match q.queue.submit_batches_async(cmds, &wait_tuples, &signal_tuples) {
        Ok(_) => LgpuResult::Success,
        Err(_) => LgpuResult::ErrorCommand,
    }
}

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

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_timeline_semaphore(
    semaphore: *mut LgpuTimelineSemaphoreWrapper,
) {
    if !semaphore.is_null() {
        drop(Box::from_raw(semaphore));
    }
}


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

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_texture(texture: *mut LgpuTextureWrapper) {
    if !texture.is_null() {
        drop(Box::from_raw(texture));
    }
}

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

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_sampler(sampler: *mut LgpuSamplerWrapper) {
    if !sampler.is_null() {
        drop(Box::from_raw(sampler));
    }
}

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
    if device.is_null() || shader.is_null() || vs_entry.is_null() || fs_entry.is_null() || out_pipeline.is_null() {
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

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_graphics_pipeline(pipeline: *mut LgpuGraphicsPipelineWrapper) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_clear_and_render_with_depth(
    encoder: *mut LgpuCommandEncoderWrapper,
    color_texture: *mut LgpuTextureWrapper,
    depth_texture: *mut LgpuTextureWrapper,
    clear_color: *const f32,
    clear_depth: f32,
) {
    if encoder.is_null() || color_texture.is_null() || depth_texture.is_null() || clear_color.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let col = &*color_texture;
    let dep = &*depth_texture;
    let color_slice = std::slice::from_raw_parts(clear_color, 4);
    let color_arr = [color_slice[0], color_slice[1], color_slice[2], color_slice[3]];

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.clear_and_render_with_depth(&col.texture, &dep.texture, color_arr, clear_depth);
    }
}

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
    if encoder.is_null() || color_texture.is_null() || depth_texture.is_null() || pipeline.is_null() || clear_color.is_null() {
        return;
    }
    let enc_wrapper = &mut *encoder;
    let col = &*color_texture;
    let dep = &*depth_texture;
    let pip = &*pipeline;
    let color_slice = std::slice::from_raw_parts(clear_color, 4);
    let color_arr = [color_slice[0], color_slice[1], color_slice[2], color_slice[3]];

    let pc_slice = if !push_constants.is_null() && push_constants_size > 0 {
        std::slice::from_raw_parts(push_constants as *const u8, push_constants_size)
    } else {
        &[]
    };

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.draw_with_depth(
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
    let color_arr = [color_slice[0], color_slice[1], color_slice[2], color_slice[3]];

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.clear_and_render_texture(&tex.texture, color_arr);
    }
}

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
    let color_arr = [color_slice[0], color_slice[1], color_slice[2], color_slice[3]];

    let pc_slice = if !push_constants.is_null() && push_constants_size > 0 {
        std::slice::from_raw_parts(push_constants as *const u8, push_constants_size)
    } else {
        &[]
    };

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.draw(&tex.texture, &pip.pipeline, vertex_count, pc_slice, color_arr);
    }
}

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
    let color_arr = [color_slice[0], color_slice[1], color_slice[2], color_slice[3]];

    if let Some(enc) = &mut enc_wrapper.encoder {
        enc.render_triangle(&tex.texture, &pip.pipeline, color_arr);
    }
}

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

#[no_mangle]
pub unsafe extern "C" fn lgpu_texture_get_width(texture: *mut LgpuTextureWrapper) -> u32 {
    if texture.is_null() {
        return 0;
    }
    let tex = &*texture;
    tex.texture.width
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_texture_get_height(texture: *mut LgpuTextureWrapper) -> u32 {
    if texture.is_null() {
        return 0;
    }
    let tex = &*texture;
    tex.texture.height
}

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
            *out_pipeline = Box::into_raw(Box::new(LgpuBindlessComputePipelineWrapper { pipeline }));
            LgpuResult::Success
        }
        Err(_) => LgpuResult::ErrorPipeline,
    }
}

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

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_bindless_compute_pipeline(
    pipeline: *mut LgpuBindlessComputePipelineWrapper,
) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_buffer(buffer: *mut LgpuBufferWrapper) {
    if !buffer.is_null() {
        drop(Box::from_raw(buffer));
    }
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_shader(shader: *mut LgpuShaderWrapper) {
    if !shader.is_null() {
        drop(Box::from_raw(shader));
    }
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_compute_pipeline(pipeline: *mut LgpuComputePipelineWrapper) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_device(device: *mut LgpuDeviceWrapper) {
    if !device.is_null() {
        drop(Box::from_raw(device));
    }
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_queue(queue: *mut LgpuQueueWrapper) {
    if !queue.is_null() {
        drop(Box::from_raw(queue));
    }
}

#[no_mangle]
pub unsafe extern "C" fn lgpu_destroy_instance(instance: *mut LgpuInstanceWrapper) {
    if !instance.is_null() {
        drop(Box::from_raw(instance));
    }
}

