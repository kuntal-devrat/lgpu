//! C99 / Zig FFI exports for lgpu

use std::ffi::{c_char, c_void, CString};
use std::ptr;

use lgpu::{BufferDesc, BufferUsage, Instance, MemoryLocation};

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
    buffer: lgpu::Buffer,
}

pub struct LgpuShaderWrapper {
    shader: lgpu::Shader,
}

pub struct LgpuComputePipelineWrapper {
    pipeline: lgpu::ComputePipeline,
}

pub struct LgpuCommandEncoderWrapper {
    encoder: Option<lgpu::CommandEncoder>,
}

pub struct LgpuCommandBufferWrapper {
    command_buffer: Option<lgpu::CommandBuffer>,
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
    if offset == 0 {
        match buf.buffer.write(slice) {
            Ok(_) => LgpuResult::Success,
            Err(_) => LgpuResult::ErrorBuffer,
        }
    } else {
        LgpuResult::ErrorGeneric
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
    if offset == 0 {
        match buf.buffer.read(slice) {
            Ok(_) => LgpuResult::Success,
            Err(_) => LgpuResult::ErrorBuffer,
        }
    } else {
        LgpuResult::ErrorGeneric
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
