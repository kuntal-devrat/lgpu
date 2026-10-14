//! Integration tests for C FFI compute pipelines, adapter info, and resource management:
//! - Adapter info querying (lgpu_get_adapter_info)
//! - Buffer initialization from host pointer (lgpu_create_buffer_init)
//! - Texture property getters (width, height, format)
//! - Pure color dynamic rendering, drawing, and single triangle primitive
//! - Texture-to-buffer memory copy (lgpu_copy_texture_to_buffer)
//! - Bindless compute pipeline creation and dispatch (lgpu_create_bindless_compute_pipeline, lgpu_dispatch_bindless_compute)

use std::ffi::{c_void, CStr, CString};
use std::ptr;
use lgpu_c::*;

const TRIANGLE_SPV: &[u32] = &[
    119734787, 65536, 28, 67, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134, 0,
    196622, 0, 1, 524303, 0, 16, 1834972022, 7235937, 9, 12, 14, 524303, 4, 63, 1834972006,
    7235937, 56, 59, 62, 196624, 63, 7, 327752, 6, 0, 35, 0, 327752, 6, 1, 35, 16, 262215,
    9, 11, 42, 262215, 12, 11, 0, 262215, 14, 30, 0, 262215, 56, 11, 15, 262215, 59, 30, 0,
    262215, 62, 30, 0, 131091, 2, 196630, 4, 32, 262167, 3, 4, 4, 262167, 5, 4, 3, 262174, 6,
    3, 5, 262165, 7, 32, 0, 262176, 10, 1, 7, 262203, 10, 9, 1, 262176, 13, 3, 3, 262203, 13,
    12, 3, 262176, 15, 3, 5, 262203, 15, 14, 3, 196641, 17, 2, 262187, 7, 18, 0, 262187, 4,
    19, 0, 262187, 4, 20, 3204448256, 262187, 4, 21, 1065353216, 458796, 3, 22, 19, 20, 19,
    21, 393260, 5, 23, 21, 19, 19, 262187, 7, 24, 1, 262187, 4, 25, 1056964608, 458796, 3,
    26, 25, 25, 19, 21, 393260, 5, 27, 19, 21, 19, 458796, 3, 28, 20, 25, 19, 21, 393260, 5,
    29, 19, 19, 21, 262176, 31, 7, 6, 196654, 6, 32, 131092, 34, 262176, 39, 7, 3, 262176,
    41, 7, 5, 262176, 57, 1, 3, 262203, 57, 56, 1, 262176, 60, 1, 5, 262203, 60, 59, 1,
    262203, 13, 62, 3, 327734, 2, 16, 0, 17, 131320, 8, 327739, 31, 30, 7, 32, 262205, 7,
    11, 9, 131321, 33, 131320, 33, 327850, 34, 35, 11, 18, 196855, 36, 0, 262394, 35, 37,
    38, 131320, 37, 327745, 39, 40, 30, 18, 196670, 40, 22, 327745, 41, 42, 30, 24, 196670,
    42, 23, 131321, 36, 131320, 38, 327850, 34, 43, 11, 24, 196855, 44, 0, 262394, 43, 45,
    46, 131320, 45, 327745, 39, 47, 30, 18, 196670, 47, 26, 327745, 41, 48, 30, 24, 196670,
    48, 27, 131321, 44, 131320, 46, 327745, 39, 49, 30, 18, 196670, 49, 28, 327745, 41, 50,
    30, 24, 196670, 50, 29, 131321, 44, 131320, 44, 131321, 36, 131320, 36, 262205, 6, 51,
    30, 327761, 3, 52, 51, 0, 196670, 12, 52, 327761, 5, 53, 51, 1, 196670, 14, 53, 65789,
    65592, 327734, 2, 63, 0, 17, 131320, 54, 262205, 3, 58, 56, 262205, 5, 61, 59, 327760,
    6, 55, 58, 61, 131321, 64, 131320, 64, 327761, 5, 65, 55, 1, 327760, 3, 66, 65, 21,
    196670, 62, 66, 65789, 65592,
];

pub const PARTICLES_100K_SPV: &[u32] = &[
    119734787, 65536, 28, 90, 0, 131089, 1, 720906, 1599492179, 1599227979, 1919906931,
    1600481121, 1717990754, 1935635045, 1634889588, 1667196263, 1936941420, 0, 393227, 1,
    1280527431, 1685353262, 808793134, 0, 196622, 0, 1, 393231, 5, 16, 1852399981, 0, 13,
    393232, 16, 17, 256, 1, 1, 327752, 5, 0, 35, 0, 327752, 5, 1, 35, 16, 327752, 5, 2, 35,
    32, 262215, 6, 6, 48, 262215, 9, 34, 0, 262215, 9, 33, 0, 196679, 10, 2, 327752, 10, 0,
    35, 0, 262215, 13, 11, 28, 131091, 2, 196630, 4, 32, 262167, 3, 4, 4, 327710, 5, 3, 3,
    3, 196637, 6, 5, 262165, 8, 32, 0, 262167, 7, 8, 3, 196638, 10, 6, 262176, 11, 12, 10,
    262203, 11, 9, 12, 262176, 14, 1, 7, 262203, 14, 13, 1, 196641, 17, 2, 262176, 18, 12,
    6, 262187, 8, 19, 0, 262187, 4, 21, 1008767022, 262187, 4, 22, 0, 262187, 4, 23,
    1062836634, 262187, 4, 24, 1092616192, 262187, 4, 25, 1084227584, 262187, 4, 26,
    1065353216, 262187, 4, 27, 1056964608, 262176, 29, 7, 5, 196654, 5, 30, 131092, 34,
    262176, 38, 12, 5, 262176, 41, 7, 3, 262176, 42, 7, 4, 262187, 8, 43, 1, 262187, 8, 60,
    2, 262167, 80, 4, 3, 327734, 2, 16, 0, 17, 131320, 12, 327739, 29, 28, 7, 30, 262205,
    7, 15, 13, 327745, 18, 20, 9, 19, 131321, 31, 131320, 31, 327761, 8, 32, 15, 0, 327748,
    8, 33, 9, 0, 327854, 34, 35, 32, 33, 196855, 36, 0, 262394, 35, 37, 36, 131320, 37,
    65789, 131320, 36, 327745, 38, 39, 20, 32, 262205, 5, 40, 39, 196670, 28, 40, 393281,
    42, 44, 28, 43, 43, 262205, 4, 45, 44, 327811, 4, 46, 45, 21, 393281, 42, 47, 28, 43,
    43, 196670, 47, 46, 393281, 42, 48, 28, 43, 19, 262205, 4, 49, 48, 393281, 42, 50, 28,
    19, 19, 262205, 4, 51, 50, 327809, 4, 52, 51, 49, 393281, 42, 53, 28, 19, 19, 196670,
    53, 52, 393281, 42, 54, 28, 43, 43, 262205, 4, 55, 54, 393281, 42, 56, 28, 19, 43,
    262205, 4, 57, 56, 327809, 4, 58, 57, 55, 393281, 42, 59, 28, 19, 43, 196670, 59, 58,
    393281, 42, 61, 28, 43, 60, 262205, 4, 62, 61, 393281, 42, 63, 28, 19, 60, 262205, 4,
    64, 63, 327809, 4, 65, 64, 62, 393281, 42, 66, 28, 19, 60, 196670, 66, 65, 393281, 42,
    67, 28, 19, 43, 262205, 4, 68, 67, 327864, 34, 69, 68, 22, 196855, 70, 0, 262394, 69,
    71, 70, 131320, 71, 393281, 42, 72, 28, 19, 43, 196670, 72, 22, 393281, 42, 73, 28, 43,
    43, 262205, 4, 74, 73, 262271, 4, 75, 74, 327813, 4, 76, 75, 23, 393281, 42, 77, 28, 43,
    43, 196670, 77, 76, 131321, 70, 131320, 70, 327745, 41, 78, 28, 43, 262205, 3, 79, 78,
    524367, 80, 81, 79, 79, 0, 1, 2, 393228, 4, 82, 1, 66, 81, 327813, 4, 83, 82, 24, 327813,
    4, 84, 82, 25, 327811, 4, 85, 26, 84, 458832, 3, 86, 83, 27, 85, 26, 327745, 41, 87, 28,
    60, 196670, 87, 86, 262205, 5, 88, 28, 327745, 38, 89, 20, 32, 196670, 89, 88, 65789,
    65592,
];

#[test]
fn test_c_adapter_info() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        let res = lgpu_create_instance(&mut instance);
        assert_eq!(res, LgpuResult::Success);

        let count = lgpu_get_adapter_count(instance);
        assert!(count >= 1, "Should have at least one GPU adapter");

        let mut info: LgpuAdapterInfo = std::mem::zeroed();
        let res = lgpu_get_adapter_info(instance, 0, &mut info);
        assert_eq!(res, LgpuResult::Success);

        let c_str = CStr::from_ptr(info.name.as_ptr());
        let adapter_name = c_str.to_str().expect("Valid UTF-8 name");
        println!("C FFI Adapter Name: {}", adapter_name);
        assert!(!adapter_name.is_empty());

        assert_eq!(info.backend, 0, "Backend should be Vulkan (0)");
        assert!(info.vendor_id > 0);
        assert!(info.supports_descriptor_indexing);
        assert!(info.max_compute_workgroup_size[0] > 0);

        lgpu_destroy_instance(instance);
    }
}

#[test]
fn test_c_buffer_init_and_bindless_id() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        let res = lgpu_create_instance(&mut instance);
        assert_eq!(res, LgpuResult::Success);

        let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
        let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
        let res = lgpu_create_device(instance, 0, &mut device, &mut queue);
        assert_eq!(res, LgpuResult::Success);

        // Host data initialization: 4 x u32 = 16 bytes
        let initial_data: [u32; 4] = [100, 200, 300, 400];
        let mut buffer: *mut LgpuBufferWrapper = ptr::null_mut();

        let res = lgpu_create_buffer_init(
            device,
            initial_data.as_ptr() as *const c_void,
            std::mem::size_of_val(&initial_data),
            8 | 1 | 2, // STORAGE | TRANSFER_SRC | TRANSFER_DST
            1,         // CPU_TO_GPU
            &mut buffer,
        );
        assert_eq!(res, LgpuResult::Success);

        let bindless_id = lgpu_buffer_get_bindless_id(buffer);
        assert!(bindless_id >= 0, "Buffer should have received a valid bindless ID");

        // Read back the data and verify contents
        let mut readback: [u32; 4] = [0; 4];
        let res = lgpu_read_buffer(
            buffer,
            0,
            readback.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(&readback),
        );
        assert_eq!(res, LgpuResult::Success);
        assert_eq!(readback, initial_data);

        // Cleanup
        lgpu_destroy_buffer(buffer);
        lgpu_destroy_queue(queue);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }
}

#[test]
fn test_c_texture_properties_and_color_rendering() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        let res = lgpu_create_instance(&mut instance);
        assert_eq!(res, LgpuResult::Success);

        let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
        let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
        let res = lgpu_create_device(instance, 0, &mut device, &mut queue);
        assert_eq!(res, LgpuResult::Success);

        // Create 64x32 color texture
        let mut texture: *mut LgpuTextureWrapper = ptr::null_mut();
        let res = lgpu_create_texture(
            device,
            64,
            32,
            1,
            0, // RGBA8_UNORM
            16 | 1 | 4, // RENDER_ATTACHMENT | COPY_SRC | TEXTURE_BINDING
            &mut texture,
        );
        assert_eq!(res, LgpuResult::Success);

        // Verify property getters
        assert_eq!(lgpu_texture_get_width(texture), 64);
        assert_eq!(lgpu_texture_get_height(texture), 32);
        assert_eq!(lgpu_texture_get_format(texture), 0);

        // Create shader & graphics pipeline without depth
        let mut shader: *mut LgpuShaderWrapper = ptr::null_mut();
        let res = lgpu_create_shader(device, TRIANGLE_SPV.as_ptr(), TRIANGLE_SPV.len(), &mut shader);
        assert_eq!(res, LgpuResult::Success);

        let vs_entry = CString::new("vs_main").unwrap();
        let fs_entry = CString::new("fs_main").unwrap();
        let mut pipeline: *mut LgpuGraphicsPipelineWrapper = ptr::null_mut();
        let res = lgpu_create_graphics_pipeline(
            device,
            shader,
            vs_entry.as_ptr(),
            fs_entry.as_ptr(),
            0,  // RGBA8_UNORM
            -1, // No depth attachment
            false,
            false,
            &mut pipeline,
        );
        assert_eq!(res, LgpuResult::Success);

        // Create destination buffer for texture copy readback
        let mut dst_buffer: *mut LgpuBufferWrapper = ptr::null_mut();
        let res = lgpu_create_buffer(
            device,
            (64 * 32 * 4) as u64,
            2, // TRANSFER_DST
            2, // GPU_TO_CPU
            &mut dst_buffer,
        );
        assert_eq!(res, LgpuResult::Success);

        // Record commands: clear, draw, render_triangle, copy to buffer
        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut encoder);
        assert_eq!(res, LgpuResult::Success);

        let clear_color = [0.1f32, 0.2, 0.3, 1.0];
        lgpu_clear_and_render_texture(encoder, texture, clear_color.as_ptr());
        lgpu_draw(encoder, texture, pipeline, 3, ptr::null(), 0, clear_color.as_ptr());
        lgpu_render_triangle(encoder, texture, pipeline, clear_color.as_ptr());
        lgpu_copy_texture_to_buffer(encoder, texture, dst_buffer);

        let cmd = lgpu_finish_encoder(encoder);
        assert!(!cmd.is_null());

        let res = lgpu_queue_submit(queue, cmd);
        assert_eq!(res, LgpuResult::Success);

        // Cleanup
        if !cmd.is_null() {
            drop(Box::from_raw(cmd));
        }
        lgpu_destroy_buffer(dst_buffer);
        lgpu_destroy_graphics_pipeline(pipeline);
        lgpu_destroy_shader(shader);
        lgpu_destroy_texture(texture);
        lgpu_destroy_queue(queue);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }
}

#[test]
fn test_c_bindless_compute_dispatch() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        let res = lgpu_create_instance(&mut instance);
        assert_eq!(res, LgpuResult::Success);

        let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
        let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
        let res = lgpu_create_device(instance, 0, &mut device, &mut queue);
        assert_eq!(res, LgpuResult::Success);

        let mut shader: *mut LgpuShaderWrapper = ptr::null_mut();
        let res = lgpu_create_shader(
            device,
            PARTICLES_100K_SPV.as_ptr(),
            PARTICLES_100K_SPV.len(),
            &mut shader,
        );
        assert_eq!(res, LgpuResult::Success);

        let mut pipeline: *mut LgpuBindlessComputePipelineWrapper = ptr::null_mut();
        let res = lgpu_create_bindless_compute_pipeline(device, shader, &mut pipeline);
        assert_eq!(res, LgpuResult::Success);

        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut encoder);
        assert_eq!(res, LgpuResult::Success);

        let push_constants = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let res = lgpu_dispatch_bindless_compute(
            encoder,
            pipeline,
            push_constants.as_ptr() as *const c_void,
            std::mem::size_of_val(&push_constants),
            1,
            1,
            1,
        );
        assert_eq!(res, LgpuResult::Success);

        let cmd = lgpu_finish_encoder(encoder);
        assert!(!cmd.is_null());

        let res = lgpu_queue_submit(queue, cmd);
        assert_eq!(res, LgpuResult::Success);

        // Cleanup
        if !cmd.is_null() {
            drop(Box::from_raw(cmd));
        }
        lgpu_destroy_bindless_compute_pipeline(pipeline);
        lgpu_destroy_shader(shader);
        lgpu_destroy_queue(queue);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }
}
