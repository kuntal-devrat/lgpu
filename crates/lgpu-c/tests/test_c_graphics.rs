//! Integration tests for C FFI graphics features:
//! - C Texture creation and bindless ID query
//! - C Sampler creation and bindless ID query
//! - C Graphics Pipeline with depth testing
//! - C Dynamic rendering with depth attachment

use std::ptr;
use std::ffi::CString;
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

#[test]
fn test_c_ffi_textures_samplers_and_depth() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        let res = lgpu_create_instance(&mut instance);
        assert_eq!(res, LgpuResult::Success);

        let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
        let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
        let res = lgpu_create_device(instance, 0, &mut device, &mut queue);
        assert_eq!(res, LgpuResult::Success);

        // 1. Test texture creation with TEXTURE_BINDING (usage = 4)
        let mut texture: *mut LgpuTextureWrapper = ptr::null_mut();
        let res = lgpu_create_texture(
            device,
            64,
            64,
            1,
            0, // RGBA8_UNORM
            4 | 16, // TEXTURE_BINDING | RENDER_ATTACHMENT
            &mut texture,
        );
        assert_eq!(res, LgpuResult::Success);
        let tex_bindless_id = lgpu_texture_get_bindless_id(texture);
        assert_eq!(tex_bindless_id, 0, "First texture bindless ID should be 0");

        // 2. Test sampler creation
        let mut sampler: *mut LgpuSamplerWrapper = ptr::null_mut();
        let res = lgpu_create_sampler(
            device,
            1, // linear
            1, // linear
            1, // linear
            0, // clamp to edge
            0,
            0,
            &mut sampler,
        );
        assert_eq!(res, LgpuResult::Success);
        let sampler_bindless_id = lgpu_sampler_get_bindless_id(sampler);
        assert_eq!(sampler_bindless_id, 0, "First sampler bindless ID should be 0");

        // 3. Test depth texture creation
        let mut depth_texture: *mut LgpuTextureWrapper = ptr::null_mut();
        let res = lgpu_create_texture(
            device,
            64,
            64,
            1,
            7, // DEPTH32_FLOAT
            16, // RENDER_ATTACHMENT
            &mut depth_texture,
        );
        assert_eq!(res, LgpuResult::Success);

        // 4. Test graphics pipeline creation with depth testing
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
            0, // RGBA8_UNORM
            7, // DEPTH32_FLOAT
            true, // depth_test
            true, // depth_write
            &mut pipeline,
        );
        assert_eq!(res, LgpuResult::Success);

        // 5. Test clear_and_render_with_depth and draw_with_depth
        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut encoder);
        assert_eq!(res, LgpuResult::Success);

        let clear_color = [0.2f32, 0.4, 0.6, 1.0];
        lgpu_clear_and_render_with_depth(encoder, texture, depth_texture, clear_color.as_ptr(), 1.0);

        lgpu_draw_with_depth(
            encoder,
            texture,
            depth_texture,
            pipeline,
            3,
            ptr::null(),
            0,
            clear_color.as_ptr(),
            1.0,
        );

        let cmd = lgpu_finish_encoder(encoder);
        assert!(!cmd.is_null());

        let res = lgpu_queue_submit(queue, cmd);
        assert_eq!(res, LgpuResult::Success);

        // 6. Cleanup
        if !cmd.is_null() {
            drop(Box::from_raw(cmd));
        }
        lgpu_destroy_graphics_pipeline(pipeline);
        lgpu_destroy_shader(shader);
        lgpu_destroy_sampler(sampler);
        lgpu_destroy_texture(depth_texture);
        lgpu_destroy_texture(texture);
        lgpu_destroy_queue(queue);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }

    println!("SUCCESS: C FFI textures, samplers, and depth testing verified.");
}
