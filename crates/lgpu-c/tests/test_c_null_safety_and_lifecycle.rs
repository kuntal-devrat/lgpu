//! C FFI null-safety, explicit handle destruction, and parameter validation tests.

use lgpu_c::*;
use std::ptr;

#[test]
fn test_c_destroy_functions_and_null_safety() {
    unsafe {
        // Destroying null pointers must be safe and no-op
        lgpu_destroy_command_encoder(ptr::null_mut());
        lgpu_destroy_command_buffer(ptr::null_mut());
        lgpu_destroy_buffer(ptr::null_mut());
        lgpu_destroy_texture(ptr::null_mut());
        lgpu_destroy_sampler(ptr::null_mut());
        lgpu_destroy_shader(ptr::null_mut());
        lgpu_destroy_compute_pipeline(ptr::null_mut());
        lgpu_destroy_graphics_pipeline(ptr::null_mut());
        lgpu_destroy_timeline_semaphore(ptr::null_mut());
        lgpu_destroy_queue(ptr::null_mut());
        lgpu_destroy_device(ptr::null_mut());
        lgpu_destroy_instance(ptr::null_mut());

        // Create actual instance and device
        let mut instance = ptr::null_mut();
        let res = lgpu_create_instance(&mut instance);
        assert_eq!(res, LgpuResult::Success);

        let mut device = ptr::null_mut();
        let mut queue = ptr::null_mut();
        let res = lgpu_create_device(instance, 0, &mut device, &mut queue);
        assert_eq!(res, LgpuResult::Success);

        // Test CommandEncoder creation and explicit destruction
        let mut encoder = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut encoder);
        assert_eq!(res, LgpuResult::Success);
        assert!(!encoder.is_null());

        let cmd = lgpu_finish_encoder(encoder);
        assert!(!cmd.is_null());

        // Destroy encoder and command buffer explicitly
        lgpu_destroy_command_encoder(encoder);
        lgpu_destroy_command_buffer(cmd);

        // Test non-aliasing validation: passing same texture for color and depth
        let mut tex = ptr::null_mut();
        let res = lgpu_create_texture(device, 64, 64, 1, 0, 1 | 4, &mut tex);
        assert_eq!(res, LgpuResult::Success);

        let mut enc = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut enc);
        assert_eq!(res, LgpuResult::Success);

        let clear_col = [0.0f32, 0.0f32, 0.0f32, 1.0f32];
        // Same pointer for color and depth: must be safely rejected (no crash, early return)
        lgpu_clear_and_render_with_depth(enc, tex, tex, clear_col.as_ptr(), 1.0);

        let finished_cmd = lgpu_finish_encoder(enc);
        lgpu_destroy_command_encoder(enc);
        if !finished_cmd.is_null() {
            lgpu_destroy_command_buffer(finished_cmd);
        }

        lgpu_destroy_texture(tex);
        lgpu_destroy_queue(queue);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }
}
