//! Integration tests for C FFI surface lifecycle, presentation, device idling, and ABI stability.

use lgpu_c::*;
use std::ptr;

#[test]
fn test_c_device_wait_idle() {
    unsafe {
        // Null safety
        assert_eq!(
            lgpu_device_wait_idle(ptr::null_mut()),
            LgpuResult::ErrorGeneric
        );

        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        assert_eq!(lgpu_create_instance(&mut instance), LgpuResult::Success);

        let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
        let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_create_device(instance, 0, &mut device, &mut queue),
            LgpuResult::Success
        );

        // Wait idle should succeed
        assert_eq!(lgpu_device_wait_idle(device), LgpuResult::Success);

        lgpu_destroy_device(device);
        lgpu_destroy_queue(queue);
        lgpu_destroy_instance(instance);
    }
}

#[test]
fn test_c_surface_lifecycle_and_presentation() {
    unsafe {
        // Null pointer safety checks
        assert_eq!(
            lgpu_create_surface_headless(ptr::null_mut(), ptr::null_mut()),
            LgpuResult::ErrorGeneric
        );
        assert_eq!(
            lgpu_surface_configure(ptr::null_mut(), ptr::null_mut(), ptr::null()),
            LgpuResult::ErrorGeneric
        );
        assert_eq!(
            lgpu_surface_get_current_texture(ptr::null_mut(), ptr::null_mut()),
            LgpuResult::ErrorGeneric
        );
        assert_eq!(
            lgpu_surface_texture_get_texture(ptr::null_mut()),
            ptr::null_mut()
        );
        assert_eq!(
            lgpu_queue_submit_surface(ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
            LgpuResult::ErrorGeneric
        );
        lgpu_surface_texture_destroy(ptr::null_mut());
        lgpu_destroy_surface(ptr::null_mut());

        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        assert_eq!(lgpu_create_instance(&mut instance), LgpuResult::Success);

        let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
        let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_create_device(instance, 0, &mut device, &mut queue),
            LgpuResult::Success
        );

        let mut surface: *mut LgpuSurfaceWrapper = ptr::null_mut();
        let res = lgpu_create_surface_headless(instance, &mut surface);
        if res != LgpuResult::Success {
            println!("Headless surface not supported on this driver, skipping windowed presentation test.");
            lgpu_destroy_device(device);
            lgpu_destroy_queue(queue);
            lgpu_destroy_instance(instance);
            return;
        }

        // Configure surface
        let config = LgpuSurfaceConfig {
            width: 640,
            height: 480,
            format: 2,     // Bgra8Unorm
            usage: 1 << 4, // RENDER_ATTACHMENT
            present_mode: LgpuPresentMode::Fifo,
        };
        assert_eq!(
            lgpu_surface_configure(surface, device, &config),
            LgpuResult::Success
        );

        // Acquire surface texture
        let mut surface_texture: *mut LgpuSurfaceTextureWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_surface_get_current_texture(surface, &mut surface_texture),
            LgpuResult::Success
        );
        assert!(!surface_texture.is_null());

        let texture = lgpu_surface_texture_get_texture(surface_texture);
        assert!(!texture.is_null());
        assert_eq!(lgpu_texture_get_width(texture), 640);
        assert_eq!(lgpu_texture_get_height(texture), 480);

        // Record a command buffer to clear the surface texture
        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_create_command_encoder(device, &mut encoder),
            LgpuResult::Success
        );

        let clear_color = [0.2f32, 0.4, 0.6, 1.0];
        lgpu_clear_and_render_texture(encoder, texture, clear_color.as_ptr());

        let cmd = lgpu_finish_encoder(encoder);
        assert!(!cmd.is_null());

        // Submit surface command and present
        assert_eq!(
            lgpu_queue_submit_surface(queue, cmd, surface_texture),
            LgpuResult::Success
        );

        // Cleanup
        lgpu_destroy_command_buffer(cmd);
        lgpu_surface_texture_destroy(surface_texture);
        lgpu_destroy_command_encoder(encoder);

        assert_eq!(lgpu_device_wait_idle(device), LgpuResult::Success);

        lgpu_destroy_surface(surface);
        lgpu_destroy_device(device);
        lgpu_destroy_queue(queue);
        lgpu_destroy_instance(instance);
    }
}

#[test]
fn test_c_adapter_info_abi_stability() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        assert_eq!(lgpu_create_instance(&mut instance), LgpuResult::Success);

        let mut info: LgpuAdapterInfo = std::mem::zeroed();
        assert_eq!(
            lgpu_get_adapter_info(instance, 0, &mut info),
            LgpuResult::Success
        );

        // Check uint8_t boolean flags
        assert!(info.supports_dynamic_rendering == 0 || info.supports_dynamic_rendering == 1);
        assert!(info.supports_timeline_semaphores == 0 || info.supports_timeline_semaphores == 1);
        assert_eq!(
            info.supports_descriptor_indexing, 1,
            "Vulkan 1.2+ must support descriptor indexing"
        );

        lgpu_destroy_instance(instance);
    }
}

#[test]
fn test_c_ffi_error_message_and_device_limits() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        assert_eq!(lgpu_create_instance(&mut instance), LgpuResult::Success);

        // 1. Query device limits on valid adapter 0
        let mut limits: LgpuDeviceLimits = std::mem::zeroed();
        assert_eq!(
            lgpu_adapter_get_limits(instance, 0, &mut limits),
            LgpuResult::Success
        );
        assert!(limits.max_buffer_size > 0);
        assert!(limits.max_texture_dimension_2d >= 1024);
        assert!(limits.max_compute_workgroup_invocations >= 64);

        // 2. Query limits on invalid adapter -> verify error code and diagnostic error message
        assert_eq!(
            lgpu_adapter_get_limits(instance, 9999, &mut limits),
            LgpuResult::ErrorNoAdapter
        );
        let mut err_buf = [0 as std::ffi::c_char; 256];
        let len = lgpu_get_last_error_message(err_buf.as_mut_ptr(), err_buf.len());
        assert!(len > 0, "Error message must have length > 0");
        let err_cstr = std::ffi::CStr::from_ptr(err_buf.as_ptr());
        let err_str = err_cstr.to_str().expect("valid utf-8 error string");
        assert!(
            err_str.contains("out of bounds"),
            "Expected 'out of bounds' in error message: {err_str}"
        );

        // 3. Null pointer safety -> verify error code and diagnostic error message
        assert_eq!(
            lgpu_create_buffer(ptr::null_mut(), 1024, 0, 0, ptr::null_mut()),
            LgpuResult::ErrorGeneric
        );
        let len2 = lgpu_get_last_error_message(err_buf.as_mut_ptr(), err_buf.len());
        assert!(len2 > 0);
        let err_cstr2 = std::ffi::CStr::from_ptr(err_buf.as_ptr());
        let err_str2 = err_cstr2.to_str().expect("valid utf-8 error string");
        assert!(
            err_str2.contains("Null pointer"),
            "Expected 'Null pointer' in error message: {err_str2}"
        );

        lgpu_destroy_instance(instance);
    }
}
