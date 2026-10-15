//! Tests for C FFI buffer-to-buffer copies and direct texture data transfers.

use lgpu_c::*;
use std::ffi::c_void;
use std::ptr;

#[test]
fn test_c_buffer_copy_and_texture_write() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        let res = lgpu_create_instance(&mut instance);
        assert_eq!(res, LgpuResult::Success);

        let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
        let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
        let res = lgpu_create_device(instance, 0, &mut device, &mut queue);
        assert_eq!(res, LgpuResult::Success);

        // 1. Buffer to buffer copy
        let initial_data: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let mut src_buf: *mut LgpuBufferWrapper = ptr::null_mut();
        let res = lgpu_create_buffer_init(
            device,
            initial_data.as_ptr() as *const c_void,
            8,
            1 | 2 | 4, // TRANSFER_SRC | TRANSFER_DST | STORAGE
            1,         // CpuToGpu
            &mut src_buf,
        );
        assert_eq!(res, LgpuResult::Success);

        let mut dst_buf: *mut LgpuBufferWrapper = ptr::null_mut();
        let res = lgpu_create_buffer(
            device,
            8,
            1 | 2 | 4, // TRANSFER_SRC | TRANSFER_DST | STORAGE
            1,         // CpuToGpu
            &mut dst_buf,
        );
        assert_eq!(res, LgpuResult::Success);

        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut encoder);
        assert_eq!(res, LgpuResult::Success);

        lgpu_copy_buffer_to_buffer(encoder, src_buf, 0, dst_buf, 0, 8);

        let cmd = lgpu_finish_encoder(encoder);
        assert!(!cmd.is_null());

        let res = lgpu_queue_submit(queue, cmd);
        assert_eq!(res, LgpuResult::Success);

        let mut readback = [0u8; 8];
        let res = lgpu_read_buffer(
            dst_buf,
            0,
            readback.as_mut_ptr() as *mut c_void,
            readback.len(),
        );
        assert_eq!(res, LgpuResult::Success);
        assert_eq!(readback, initial_data);

        // 2. Direct texture write via lgpu_device_write_texture
        let mut tex: *mut LgpuTextureWrapper = ptr::null_mut();
        let res = lgpu_create_texture(
            device,
            4,
            4,
            1,
            0,         // RGBA8_UNORM
            1 | 2 | 4, // COPY_SRC | COPY_DST | TEXTURE_BINDING
            &mut tex,
        );
        assert_eq!(res, LgpuResult::Success);

        let pixels = [255u8; 4 * 4 * 4];
        let res = lgpu_device_write_texture(device, tex, pixels.as_ptr(), pixels.len());
        assert_eq!(res, LgpuResult::Success);

        // Cleanup
        drop(Box::from_raw(cmd));
        lgpu_destroy_texture(tex);
        lgpu_destroy_buffer(src_buf);
        lgpu_destroy_buffer(dst_buf);
        lgpu_destroy_queue(queue);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }
}
