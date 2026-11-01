//! Integration tests for the extended C API: texture dimensions, staged uploads, timestamps,
//! comparison samplers, vertex bindings, debug labels and the async compute queue.

use lgpu_c::*;
use std::ffi::{c_void, CString};
use std::ptr;

unsafe fn create_device() -> Option<(
    *mut LgpuInstanceWrapper,
    *mut LgpuDeviceWrapper,
    *mut LgpuQueueWrapper,
)> {
    let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
    if lgpu_create_instance(&mut instance) != LgpuResult::Success {
        eprintln!("Skipping test: no Vulkan instance");
        return None;
    }
    let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
    let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
    if lgpu_create_device(instance, 0, &mut device, &mut queue) != LgpuResult::Success {
        eprintln!("Skipping test: no Vulkan device");
        lgpu_destroy_instance(instance);
        return None;
    }
    Some((instance, device, queue))
}

#[test]
fn test_c_texture_dimensions_uploads_and_samplers() {
    unsafe {
        let Some((instance, device, _queue)) = create_device() else {
            return;
        };

        // 3D texture: 4x4x4, three mip levels, sampled with mips generated on the GPU.
        let mut volume: *mut LgpuTextureWrapper = ptr::null_mut();
        let res = lgpu_create_texture_ex(
            device,
            4,
            4,
            4,
            1,
            2, // 3D
            3,
            1,
            0,         // Rgba8Unorm
            2 | 1 | 4, // COPY_DST | COPY_SRC | TEXTURE_BINDING
            &mut volume,
        );
        assert_eq!(res, LgpuResult::Success, "3D texture creation must succeed");

        let texels = vec![180u8; 4 * 4 * 4 * 4];
        let res =
            lgpu_device_write_texture_mip(device, volume, 0, texels.as_ptr(), texels.len(), 0);
        assert_eq!(res, LgpuResult::Success, "3D mip upload must succeed");

        let res = lgpu_device_generate_mipmaps(device, volume);
        assert_eq!(
            res,
            LgpuResult::Success,
            "cube/volume mip generation must succeed"
        );

        // Cube map with six faces.
        let mut cube: *mut LgpuTextureWrapper = ptr::null_mut();
        let res = lgpu_create_texture_ex(
            device, 8, 8, 1, 6, 3, // cube
            1, 1, 0, 4, // TEXTURE_BINDING
            &mut cube,
        );
        assert_eq!(res, LgpuResult::Success, "cube map creation must succeed");

        // Non-square cube maps are rejected.
        let mut bad_cube: *mut LgpuTextureWrapper = ptr::null_mut();
        let res = lgpu_create_texture_ex(device, 8, 4, 1, 6, 3, 1, 1, 0, 4, &mut bad_cube);
        assert_eq!(res, LgpuResult::ErrorGeneric);

        // Depth/stencil format usable as an attachment.
        let mut depth: *mut LgpuTextureWrapper = ptr::null_mut();
        let res = lgpu_create_texture_ex(
            device, 32, 32, 1, 1, 0, 1, 1, 8,  // Depth24Stencil8
            16, // RENDER_ATTACHMENT
            &mut depth,
        );
        assert_eq!(
            res,
            LgpuResult::Success,
            "D24S8 attachment must be creatable"
        );
        assert_eq!(lgpu_texture_get_format(depth), 8);

        // Comparison sampler for shadow maps.
        let mut sampler: *mut LgpuSamplerWrapper = ptr::null_mut();
        let res = lgpu_create_sampler_ex(device, 0, 0, 0, 0, 0, 0, true, 1, &mut sampler);
        assert_eq!(res, LgpuResult::Success);
        assert!(lgpu_sampler_get_bindless_id(sampler) >= 0);

        lgpu_destroy_sampler(sampler);
        lgpu_destroy_texture(depth);
        lgpu_destroy_texture(cube);
        lgpu_destroy_texture(volume);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }
}

#[test]
fn test_c_timestamps_vertex_bindings_and_device_local_buffers() {
    unsafe {
        let Some((instance, device, queue)) = create_device() else {
            return;
        };

        // Device-local buffer uploaded through staging.
        let payload: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut local: *mut LgpuBufferWrapper = ptr::null_mut();
        let res = lgpu_device_create_buffer_init_device_local(
            device,
            std::mem::size_of_val(&payload) as u64,
            4 | 8, // VERTEX | STORAGE
            payload.as_ptr() as *const u8,
            std::mem::size_of_val(&payload),
            &mut local,
        );
        assert_eq!(res, LgpuResult::Success, "device-local upload must succeed");

        // Device-local memory is not host visible: copy it into a mappable staging buffer.
        let mut staging: *mut LgpuBufferWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_create_buffer(
                device,
                std::mem::size_of_val(&payload) as u64,
                1 | 2, // TRANSFER_SRC | TRANSFER_DST
                1,     // CpuToGpu
                &mut staging,
            ),
            LgpuResult::Success
        );

        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_create_command_encoder(device, &mut encoder),
            LgpuResult::Success
        );
        lgpu_copy_buffer_to_buffer(
            encoder,
            local,
            0,
            staging,
            0,
            std::mem::size_of_val(&payload) as u64,
        );
        let command_buffer = lgpu_finish_encoder(encoder);
        assert!(!command_buffer.is_null());
        assert_eq!(
            lgpu_queue_submit(queue, command_buffer),
            LgpuResult::Success
        );

        let mut readback = [0u32; 8];
        let res = lgpu_read_buffer(
            staging,
            0,
            readback.as_mut_ptr() as *mut c_void,
            std::mem::size_of_val(&readback),
        );
        assert_eq!(res, LgpuResult::Success);
        assert_eq!(readback, payload, "device-local contents must round-trip");

        // Timestamp pool written and read back around a real submission.
        let mut pool: *mut LgpuTimestampPoolWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_create_timestamp_pool(device, 2, &mut pool),
            LgpuResult::Success
        );
        assert_eq!(lgpu_timestamp_pool_count(pool), 2);

        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_create_command_encoder(device, &mut encoder),
            LgpuResult::Success
        );
        lgpu_encoder_bind_vertex_buffer(encoder, 0, local, 0);
        lgpu_encoder_unbind_vertex_buffers(encoder);

        let label = CString::new("c-binding-test").unwrap();
        lgpu_encoder_begin_debug_label(encoder, label.as_ptr(), 0.1, 0.9, 0.2, 1.0);
        lgpu_encoder_end_debug_label(encoder);

        lgpu_encoder_write_timestamp(encoder, pool, 0);
        lgpu_encoder_memory_barrier(encoder);
        lgpu_encoder_write_timestamp(encoder, pool, 1);

        let command_buffer = lgpu_finish_encoder(encoder);
        assert!(!command_buffer.is_null());
        assert_eq!(
            lgpu_queue_submit(queue, command_buffer),
            LgpuResult::Success
        );

        let mut timestamps = [0u64; 2];
        let res = lgpu_timestamp_pool_read(pool, timestamps.as_mut_ptr(), 2);
        assert_eq!(res, LgpuResult::Success, "timestamps must be readable");
        assert!(
            timestamps[1] >= timestamps[0],
            "timestamps must not go backwards"
        );

        lgpu_destroy_timestamp_pool(pool);
        lgpu_destroy_buffer(staging);
        lgpu_destroy_buffer(local);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }
}

#[test]
fn test_c_async_compute_queue() {
    unsafe {
        let Some((instance, device, _queue)) = create_device() else {
            return;
        };

        if !lgpu_device_has_async_compute_queue(device) {
            eprintln!("Skipping test: adapter has no separate compute queue family");
            lgpu_destroy_device(device);
            lgpu_destroy_instance(instance);
            return;
        }

        let mut compute_queue: *mut LgpuQueueWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_device_get_compute_queue(device, &mut compute_queue),
            LgpuResult::Success
        );
        assert!(!compute_queue.is_null());

        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        assert_eq!(
            lgpu_create_compute_encoder(device, &mut encoder),
            LgpuResult::Success
        );
        lgpu_encoder_memory_barrier(encoder);

        let command_buffer = lgpu_finish_encoder(encoder);
        assert!(!command_buffer.is_null());
        assert_eq!(
            lgpu_queue_submit(compute_queue, command_buffer),
            LgpuResult::Success,
            "submitting recorded compute work must succeed"
        );

        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }
}
