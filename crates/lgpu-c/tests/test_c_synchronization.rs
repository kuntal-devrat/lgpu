//! Integration tests for C FFI timeline semaphores and pipeline barriers.

use std::ptr;
use lgpu_c::*;

#[test]
fn test_c_ffi_timeline_semaphores_and_barriers() {
    unsafe {
        let mut instance: *mut LgpuInstanceWrapper = ptr::null_mut();
        let res = lgpu_create_instance(&mut instance);
        assert_eq!(res, LgpuResult::Success);

        let mut device: *mut LgpuDeviceWrapper = ptr::null_mut();
        let mut queue: *mut LgpuQueueWrapper = ptr::null_mut();
        let res = lgpu_create_device(instance, 0, &mut device, &mut queue);
        assert_eq!(res, LgpuResult::Success);

        // 1. Test timeline semaphore creation with initial value 100
        let mut semaphore: *mut LgpuTimelineSemaphoreWrapper = ptr::null_mut();
        let res = lgpu_create_timeline_semaphore(device, 100, &mut semaphore);
        assert_eq!(res, LgpuResult::Success);
        assert!(!semaphore.is_null());

        let mut val: u64 = 0;
        let res = lgpu_get_timeline_semaphore_value(semaphore, &mut val);
        assert_eq!(res, LgpuResult::Success);
        assert_eq!(val, 100);

        // Signal new value from host
        let res = lgpu_signal_timeline_semaphore(semaphore, 200);
        assert_eq!(res, LgpuResult::Success);

        let res = lgpu_get_timeline_semaphore_value(semaphore, &mut val);
        assert_eq!(res, LgpuResult::Success);
        assert_eq!(val, 200);

        // Wait on host
        let res = lgpu_wait_timeline_semaphore(semaphore, 200, 1_000_000_000);
        assert_eq!(res, LgpuResult::Success);

        // 2. Test buffer and barriers
        let mut buffer: *mut LgpuBufferWrapper = ptr::null_mut();
        let res = lgpu_create_buffer(device, 256, 1 | 2 | 8, 1, &mut buffer);
        assert_eq!(res, LgpuResult::Success);

        let mut encoder: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut encoder);
        assert_eq!(res, LgpuResult::Success);

        lgpu_encoder_memory_barrier(encoder);
        lgpu_encoder_buffer_barrier(encoder, buffer);

        let cmd1 = lgpu_finish_encoder(encoder);
        assert!(!cmd1.is_null());

        // 3. Test batch submissions via C FFI
        let mut encoder2: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut encoder2);
        assert_eq!(res, LgpuResult::Success);
        lgpu_encoder_memory_barrier(encoder2);
        let cmd2 = lgpu_finish_encoder(encoder2);
        assert!(!cmd2.is_null());

        let cmds = [cmd1, cmd2];
        let res = lgpu_queue_submit_batches(queue, cmds.as_ptr(), 2);
        assert_eq!(res, LgpuResult::Success);

        // 4. Test async submit with timeline semaphore via C FFI
        let mut encoder3: *mut LgpuCommandEncoderWrapper = ptr::null_mut();
        let res = lgpu_create_command_encoder(device, &mut encoder3);
        assert_eq!(res, LgpuResult::Success);
        lgpu_encoder_memory_barrier(encoder3);
        let cmd3 = lgpu_finish_encoder(encoder3);
        assert!(!cmd3.is_null());

        let wait_sems = [semaphore];
        let wait_vals = [200u64];
        let signal_sems = [semaphore];
        let signal_vals = [300u64];

        let res = lgpu_queue_submit_async(
            queue,
            &cmd3,
            1,
            wait_sems.as_ptr(),
            wait_vals.as_ptr(),
            1,
            signal_sems.as_ptr(),
            signal_vals.as_ptr(),
            1,
        );
        assert_eq!(res, LgpuResult::Success);

        // Wait on host for async completion
        let res = lgpu_wait_timeline_semaphore(semaphore, 300, 5_000_000_000);
        assert_eq!(res, LgpuResult::Success);

        let res = lgpu_get_timeline_semaphore_value(semaphore, &mut val);
        assert_eq!(res, LgpuResult::Success);
        assert_eq!(val, 300);

        // 5. Cleanup
        lgpu_destroy_timeline_semaphore(semaphore);
        lgpu_destroy_buffer(buffer);
        lgpu_destroy_command_encoder(encoder);
        lgpu_destroy_command_encoder(encoder2);
        lgpu_destroy_command_encoder(encoder3);
        lgpu_destroy_command_buffer(cmd1);
        lgpu_destroy_command_buffer(cmd2);
        lgpu_destroy_command_buffer(cmd3);
        lgpu_destroy_queue(queue);
        lgpu_destroy_device(device);
        lgpu_destroy_instance(instance);
    }

    println!("SUCCESS: C FFI timeline semaphores, batch submissions, and barriers verified.");
}
