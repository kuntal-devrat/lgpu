const std = @import("std");

const lgpu = @cImport({
    @cInclude("lgpu.h");
});

pub fn main() !void {
    std.debug.print("=== lgpu Zig Example ===\n", .{});
    std.debug.print("lgpu ABI Version: {d}\n", .{lgpu.lgpu_version()});

    // 1. Create Instance
    var instance: lgpu.lgpu_instance_t = null;
    var res = lgpu.lgpu_create_instance(&instance);
    if (res != lgpu.LGPU_SUCCESS or instance == null) {
        std.debug.print("Failed to initialize lgpu instance (error: {d})\n", .{res});
        return error.LgpuInitializationFailed;
    }
    defer lgpu.lgpu_destroy_instance(instance);

    // 2. Query Adapters
    const count = lgpu.lgpu_get_adapter_count(instance);
    std.debug.print("Found {d} GPU adapter(s).\n", .{count});
    if (count == 0) return error.NoGpuFound;

    var info: lgpu.lgpu_adapter_info_t = std.mem.zeroes(lgpu.lgpu_adapter_info_t);
    res = lgpu.lgpu_get_adapter_info(instance, 0, &info);
    if (res != lgpu.LGPU_SUCCESS) return error.AdapterInfoFailed;

    const name_len = std.mem.indexOfScalar(u8, &info.name, 0) orelse info.name.len;
    std.debug.print("Using GPU: {s}\n", .{info.name[0..name_len]});
    std.debug.print("  Dynamic Rendering: {}\n", .{info.supports_dynamic_rendering != 0});
    std.debug.print("  Timeline Semaphores: {}\n", .{info.supports_timeline_semaphores != 0});
    std.debug.print("  Descriptor Indexing: {}\n", .{info.supports_descriptor_indexing != 0});

    // 3. Create Device & Queue
    var device: lgpu.lgpu_device_t = null;
    var queue: lgpu.lgpu_queue_t = null;
    res = lgpu.lgpu_create_device(instance, 0, &device, &queue);
    if (res != lgpu.LGPU_SUCCESS) return error.DeviceCreationFailed;
    defer lgpu.lgpu_destroy_device(device);
    defer lgpu.lgpu_destroy_queue(queue);

    // 4. Host-Initialized Buffer
    const input_data = [_]f32{ 3.14159, 2.71828, 1.41421, 1.61803 };
    const data_bytes = input_data.len * @sizeOf(f32);

    var src_buffer: lgpu.lgpu_buffer_t = null;
    res = lgpu.lgpu_create_buffer_init(
        device,
        &input_data,
        data_bytes,
        lgpu.LGPU_USAGE_STORAGE | lgpu.LGPU_USAGE_TRANSFER_SRC,
        lgpu.LGPU_MEMORY_CPU_TO_GPU,
        &src_buffer,
    );
    if (res != lgpu.LGPU_SUCCESS) return error.BufferCreationFailed;
    defer lgpu.lgpu_destroy_buffer(src_buffer);

    const slot = lgpu.lgpu_buffer_get_bindless_id(src_buffer);
    std.debug.print("Created buffer with {d} floats. Bindless slot: {d}\n", .{ input_data.len, slot });

    // 5. Destination Buffer
    var dst_buffer: lgpu.lgpu_buffer_t = null;
    res = lgpu.lgpu_create_buffer(
        device,
        data_bytes,
        lgpu.LGPU_USAGE_STORAGE | lgpu.LGPU_USAGE_TRANSFER_DST,
        lgpu.LGPU_MEMORY_GPU_TO_CPU,
        &dst_buffer,
    );
    if (res != lgpu.LGPU_SUCCESS) return error.BufferCreationFailed;
    defer lgpu.lgpu_destroy_buffer(dst_buffer);

    // 6. Record & Submit Copy
    var encoder: lgpu.lgpu_command_encoder_t = null;
    res = lgpu.lgpu_create_command_encoder(device, &encoder);
    if (res != lgpu.LGPU_SUCCESS) return error.EncoderFailed;

    lgpu.lgpu_copy_buffer_to_buffer(encoder, src_buffer, 0, dst_buffer, 0, data_bytes);

    const cmd = lgpu.lgpu_finish_encoder(encoder);
    if (cmd == null) return error.FinishFailed;

    res = lgpu.lgpu_queue_submit(queue, cmd);
    if (res != lgpu.LGPU_SUCCESS) return error.SubmitFailed;

    _ = lgpu.lgpu_device_wait_idle(device);

    // 7. Read Back
    var output_data: [4]f32 = undefined;
    res = lgpu.lgpu_read_buffer(dst_buffer, 0, &output_data, data_bytes);
    if (res != lgpu.LGPU_SUCCESS) return error.ReadBufferFailed;

    std.debug.print("GPU buffer readback verified in Zig: [{d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        output_data[0], output_data[1], output_data[2], output_data[3],
    });

    std.debug.print("\nZig example completed successfully.\n", .{});
}
