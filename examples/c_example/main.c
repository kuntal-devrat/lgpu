/**
 * @file main.c
 * @brief C99 example demonstrating lgpu C API:
 *  - Adapter hardware info query
 *  - Host initialized buffers (lgpu_create_buffer_init)
 *  - Bindless buffer, texture, and sampler descriptor slot indexing
 *  - Texture creation and clear/render
 *  - Dynamic rendering and triangle rasterization
 *  - Texture-to-buffer readback
 *  - Timeline semaphores synchronization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../../crates/lgpu-c/include/lgpu.h"

static const uint32_t TRIANGLE_SPV[] = {
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
    196670, 62, 66, 65789, 65592
};

int main(void) {
    printf("=== lgpu C99 Example ===\n");
    printf("lgpu ABI Version: %u\n", lgpu_version());

    // 1. Create Instance
    lgpu_instance_t instance = NULL;
    lgpu_result_t res = lgpu_create_instance(&instance);
    if (res != LGPU_SUCCESS || !instance) {
        fprintf(stderr, "Failed to create lgpu instance (error %d)\n", res);
        return 1;
    }

    // 2. Query Adapters and Hardware Information
    uint32_t adapter_count = lgpu_get_adapter_count(instance);
    printf("Discovered %u GPU Adapter(s):\n", adapter_count);
    assert(adapter_count > 0);

    lgpu_adapter_info_t info;
    memset(&info, 0, sizeof(info));
    res = lgpu_get_adapter_info(instance, 0, &info);
    assert(res == LGPU_SUCCESS);
    printf("  [0] %s\n", info.name);
    printf("      Backend: %u | DeviceType: %u | VendorID: 0x%04x | DeviceID: 0x%04x\n",
           info.backend, info.device_type, info.vendor_id, info.device_id);
    printf("      Dynamic Rendering: %s | Timeline Semaphores: %s | Descriptor Indexing: %s\n",
           info.supports_dynamic_rendering ? "YES" : "NO",
           info.supports_timeline_semaphores ? "YES" : "NO",
           info.supports_descriptor_indexing ? "YES" : "NO");
    printf("      Max Workgroup Size: [%u, %u, %u], Max Invocations: %u\n",
           info.max_compute_workgroup_size[0],
           info.max_compute_workgroup_size[1],
           info.max_compute_workgroup_size[2],
           info.max_compute_workgroup_invocations);

    // 3. Request Logical Device and Queue
    lgpu_device_t device = NULL;
    lgpu_queue_t queue = NULL;
    res = lgpu_create_device(instance, 0, &device, &queue);
    assert(res == LGPU_SUCCESS && device && queue);
    printf("\nSuccessfully initialized logical device & queue from C!\n");

    // 4. Test Buffer Init with Host Memory (lgpu_create_buffer_init)
    const float init_floats[4] = { 1.25f, 2.5f, 5.0f, 10.0f };
    lgpu_buffer_t init_buf = NULL;
    res = lgpu_create_buffer_init(
        device,
        init_floats,
        sizeof(init_floats),
        LGPU_USAGE_STORAGE | LGPU_USAGE_TRANSFER_SRC | LGPU_USAGE_TRANSFER_DST,
        LGPU_MEMORY_CPU_TO_GPU,
        &init_buf
    );
    assert(res == LGPU_SUCCESS && init_buf);

    int32_t buf_bindless_slot = lgpu_buffer_get_bindless_id(init_buf);
    printf("Created buffer initialized with host memory. Bindless slot: %d\n", buf_bindless_slot);
    assert(buf_bindless_slot >= 0);

    float read_floats[4] = { 0 };
    res = lgpu_read_buffer(init_buf, 0, read_floats, sizeof(read_floats));
    assert(res == LGPU_SUCCESS);
    for (int i = 0; i < 4; i++) {
        assert(read_floats[i] == init_floats[i]);
    }
    printf("Buffer initialization readback verified: [%.2f, %.2f, %.2f, %.2f]\n",
           read_floats[0], read_floats[1], read_floats[2], read_floats[3]);

    // 5. Test Texture Creation and Property Getters
    lgpu_texture_t color_texture = NULL;
    res = lgpu_create_texture(
        device,
        128,
        64,
        1,
        LGPU_FORMAT_RGBA8_UNORM,
        LGPU_TEX_USAGE_RENDER_ATTACHMENT | LGPU_TEX_USAGE_COPY_SRC | LGPU_TEX_USAGE_TEXTURE_BINDING,
        &color_texture
    );
    assert(res == LGPU_SUCCESS && color_texture);

    uint32_t tex_w = lgpu_texture_get_width(color_texture);
    uint32_t tex_h = lgpu_texture_get_height(color_texture);
    int32_t tex_fmt = lgpu_texture_get_format(color_texture);
    int32_t tex_slot = lgpu_texture_get_bindless_id(color_texture);
    printf("Color Texture: %ux%u, Format: %d, Bindless Slot: %d\n", tex_w, tex_h, tex_fmt, tex_slot);
    assert(tex_w == 128 && tex_h == 64 && tex_fmt == LGPU_FORMAT_RGBA8_UNORM);
    assert(tex_slot >= 0);

    // 6. Test Sampler Creation
    lgpu_sampler_t sampler = NULL;
    res = lgpu_create_sampler(
        device,
        LGPU_FILTER_LINEAR,
        LGPU_FILTER_LINEAR,
        LGPU_FILTER_LINEAR,
        LGPU_ADDRESS_CLAMP_TO_EDGE,
        LGPU_ADDRESS_CLAMP_TO_EDGE,
        LGPU_ADDRESS_CLAMP_TO_EDGE,
        &sampler
    );
    assert(res == LGPU_SUCCESS && sampler);
    int32_t sampler_slot = lgpu_sampler_get_bindless_id(sampler);
    printf("Sampler Created: Linear / ClampToEdge, Bindless Slot: %d\n", sampler_slot);
    assert(sampler_slot >= 0);

    // 7. Test Timeline Semaphore
    lgpu_timeline_semaphore_t sem = NULL;
    res = lgpu_create_timeline_semaphore(device, 10, &sem);
    assert(res == LGPU_SUCCESS && sem);

    uint64_t sem_val = 0;
    res = lgpu_get_timeline_semaphore_value(sem, &sem_val);
    assert(res == LGPU_SUCCESS && sem_val == 10);

    res = lgpu_signal_timeline_semaphore(sem, 25);
    assert(res == LGPU_SUCCESS);

    res = lgpu_wait_timeline_semaphore(sem, 25, 1000000000ULL); // 1 sec timeout
    assert(res == LGPU_SUCCESS);
    printf("Timeline Semaphore Host Signaling and Waiting verified (value: 25)\n");

    // 8. Test Graphics Pipeline and Triangle Drawing
    lgpu_shader_t shader = NULL;
    res = lgpu_create_shader(device, TRIANGLE_SPV, sizeof(TRIANGLE_SPV) / sizeof(uint32_t), &shader);
    assert(res == LGPU_SUCCESS && shader);

    lgpu_graphics_pipeline_t pipeline = NULL;
    res = lgpu_create_graphics_pipeline(
        device,
        shader,
        "vs_main",
        "fs_main",
        LGPU_FORMAT_RGBA8_UNORM,
        -1, // No depth
        false,
        false,
        &pipeline
    );
    assert(res == LGPU_SUCCESS && pipeline);

    // Allocate readback buffer for texture
    lgpu_buffer_t readback_buf = NULL;
    res = lgpu_create_buffer(
        device,
        128 * 64 * 4,
        LGPU_USAGE_TRANSFER_DST,
        LGPU_MEMORY_GPU_TO_CPU,
        &readback_buf
    );
    assert(res == LGPU_SUCCESS && readback_buf);

    // Encode clear, triangle draw, and copy to buffer
    lgpu_command_encoder_t encoder = NULL;
    res = lgpu_create_command_encoder(device, &encoder);
    assert(res == LGPU_SUCCESS && encoder);

    float clear_color[4] = { 0.2f, 0.4f, 0.8f, 1.0f };
    lgpu_clear_and_render_texture(encoder, color_texture, clear_color);
    lgpu_render_triangle(encoder, color_texture, pipeline, clear_color);
    lgpu_copy_texture_to_buffer(encoder, color_texture, readback_buf);

    lgpu_command_buffer_t cmd = lgpu_finish_encoder(encoder);
    assert(cmd != NULL);

    res = lgpu_queue_submit(queue, cmd);
    assert(res == LGPU_SUCCESS);
    printf("Rendered triangle to texture and copied to buffer successfully!\n");

    // 9. Clean up all resources
    lgpu_destroy_buffer(readback_buf);
    lgpu_destroy_graphics_pipeline(pipeline);
    lgpu_destroy_shader(shader);
    lgpu_destroy_timeline_semaphore(sem);
    lgpu_destroy_sampler(sampler);
    lgpu_destroy_texture(color_texture);
    lgpu_destroy_buffer(init_buf);
    lgpu_destroy_queue(queue);
    lgpu_destroy_device(device);
    lgpu_destroy_instance(instance);

    printf("\nC99 example completed successfully.\n");
    return 0;
}
