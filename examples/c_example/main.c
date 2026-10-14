/**
 * @file main.c
 * @brief Standalone C99 program demonstrating usage of lgpu via lgpu.h
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../../crates/lgpu-c/include/lgpu.h"

int main(void) {
    printf("=== lgpu: Pure C99 ABI Demonstration ===\n");
    printf("lgpu ABI Version: %u\n", lgpu_version());

    // 1. Create Instance
    lgpu_instance_t instance = NULL;
    lgpu_result_t res = lgpu_create_instance(&instance);
    if (res != LGPU_SUCCESS || !instance) {
        fprintf(stderr, "Failed to create lgpu instance (error %d)\n", res);
        return 1;
    }

    uint32_t adapter_count = lgpu_get_adapter_count(instance);
    printf("Discovered %u GPU Adapter(s):\n", adapter_count);
    for (uint32_t i = 0; i < adapter_count; i++) {
        const char* name = lgpu_get_adapter_name(instance, i);
        printf("  [%u] %s\n", i, name ? name : "Unknown");
    }

    if (adapter_count == 0) {
        fprintf(stderr, "No GPU adapter found!\n");
        lgpu_destroy_instance(instance);
        return 1;
    }

    // 2. Request Logical Device and Queue
    lgpu_device_t device = NULL;
    lgpu_queue_t queue = NULL;
    res = lgpu_create_device(instance, 0, &device, &queue);
    if (res != LGPU_SUCCESS || !device || !queue) {
        fprintf(stderr, "Failed to create device/queue (error %d)\n", res);
        lgpu_destroy_instance(instance);
        return 1;
    }
    printf("Successfully initialized device and queue from C!\n");

    // 3. Allocate a GPU Buffer from C
    const uint64_t size = 256;
    lgpu_buffer_t buffer = NULL;
    res = lgpu_create_buffer(
        device,
        size,
        LGPU_USAGE_STORAGE | LGPU_USAGE_TRANSFER_SRC | LGPU_USAGE_TRANSFER_DST,
        LGPU_MEMORY_CPU_TO_GPU,
        &buffer
    );
    if (res != LGPU_SUCCESS || !buffer) {
        fprintf(stderr, "Failed to create buffer from C\n");
        lgpu_destroy_device(device);
        lgpu_destroy_queue(queue);
        lgpu_destroy_instance(instance);
        return 1;
    }

    int32_t slot = lgpu_buffer_get_bindless_id(buffer);
    printf("Buffer allocated with Bindless Global Slot: %d\n", slot);
    assert(slot >= 0);

    // 4. Write data from C
    float input_data[4] = { 42.0f, 13.37f, 100.0f, 999.0f };
    res = lgpu_write_buffer(buffer, 0, input_data, sizeof(input_data));
    assert(res == LGPU_SUCCESS);

    // 5. Read back data to C
    float output_data[4] = { 0 };
    res = lgpu_read_buffer(buffer, 0, output_data, sizeof(output_data));
    assert(res == LGPU_SUCCESS);

    printf("Readback Verification:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d] Read: %.2f | Expected: %.2f\n", i, output_data[i], input_data[i]);
        assert(output_data[i] == input_data[i]);
    }

    // 6. Clean up
    lgpu_destroy_buffer(buffer);
    lgpu_destroy_device(device);
    lgpu_destroy_queue(queue);
    lgpu_destroy_instance(instance);

    printf("SUCCESS! Pure C99 program compiled and verified with lgpu.\n");
    return 0;
}
