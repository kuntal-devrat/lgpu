/**
 * @file main.cpp
 * @brief Modern C++17 example using lgpu C ABI via lgpu.h.
 * 
 * Demonstrates:
 *  - Adapter discovery and hardware capabilities query
 *  - RAII resource management for handles
 *  - Buffer creation, host initialization, and GPU-to-GPU memory copies
 *  - Command encoding and queue execution
 *  - Safe memory readback and verification
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cassert>

#include "../../crates/lgpu-c/include/lgpu.h"

// Custom deleters for modern C++ RAII
struct InstanceDeleter { void operator()(lgpu_instance_s* p) const { if (p) lgpu_destroy_instance(p); } };
struct DeviceDeleter   { void operator()(lgpu_device_s* p)   const { if (p) lgpu_destroy_device(p); } };
struct QueueDeleter    { void operator()(lgpu_queue_s* p)    const { if (p) lgpu_destroy_queue(p); } };
struct BufferDeleter   { void operator()(lgpu_buffer_s* p)   const { if (p) lgpu_destroy_buffer(p); } };

using UniqueInstance = std::unique_ptr<lgpu_instance_s, InstanceDeleter>;
using UniqueDevice   = std::unique_ptr<lgpu_device_s, DeviceDeleter>;
using UniqueQueue    = std::unique_ptr<lgpu_queue_s, QueueDeleter>;
using UniqueBuffer   = std::unique_ptr<lgpu_buffer_s, BufferDeleter>;

int main() {
    std::cout << "=== lgpu C++17 Example ===" << std::endl;
    std::cout << "lgpu ABI Version: " << lgpu_version() << std::endl;

    // 1. Initialize Instance
    lgpu_instance_t raw_instance = nullptr;
    lgpu_result_t res = lgpu_create_instance(&raw_instance);
    if (res != LGPU_SUCCESS || !raw_instance) {
        std::cerr << "Failed to create Vulkan instance (error " << res << ")" << std::endl;
        return 1;
    }
    UniqueInstance instance(raw_instance);

    // 2. Query Adapters
    uint32_t adapter_count = lgpu_get_adapter_count(instance.get());
    std::cout << "Found " << adapter_count << " GPU adapter(s)." << std::endl;
    if (adapter_count == 0) {
        std::cerr << "No compatible Vulkan 1.3 GPU found." << std::endl;
        return 1;
    }

    lgpu_adapter_info_t info{};
    lgpu_get_adapter_info(instance.get(), 0, &info);
    std::cout << "Using GPU [0]: " << info.name << std::endl;
    std::cout << "  Dynamic Rendering: " << (info.supports_dynamic_rendering ? "Supported" : "No") << std::endl;
    std::cout << "  Timeline Semaphores: " << (info.supports_timeline_semaphores ? "Supported" : "No") << std::endl;
    std::cout << "  Descriptor Indexing: " << (info.supports_descriptor_indexing ? "Supported" : "No") << std::endl;

    // 3. Create Device & Queue
    lgpu_device_t raw_device = nullptr;
    lgpu_queue_t raw_queue = nullptr;
    res = lgpu_create_device(instance.get(), 0, &raw_device, &raw_queue);
    if (res != LGPU_SUCCESS) {
        std::cerr << "Failed to create logical device." << std::endl;
        return 1;
    }
    UniqueDevice device(raw_device);
    UniqueQueue queue(raw_queue);

    // 4. Create Source Buffer with Initial Host Data
    const std::vector<float> input_data = { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };
    size_t data_bytes = input_data.size() * sizeof(float);

    lgpu_buffer_t raw_src_buf = nullptr;
    res = lgpu_create_buffer_init(
        device.get(),
        input_data.data(),
        data_bytes,
        LGPU_USAGE_STORAGE | LGPU_USAGE_TRANSFER_SRC,
        LGPU_MEMORY_CPU_TO_GPU,
        &raw_src_buf
    );
    assert(res == LGPU_SUCCESS);
    UniqueBuffer src_buffer(raw_src_buf);

    int32_t src_slot = lgpu_buffer_get_bindless_id(src_buffer.get());
    std::cout << "Created source buffer initialized with " << input_data.size() 
              << " floats. Bindless slot: " << src_slot << std::endl;

    // 5. Create Destination Buffer
    lgpu_buffer_t raw_dst_buf = nullptr;
    res = lgpu_create_buffer(
        device.get(),
        data_bytes,
        LGPU_USAGE_STORAGE | LGPU_USAGE_TRANSFER_DST,
        LGPU_MEMORY_GPU_TO_CPU,
        &raw_dst_buf
    );
    assert(res == LGPU_SUCCESS);
    UniqueBuffer dst_buffer(raw_dst_buf);

    // 6. Record Copy Command & Submit
    lgpu_command_encoder_t encoder = nullptr;
    res = lgpu_create_command_encoder(device.get(), &encoder);
    assert(res == LGPU_SUCCESS);

    lgpu_copy_buffer_to_buffer(encoder, src_buffer.get(), 0, dst_buffer.get(), 0, data_bytes);

    lgpu_command_buffer_t cmd = lgpu_finish_encoder(encoder);
    assert(cmd != nullptr);

    res = lgpu_queue_submit(queue.get(), cmd);
    assert(res == LGPU_SUCCESS);

    res = lgpu_device_wait_idle(device.get());
    assert(res == LGPU_SUCCESS);

    // 7. Read Back and Verify
    std::vector<float> output_data(input_data.size(), 0.0f);
    res = lgpu_read_buffer(dst_buffer.get(), 0, output_data.data(), data_bytes);
    assert(res == LGPU_SUCCESS);

    std::cout << "GPU buffer copy verified: [";
    for (size_t i = 0; i < output_data.size(); ++i) {
        std::cout << output_data[i] << (i + 1 < output_data.size() ? ", " : "");
        assert(output_data[i] == input_data[i]);
    }
    std::cout << "]" << std::endl;

    std::cout << "\nC++17 example completed successfully." << std::endl;
    return 0;
}
