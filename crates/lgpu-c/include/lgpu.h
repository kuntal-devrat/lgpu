/**
 * @file lgpu.h
 * @brief Universal C99 / C++ / Zig / Odin header for lgpu (Lean GPU).
 * 
 * An ultra-lightweight, bindless-first cross-platform GPU abstraction.
 */

#ifndef LGPU_H
#define LGPU_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Handles === */
typedef struct lgpu_instance_s* lgpu_instance_t;
typedef struct lgpu_device_s*   lgpu_device_t;
typedef struct lgpu_queue_s*    lgpu_queue_t;
typedef struct lgpu_buffer_s*   lgpu_buffer_t;
typedef struct lgpu_shader_s*   lgpu_shader_t;
typedef struct lgpu_compute_pipeline_s* lgpu_compute_pipeline_t;
typedef struct lgpu_command_encoder_s* lgpu_command_encoder_t;
typedef struct lgpu_command_buffer_s*  lgpu_command_buffer_t;

/* === Enums === */
typedef enum lgpu_result {
    LGPU_SUCCESS = 0,
    LGPU_ERROR_NO_ADAPTER = -1,
    LGPU_ERROR_DEVICE_CREATION = -2,
    LGPU_ERROR_BUFFER = -3,
    LGPU_ERROR_SHADER = -4,
    LGPU_ERROR_PIPELINE = -5,
    LGPU_ERROR_COMMAND = -6,
    LGPU_ERROR_GENERIC = -99,
} lgpu_result_t;

typedef enum lgpu_memory_location {
    LGPU_MEMORY_GPU_ONLY = 0,
    LGPU_MEMORY_CPU_TO_GPU = 1,
    LGPU_MEMORY_GPU_TO_CPU = 2,
} lgpu_memory_location_t;

enum lgpu_buffer_usage {
    LGPU_USAGE_TRANSFER_SRC = 1 << 0,
    LGPU_USAGE_TRANSFER_DST = 1 << 1,
    LGPU_USAGE_UNIFORM      = 1 << 2,
    LGPU_USAGE_STORAGE      = 1 << 3,
    LGPU_USAGE_VERTEX       = 1 << 4,
    LGPU_USAGE_INDEX        = 1 << 5,
};
typedef uint32_t lgpu_buffer_usage_t;

/* === API Functions === */

/** Returns the lgpu ABI version */
uint32_t lgpu_version(void);

/** Create a new lgpu instance */
lgpu_result_t lgpu_create_instance(lgpu_instance_t* out_instance);

/** Get adapter count */
uint32_t lgpu_get_adapter_count(lgpu_instance_t instance);

/** Get adapter name */
const char* lgpu_get_adapter_name(lgpu_instance_t instance, uint32_t index);

/** Request logical device and queue */
lgpu_result_t lgpu_create_device(
    lgpu_instance_t instance,
    uint32_t adapter_index,
    lgpu_device_t* out_device,
    lgpu_queue_t* out_queue
);

/** Create a GPU buffer */
lgpu_result_t lgpu_create_buffer(
    lgpu_device_t device,
    uint64_t size,
    lgpu_buffer_usage_t usage,
    lgpu_memory_location_t location,
    lgpu_buffer_t* out_buffer
);

/** Get bindless slot ID for a buffer (-1 if none) */
int32_t lgpu_buffer_get_bindless_id(lgpu_buffer_t buffer);

/** Write data into host-mapped buffer */
lgpu_result_t lgpu_write_buffer(
    lgpu_buffer_t buffer,
    uint64_t offset,
    const void* data,
    size_t size
);

/** Read data from host-mapped buffer */
lgpu_result_t lgpu_read_buffer(
    lgpu_buffer_t buffer,
    uint64_t offset,
    void* data,
    size_t size
);

/** Create a shader module from SPIR-V bytecode */
lgpu_result_t lgpu_create_shader(
    lgpu_device_t device,
    const uint32_t* spirv_words,
    size_t word_count,
    lgpu_shader_t* out_shader
);

/** Create a compute pipeline */
lgpu_result_t lgpu_create_compute_pipeline(
    lgpu_device_t device,
    lgpu_shader_t shader,
    uint32_t buffer_count,
    lgpu_compute_pipeline_t* out_pipeline
);

/** Create a command encoder */
lgpu_result_t lgpu_create_command_encoder(
    lgpu_device_t device,
    lgpu_command_encoder_t* out_encoder
);

/** Record a compute dispatch */
void lgpu_dispatch_compute(
    lgpu_command_encoder_t encoder,
    lgpu_compute_pipeline_t pipeline,
    const lgpu_buffer_t* buffers,
    uint32_t buffer_count,
    uint32_t x, uint32_t y, uint32_t z
);

/** Finish encoding and produce a command buffer */
lgpu_command_buffer_t lgpu_finish_encoder(lgpu_command_encoder_t encoder);

/** Submit command buffer to queue and wait */
lgpu_result_t lgpu_queue_submit(
    lgpu_queue_t queue,
    lgpu_command_buffer_t command_buffer
);

/* === Resource Destruction === */
void lgpu_destroy_buffer(lgpu_buffer_t buffer);
void lgpu_destroy_shader(lgpu_shader_t shader);
void lgpu_destroy_compute_pipeline(lgpu_compute_pipeline_t pipeline);
void lgpu_destroy_device(lgpu_device_t device);
void lgpu_destroy_queue(lgpu_queue_t queue);
void lgpu_destroy_instance(lgpu_instance_t instance);

#ifdef __cplusplus
}
#endif

#endif /* LGPU_H */
