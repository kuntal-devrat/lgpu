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
typedef struct lgpu_timeline_semaphore_s* lgpu_timeline_semaphore_t;
typedef struct lgpu_texture_s*         lgpu_texture_t;
typedef struct lgpu_sampler_s*         lgpu_sampler_t;
typedef struct lgpu_graphics_pipeline_s* lgpu_graphics_pipeline_t;
typedef struct lgpu_bindless_compute_pipeline_s* lgpu_bindless_compute_pipeline_t;
typedef struct lgpu_surface_s*         lgpu_surface_t;
typedef struct lgpu_surface_texture_s* lgpu_surface_texture_t;

typedef struct lgpu_adapter_info {
    char name[256];
    uint32_t backend;
    uint32_t device_type;
    uint32_t vendor_id;
    uint32_t device_id;
    uint8_t supports_dynamic_rendering;
    uint8_t supports_timeline_semaphores;
    uint8_t supports_descriptor_indexing;
    uint8_t _padding;
    uint32_t max_compute_workgroup_invocations;
    uint32_t max_compute_workgroup_size[3];
} lgpu_adapter_info_t;


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

typedef enum lgpu_texture_format {
    LGPU_FORMAT_RGBA8_UNORM = 0,
    LGPU_FORMAT_RGBA8_SRGB = 1,
    LGPU_FORMAT_BGRA8_UNORM = 2,
    LGPU_FORMAT_BGRA8_SRGB = 3,
    LGPU_FORMAT_R32_FLOAT = 4,
    LGPU_FORMAT_RGBA16_FLOAT = 5,
    LGPU_FORMAT_RGBA32_FLOAT = 6,
    LGPU_FORMAT_DEPTH32_FLOAT = 7,
} lgpu_texture_format_t;

enum lgpu_texture_usage {
    LGPU_TEX_USAGE_COPY_SRC          = 1 << 0,
    LGPU_TEX_USAGE_COPY_DST          = 1 << 1,
    LGPU_TEX_USAGE_TEXTURE_BINDING   = 1 << 2,
    LGPU_TEX_USAGE_STORAGE_BINDING   = 1 << 3,
    LGPU_TEX_USAGE_RENDER_ATTACHMENT = 1 << 4,
};
typedef uint32_t lgpu_texture_usage_t;

typedef enum lgpu_filter_mode {
    LGPU_FILTER_NEAREST = 0,
    LGPU_FILTER_LINEAR = 1,
} lgpu_filter_mode_t;

typedef enum lgpu_address_mode {
    LGPU_ADDRESS_CLAMP_TO_EDGE = 0,
    LGPU_ADDRESS_REPEAT = 1,
    LGPU_ADDRESS_MIRROR_REPEAT = 2,
} lgpu_address_mode_t;

typedef enum lgpu_present_mode {
    LGPU_PRESENT_MODE_IMMEDIATE = 0,
    LGPU_PRESENT_MODE_MAILBOX = 1,
    LGPU_PRESENT_MODE_FIFO = 2,
} lgpu_present_mode_t;

typedef struct lgpu_surface_config {
    uint32_t width;
    uint32_t height;
    lgpu_texture_format_t format;
    lgpu_texture_usage_t usage;
    lgpu_present_mode_t present_mode;
} lgpu_surface_config_t;


/* === API Functions === */

/** Returns the lgpu ABI version */
uint32_t lgpu_version(void);

/** Create a new lgpu instance */
lgpu_result_t lgpu_create_instance(lgpu_instance_t* out_instance);

/** Get adapter count */
uint32_t lgpu_get_adapter_count(lgpu_instance_t instance);

/** Get adapter name */
const char* lgpu_get_adapter_name(lgpu_instance_t instance, uint32_t index);

/** Get detailed adapter hardware capabilities */
lgpu_result_t lgpu_get_adapter_info(
    lgpu_instance_t instance,
    uint32_t index,
    lgpu_adapter_info_t* out_info
);

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

/** Create a GPU buffer initialized with host memory */
lgpu_result_t lgpu_create_buffer_init(
    lgpu_device_t device,
    const void* data,
    size_t size,
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

/** Submit multiple command buffers atomically and wait for completion */
lgpu_result_t lgpu_queue_submit_batches(
    lgpu_queue_t queue,
    const lgpu_command_buffer_t* command_buffers,
    uint32_t count
);

/** Asynchronously submit command buffers with timeline semaphore dependencies */
lgpu_result_t lgpu_queue_submit_async(
    lgpu_queue_t queue,
    const lgpu_command_buffer_t* command_buffers,
    uint32_t count,
    const lgpu_timeline_semaphore_t* wait_semaphores,
    const uint64_t* wait_values,
    uint32_t wait_count,
    const lgpu_timeline_semaphore_t* signal_semaphores,
    const uint64_t* signal_values,
    uint32_t signal_count
);

/** Create a timeline semaphore */
lgpu_result_t lgpu_create_timeline_semaphore(
    lgpu_device_t device,
    uint64_t initial_value,
    lgpu_timeline_semaphore_t* out_semaphore
);

/** Get current value of timeline semaphore */
lgpu_result_t lgpu_get_timeline_semaphore_value(
    lgpu_timeline_semaphore_t semaphore,
    uint64_t* out_value
);

/** Wait on host for timeline semaphore to reach value (timeout in nanoseconds, UINT64_MAX for infinite) */
lgpu_result_t lgpu_wait_timeline_semaphore(
    lgpu_timeline_semaphore_t semaphore,
    uint64_t value,
    uint64_t timeout_nanos
);

/** Signal timeline semaphore from host with value */
lgpu_result_t lgpu_signal_timeline_semaphore(
    lgpu_timeline_semaphore_t semaphore,
    uint64_t value
);

/** Record global GPU execution and memory barrier */
void lgpu_encoder_memory_barrier(lgpu_command_encoder_t encoder);

/** Record GPU buffer memory barrier */
void lgpu_encoder_buffer_barrier(
    lgpu_command_encoder_t encoder,
    lgpu_buffer_t buffer
);

/** Create a GPU texture */
lgpu_result_t lgpu_create_texture(
    lgpu_device_t device,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    lgpu_texture_format_t format,
    lgpu_texture_usage_t usage,
    lgpu_texture_t* out_texture
);

/** Get bindless slot ID for a texture (-1 if none) */
int32_t lgpu_texture_get_bindless_id(lgpu_texture_t texture);

/** Create a GPU sampler */
lgpu_result_t lgpu_create_sampler(
    lgpu_device_t device,
    lgpu_filter_mode_t min_filter,
    lgpu_filter_mode_t mag_filter,
    lgpu_filter_mode_t mipmap_filter,
    lgpu_address_mode_t address_u,
    lgpu_address_mode_t address_v,
    lgpu_address_mode_t address_w,
    lgpu_sampler_t* out_sampler
);

/** Get bindless slot ID for a sampler (-1 if none) */
int32_t lgpu_sampler_get_bindless_id(lgpu_sampler_t sampler);

/** Create a graphics pipeline with optional depth testing */
lgpu_result_t lgpu_create_graphics_pipeline(
    lgpu_device_t device,
    lgpu_shader_t shader,
    const char* vs_entry,
    const char* fs_entry,
    lgpu_texture_format_t color_format,
    int32_t depth_format, /* -1 if no depth attachment */
    bool depth_test,
    bool depth_write,
    lgpu_graphics_pipeline_t* out_pipeline
);

/** Clear color and depth attachments */
void lgpu_clear_and_render_with_depth(
    lgpu_command_encoder_t encoder,
    lgpu_texture_t color_texture,
    lgpu_texture_t depth_texture,
    const float clear_color[4],
    float clear_depth
);

/** Draw primitives with depth attachment */
void lgpu_draw_with_depth(
    lgpu_command_encoder_t encoder,
    lgpu_texture_t color_texture,
    lgpu_texture_t depth_texture,
    lgpu_graphics_pipeline_t pipeline,
    uint32_t vertex_count,
    const void* push_constants,
    size_t push_constants_size,
    const float clear_color[4],
    float clear_depth
);

/** Clear color texture */
void lgpu_clear_and_render_texture(
    lgpu_command_encoder_t encoder,
    lgpu_texture_t texture,
    const float clear_color[4]
);

/** Draw primitives to color texture */
void lgpu_draw(
    lgpu_command_encoder_t encoder,
    lgpu_texture_t texture,
    lgpu_graphics_pipeline_t pipeline,
    uint32_t vertex_count,
    const void* push_constants,
    size_t push_constants_size,
    const float clear_color[4]
);

/** Draw single 3-vertex triangle to color texture */
void lgpu_render_triangle(
    lgpu_command_encoder_t encoder,
    lgpu_texture_t texture,
    lgpu_graphics_pipeline_t pipeline,
    const float clear_color[4]
);

/** Copy texture memory to buffer */
void lgpu_copy_texture_to_buffer(
    lgpu_command_encoder_t encoder,
    lgpu_texture_t texture,
    lgpu_buffer_t buffer
);

/** Copy buffer to buffer on GPU */
void lgpu_copy_buffer_to_buffer(
    lgpu_command_encoder_t encoder,
    lgpu_buffer_t source,
    uint64_t source_offset,
    lgpu_buffer_t destination,
    uint64_t destination_offset,
    uint64_t size
);

/** Copy buffer pixel data to texture */
void lgpu_copy_buffer_to_texture(
    lgpu_command_encoder_t encoder,
    lgpu_buffer_t buffer,
    lgpu_texture_t texture
);

/** Record an indirect compute dispatch */
lgpu_result_t lgpu_dispatch_indirect(
    lgpu_command_encoder_t encoder,
    lgpu_bindless_compute_pipeline_t pipeline,
    lgpu_buffer_t indirect_buffer,
    uint64_t indirect_offset,
    const void* push_constants,
    size_t push_constants_size
);

/** Draw indexed primitives */
void lgpu_draw_indexed(
    lgpu_command_encoder_t encoder,
    lgpu_texture_t texture,
    lgpu_graphics_pipeline_t pipeline,
    lgpu_buffer_t index_buffer,
    uint32_t index_count,
    uint32_t instance_count,
    uint32_t first_index,
    int32_t vertex_offset,
    uint32_t first_instance,
    const void* push_constants,
    size_t push_constants_size,
    const float clear_color[4]
);

/** Directly upload CPU memory to GPU texture */
lgpu_result_t lgpu_device_write_texture(
    lgpu_device_t device,
    lgpu_texture_t texture,
    const uint8_t* data,
    size_t data_len
);

/** Get texture width */
uint32_t lgpu_texture_get_width(lgpu_texture_t texture);

/** Get texture height */
uint32_t lgpu_texture_get_height(lgpu_texture_t texture);

/** Get texture format */
int32_t lgpu_texture_get_format(lgpu_texture_t texture);

/** Create a bindless compute pipeline */
lgpu_result_t lgpu_create_bindless_compute_pipeline(
    lgpu_device_t device,
    lgpu_shader_t shader,
    lgpu_bindless_compute_pipeline_t* out_pipeline
);

/** Record a bindless compute dispatch */
lgpu_result_t lgpu_dispatch_bindless_compute(
    lgpu_command_encoder_t encoder,
    lgpu_bindless_compute_pipeline_t pipeline,
    const void* push_constants,
    size_t push_constants_size,
    uint32_t x, uint32_t y, uint32_t z
);

/** Wait for all device operations to complete */
lgpu_result_t lgpu_device_wait_idle(lgpu_device_t device);

/** Create a headless surface */
lgpu_result_t lgpu_create_surface_headless(
    lgpu_instance_t instance,
    lgpu_surface_t* out_surface
);

/** Create an Xlib window surface */
lgpu_result_t lgpu_create_surface_from_xlib(
    lgpu_instance_t instance,
    void* dpy,
    unsigned long window,
    lgpu_surface_t* out_surface
);

/** Create an XCB window surface */
lgpu_result_t lgpu_create_surface_from_xcb(
    lgpu_instance_t instance,
    void* connection,
    uint32_t window,
    lgpu_surface_t* out_surface
);

/** Create a Wayland window surface */
lgpu_result_t lgpu_create_surface_from_wayland(
    lgpu_instance_t instance,
    void* display,
    void* surface,
    lgpu_surface_t* out_surface
);

/** Create a Win32 window surface */
lgpu_result_t lgpu_create_surface_from_win32(
    lgpu_instance_t instance,
    void* hinstance,
    void* hwnd,
    lgpu_surface_t* out_surface
);

/** Configure surface width, height, format, and present mode */
lgpu_result_t lgpu_surface_configure(
    lgpu_surface_t surface,
    lgpu_device_t device,
    const lgpu_surface_config_t* config
);

/** Acquire next swapchain surface texture */
lgpu_result_t lgpu_surface_get_current_texture(
    lgpu_surface_t surface,
    lgpu_surface_texture_t* out_surface_texture
);

/** Get underlying texture handle from surface texture */
lgpu_texture_t lgpu_surface_texture_get_texture(
    lgpu_surface_texture_t surface_texture
);

/** Submit command buffer and present swapchain image */
lgpu_result_t lgpu_queue_submit_surface(
    lgpu_queue_t queue,
    lgpu_command_buffer_t command_buffer,
    lgpu_surface_texture_t surface_texture
);

/** Free surface texture wrapper */
void lgpu_surface_texture_destroy(lgpu_surface_texture_t surface_texture);

/** Destroy window surface */
void lgpu_destroy_surface(lgpu_surface_t surface);

/* === Resource Destruction === */
void lgpu_destroy_command_encoder(lgpu_command_encoder_t encoder);
void lgpu_destroy_command_buffer(lgpu_command_buffer_t command_buffer);
void lgpu_destroy_buffer(lgpu_buffer_t buffer);
void lgpu_destroy_texture(lgpu_texture_t texture);
void lgpu_destroy_sampler(lgpu_sampler_t sampler);
void lgpu_destroy_shader(lgpu_shader_t shader);
void lgpu_destroy_compute_pipeline(lgpu_compute_pipeline_t pipeline);
void lgpu_destroy_bindless_compute_pipeline(lgpu_bindless_compute_pipeline_t pipeline);
void lgpu_destroy_graphics_pipeline(lgpu_graphics_pipeline_t pipeline);
void lgpu_destroy_timeline_semaphore(lgpu_timeline_semaphore_t semaphore);
void lgpu_destroy_device(lgpu_device_t device);
void lgpu_destroy_queue(lgpu_queue_t queue);
void lgpu_destroy_instance(lgpu_instance_t instance);

#ifdef __cplusplus
}
#endif

#endif /* LGPU_H */

