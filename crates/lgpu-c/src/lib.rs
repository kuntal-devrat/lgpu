//! C99 / Zig FFI exports for lgpu

#[no_mangle]
pub extern "C" fn lgpu_version() -> u32 {
    1
}
