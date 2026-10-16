# Contributing to `lgpu` (Lean GPU)

Thank you for your interest in contributing to `lgpu`!  
`lgpu` is an ultra-lightweight, bindless-first, cross-platform GPU abstraction designed for modern systems.

Our core guiding principles are:
1. **Sub-3-second clean compile times**: Keep dependencies minimal (< 10 total crates).
2. **Lean binary footprint**: Release binaries should stay under 500 KB.
3. **Zero sandbox bloat**: Embrace modern GPU hardware as it exists today (dynamic rendering, bindless descriptor indexing, timeline semaphores).
4. **First-class multi-language support**: First-class Safe Rust, C99, C++, and Zig interfaces.

---

## 📁 Repository Structure

```
lgpu/
├── crates/
│   ├── lgpu/          # Safe, ergonomic high-level Rust API
│   ├── lgpu-hal/      # Ultra-low-overhead Hardware Abstraction Layer (Vulkan, Metal, D3D12)
│   └── lgpu-c/        # C-ABI bindings and single-header C99/Zig export (include/lgpu.h)
├── examples/          # Real-world showcases and test programs
│   ├── 01_headless_compute/
│   ├── 02_clear_screen/
│   ├── 03_bindless_triangle/
│   ├── 04_bindless_100k/
│   └── 05_window_shader/
└── tests/             # Comprehensive integration test suites
```

---

## 🛠️ Development Setup & Workflow

### Prerequisites
- **Rust Toolchain**: 1.80+ (stable).
- **Vulkan Driver**: Vulkan 1.2+ with Dynamic Rendering and Descriptor Indexing support (standard on modern Linux Mesa, Intel, NVIDIA, and AMD drivers).

### Common Commands

```bash
# Verify all targets build cleanly
cargo check --workspace --all-targets

# Run the complete test suite
cargo test --all

# Run linter
cargo clippy --workspace --all-targets

# Run an interactive example
cargo run --release -p example-window-shader
```

---

## 📐 Coding Guidelines

1. **Zero Compiler Warnings**: All code must compile with 0 warnings on stable Rust.
2. **Error Propagation**: Never swallow errors silently. Always return `Result<T, Error>` or `Result<T, HalError>` rather than panicking or executing a no-op return.
3. **Thread Safety & Invariants**:
   - `Device`, `Queue`, `Buffer`, `Texture`, and `Sampler` must be `Send + Sync`.
   - Protect concurrent internal state (e.g. host-mapped pointers, texture layouts) with synchronization primitives (`std::sync::Mutex`).
4. **C FFI Parity**:
   - Any new feature exposed to C in `crates/lgpu-c/src/lib.rs` must also be reflected in `crates/lgpu-c/include/lgpu.h`.
   - Maintain ABI stability: use fixed-width types (`uint8_t`, `uint32_t`, `uint64_t`) instead of implementation-dependent types like `bool`.
   - Every `extern "C"` function must include a rustdoc `/// # Safety` section.
5. **Backwards Compatibility**:
   - Do not break existing API calls without strong rationale and clear deprecation paths.

---

## 🚀 Submitting a Pull Request

1. **Fork & Branch**: Create a feature branch off `main` (`git checkout -b feature/my-feature`).
2. **Add Tests**: Accompany any new feature or bug fix with integration tests under `tests/`.
3. **Verify**: Ensure `cargo check --workspace --all-targets` and `cargo test --all` pass.
4. **Changelog**: Update `CHANGELOG.md` under `[Unreleased]` with your changes.
5. **Open PR**: Submit your PR with a clear description of the problem, solution, and test results.
