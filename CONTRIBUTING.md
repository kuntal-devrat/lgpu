# Contributing to `lgpu`

Thank you for your interest in contributing to `lgpu`!  
`lgpu` is a lightweight, bindless-first Vulkan 1.3 wrapper for Rust and C99 designed for fast clean builds and minimal boilerplate.

---

## Repository Structure

```text
lgpu/
├── crates/
│   ├── lgpu/          # Safe, ergonomic high-level Rust API
│   ├── lgpu-hal/      # Vulkan 1.3 hardware abstraction layer
│   └── lgpu-c/        # C-ABI bindings and single-header C99 export (include/lgpu.h)
├── examples/          # Hardware examples and demos
│   ├── 01_headless_compute/
│   ├── 02_clear_screen/
│   ├── 03_bindless_triangle/
│   ├── 04_bindless_100k/
│   ├── 05_window_shader/
│   ├── 06_voxel_renderer/
│   └── c_example/
└── docs/              # Technical documentation and architecture guides
```

---

## Development Setup

### Prerequisites
- **Rust**: 1.80+ (stable).
- **Vulkan Driver**: Vulkan 1.3+ driver supporting Dynamic Rendering (`VK_KHR_dynamic_rendering`) and Descriptor Indexing (standard on modern Linux Mesa, Intel, NVIDIA, AMD drivers, and macOS via MoltenVK).

### Common Commands

```bash
# Verify all targets build cleanly
cargo check --workspace --all-targets

# Run the complete test suite
cargo test --workspace

# Run linter
cargo clippy --workspace --all-targets --all-features

# Run format check
cargo fmt --all -- --check

# Run the voxel demo
cargo run --release -p example-voxel-renderer
```

---

## Guidelines

1. **Zero Compiler Warnings**: Code should compile with 0 warnings on stable Rust with `cargo clippy`.
2. **Error Propagation**: Return descriptive `Result<T, Error>` or `Result<T, HalError>` rather than panicking.
3. **Thread Safety & Invariants**:
   - `Device`, `Queue`, `Buffer`, `Texture`, and `Sampler` must implement `Send + Sync`.
   - `CommandEncoder` is `Send` but not `Sync`.
   - Host-mapped memory writes and concurrent texture transitions must be protected with appropriate synchronization.
4. **C FFI Parity**:
   - Features added to `crates/lgpu-c/src/lib.rs` should be reflected in `crates/lgpu-c/include/lgpu.h`.
   - Use ABI-stable fixed-width types (`uint8_t`, `uint32_t`, `uint64_t`).
5. **Tests**: Accompany any non-trivial fix or feature with an integration test.

---

## Submitting a Pull Request

1. Create a feature branch off `main` (`git checkout -b feature/my-feature`).
2. Verify that `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --all-features`, and `cargo test --workspace` all pass.
3. Open a PR with a description of the changes and any relevant hardware/driver details.

---

## CI & Release Lifecycle

To protect repository stability and prevent unintended crate publishing as contributors submit code:

1. **Pull Requests & Main Branch CI (`ci.yml`)**:
   - Every pull request and push to `main` triggers automated CI validation.
   - Runs formatting checks, Clippy lints (`-D warnings`), cross-platform test matrices (Linux Mesa/Lavapipe, Windows, macOS), doc builds, and dry-run package checks (`cargo publish --dry-run`).
   - **No Releases**: PRs and merges to `main` **never** publish to crates.io or create GitHub releases. Contributions can be merged freely without version conflicts.

2. **Releases & Publishing (`release.yml`)**:
   - Releases are completely decoupled into a dedicated workflow.
   - Triggered exclusively when a repository maintainer pushes a semantic version tag (`git tag v0.X.Y && git push origin v0.X.Y`) or manually runs the release workflow with `workflow_dispatch`.
   - Built-in crates.io idempotency checks automatically detect whether crate versions are already live on crates.io, skipping published crates and preventing pipeline failures.

