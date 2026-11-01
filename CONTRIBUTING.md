# Contributing to `lgpu`

`lgpu` is a small Vulkan 1.3 wrapper for Rust and C99. Patches that keep the dependency
set small, the build fast and the API explicit are the easiest to review.

## Repository layout

```text
lgpu/
├── crates/
│   ├── lgpu/          # high-level Rust API
│   ├── lgpu-hal/      # Vulkan 1.3 hardware abstraction layer
│   └── lgpu-c/        # C ABI bindings and lgpu.h
├── examples/          # examples, including c_example, cpp_example and zig_example
└── docs/              # architecture and safety notes
```

## Setup

- Rust 1.80 or newer (stable).
- A Vulkan 1.3 driver with dynamic rendering (`VK_KHR_dynamic_rendering`) and descriptor
  indexing. This covers current Mesa, AMD, Intel and NVIDIA drivers, plus MoltenVK on
  macOS. Without a device, the integration tests skip themselves.

Commands used in CI:

```bash
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets --all-features
cargo fmt --all -- --check
cargo run --release -p example-voxel-renderer
```

## Guidelines

1. Code should build without warnings on stable Rust, including `cargo clippy`.
2. Return `Result<T, Error>` or `Result<T, HalError>` with a useful message instead of
   panicking.
3. Keep thread safety as documented: `Device`, `Queue`, `Buffer`, `Texture` and `Sampler`
   are `Send + Sync`; `CommandEncoder` is `Send` but not `Sync`. Host mapped writes and
   image layout transitions must stay behind their mutexes.
4. When you add a function to `crates/lgpu-c/src/lib.rs`, update
   `crates/lgpu-c/include/lgpu.h` as well and use fixed width types for the ABI.
5. Add an integration test for any non-trivial fix or feature.

## Pull requests

1. Branch from `main`.
2. Make sure the four commands above pass.
3. Describe the change and, when it matters, the GPU and driver you tested on.

## CI and releases

Pull requests and pushes to `main` run formatting, clippy with `-D warnings`, tests on
Linux, Windows and macOS, the C and C++ example builds, and the doc build. They never
publish crates or create releases.

Publishing is a separate workflow. It runs when a maintainer pushes a version tag such as
`v0.1.5`, or starts it manually with `workflow_dispatch`. A crates.io check skips versions
that are already published, so re-running the workflow is safe.
