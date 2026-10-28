#!/usr/bin/env bash
set -euo pipefail

# Generates C99 headers using cbindgen if available
if command -v cbindgen &> /dev/null; then
    echo "Running cbindgen for lgpu-c..."
    cbindgen --config crates/lgpu-c/cbindgen.toml --crate lgpu-c --output crates/lgpu-c/include/lgpu_generated.h
    echo "C99 bindings generated at crates/lgpu-c/include/lgpu_generated.h"
else
    echo "cbindgen not installed. Install with: cargo install cbindgen"
fi
