#!/usr/bin/env bash
set -euo pipefail

# Release helper script for lgpu maintainers
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.1.3

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version> (e.g. $0 0.1.3)"
    exit 1
fi

NEW_VERSION="$1"

# Validate semver format (X.Y.Z)
if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must follow semantic versioning format X.Y.Z (got: $NEW_VERSION)"
    exit 1
fi

# Ensure clean working directory
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory has uncommitted changes. Please commit or stash first."
    exit 1
fi

# Ensure on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Warning: You are currently on branch '$CURRENT_BRANCH', expected 'main'."
    read -r -p "Continue anyway? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "============================================================"
echo "Preparing release for lgpu v$NEW_VERSION"
echo "============================================================"

# 1. Update version in root Cargo.toml
sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" Cargo.toml

# 2. Update inter-crate dependency versions
sed -i "s/lgpu-hal = { path = \"..\/lgpu-hal\", version = \".*\"/lgpu-hal = { path = \"..\/lgpu-hal\", version = \"$NEW_VERSION\"/" crates/lgpu/Cargo.toml
sed -i "s/lgpu = { path = \"..\/lgpu\", version = \".*\"/lgpu = { path = \"..\/lgpu\", version = \"$NEW_VERSION\"/" crates/lgpu-c/Cargo.toml

# 3. Update Cargo.lock
cargo check --workspace --all-targets

# 4. Verify formatting & lints
echo "Running formatting & lint verification..."
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings

# 5. Commit & Tag
echo "Committing version bump..."
git add Cargo.toml Cargo.lock crates/lgpu/Cargo.toml crates/lgpu-c/Cargo.toml
git commit -m "chore: release v$NEW_VERSION"

echo "Creating tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

echo "============================================================"
echo "Release v$NEW_VERSION prepared successfully!"
echo "To publish, push the commit and tag to GitHub:"
echo "  git push origin main && git push origin v$NEW_VERSION"
echo "This will trigger the Release & Publish workflow."
echo "============================================================"
