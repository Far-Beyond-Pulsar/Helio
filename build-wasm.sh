#!/usr/bin/env bash
# build-wasm.sh
#
# Pre-builds all helio-wasm-examples WASM demos and runs wasm-bindgen on each.
# Outputs land in target/wasm-prebuilt/<name>/ where build.rs copies them from.
#
# Usage:
#   ./build-wasm.sh            # build all examples
#   ./build-wasm.sh sdf_blend  # build one example
#
# After this finishes, run:
#   cargo run --release --bin helio-wasm-server -p helio-wasm-app

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WASM_TARGET="wasm32-unknown-unknown"

declare -a ALL_EXAMPLES=(
    render_v2_basic
    render_v2_sky
    sdf_blend
    sdf_boolean
    sdf_demo
    sdf_grid
    sdf_morph
    sdf_move
    sdf_multi
    sdf_pulse
    sdf_terrain
)

# Filter to a specific example if one was given on the command line
if [[ $# -gt 0 ]]; then
    FOUND=0
    for ex in "${ALL_EXAMPLES[@]}"; do
        [[ "$ex" == "$1" ]] && FOUND=1
    done
    if [[ $FOUND -eq 0 ]]; then
        echo "Unknown example '$1'. Valid: ${ALL_EXAMPLES[*]}" >&2
        exit 1
    fi
    EXAMPLES=("$1")
else
    EXAMPLES=("${ALL_EXAMPLES[@]}")
fi

FAILED=()

for name in "${EXAMPLES[@]}"; do
    echo ""
    echo "════ $name ════"

    target_dir="$SCRIPT_DIR/target/wasm-build/ex/$name"
    out_dir="$SCRIPT_DIR/target/wasm-prebuilt/$name"
    mkdir -p "$target_dir" "$out_dir"

    echo "  cargo build --release --lib -p helio-wasm-examples --features $name"
    if ! cargo build --release --lib \
        -p helio-wasm-examples \
        --target "$WASM_TARGET" \
        --no-default-features \
        --features "$name" \
        --target-dir "$target_dir"; then
        echo "  FAILED cargo build for $name" >&2
        FAILED+=("$name")
        continue
    fi

    wasm_path="$target_dir/$WASM_TARGET/release/helio_wasm_examples.wasm"
    if [[ ! -f "$wasm_path" ]]; then
        echo "  FAILED: wasm not found at $wasm_path" >&2
        FAILED+=("$name")
        continue
    fi
    wasm_size=$(wc -c < "$wasm_path")
    echo "  wasm OK ($wasm_size bytes)"

    echo "  wasm-bindgen --out-dir $out_dir --target web"
    if ! wasm-bindgen "$wasm_path" --out-dir "$out_dir" --target web; then
        echo "  FAILED wasm-bindgen for $name" >&2
        FAILED+=("$name")
        continue
    fi

    echo "  OK -> $out_dir"
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED examples: ${FAILED[*]}" >&2
    exit 1
else
    echo "All examples built successfully."
    echo ""
    echo "Now run:"
    echo "  cargo run --release --bin helio-wasm-server -p helio-wasm-app"
fi
