// build.rs for helio-wasm-app
//
// Does NOT spawn any cargo processes (that causes a jobserver deadlock).
// All WASM compilation is done by build-wasm.ps1 (workspace root) which
// writes bindgen output to target/wasm-prebuilt/<name>/.
//
// This script only:
//   1. Copies pre-built JS + WASM into OUT_DIR/<name>/ so that
//      include_bytes! in server.rs can find them.
//   2. Registers rerun-if-changed on those pre-built files.
//
// Workflow:
//   1. ./build-wasm.ps1          (builds WASM outside cargo, no deadlock)
//   2. cargo run --release --bin helio-wasm-server -p helio-wasm-app

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cargo_out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Pre-built artifacts live in <workspace>/target/wasm-prebuilt/<name>/
    // crates/helio-wasm-app  ->  crates  ->  workspace root
    let workspace_root = manifest_dir
        .parent().expect("crate inside crates/")
        .parent().expect("crates/ inside workspace root");
    let prebuilt = workspace_root.join("target").join("wasm-prebuilt");

    println!("cargo:warning=[build.rs] prebuilt dir = {:?}", prebuilt);
    println!("cargo:warning=[build.rs] OUT_DIR      = {:?}", cargo_out_dir);

    let examples: &[(&str, &str)] = &[
        ("render_v2_basic", "Basic Render"),
        ("render_v2_sky",   "Volumetric Sky"),
        ("sdf_blend",       "SDF Blend"),
        ("sdf_boolean",     "SDF Boolean"),
        ("sdf_demo",        "SDF Demo"),
        ("sdf_grid",        "SDF Grid Wave"),
        ("sdf_morph",       "SDF Morph"),
        ("sdf_move",        "SDF Move"),
        ("sdf_multi",       "SDF Multi"),
        ("sdf_pulse",       "SDF Pulse"),
        ("sdf_terrain",     "SDF Terrain"),
    ];

    for (name, _title) in examples {
        let src_dir = prebuilt.join(name);
        let dst_dir = cargo_out_dir.join(name);
        std::fs::create_dir_all(&dst_dir)
            .unwrap_or_else(|e| panic!("create OUT_DIR/{}: {}", name, e));

        for filename in &["helio_wasm_examples.js", "helio_wasm_examples_bg.wasm"] {
            let src = src_dir.join(filename);
            if !src.exists() {
                panic!(
                    "\n\
                     [build.rs] Missing pre-built artifact:\n  {:?}\n\n\
                     Run `./build-wasm.ps1` from the workspace root first,\n\
                     then re-run `cargo run --release --bin helio-wasm-server -p helio-wasm-app`.\n",
                    src
                );
            }
            let dst = dst_dir.join(filename);
            std::fs::copy(&src, &dst)
                .unwrap_or_else(|e| panic!("copy {} for {}: {}", filename, name, e));

            // Re-trigger this script whenever the pre-built file changes.
            println!("cargo:rerun-if-changed={}", src.display());
        }
        println!("cargo:warning=[build.rs] {} OK", name);
    }

    println!("cargo:warning=[build.rs] all examples copied into OUT_DIR -- done");
}
