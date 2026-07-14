use std::env;

fn enabled(feature: &str) -> bool {
    env::var_os(format!("CARGO_FEATURE_{feature}")).is_some()
}

fn emit(cfg: &str) {
    println!("cargo:rustc-cfg={cfg}");
}

fn main() {
    let cfgs = [
        "web",
        "webgpu",
        "send_sync",
        "supports_64bit_atomics",
        "std",
        "no_std",
        "native",
        "Emscripten",
        "wgpu_core",
        "custom",
        "naga",
        "webgl",
        "dx12",
        "metal",
        "vulkan",
        "drm",
        "gles",
        "noop",
        "static_dxc",
    ];
    for cfg in cfgs {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }
    println!(
        "cargo:rustc-check-cfg=cfg(feature, values(\"glsl\", \"naga-ir\", \"noop\", \"spirv\"))"
    );

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let target_atomics = env::var("CARGO_CFG_TARGET_HAS_ATOMIC").unwrap_or_default();
    let web = target_arch == "wasm32" && target_os != "emscripten" && enabled("WEB");
    let send_sync = enabled("FRAGILE_SEND_SYNC_NON_ATOMIC_WASM")
        && !target_features
            .split(',')
            .any(|feature| feature == "atomics");
    let uses_std = enabled("STD")
        || send_sync
        || env::var("CARGO_CFG_PANIC").is_ok_and(|panic| panic == "unwind");

    if web {
        emit("web");
    }
    if web && enabled("WEBGPU") {
        emit("webgpu");
    }
    if send_sync {
        emit("send_sync");
    }
    if target_atomics.split(',').any(|width| width == "64") {
        emit("supports_64bit_atomics");
    }
    emit(if uses_std { "std" } else { "no_std" });
}
