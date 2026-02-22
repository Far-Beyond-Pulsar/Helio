fn main() {
    // Recompile when any shader files change
    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=shaders/radiance_cascade_trace.wgsl");
    println!("cargo:rerun-if-changed=shaders/radiance_lookup.wgsl");
}
