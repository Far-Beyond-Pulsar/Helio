fn main() {
    // Recompile when any shader files change
    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=shaders/probe_placement.wgsl");
    println!("cargo:rerun-if-changed=shaders/radiance_injection.wgsl");
    println!("cargo:rerun-if-changed=shaders/radiance_propagation.wgsl");
    println!("cargo:rerun-if-changed=shaders/temporal_blend.wgsl");
    println!("cargo:rerun-if-changed=shaders/gi_functions.wgsl");
    println!("cargo:rerun-if-changed=shaders/gi_sampling.wgsl");
}
