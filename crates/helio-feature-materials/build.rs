fn main() {
    // Recompile when any shader files change
    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=shaders/material_bindings.wgsl");
    println!("cargo:rerun-if-changed=shaders/material_functions.wgsl");
}
