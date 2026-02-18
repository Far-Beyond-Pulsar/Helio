fn main() {
    // Recompile when any shader files change
    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=shaders/shadow_functions.wgsl");
}
