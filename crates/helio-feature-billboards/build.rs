fn main() {
    // Recompile when any shader files change
    println!("cargo:rerun-if-changed=shaders/");
    println!("cargo:rerun-if-changed=shaders/billboard_vertex.wgsl");
    println!("cargo:rerun-if-changed=shaders/billboard_fragment.wgsl");
}
