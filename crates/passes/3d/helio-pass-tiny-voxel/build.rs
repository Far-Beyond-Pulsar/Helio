use std::{env, fs, path::PathBuf};
fn main() {
    let paths = [
        "src/landforms/sampling.wgsl",
        "src/landforms/volume.wgsl",
        "src/planet.wgsl",
        "src/landforms/bounds.wgsl",
    ];
    for path in paths {
        println!("cargo:rerun-if-changed={path}");
    }
    let sampling = fs::read_to_string(paths[0])
        .unwrap()
        .replace("@group(0) @binding(0)", "@group(0) @binding(20)")
        .replace("@group(0) @binding(1)", "@group(0) @binding(21)");
    let volume = fs::read_to_string(paths[1])
        .unwrap()
        .replace("@group(0) @binding(5)", "@group(0) @binding(23)");
    let bounds = fs::read_to_string(paths[3])
        .unwrap()
        .replace("@group(0) @binding(4)", "@group(0) @binding(22)");
    let source = format!(
        "{sampling}\n{bounds}\n{volume}\n{}",
        fs::read_to_string(paths[2]).unwrap()
    );
    fs::write(
        PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("planet.wgsl"),
        source,
    )
    .unwrap();
}
