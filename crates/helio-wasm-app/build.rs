use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only run the post-processing when compiling for wasm32-unknown-unknown
    let target = env::var("TARGET").unwrap_or_default();
    if target != "wasm32-unknown-unknown" {
        return;
    }

    let profile = env::var("PROFILE").unwrap();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // location of the compiled wasm file
    let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into());
    let mut wasm_path = PathBuf::from(&target_dir);
    wasm_path.push(&target);
    wasm_path.push(&profile);
    wasm_path.push("helio_wasm_app.wasm");

    // output directory for generated bindings
    let mut out_dir = manifest_dir.clone();
    out_dir.push("pkg");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-env-changed=TARGET");

    // if the wasm artifact isn't present yet, try to build it in a separate
    // target-dir so we can run wasm-bindgen in a single `cargo build`
    if !wasm_path.exists() && env::var("HELIO_BUILD_WASM_INNER").is_err() {
        eprintln!("[build.rs] wasm module not found; invoking cargo to compile wasm");
        let status = Command::new("cargo") // cargo build --release -p helio-wasm-app --target wasm32-unknown-unknown --target-dir target/wasm-build
            .arg("build")
            .arg("--release")
            .arg("-p")
            .arg("helio-wasm-app")
            .arg("--target")
            .arg("wasm32-unknown-unknown")
            .arg("--target-dir")
            .arg("target/wasm-build")
            .env("HELIO_BUILD_WASM_INNER", "1")
            .status()
            .expect("failed to spawn inner cargo build");
        if !status.success() {
            panic!("inner cargo build failed");
        }
        // after inner build the artifact should exist in the other folder; copy
        // it to the normal target so future accesses work as expected
        let mut src = PathBuf::from("target/wasm-build");
        src.push(&target);
        src.push(&profile);
        src.push("helio_wasm_app.wasm");
        if src.exists() {
            std::fs::create_dir_all(wasm_path.parent().unwrap()).unwrap();
            std::fs::copy(&src, &wasm_path).unwrap();
        }
    }

    if !wasm_path.exists() {
        // still missing, give up and let the actual compile produce it
        return;
    }

    std::fs::create_dir_all(&out_dir).unwrap();

    // run wasm-bindgen on the generated module.  Prefer invoking the CLI
    // binary directly (installed via `cargo install wasm-bindgen-cli`) because
    // cargo run only works for workspace members.  If the binary isn't found we
    // fall back to `cargo run --package wasm-bindgen-cli` so people who have
    // added it to the workspace (e.g. for CI) still work.
    let mut cmd = Command::new("wasm-bindgen");
    cmd.arg(&wasm_path)
        .arg("--out-dir")
        .arg(&out_dir)
        .arg("--target")
        .arg("web");

    let status = match cmd.status() {
        Ok(s) if s.success() => s,
        _ => {
            eprintln!("[build.rs] `wasm-bindgen` binary not found, trying `cargo run`...");
            // attempt to run the CLI via cargo run; this works only if the user has
            // added wasm-bindgen-cli as a workspace member.  if that also fails we
            // attempt to install the tool.
            let s = Command::new("cargo")
                .arg("run")
                .arg("--package")
                .arg("wasm-bindgen-cli")
                .arg("--")
                .arg(&wasm_path)
                .arg("--out-dir")
                .arg(&out_dir)
                .arg("--target")
                .arg("web")
                .status();
            match s {
                Ok(s) if s.success() => s,
                _ => {
                    eprintln!("[build.rs] cargo run failed; installing wasm-bindgen-cli...");
                    let install = Command::new("cargo")
                        .arg("install")
                        .arg("--locked")
                        .arg("wasm-bindgen-cli")
                        .status()
                        .expect("failed to install wasm-bindgen-cli");
                    if !install.success() {
                        panic!("unable to install wasm-bindgen-cli");
                    }
                    // retry the binary now that it should be on PATH
                    let s2 = Command::new("wasm-bindgen")
                        .arg(&wasm_path)
                        .arg("--out-dir").arg(&out_dir)
                        .arg("--target").arg("web")
                        .status()
                        .expect("failed to run wasm-bindgen after install");
                    if !s2.success() {
                        panic!("wasm-bindgen returned error after install");
                    }
                    s2
                }
            }
        }
    };

    // generate a minimal index.html if one doesn't exist
    let mut index = out_dir.clone();
    index.push("index.html");
    if !index.exists() {
        std::fs::write(
            &index,
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Helio WASM App</title>
</head>
<body>
<script type="module">
    import init from './helio_wasm_app.js';
    async function run() {
        await init();
    }
    run();
</script>
</body>
</html>"#,
        )
        .unwrap();
    }
}
