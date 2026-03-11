use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Different logic depending on the target.  When compiling for the
    // host we want to make sure the wasm assets are built as well so that the
    // `pkg/` directory is populated before the server binary runs.  To achieve
    // this we always run the wasm build steps below regardless of the current
    // target triple.  We still copy the native server exe when building for a
    // host target, but we no longer bail out early.
    let target = env::var("TARGET").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // the triple we use for building the wasm library; we never actually
    // compile anything for the host under this path, so hardcode it here.
    let wasm_target = "wasm32-unknown-unknown";

    // location of the compiled wasm file (always use wasm_target, even when
    // building the host binary)
    let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into());
    let mut wasm_path = PathBuf::from(&target_dir);
    wasm_path.push(wasm_target);
    wasm_path.push(&profile);
    wasm_path.push("helio_wasm_app.wasm");

    // output directory for generated bindings
    let mut out_dir = manifest_dir.clone();
    out_dir.push("pkg");

    // if we are building a host binary, copy the server exe into pkg so that
    // the `pkg` directory from a release build contains both parts.  this used
    // to be the only thing we did in the non-wasm case; the wasm build was
    // skipped entirely.  by dropping the early return below we can run the
    // normal wasm-oriented logic afterwards as well.
    if target != wasm_target {
        // copy native server binary if it exists
        let mut server_path = PathBuf::from(&target_dir);
        server_path.push(&target);
        server_path.push(&profile);
        let server_name = if cfg!(windows) { "helio-wasm-server.exe" } else { "helio-wasm-server" };
        server_path.push(server_name);
        if server_path.exists() {
            std::fs::create_dir_all(&out_dir).ok();
            let _ = std::fs::copy(&server_path, out_dir.join(server_name));
        }
        // continue on to the wasm build steps rather than returning early
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-env-changed=TARGET");

    // Always attempt to copy any existing outputs into the pkg directory.
    // This ensures running `cargo build` without rebuilding still updates the
    // artifacts (useful for incremental development or CI scripts).
    {
        // copy wasm if available
        if wasm_path.exists() {
            let _ = std::fs::create_dir_all(wasm_path.parent().unwrap());
            let _ = std::fs::copy(&wasm_path, &wasm_path); // no-op but ensures path exists
        }
        // copy host server binary from the regular target directory as well
        let host_trip = env::var("HOST").unwrap_or_else(|_| "".into());
        let mut server_src = PathBuf::from(&target_dir);
        server_src.push(&host_trip);
        server_src.push(&profile);
        let server_name = if cfg!(windows) { "helio-wasm-server.exe" } else { "helio-wasm-server" };
        server_src.push(server_name);
        if server_src.exists() {
            std::fs::create_dir_all(&out_dir).ok();
            let _ = std::fs::copy(&server_src, out_dir.join(server_name));
        }
    }

    // Always rebuild the wasm library in an inner cargo invocation unless
    // we're already the inner build.  This ensures that changes to any Rust
    // source are reflected in the output even if the old wasm file already
    // exists from a previous run.
    if env::var("HELIO_BUILD_WASM_INNER").is_err() {
        eprintln!("[build.rs] invoking inner cargo to compile wasm");

        // Collect env vars that cargo injects for the *host* build-script
        // target.  If they are inherited by the inner cross-compilation they
        // corrupt feature/cfg resolution for wasm32 and cause link failures.
        let poisoned: Vec<String> = env::vars()
            .map(|(k, _)| k)
            .filter(|k| {
                k == "RUSTFLAGS"
                    || k == "CARGO_ENCODED_RUSTFLAGS"
                    || k.starts_with("CARGO_FEATURE_")
                    || k.starts_with("CARGO_CFG_")
                    || k == "CARGO_PRIMARY_PACKAGE"
            })
            .collect();

        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .arg("--release")
            .arg("--lib")
            .arg("-p")
            .arg("helio-wasm-app")
            .arg("--target")
            .arg("wasm32-unknown-unknown")
            .arg("--target-dir")
            .arg("target/wasm-build")
            .env("HELIO_BUILD_WASM_INNER", "1");
        for key in &poisoned {
            cmd.env_remove(key);
        }
        let status = cmd.status().expect("failed to spawn inner cargo build");
        if !status.success() {
            panic!("inner cargo build failed");
        }
        // make sure native server also gets rebuilt in the inner directory
        let _ = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("-p")
            .arg("helio-wasm-app")
            .arg("--target-dir")
            .arg("target/wasm-build")
            .env("HELIO_BUILD_WASM_INNER", "1")
            .status();

        // copy the freshly-built wasm to the normal target path
        let mut src = PathBuf::from("target/wasm-build");
        src.push(wasm_target);
        src.push(&profile);
        src.push("helio_wasm_app.wasm");
        if src.exists() {
            std::fs::create_dir_all(wasm_path.parent().unwrap()).unwrap();
            std::fs::copy(&src, &wasm_path).unwrap();
        }
        // also copy the host server exe into pkg
        let host_trip = env::var("HOST").unwrap_or_else(|_| "".into());
        let mut server_src = PathBuf::from("target/wasm-build");
        server_src.push(&host_trip);
        server_src.push(&profile);
        let server_name = if cfg!(windows) { "helio-wasm-server.exe" } else { "helio-wasm-server" };
        server_src.push(server_name);
        if server_src.exists() {
            std::fs::create_dir_all(&out_dir).unwrap();
            let _ = std::fs::copy(&server_src, out_dir.join(server_name));
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

    if !cmd.status().map(|s| s.success()).unwrap_or(false) {
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
            Ok(s) if s.success() => (),
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
            }
        }
    }

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
