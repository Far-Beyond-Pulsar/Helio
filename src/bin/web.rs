//! `cargo run --bin web` — build all WASM demos and serve them locally.
//!
//! 1. Runs `wasm-pack build` for every demo in `helio-web-demos`.
//! 2. Spawns a static file server on `http://127.0.0.1:8000`.
//!
//! Requires `wasm-pack` and a wasm-capable `CC` in PATH.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

const DEMOS: &[&str] = &[
    "render_v2_basic",
    "render_v2_sky",
    "debug_shapes",
    "indoor_room",
    "indoor_corridor",
    "outdoor_night",
    "outdoor_canyon",
    "indoor_cathedral",
    "indoor_server_room",
    "outdoor_city",
    "outdoor_volcano",
    "space_station",
    "light_benchmark",
    "hlfs_benchmark",
    "sdf_demo",
    "rc_benchmark",
    "load_fbx",
    "load_fbx_embedded",
    "ship_flight",
    "simple_graph",
    "outdoor_rocks",
    "editor_demo",
];

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_base = manifest_dir.join("target/wasm-prebuilt");

    for name in DEMOS {
        let out_dir = out_base.join(name);
        println!("═══ {name} ═══");

        let status = Command::new("wasm-pack")
            .args([
                "build",
                "--release",
                "--target",
                "web",
                "--no-default-features",
                "--features",
                name,
            ])
            .current_dir(manifest_dir.join("crates/helio-web-demos"))
            .env("CC", std::env::var("CC").unwrap_or_default())
            .status()
            .expect("wasm-pack not found. Install with: cargo install wasm-pack");

        if !status.success() {
            eprintln!("  FAILED: {name}");
            std::process::exit(1);
        }

        let pkg_dir = manifest_dir.join("crates/helio-web-demos/pkg");
        if pkg_dir.exists() {
            let _ = std::fs::remove_dir_all(&out_dir);
            if let Err(e) = std::fs::rename(&pkg_dir, &out_dir) {
                eprintln!("  mv failed: {e}");
                std::process::exit(1);
            }
        }

        let wasm = out_dir.join("helio_web_demos_bg.wasm");
        let size_kb = std::fs::metadata(&wasm)
            .ok()
            .map(|m| m.len() / 1024)
            .unwrap_or(0);
        println!("  OK ({size_kb} KiB)");
    }

    let addr = "127.0.0.1:8000";
    println!("\nServing demos at http://{addr}/");
    serve(&out_base, addr);
}

fn serve(root: &PathBuf, addr: &str) {
    let server = tiny_http::Server::http(addr).unwrap();
    loop {
        let request = match server.recv() {
            Ok(r) => r,
            Err(_) => break,
        };
        let url = request.url().to_string();
        let path = {
            let stripped = url.trim_start_matches('/');
            let candidate = root.join(stripped);
            if candidate.is_dir() {
                candidate.join("index.html")
            } else {
                candidate
            }
        };

        let (status, contents) = match std::fs::read(&path) {
            Ok(data) => (tiny_http::StatusCode(200), data),
            Err(_) => {
                let msg = b"404 Not Found\n";
                (tiny_http::StatusCode(404), msg.to_vec())
            }
        };

        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let cts = mime_for(ext);
        let response = tiny_http::Response::from_data(contents)
            .with_status_code(status)
            .with_header(
                tiny_http::Header::from_bytes(&b"Content-Type"[..], cts.as_bytes()).unwrap(),
            );
        let _ = request.respond(response);
    }
}

fn mime_for(ext: &str) -> &'static str {
    match ext {
        "html" => "text/html; charset=utf-8",
        "js" => "application/javascript",
        "wasm" => "application/wasm",
        "css" => "text/css; charset=utf-8",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}
