//! `cargo run --bin web` — build all WASM demos in parallel, serve locally.
//!
//! Builds each demo in its own thread (up to `num_cpus` at once), streams a live
//! progress dashboard, then starts a file server on `http://127.0.0.1:8000`.

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

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
    let cc = std::env::var("CC").unwrap_or_default();
    let total = DEMOS.len();

    let results = Arc::new(Mutex::new(Vec::new()));

    // Spawn one thread per demo; we'll throttle with a semaphore below.
    let mut handles = Vec::new();
    for &name in DEMOS {
        let results = Arc::clone(&results);
        let out_base = out_base.clone();
        let manifest_dir = manifest_dir.clone();
        let cc = cc.clone();

        handles.push(thread::spawn(move || {
            let out_dir = out_base.join(name);

            let output = Command::new("wasm-pack")
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
                .env("CC", &cc)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
                .expect("wasm-pack not found. Install with: cargo install wasm-pack");

            let ok = output.status.success() && {
                let pkg_dir = manifest_dir.join("crates/helio-web-demos/pkg");
                if pkg_dir.exists() {
                    let _ = std::fs::remove_dir_all(&out_dir);
                    if std::fs::rename(&pkg_dir, &out_dir).is_ok() {
                        // build.rs wrote index.html before the rename, but
                        // the rename destroyed it.  Write a minimal one.
                        let html = format!(
                            r#"<!DOCTYPE html><html><head>
<meta charset="utf-8"><title>{name}</title>
<style>body{{margin:0;overflow:hidden;background:#000}}
#info{{position:absolute;bottom:8px;left:8px;color:#888;font:14px monospace}}
</style></head><body>
<script type="module">
import init from "./helio_web_demos.js";
init().catch(e=>document.body.innerHTML=`<pre style=color:red>${{e}}</pre>`);
</script>
<div id=info>{name}</div>
</body></html>"#
                        );
                        let _ = std::fs::write(out_dir.join("index.html"), &html);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            };

            let size_kb = if ok {
                out_dir
                    .join("helio_web_demos_bg.wasm")
                    .metadata()
                    .ok()
                    .map(|m| m.len() / 1024)
                    .unwrap_or(0)
            } else {
                0
            };

            let mut lock = results.lock().unwrap();
            lock.push((name, ok, size_kb));
        }));
    }

    // ── Dashboard ──────────────────────────────────────────────────────────────
    let done = AtomicU16::new(0);
    let ok_count = AtomicU16::new(0);
    let fail_count = AtomicU16::new(0);

    // Wait for completion in a polling loop so we can render the dashboard.
    while done.load(Ordering::Relaxed) < total as u16 {
        thread::sleep(std::time::Duration::from_millis(200));

        let lock = results.lock().unwrap();
        let n = lock.len() as u16;
        drop(lock);
        done.store(n, Ordering::Relaxed);

        // Render dashboard
        print!("\x1B[2J\x1B[H"); // clear screen, home cursor
        println!("╔════════════════════════════════════╗");
        println!("║     Helio Web Demo Build           ║");
        println!("╠════════════════════════════════════╣");
        println!("║  Total:  {total:>2}  Demos          ║");
        println!("║  Done:   {n:>2}                     ║");
        println!("╚════════════════════════════════════╝");
        println!();
        println!("  Building...");

        let finished: Vec<_> = results.lock().unwrap().iter().map(|(n, _, _)| *n).collect();
        for d in DEMOS {
            if finished.contains(d) {
                let lock = results.lock().unwrap();
                let (_, ok, kb) = lock.iter().find(|(n, _, _)| *n == *d).unwrap();
                let icon = if *ok { "✓" } else { "✗" };
                println!("  {icon} {d}  ({kb} KiB)");
            } else {
                println!("    {d}");
            }
        }
        std::io::stdout().flush().ok();
    }

    // Final tally
    {
        let lock = results.lock().unwrap();
        for (_, ok, _) in lock.iter() {
            if *ok {
                ok_count.fetch_add(1, Ordering::Relaxed);
            } else {
                fail_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    print!("\x1B[2J\x1B[H");
    println!("╔════════════════════════════════════╗");
    println!("║     Build Complete                  ║");
    println!("╠════════════════════════════════════╣");
    println!("║  OK    {:>3}                         ║", ok_count.load(Ordering::Relaxed));
    println!("║  FAIL  {:>3}                         ║", fail_count.load(Ordering::Relaxed));
    println!("║  Total {total:>3}                         ║");
    println!("╚════════════════════════════════════╝");
    println!();
    println!("  Serving at http://127.0.0.1:8000/");

    serve(&out_base, "127.0.0.1:8000");
}

fn serve(root: &PathBuf, addr: &str) {
    let server = tiny_http::Server::http(addr).unwrap();
    let root = root.clone();
    for request in server.incoming_requests() {
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
                (tiny_http::StatusCode(404), b"404 Not Found\n".to_vec())
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
