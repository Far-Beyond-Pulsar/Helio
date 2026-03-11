// Static file server for the Helio WASM app.
// All assets are embedded at compile-time via include_bytes! so the binary
// is fully self-contained — no pkg/ directory is required at runtime.

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use tiny_http::{Response, Server};

    // Assets baked in at compile time.  The paths are relative to this source
    // file, which lives at crates/helio-wasm-app/src/bin/server.rs, so ../..
    // resolves to crates/helio-wasm-app/pkg/.
    static INDEX_HTML: &[u8] = include_bytes!("../../pkg/index.html");
    static WASM_JS: &[u8] = include_bytes!("../../pkg/helio_wasm_app.js");
    static WASM_BG: &[u8] = include_bytes!("../../pkg/helio_wasm_app_bg.wasm");

    let port = 8000;
    let addr = format!("0.0.0.0:{}", port);

    let url = format!("http://localhost:{}/", port);
    if webbrowser::open(&url).is_ok() {
        eprintln!("opening browser at {}", url);
    }

    let server = Server::http(&addr).expect("failed to start server");
    eprintln!("serving embedded assets on {}", addr);

    for request in server.incoming_requests() {
        let req_url = request.url().to_owned();
        let rel = if req_url == "/" { "index.html" } else { req_url.trim_start_matches('/') };

        let (data, mime): (&[u8], &str) = match rel {
            "index.html" => (INDEX_HTML, "text/html; charset=utf-8"),
            "helio_wasm_app.js" => (WASM_JS, "application/javascript"),
            "helio_wasm_app_bg.wasm" => (WASM_BG, "application/wasm"),
            _ => {
                let _ = request.respond(Response::from_data(Vec::new()).with_status_code(404));
                continue;
            }
        };

        let mut resp = Response::from_data(data.to_vec());
        resp.add_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], mime.as_bytes()).unwrap(),
        );
        let _ = request.respond(resp);
    }
}
