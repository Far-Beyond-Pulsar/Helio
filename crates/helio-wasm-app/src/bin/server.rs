// simple static file server for the generated `pkg/` directory.
// this binary is only built for the host (non-wasm) target.

// When compiling for wasm the file is still parsed, so provide a nop `main`
// to satisfy the compiler.  the real server code is gated below.
#[cfg(target_arch = "wasm32")]
fn main() {
    // nothing
}

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use tiny_http::{Server, Response};

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let port = 8000;
    let addr = format!("0.0.0.0:{}", port);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("pkg");

    if !root.exists() {
        eprintln!("pkg directory does not exist; build the wasm target first");
        std::process::exit(1);
    }

    let url = format!("http://localhost:{}/", port);
    if webbrowser::open(&url).is_ok() {
        eprintln!("opening browser at {}", url);
    }

    let server = Server::http(&addr).expect("failed to start server");
    eprintln!("serving {} on {}", root.display(), addr);

    for request in server.incoming_requests() {
        let req_url = request.url();
        // map "/" to index.html
        let rel = if req_url == "/" { "index.html" } else { &req_url[1..] };
        let mut path = root.clone();
        path.push(rel);
        if path.is_dir() {
            path.push("index.html");
        }

        // helper that attaches a couple common headers to each response
        fn add_common_headers(r: Response<std::io::Cursor<Vec<u8>>>) -> Response<std::io::Cursor<Vec<u8>>> {
            if let Ok(h) = tiny_http::Header::from_bytes(&b"Permissions-Policy"[..], b"browsing-topics=()") {
                r.with_header(h)
            } else {
                r
            }
        }

        // build response using in-memory bytes so we always return the same
        // `Response<Cursor<Vec<u8>>>` type and avoid mismatched generics.
        let response = if path.exists() {
            match std::fs::read(&path) {
                Ok(data) => {
                    let mime = mime_guess::from_path(&path).first_or_octet_stream();
                    let mut resp = Response::from_data(data);
                    resp.add_header(
                        tiny_http::Header::from_bytes(&b"Content-Type"[..], mime.as_ref()).unwrap(),
                    );
                    add_common_headers(resp)
                }
                Err(_) => add_common_headers(Response::from_data(Vec::new()).with_status_code(500)),
            }
        } else {
            add_common_headers(Response::from_data(Vec::new()).with_status_code(404))
        };
        let _ = request.respond(response);
    }
}
