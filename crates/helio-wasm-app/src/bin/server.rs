// simple static file server for the generated `pkg/` directory.
// this binary is only built for the host (non-wasm) target.

#![cfg(not(target_arch = "wasm32"))]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use tiny_http::{Server, Response};

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

        let response = if path.exists() {
            match std::fs::File::open(&path) {
                Ok(file) => {
                    let mime = mime_guess::from_path(&path).first_or_octet_stream();
                    Response::from_file(file)
                        .with_header(
                            tiny_http::Header::from_bytes(&b"Content-Type"[..], mime.as_ref()).unwrap(),
                        )
                }
                Err(_) => Response::empty(500),
            }
        } else {
            Response::empty(404)
        };
        let _ = request.respond(response);
    }
}
