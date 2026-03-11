// Static file server for Helio WASM examples.
// All assets are embedded at compile-time via include_bytes! so the binary
// is fully self-contained — no pkg/ directory is required at runtime.
//
// Routes:
//   GET /                           → landing page
//   GET /<example>/                 → per-example wrapper page
//   GET /<example>/index.html       → per-example wrapper page
//   GET /<example>/helio_wasm_examples.js       → bindgen JS glue
//   GET /<example>/helio_wasm_examples_bg.wasm  → compiled WASM

#[cfg(target_arch = "wasm32")]
fn main() {}

// ── Embedded static assets (produced by build.rs into pkg/<name>/) ──────────
// include_bytes! paths are relative to this source file (src/bin/server.rs),
// so ../../pkg/<name>/... resolves to crates/helio-wasm-app/pkg/<name>/...

#[cfg(not(target_arch = "wasm32"))]
fn lookup_asset(rel: &str) -> Option<(&'static [u8], &'static str)> {
    macro_rules! js   { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, "/helio_wasm_examples.js"))   } }
    macro_rules! wasm { ($n:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $n, "/helio_wasm_examples_bg.wasm")) } }
    match rel {
        "render_v2_basic/helio_wasm_examples.js"        => Some((js!("render_v2_basic"),   "application/javascript")),
        "render_v2_basic/helio_wasm_examples_bg.wasm"   => Some((wasm!("render_v2_basic"), "application/wasm")),
        "render_v2_sky/helio_wasm_examples.js"          => Some((js!("render_v2_sky"),     "application/javascript")),
        "render_v2_sky/helio_wasm_examples_bg.wasm"     => Some((wasm!("render_v2_sky"),   "application/wasm")),
        "sdf_blend/helio_wasm_examples.js"              => Some((js!("sdf_blend"),         "application/javascript")),
        "sdf_blend/helio_wasm_examples_bg.wasm"         => Some((wasm!("sdf_blend"),       "application/wasm")),
        "sdf_boolean/helio_wasm_examples.js"            => Some((js!("sdf_boolean"),       "application/javascript")),
        "sdf_boolean/helio_wasm_examples_bg.wasm"       => Some((wasm!("sdf_boolean"),     "application/wasm")),
        "sdf_demo/helio_wasm_examples.js"               => Some((js!("sdf_demo"),          "application/javascript")),
        "sdf_demo/helio_wasm_examples_bg.wasm"          => Some((wasm!("sdf_demo"),        "application/wasm")),
        "sdf_grid/helio_wasm_examples.js"               => Some((js!("sdf_grid"),          "application/javascript")),
        "sdf_grid/helio_wasm_examples_bg.wasm"          => Some((wasm!("sdf_grid"),        "application/wasm")),
        "sdf_morph/helio_wasm_examples.js"              => Some((js!("sdf_morph"),         "application/javascript")),
        "sdf_morph/helio_wasm_examples_bg.wasm"         => Some((wasm!("sdf_morph"),       "application/wasm")),
        "sdf_move/helio_wasm_examples.js"               => Some((js!("sdf_move"),          "application/javascript")),
        "sdf_move/helio_wasm_examples_bg.wasm"          => Some((wasm!("sdf_move"),        "application/wasm")),
        "sdf_multi/helio_wasm_examples.js"              => Some((js!("sdf_multi"),         "application/javascript")),
        "sdf_multi/helio_wasm_examples_bg.wasm"         => Some((wasm!("sdf_multi"),       "application/wasm")),
        "sdf_pulse/helio_wasm_examples.js"              => Some((js!("sdf_pulse"),         "application/javascript")),
        "sdf_pulse/helio_wasm_examples_bg.wasm"         => Some((wasm!("sdf_pulse"),       "application/wasm")),
        "sdf_terrain/helio_wasm_examples.js"            => Some((js!("sdf_terrain"),       "application/javascript")),
        "sdf_terrain/helio_wasm_examples_bg.wasm"       => Some((wasm!("sdf_terrain"),     "application/wasm")),
        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn build_example_page(_name: &str, title: &str) -> Vec<u8> {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Helio – {title}</title>
  <style>body{{margin:0;background:#111;}}</style>
</head>
<body>
  <a href="../" style="position:fixed;top:8px;left:8px;color:#89b4fa;font-family:monospace;font-size:13px;z-index:10;">← examples</a>
  <script type="module">
    import init from './helio_wasm_examples.js';
    init();
  </script>
</body>
</html>"#,
        title = title,
    )
    .into_bytes()
}

#[cfg(not(target_arch = "wasm32"))]
fn build_landing_page(examples: &[(&str, &str)]) -> Vec<u8> {
    let mut cards = String::new();
    for (name, title) in examples {
        cards.push_str(&format!(
            "    <a class=\"card\" href=\"./{name}/\"><div class=\"card-body\"><div class=\"card-title\">{name}</div><div class=\"card-desc\">{title}</div></div></a>\n",
            name = name, title = title,
        ));
    }
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Helio — WebGPU Examples</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0f0f13;color:#cdd6f4;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}}
    header{{padding:2.5rem 2rem 1rem;border-bottom:1px solid #313244}}
    header h1{{font-size:2rem;font-weight:700;letter-spacing:.04em}}
    header p{{margin-top:.4rem;color:#a6adc8;font-size:.95rem}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:1.25rem;padding:2rem}}
    .card{{background:#1e1e2e;border:1px solid #313244;border-radius:.75rem;overflow:hidden;
          transition:border-color .15s,transform .15s;text-decoration:none;color:inherit;display:block}}
    .card:hover{{border-color:#89b4fa;transform:translateY(-2px)}}
    .card-body{{padding:1.1rem 1.25rem 1.3rem}}
    .card-title{{font-size:1rem;font-weight:600;font-family:monospace;color:#89b4fa}}
    .card-desc{{margin-top:.35rem;font-size:.85rem;color:#a6adc8;line-height:1.5}}
    footer{{padding:1.5rem 2rem;border-top:1px solid #313244;color:#585b70;font-size:.8rem;text-align:center}}
  </style>
</head>
<body>
  <header>
    <h1>Helio — WebGPU Examples</h1>
    <p>Interactive demos running in the browser via WebGPU. Requires a browser with WebGPU support.</p>
  </header>
  <div class="grid">
{cards}  </div>
  <footer>Built with Helio render engine · WebGPU · Rust/WASM</footer>
</body>
</html>"#,
        cards = cards,
    )
    .into_bytes()
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use tiny_http::{Header, Response, Server};

    const EXAMPLES: &[(&str, &str)] = &[
        ("render_v2_basic", "Basic Render"),
        ("render_v2_sky",   "Volumetric Sky"),
        ("sdf_blend",       "SDF Blend"),
        ("sdf_boolean",     "SDF Boolean"),
        ("sdf_demo",        "SDF Demo"),
        ("sdf_grid",        "SDF Grid Wave"),
        ("sdf_morph",       "SDF Morph"),
        ("sdf_move",        "SDF Move"),
        ("sdf_multi",       "SDF Multi"),
        ("sdf_pulse",       "SDF Pulse"),
        ("sdf_terrain",     "SDF Terrain"),
    ];

    let port = 8000;
    let addr = format!("0.0.0.0:{}", port);
    let url  = format!("http://localhost:{}/", port);

    if webbrowser::open(&url).is_ok() {
        eprintln!("opening browser at {}", url);
    }

    let server = Server::http(&addr).expect("failed to start server");
    eprintln!("serving {} examples on {}", EXAMPLES.len(), addr);

    for request in server.incoming_requests() {
        let req_url = request.url().to_owned();
        // Strip query string / fragment
        let path = req_url.split('?').next().unwrap_or("/");
        let rel  = path.trim_start_matches('/');

        // Helper: respond with bytes and a MIME type.
        // (We define a local closure to keep the loop body readable.)
        let respond = |data: Vec<u8>, mime: &str| {
            let mut resp = Response::from_data(data);
            resp.add_header(
                Header::from_bytes(b"Content-Type", mime.as_bytes()).unwrap(),
            );
            resp
        };

        // ── Landing page ────────────────────────────────────────────────────
        if rel.is_empty() || rel == "index.html" {
            let _ = request.respond(respond(build_landing_page(EXAMPLES), "text/html; charset=utf-8"));
            continue;
        }

        // ── Split into <example> / <rest> ───────────────────────────────────
        let (first, rest) = rel.split_once('/').unwrap_or((rel, ""));

        if let Some(&(name, title)) = EXAMPLES.iter().find(|(n, _)| *n == first) {
            if rest.is_empty() || rest == "index.html" {
                // Per-example wrapper page
                let _ = request.respond(respond(build_example_page(name, title), "text/html; charset=utf-8"));
                continue;
            }

            // Static asset (JS glue / WASM binary)
            if let Some((data, mime)) = lookup_asset(rel) {
                let _ = request.respond(respond(data.to_vec(), mime));
                continue;
            }
        }

        // Not found
        let _ = request.respond(Response::from_data(Vec::<u8>::new()).with_status_code(404));
    }
}

