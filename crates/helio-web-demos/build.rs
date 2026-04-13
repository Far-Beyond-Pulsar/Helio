//! Build script for `helio-web-demos`.
//!
//! Generates one `<name>.html` landing page per demo and one `index.html`
//! master listing, writing them to:
//!
//!   1. `$OUT_DIR/<name>.html` — embedded at compile-time via
//!      `include_str!(concat!(env!("OUT_DIR"), "/<name>.html"))` in lib.rs.
//!
//!   2. `<workspace>/target/wasm-prebuilt/<name>/index.html` — copied as a
//!      side-effect so the web server can serve them directly alongside the
//!      wasm-bindgen output without extra scripting.

use std::fs;
use std::path::PathBuf;

// ── Demo catalogue ─────────────────────────────────────────────────────────────

struct Demo {
    name: &'static str,
    title: &'static str,
    description: &'static str,
    controls: &'static str,
}

const DEMOS: &[Demo] = &[
    Demo { name: "render_v2_basic",     title: "Basic Render",               description: "Three lit cubes — the minimal Helio render pipeline.",            controls: "WASD / Space / Shift — fly &nbsp;·&nbsp; Mouse — look &nbsp;·&nbsp; Click — grab cursor" },
    Demo { name: "render_v2_sky",       title: "Volumetric Sky",             description: "Real-time atmospheric scattering with a moving sun.",             controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Q/E — rotate sun &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "debug_shapes",        title: "Debug Shapes",               description: "Coloured debug boxes exercising the geometry pass.",              controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "indoor_room",         title: "Indoor Room",                description: "Furnished room with a flickering point light.",                   controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "indoor_corridor",     title: "Indoor Corridor",            description: "40 m corridor with overhead fluorescent lighting.",               controls: "WASD/Space/Shift — walk &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "outdoor_night",       title: "Outdoor Night",              description: "City block under streetlamps at night.",                          controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "outdoor_canyon",      title: "Outdoor Canyon",             description: "Desert canyon with a campfire and a dynamic sky.",                controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Q/E — sun &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "indoor_cathedral",    title: "Indoor Cathedral",           description: "Gothic nave lit by candles and flickering torch sconces.",        controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "indoor_server_room",  title: "Indoor Server Room",         description: "Datacenter server racks with animated LED indicators.",           controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "outdoor_city",        title: "Outdoor City",               description: "Procedural night city with street lamps and rooftop beacons.",    controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "outdoor_volcano",     title: "Outdoor Volcano",            description: "Active lava field with pulsing vent glow.",                       controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "space_station",       title: "Space Station",              description: "Orbiting station with solar arrays and navigation lights.",       controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "light_benchmark",     title: "Light Benchmark",            description: "128 animated point lights — deferred lighting stress test.",      controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "rc_benchmark",        title: "Radiance Cascades",          description: "Cornell box global illumination benchmark.",                      controls: "+/- — intensity &nbsp;·&nbsp; WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "sdf_demo",            title: "SDF Demo",                   description: "Signed-distance field clipmap with live sphere edits.",           controls: "F — free fly / orbit &nbsp;·&nbsp; WASD — move &nbsp;·&nbsp; Mouse — look/orbit" },
    Demo { name: "load_fbx",            title: "Load FBX",                   description: "FBX asset loading (placeholder scene on WASM).",                  controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "load_fbx_embedded",   title: "Load FBX (Embedded)",        description: "FBX loaded from bytes embedded at compile time.",                 controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "ship_flight",         title: "Ship Flight",                description: "6-DoF spaceship through an asteroid field.",                      controls: "WASD — thrust &nbsp;·&nbsp; Q/E — roll &nbsp;·&nbsp; Space/Shift — lift &nbsp;·&nbsp; Mouse — aim" },
    Demo { name: "simple_graph",        title: "Simple Graph",               description: "Minimal fly-camera around a lit unit cube.",                      controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "outdoor_rocks",       title: "Outdoor Rocks",              description: "Scattered rocks with an embedded FBX ship and dynamic sun.",      controls: "WASD/Space/Shift — fly &nbsp;·&nbsp; Q/E — sun &nbsp;·&nbsp; Mouse — look" },
    Demo { name: "editor_demo",          title: "Editor Demo",                description: "BVH ray-picking and transform gizmo (translate / rotate / scale).", controls: "RMB hold — fly &nbsp;·&nbsp; LMB — pick/drag &nbsp;·&nbsp; G/R/S — gizmo mode &nbsp;·&nbsp; Ctrl+D — duplicate &nbsp;·&nbsp; Del — delete &nbsp;·&nbsp; Tab — grid" },
];

// ── HTML templates ─────────────────────────────────────────────────────────────

fn demo_html(demo: &Demo) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} — Helio</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      width: 100%; height: 100%;
      overflow: hidden;
      background: #000;
      font-family: system-ui, sans-serif;
    }}
    #loading {{
      position: fixed; inset: 0;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      background: #000; color: #aaa;
      font-size: 14px; gap: 16px;
      transition: opacity 0.4s;
      z-index: 10;
    }}
    #loading.hidden {{ opacity: 0; pointer-events: none; }}
    .spinner {{
      width: 32px; height: 32px;
      border: 3px solid #333;
      border-top-color: #888;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    #controls {{
      position: fixed; bottom: 10px; left: 50%;
      transform: translateX(-50%);
      color: #555; font-size: 11px;
      pointer-events: none;
      transition: opacity 1s;
      white-space: nowrap;
    }}
    #controls.fade {{ opacity: 0; }}
    a.back {{
      position: fixed; top: 10px; left: 12px;
      color: #444; font-size: 11px; text-decoration: none;
      transition: color 0.2s;
    }}
    a.back:hover {{ color: #aaa; }}
  </style>
</head>
<body>
  <div id="loading">
    <div class="spinner"></div>
    <span>Loading {title}…</span>
  </div>
  <div id="controls">{controls}</div>
  <a class="back" href="../index.html">← all demos</a>

  <script type="module">
    import init from './helio_web_demos.js';

    try {{
      await init();
    }} catch (e) {{
      document.getElementById('loading').innerHTML =
        '<span style="color:#c44">Failed to load WASM: ' + e + '</span>';
      throw e;
    }}

    // Hide loading overlay once the WASM module has initialised
    const overlay = document.getElementById('loading');
    overlay.classList.add('hidden');
    setTimeout(() => overlay.remove(), 500);

    // Fade controls hint after 5 s
    const ctrl = document.getElementById('controls');
    setTimeout(() => ctrl.classList.add('fade'), 5000);
  </script>
</body>
</html>
"#,
        title = demo.title,
        controls = demo.controls,
    )
}

fn index_html(demos: &[Demo]) -> String {
    let cards: String = demos
        .iter()
        .map(|d| {
            format!(
                r#"    <a class="card" href="{name}/">
      <div class="card-title">{title}</div>
      <div class="card-desc">{description}</div>
    </a>
"#,
                name = d.name,
                title = d.title,
                description = d.description,
            )
        })
        .collect();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Helio — Web Demos</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ background: #0a0a0c; color: #ccc; font-family: system-ui, sans-serif; }}
    body {{ max-width: 960px; margin: 0 auto; padding: 48px 20px; }}
    h1 {{ font-size: 28px; font-weight: 700; color: #eee; margin-bottom: 6px; }}
    .subtitle {{ color: #555; font-size: 14px; margin-bottom: 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
    .card {{
      display: flex; flex-direction: column; gap: 6px;
      background: #111; border: 1px solid #1e1e22;
      border-radius: 8px; padding: 18px 20px;
      text-decoration: none; color: inherit;
      transition: border-color 0.15s, background 0.15s;
    }}
    .card:hover {{ background: #161618; border-color: #3a3a44; }}
    .card-title {{ font-size: 15px; font-weight: 600; color: #ddd; }}
    .card-desc  {{ font-size: 12px; color: #666; line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>Helio Web Demos</h1>
  <p class="subtitle">Real-time rendering demos compiled to WebAssembly.</p>
  <div class="grid">
{cards}  </div>
</body>
</html>
"#,
        cards = cards,
    )
}

// ── main ───────────────────────────────────────────────────────────────────────

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let manifest_dir =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));

    // Derive workspace root (crates/helio-web-demos -> ../../)
    let workspace_root = manifest_dir.join("..").join("..").canonicalize().ok();

    // Generate per-demo HTML files
    for demo in DEMOS {
        let html = demo_html(demo);

        // 1. Write to OUT_DIR  (picked up by include_str! macros in lib.rs)
        let out_path = out_dir.join(format!("{}.html", demo.name));
        fs::write(&out_path, &html).expect("write demo html to OUT_DIR");

        // 2. Side-effect copy to target/wasm-prebuilt/<name>/index.html so
        //    the WASM output directory is ready to serve immediately.
        if let Some(ref root) = workspace_root {
            let prebuilt_dir = root.join("target").join("wasm-prebuilt").join(demo.name);
            if let Err(e) = fs::create_dir_all(&prebuilt_dir) {
                eprintln!(
                    "cargo:warning=could not create {}: {e}",
                    prebuilt_dir.display()
                );
            } else {
                let dest = prebuilt_dir.join("index.html");
                if let Err(e) = fs::write(&dest, &html) {
                    eprintln!("cargo:warning=could not write {}: {e}", dest.display());
                }
            }
        }
    }

    // Generate master index.html
    let index = index_html(DEMOS);
    let out_index = out_dir.join("index.html");
    fs::write(&out_index, &index).expect("write index.html to OUT_DIR");

    if let Some(ref root) = workspace_root {
        let prebuilt_root = root.join("target").join("wasm-prebuilt");
        let _ = fs::create_dir_all(&prebuilt_root);
        if let Err(e) = fs::write(prebuilt_root.join("index.html"), &index) {
            eprintln!("cargo:warning=could not write prebuilt index.html: {e}");
        }
    }
}

