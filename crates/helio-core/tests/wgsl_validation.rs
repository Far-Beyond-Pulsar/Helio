//! Parses and validates every `.wgsl` shader in the workspace.
//!
//! wgpu only compiles a shader at `create_shader_module`, i.e. at runtime with a
//! live device. Nothing in `cargo check`, `cargo build`, or `cargo test` looks at
//! WGSL, so a shader that cannot possibly compile ships silently and only fails
//! on the machine that runs it — or never, if the pass is not wired up.
//!
//! That is not hypothetical: `ssr_denoise.wgsl` sat in-tree unable to compile
//! (it negated a `u32`) for as long as it existed, because `SsrPass` never built
//! a pipeline from it.
//!
//! This test walks the repo rather than taking an explicit list, so a new shader
//! is covered the moment it is added.
//!
//! Sources go through `helio_core::shader::resolve_with` first — the same call
//! (modulo which snippets a given pass registers) the runtime makes — so a
//! prelude/snippet-using shader is validated as the GPU will see it, not as
//! the bare file on disk. The snippet list here (HIZ, PBR, wind) is this
//! test's own aggregation of every pass-owned snippet in the workspace, not
//! something `helio-core` itself knows about — `resolve_with` never fails
//! for a marker with no matching snippet, so listing all of them here is
//! harmless for shaders that only use a subset.

use std::path::{Path, PathBuf};

const SNIPPETS: &[helio_core::shader::ShaderSnippet] = &[
    helio_pass_hiz::HIZ_SNIPPET,
    helio_mats::PBR_EVAL_SNIPPET,
    helio_mats::SPECULAR_AA_SNIPPET,
    helio_pass_foliage_place::WIND_SNIPPET,
    helio_pass_foliage_place::COMMON_SNIPPET,
    helio_pass_sky::atmosphere::ATMOSPHERE_SNIPPET,
    helio_pass_sky::atmosphere::ATMOSPHERE_SKY_TABLE_SNIPPET,
    helio_pass_sky::clouds::CLOUD_SHADOW_SNIPPET,
    helio_pass_sky::clouds::CLOUDS_SNIPPET,
    helio_pass_gi::GI_COMMON_SNIPPET,
    helio_pass_gi::GI_SAMPLE_SNIPPET,
    helio_pass_shadow_matrix::SUN_OCCLUSION_SNIPPET,
    helio_pass_water_sim::pipeline::WATER_SHORE_SNIPPET,
    helio_pass_water_sim::pipeline::WATER_SKY_SNIPPET,
    helio_pass_water_sim::surface_query::WATER_QUERY_SNIPPET,
];

use naga::valid::{Capabilities, ValidationFlags, Validator};

fn workspace_root() -> PathBuf {
    // crates/helio-core -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("helio-core should live at <root>/crates/helio-core")
        .to_path_buf()
}

fn collect_wgsl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `target` holds vendored deps' shaders, which are not ours to police.
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            collect_wgsl(&path, out);
        } else if path.extension().is_some_and(|e| e == "wgsl") {
            out.push(path);
        }
    }
}

/// A standalone shader has at least one entry point. A file with none is a
/// fragment that gets string-concatenated into a host shader before compiling
/// (e.g. `vhs_effects.wgsl`, pulled in as `VHS_SHADER_SNIPPET`), so it refers to
/// bindings it does not declare and cannot validate on its own.
fn is_fragment(source: &str) -> bool {
    !(source.contains("@vertex") || source.contains("@fragment") || source.contains("@compute"))
}

/// An entry-point-bearing piece its crate assembles with shared declarations
/// (or a terrain program, or a row stride): validated as composed, by
/// `composed_shaders_parse_and_validate` or by its pass, not on its own.
/// Keep this implementation detail in the validator rather than adding
/// test-only comments to production shader sources.
fn is_composed(path: &Path) -> bool {
    const COMPOSED: &[&str] = &[
        "helio-pass-hlfs/shaders/composite.wgsl",
        "helio-pass-hlfs/shaders/light_grid.wgsl",
        "helio-pass-hlfs/shaders/sample.wgsl",
        "helio-pass-hlfs/shaders/spatial.wgsl",
        "helio-pass-hlfs/shaders/temporal.wgsl",
        "helio-pass-transparent/shaders/transparent_base.wgsl",
        // Assembled by `helio-pass-voxel-planet`'s `chunks::shader_sources`
        // (a terrain program generated in Rust, the noise library, walk
        // templates); validated when the pass builds its pipelines.
        "helio-pass-voxel-planet/shaders/chunk_generate.wgsl",
        "helio-pass-voxel-planet/shaders/chunk_draw.wgsl",
        // The virtual geometry's kernels, assembled by their crates from
        // shared declaration pieces (`include_str!` constants joined in
        // Rust); validated when their passes build their pipelines (the
        // crates' GPU tests).
        "helio-component/src/virtual_geometry_instances.wgsl",
        "helio-pass-vg-bvh/shaders/vg_bvh_build.wgsl",
        "helio-pass-vg-cull/shaders/vg_cull.wgsl",
        "helio-pass-vg-dense/shaders/args.wgsl",
        "helio-pass-vg-dense/shaders/bin.wgsl",
        "helio-pass-vg-dense/shaders/collect.wgsl",
        "helio-pass-vg-dense/shaders/footprint.wgsl",
        "helio-pass-vg-dense/shaders/peel.wgsl",
        "helio-pass-vg-dense/shaders/rt.wgsl",
        "helio-pass-vg-dense/shaders/tiles.wgsl",
        "helio-pass-vg-raster/shaders/vg_dense_count.wgsl",
        "helio-pass-vg-raster/shaders/vg_hw_raster.wgsl",
        "helio-pass-vg-raster/shaders/vg_hw_raster_masked.wgsl",
        "helio-pass-vg-raster/shaders/vg_sw_bin.wgsl",
        "helio-pass-vg-raster/shaders/vg_sw_raster.wgsl",
        "helio-pass-vg-raster/shaders/vg_sw_raster_masked.wgsl",
        "helio-pass-vg-raster/shaders/vg_tile_raster.wgsl",
        "helio-pass-vg-shade/shaders/vg_debug.wgsl",
        "helio-pass-vg-shade/shaders/vg_resolve.wgsl",
        "helio-pass-vg-shade/shaders/vg_shade.wgsl",
        // Prefixed with the cut-outs' alpha evaluator (`helio_mats::vg_alpha`,
        // in the device's material binding mode); validated composed, both
        // modes, by `helio-pass-gi`'s `shaders_validate`.
        "helio-pass-gi/shaders/gi_mesh_trace.wgsl",
        // The water surface's query kernel, a snippet of `surface.wgsl`
        // (`//!use water_query`): validated composed with it.
        "helio-pass-water-sim/shaders/water_query.wgsl",
    ];
    let normalized = path.to_string_lossy().replace('\\', "/");
    COMPOSED.iter().any(|suffix| normalized.ends_with(suffix))
}

/// Assembles the shaders their passes build from more than one piece, the
/// same way the pass does, so they are validated as the GPU sees them.
fn compose(root: &Path, path: &Path, source: String) -> String {
    let normalized = path.to_string_lossy().replace('\\', "/");
    // `SceneJoin` and `EnvironmentJoin` prefix the shared join declarations.
    const JOINS: &[&str] = &[
        "helio-default-passes/shaders/scene_join_meshes.wgsl",
        "helio-default-passes/shaders/scene_join_lights.wgsl",
        "helio-default-passes/shaders/environment_join.wgsl",
        "helio-default-passes/shaders/hierarchy_resolve.wgsl",
        "helio-default-passes/shaders/scene_join_moved.wgsl",
    ];
    if JOINS.iter().any(|suffix| normalized.ends_with(suffix)) {
        let common = root.join("crates/helio-default-passes/shaders/scene_join_common.wgsl");
        let common = std::fs::read_to_string(common).expect("scene_join_common.wgsl");
        return format!("{common}\n{source}");
    }
    // The depth passes prefix the alpha-tested depth material's declarations
    // (`helio_pass_gbuffer::depth_material::DepthMaterialBindings::shader_source`).
    const DEPTH_MATERIAL: &[&str] = &[
        "helio-pass-depth-prepass/shaders/depth_prepass.wgsl",
        "helio-pass-shadow/shaders/shadow.wgsl",
    ];
    if DEPTH_MATERIAL
        .iter()
        .any(|suffix| normalized.ends_with(suffix))
    {
        let material = root.join("crates/passes/3d/helio-pass-gbuffer/shaders/depth_material.wgsl");
        let material = std::fs::read_to_string(material).expect("depth_material.wgsl");
        return format!(
            "{material}
{source}"
        );
    }
    // `helio_pass_volumetric_fog::shader_source` sizes the post-process row's
    // words around its fog block from its Rust ABI; any lengths validate the
    // same.
    if normalized.ends_with("helio-pass-volumetric-fog/shaders/volumetric_fog.wgsl") {
        return source
            .replace("__PP_HEAD_VEC4__", "1")
            .replace("__PP_TAIL_VEC4__", "1");
    }
    source
}

/// Maps a line number in resolved source back to the original file, so a
/// diagnostic in a prelude-using shader points at a line that actually exists in
/// the file the reader will open.
fn describe_location(source: &str, rendered: &str) -> String {
    let prepended = helio_core::shader::expanded_lines_with(source, SNIPPETS);
    if prepended > 0 {
        format!(" (expanded: subtract {prepended} lines for the original file)")
    } else {
        String::new()
    }
    .to_string()
        + "\n"
        + rendered
}

#[test]
fn every_wgsl_shader_parses_and_validates() {
    let root = workspace_root();
    let mut shaders = Vec::new();
    collect_wgsl(&root.join("crates"), &mut shaders);
    shaders.sort();

    assert!(
        !shaders.is_empty(),
        "found no .wgsl files under {}; the walk is probably broken, and a test \
         that silently checks nothing is worse than no test",
        root.display()
    );

    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut skipped = Vec::new();

    for path in &shaders {
        let rel = path.strip_prefix(&root).unwrap_or(path);
        let source = std::fs::read_to_string(path).expect("shader should be readable");
        let source = compose(&root, path, source);

        if is_composed(path) {
            continue;
        }
        if is_fragment(&source) {
            skipped.push(rel.display().to_string());
            continue;
        }
        checked += 1;

        // Exactly what create_shader_module would receive.
        let resolved = helio_core::shader::resolve_with(&source, SNIPPETS);

        let module = match naga::front::wgsl::parse_str(&resolved) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!(
                    "{}:{}",
                    rel.display(),
                    describe_location(&source, &e.emit_to_string(&resolved))
                ));
                continue;
            }
        };

        if let Err(e) =
            Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)
        {
            failures.push(format!(
                "{}:{}",
                rel.display(),
                describe_location(&source, &format!("{e:?}"))
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {checked} standalone shaders failed to compile:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );

    // Guard the skip heuristic: fragments are deliberate source modules and
    // composed shaders. If this trips, the source-module set has grown enough
    // to require an explicit audit. Last audited at 56 (2026-10-28): Radiant
    // templates, shared prelude/common pieces, the voxel planet's source
    // pieces (volumes.wgsl for caves and cliffs, planet_maps.wgsl for the
    // baked maps, joined them), and the virtual geometry's declaration pieces
    // (vg-format, vg-stream, vg-bvh, vg-dense, vg-raster, vg-shade,
    // helio-mats' vg_alpha). 58 (2026-10-30): Ultra-Water's snippets
    // (water-sim's water_sky.wgsl and water_surface_shared.wgsl).
    assert!(
        skipped.len() <= 58,
        "{} shaders were skipped as fragments/composed modules, which is more than expected:\n{}",
        skipped.len(),
        skipped.join("\n")
    );
}

/// Shaders their crates assemble from pieces (shared declarations, the
/// terrain program, row strides), validated exactly as each crate compiles
/// them.
#[test]
fn composed_shaders_parse_and_validate() {
    let mut sources = helio_pass_voxel_planet::engine::validation_sources();
    sources.extend(helio_default_passes::scene_join::validation_sources());
    sources.push((
        "volumetric_fog.wgsl".into(),
        helio_pass_volumetric_fog::shader_source(),
    ));
    let failures: Vec<String> = sources
        .iter()
        .filter_map(|(name, source)| {
            // The fog's source holds its snippets already (`shader_source`):
            // its pass resolves the prelude alone.
            let source = if name == "volumetric_fog.wgsl" {
                helio_core::shader::resolve(source).into_owned()
            } else {
                helio_core::shader::resolve_with(source, SNIPPETS).into_owned()
            };
            let module = match naga::front::wgsl::parse_str(&source) {
                Ok(module) => module,
                Err(e) => {
                    return Some(format!(
                        "{name}:
{}",
                        e.emit_to_string(&source)
                    ))
                }
            };
            Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .err()
                .map(|e| format!("{name}: {e:?}"))
        })
        .collect();
    assert!(sources.len() >= 10, "{} composed shaders", sources.len());
    assert!(
        failures.is_empty(),
        "{} composed shaders failed:

{}",
        failures.len(),
        failures.join(
            "

"
        )
    );
}
