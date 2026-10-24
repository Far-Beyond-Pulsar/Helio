//! Helio's tunable settings, described for hosts.
//!
//! Every setting a host can change (the renderer's graph and quality, the
//! voxel worlds' residency) is listed here once, with its label, page,
//! kind, range and default, so an editor or a config file exposes all of
//! them without wiring each by hand: a host builds its UI from [`GROUPS`],
//! hands the values back through [`Group::resolve`], and the crate that owns
//! a setting reads it from the [`Resolved`] values (`helio::RendererConfig::
//! with_settings`, the voxel pass's `ResidencyConfig::from_settings`).
//!
//! The defaults are the engine's: each owning crate tests that resolving
//! nothing gives its `Default`.

use std::borrow::Cow;

/// A setting's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    /// Integers too ([`Kind::Integer`]).
    Number(f64),
    /// A [`Kind::Choice`]'s value.
    Text(Cow<'static, str>),
    /// Linear RGBA.
    Color([f32; 4]),
}

/// One option of a [`Kind::Choice`].
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub value: &'static str,
    pub label: &'static str,
}

const fn choice(value: &'static str, label: &'static str) -> Choice {
    Choice { value, label }
}

/// What a setting holds, and so the widget that edits it.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Toggle,
    Number { min: f64, max: f64, step: f64 },
    Integer { min: i64, max: i64 },
    Choice(&'static [Choice]),
    Color,
}

/// When a change takes effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Applies {
    /// The next frame.
    Live,
    /// When the renderer's graph is rebuilt (its passes and their GPU
    /// allocations depend on it).
    Rebuild,
}

/// One setting.
#[derive(Clone, Debug)]
pub struct Setting {
    pub key: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    /// The page within the group (Quality, Shadows, ...).
    pub page: &'static str,
    pub kind: Kind,
    pub default: Value,
    pub applies: Applies,
}

/// Settings one owner reads.
#[derive(Debug)]
pub struct Group {
    /// Stable id (a config namespace's owner).
    pub key: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub settings: &'static [Setting],
}

/// Every group, in the order a UI lists them.
pub static GROUPS: &[&Group] = &[&RENDERER, &VOXEL_RESIDENCY];

impl Group {
    pub fn setting(&self, key: &str) -> Option<&'static Setting> {
        self.settings.iter().find(|s| s.key == key)
    }

    /// The group's values: the host's where `lookup` has a valid one for
    /// the setting's kind (numbers clamped into range), the default
    /// otherwise.
    pub fn resolve(&'static self, lookup: impl Fn(&str) -> Option<Value>) -> Resolved {
        let values = self
            .settings
            .iter()
            .map(|s| lookup(s.key).and_then(|v| s.accept(v)).unwrap_or_else(|| s.default.clone()))
            .collect();
        Resolved { group: self, values }
    }

    /// The defaults.
    pub fn defaults(&'static self) -> Resolved {
        self.resolve(|_| None)
    }
}

impl Setting {
    /// `value` as this setting holds it, or `None` if it cannot.
    pub fn accept(&self, value: Value) -> Option<Value> {
        match (self.kind, value) {
            (Kind::Toggle, Value::Bool(b)) => Some(Value::Bool(b)),
            (Kind::Number { min, max, .. }, Value::Number(n)) if n.is_finite() => Some(Value::Number(n.clamp(min, max))),
            (Kind::Integer { min, max }, Value::Number(n)) if n.is_finite() => {
                Some(Value::Number(n.round().clamp(min as f64, max as f64)))
            }
            // Hosts that store numbers as text (dropdowns of sizes).
            (Kind::Integer { .. } | Kind::Number { .. }, Value::Text(t)) => {
                self.accept(Value::Number(t.trim().parse().ok()?))
            }
            (Kind::Choice(options), Value::Text(t)) => {
                options.iter().find(|o| o.value == t).map(|o| Value::Text(Cow::Borrowed(o.value)))
            }
            (Kind::Color, Value::Color(c)) if c.iter().all(|x| x.is_finite()) => Some(Value::Color(c)),
            _ => None,
        }
    }
}

/// A group's values, one per setting.
#[derive(Clone, Debug)]
pub struct Resolved {
    group: &'static Group,
    values: Vec<Value>,
}

impl PartialEq for Resolved {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.group, other.group) && self.values == other.values
    }
}

impl Resolved {
    pub fn group(&self) -> &'static Group {
        self.group
    }

    pub fn get(&self, key: &str) -> &Value {
        let i = self
            .group
            .settings
            .iter()
            .position(|s| s.key == key)
            .unwrap_or_else(|| panic!("no setting `{key}` in {}", self.group.key));
        &self.values[i]
    }

    pub fn bool(&self, key: &str) -> bool {
        match self.get(key) {
            Value::Bool(b) => *b,
            other => panic!("`{key}` is {other:?}, not a toggle"),
        }
    }

    pub fn number(&self, key: &str) -> f64 {
        match self.get(key) {
            Value::Number(n) => *n,
            other => panic!("`{key}` is {other:?}, not a number"),
        }
    }

    /// A number where 0 means "the engine's choice".
    pub fn optional(&self, key: &str) -> Option<f64> {
        Some(self.number(key)).filter(|n| *n > 0.0)
    }

    pub fn text(&self, key: &str) -> &str {
        match self.get(key) {
            Value::Text(t) => t,
            other => panic!("`{key}` is {other:?}, not a choice"),
        }
    }

    pub fn color(&self, key: &str) -> [f32; 4] {
        match self.get(key) {
            Value::Color(c) => *c,
            other => panic!("`{key}` is {other:?}, not a color"),
        }
    }

    /// Whether the values a rebuild reads are the same in both: if not, the
    /// host rebuilds the renderer for `self`.
    pub fn same_graph(&self, other: &Resolved) -> bool {
        std::ptr::eq(self.group, other.group)
            && self
                .group
                .settings
                .iter()
                .zip(self.values.iter().zip(&other.values))
                .all(|(s, (a, b))| s.applies == Applies::Live || a == b)
    }
}

const fn toggle(key: &'static str, label: &'static str, page: &'static str, default: bool, description: &'static str) -> Setting {
    Setting { key, label, description, page, kind: Kind::Toggle, default: Value::Bool(default), applies: Applies::Rebuild }
}

#[allow(clippy::too_many_arguments)]
const fn number(
    key: &'static str,
    label: &'static str,
    page: &'static str,
    (min, max, step): (f64, f64, f64),
    default: f64,
    description: &'static str,
) -> Setting {
    Setting { key, label, description, page, kind: Kind::Number { min, max, step }, default: Value::Number(default), applies: Applies::Rebuild }
}

const fn integer(
    key: &'static str,
    label: &'static str,
    page: &'static str,
    (min, max): (i64, i64),
    default: i64,
    description: &'static str,
) -> Setting {
    Setting { key, label, description, page, kind: Kind::Integer { min, max }, default: Value::Number(default as f64), applies: Applies::Rebuild }
}

const fn pick(
    key: &'static str,
    label: &'static str,
    page: &'static str,
    options: &'static [Choice],
    default: &'static str,
    description: &'static str,
) -> Setting {
    Setting { key, label, description, page, kind: Kind::Choice(options), default: Value::Text(Cow::Borrowed(default)), applies: Applies::Rebuild }
}

const fn live(mut setting: Setting) -> Setting {
    setting.applies = Applies::Live;
    setting
}

const QUALITY_TIERS: &[Choice] = &[choice("low", "Low"), choice("medium", "Medium"), choice("high", "High"), choice("ultra", "Ultra")];

/// The renderer: its graph (`helio::RendererConfig`) and its quality.
pub static RENDERER: Group = Group {
    key: "rendering",
    title: "Rendering",
    description: "Renderer quality, effects and budgets",
    settings: &[
        live(pick(
            "tsr_quality",
            "Temporal Super Resolution",
            "Quality",
            &[
                choice("auto", "Automatic"),
                choice("off", "Off"),
                choice("performance", "Performance"),
                choice("balanced", "Balanced"),
                choice("quality", "Quality"),
                choice("native", "Native"),
            ],
            "auto",
            "Temporal upscaling preset; its preset sets the render scale. Automatic: the scene's \
             (a voxel world's by resolution), off when it has none",
        )),
        live(number(
            "render_scale",
            "Render Scale",
            "Quality",
            (0.25, 1.0, 0.05),
            0.75,
            "Internal resolution while temporal super resolution is off",
        )),
        pick(
            "render_mode",
            "Render Mode",
            "Quality",
            &[choice("deferred", "Deferred"), choice("forward_opaque", "Forward Opaque"), choice("forward_only", "Forward Only")],
            "deferred",
            "Renderer path",
        ),
        pick("shadow_quality", "Shadow Quality", "Shadows", QUALITY_TIERS, "medium", "Shadow filtering quality"),
        number(
            "shadow_memory_mb",
            "Shadow Memory (MB)",
            "Shadows",
            (16.0, 2048.0, 16.0),
            160.0,
            "GPU memory of the shadow atlases; sets their size",
        ),
        pick(
            "shadow_max_resolution",
            "Shadow Max Resolution",
            "Shadows",
            &[choice("512", "512"), choice("1024", "1024"), choice("2048", "2048"), choice("4096", "4096"), choice("8192", "8192")],
            "2048",
            "Largest shadow tile a light gets",
        ),
        integer(
            "shadow_updates_per_frame",
            "Shadow Updates per Frame",
            "Shadows",
            (1, 1024),
            32,
            "Shadow faces redrawn a frame at most",
        ),
        number(
            "shadow_update_megatexels",
            "Shadow Update Texels (M)",
            "Shadows",
            (0.25, 64.0, 0.25),
            4.0,
            "Millions of shadow texels redrawn a frame at most",
        ),
        number(
            "shadow_max_distance",
            "Shadow Distance",
            "Shadows",
            (10.0, 100_000.0, 10.0),
            500.0,
            "Metres from the eye that local lights cast shadows",
        ),
        integer("shadow_fade_frames", "Shadow Fade Frames", "Shadows", (0, 240), 12, "Frames a shadow fades in or out over"),
        number(
            "shadow_hysteresis",
            "Shadow Hysteresis",
            "Shadows",
            (0.0, 1.0, 0.05),
            0.2,
            "How much a light's shadow resolution must change before its tile is reallocated",
        ),
        toggle(
            "shadow_page_invalidation",
            "Shadow Page Invalidation",
            "Shadows",
            true,
            "Redraw only the shadow pages a moving caster touches",
        ),
        toggle(
            "environment_reflections",
            "Environment Reflections",
            "Reflections",
            true,
            "Environment cubemap specular, the base reflection layer (metals are black without it)",
        ),
        toggle("screen_space_reflections", "Screen Space Reflections", "Reflections", false, "Screen-space reflections"),
        toggle("planar_reflections", "Planar Reflections", "Reflections", false, "Render authored planar reflection surfaces"),
        number(
            "gi_probe_spacing",
            "GI Probe Spacing",
            "Global Illumination",
            (0.0, 8.0, 0.25),
            0.0,
            "Metres between the finest GI probes (each coarser level doubles it); 0: the renderer's choice \
             (1 m over voxel terrain, 0.5 m in mesh scenes)",
        ),
        toggle("foliage", "Foliage", "Foliage", true, "Draw scattered foliage (grass, flowers)"),
        pick(
            "foliage_quality",
            "Foliage Quality",
            "Foliage",
            QUALITY_TIERS,
            "medium",
            "Foliage range, density and LOD distances",
        ),
        number(
            "foliage_range",
            "Foliage Range",
            "Foliage",
            (0.0, 4000.0, 10.0),
            0.0,
            "Metres foliage reaches; 0: the quality's",
        ),
        number(
            "foliage_tufts_per_m2",
            "Foliage Max Density",
            "Foliage",
            (0.0, 64.0, 0.5),
            0.0,
            "Densest foliage in tufts a square metre (denser types are thinned); 0: 8",
        ),
        number(
            "foliage_blades_per_tuft",
            "Foliage Blades per Tuft",
            "Foliage",
            (0.0, 64.0, 1.0),
            0.0,
            "Blades a tuft averages, sizing the near band; 0: 8",
        ),
        pick(
            "hdr_output_mode",
            "HDR Output",
            "Output",
            &[choice("ldr", "SDR"), choice("hdr10", "HDR10"), choice("scrgb", "scRGB"), choice("passthrough", "Passthrough")],
            "ldr",
            "Display output encoding",
        ),
        Setting {
            key: "background_color",
            label: "Background Color",
            description: "Colour behind the scene where no sky is drawn",
            page: "Output",
            kind: Kind::Color,
            default: Value::Color([0.0, 0.0, 0.0, 1.0]),
            applies: Applies::Rebuild,
        },
        toggle("portals", "Portals", "Advanced", true, "Portal passes (about 10 MB of GPU memory)"),
        pick(
            "perf_overlay",
            "Performance Overlay",
            "Advanced",
            &[
                choice("disabled", "Off"),
                choice("pass_overdraw", "Pass Overdraw"),
                choice("shader_complexity", "Shader Complexity"),
                choice("tile_light_count", "Tile Light Count"),
                choice("pass_output", "Pass Output"),
            ],
            "disabled",
            "Overlay drawn over the image",
        ),
    ],
};

/// Voxel worlds' residency: which chunks are generated and kept, and the
/// time generation gets (`helio_pass_voxel_planet::chunks::ResidencyConfig`).
pub static VOXEL_RESIDENCY: Group = Group {
    key: "voxels",
    title: "Voxels",
    description: "Voxel world detail, streaming and generation budgets",
    settings: &[
        number(
            "lod_pixels",
            "Detail (pixels a cell)",
            "Detail",
            (0.25, 16.0, 0.25),
            1.0,
            "Pixels a voxel may cover on screen before a finer level is wanted",
        ),
        number(
            "offscreen_pixels",
            "Off-screen Detail",
            "Detail",
            (0.25, 64.0, 0.25),
            2.0,
            "Pixels a cell may cover out of view",
        ),
        number(
            "unseen_pixels",
            "Hidden Detail",
            "Detail",
            (1.0, 64.0, 0.25),
            2.0,
            "Pixels a cell may cover where no ray met it lately (1: rays decide nothing)",
        ),
        number("near_cells", "Near Cells", "Detail", (0.0, 16.0, 0.5), 2.0, "Cells around the eye kept at full detail"),
        number(
            "coarsen",
            "Coarsen Threshold",
            "Detail",
            (0.1, 1.0, 0.05),
            0.8,
            "Fraction of the detail target below which a level coarsens (hysteresis)",
        ),
        number("view_guard", "View Guard", "Detail", (0.0, 1.5, 0.05), 0.35, "Radians past the view's edge still counted in view"),
        number(
            "view_distance_km",
            "View Distance (km)",
            "Detail",
            (1.0, 100_000.0, 1.0),
            400.0,
            "Kilometres chunks are planned out to",
        ),
        number(
            "motion_flow",
            "Motion Flow",
            "Motion",
            (0.0, 256.0, 1.0),
            16.0,
            "Pixels a frame of motion per extra pixel a cell may cover (0: motion changes nothing)",
        ),
        number("motion_max", "Motion Max", "Motion", (1.0, 16.0, 0.25), 3.0, "Most a cell's pixels grow with motion"),
        number(
            "prefetch_seconds",
            "Prefetch",
            "Motion",
            (0.0, 8.0, 0.1),
            0.5,
            "Seconds ahead along the motion that chunks are planned",
        ),
        integer("seen_frames", "Seen Frames", "Streaming", (1, 10_000), 90, "Frames a ray's visit counts"),
        integer(
            "keep_frames",
            "Keep Frames",
            "Streaming",
            (1, 100_000),
            120,
            "Frames a chunk out of the plan stays before it may be replaced",
        ),
        integer(
            "release_frames",
            "Release Frames",
            "Streaming",
            (1, 100_000),
            600,
            "Frames unvisited before a chunk's memory is released",
        ),
        integer(
            "footprints_per_frame",
            "Footprints per Frame",
            "Generation",
            (1, 65_536),
            256,
            "Chunk footprints generated a frame at most",
        ),
        number(
            "generate_ms",
            "Generation Budget (ms)",
            "Generation",
            (0.0, 100.0, 0.25),
            3.0,
            "GPU milliseconds of generation a frame at 60 fps; 0: unbounded",
        ),
        number(
            "frame_ms",
            "Frame Budget (ms)",
            "Generation",
            (0.0, 1000.0, 0.5),
            16.0,
            "Milliseconds a frame may take: generation grows into what the frame leaves; 0: no growth",
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_accepted_by_their_settings() {
        for group in GROUPS {
            for s in group.settings {
                assert_eq!(s.accept(s.default.clone()).as_ref(), Some(&s.default), "{}.{}", group.key, s.key);
            }
            let keys: std::collections::HashSet<_> = group.settings.iter().map(|s| s.key).collect();
            assert_eq!(keys.len(), group.settings.len(), "duplicate key in {}", group.key);
        }
    }

    #[test]
    fn host_values_are_clamped_parsed_or_dropped() {
        let r = RENDERER.resolve(|key| match key {
            "render_scale" => Some(Value::Number(4.0)),
            "shadow_updates_per_frame" => Some(Value::Text("64".into())),
            "shadow_quality" => Some(Value::Text("bogus".into())),
            "foliage" => Some(Value::Number(1.0)),
            _ => None,
        });
        assert_eq!(r.number("render_scale"), 1.0);
        assert_eq!(r.number("shadow_updates_per_frame"), 64.0);
        assert_eq!(r.text("shadow_quality"), "medium");
        assert!(r.bool("foliage"));
    }

    #[test]
    fn live_settings_do_not_rebuild() {
        let a = RENDERER.defaults();
        let b = RENDERER.resolve(|key| (key == "render_scale").then_some(Value::Number(0.5)));
        let c = RENDERER.resolve(|key| (key == "foliage").then_some(Value::Bool(false)));
        assert!(a.same_graph(&b));
        assert!(!a.same_graph(&c));
    }
}
